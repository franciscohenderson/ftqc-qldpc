"""cffi (ABI mode) binding of the C++20 fast-tier decoder (qdecoder.dll / .so) + build helper."""
from __future__ import annotations

import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from cffi import FFI

_DIR = Path(__file__).parent
_SRC = _DIR / "cpp" / "qdecoder.cpp"
_LIB = _DIR / ("qdecoder.dll" if sys.platform == "win32" else "qdecoder.so")

_CDEF = """
void* qd_create(int m, int n, const int* row_ptr, const int* col_idx, const double* prior,
                int max_iter, double llr_scale, int osd_lambda);
void qd_destroy(void* h);
int qd_lanes(void);
int qd_osd_external(void* h, const uint8_t* synd, const int16_t* post, uint8_t* out);
int qd_channel_llr(void* h, int16_t* out);
void qd_decode_batch(void* h, int shots, const uint8_t* synd, uint8_t* out, int* iters,
                     uint8_t* status, uint64_t* latency_ns, int use_osd);
void qd_decode_stream(void* h, int shots, const uint8_t* synd, uint8_t* out, int* iters,
                      uint8_t* status, uint64_t* bp_ns, uint64_t* total_ns, int use_osd);
void qd_bp_posterior(void* h, const uint8_t* synd, int16_t* post, int* iters, uint8_t* status);
typedef struct { void* dec; int nrows; const int* rows; int ngroups; const int* groups; const int* gcols; int last; } QdWin;
void qd_window_decode(const QdWin* wins, int nwin, int B, int nDet, int nMech, const int* hc_colptr, const int* hc_rowidx,
                      uint8_t* syn, uint8_t* err, uint8_t* failed, uint8_t* shot_status,
                      uint64_t* bp_ns, uint64_t* post_ns, uint64_t* prop_ns, int64_t* counts);
"""
_ffi = FFI()
_ffi.cdef(_CDEF)
_lib = None


def build(lanes: int | None = None, force: bool = False) -> Path:
    """Compile the shared library with `zig c++` (pip install ziglang) or clang++/g++ if present."""
    if _LIB.exists() and not force and not lanes and _LIB.stat().st_mtime >= _SRC.stat().st_mtime:
        return _LIB
    flags = ["-std=c++20", "-O3", "-march=native", "-DNDEBUG", "-shared"]
    if sys.platform != "win32":
        flags.append("-fPIC")
    if lanes:
        flags.append(f"-DQD_LANES={lanes}")
    cmds = [[sys.executable, "-m", "ziglang", "c++"], ["clang++"]]
    err = ""
    for base in cmds:
        r = subprocess.run(base + flags + ["-o", str(_LIB), str(_SRC)], capture_output=True, text=True)
        if r.returncode == 0 and _LIB.exists():
            return _LIB
        err = r.stderr[-2000:]
    raise RuntimeError(f"could not build decoder: {err}")


def _load():
    global _lib
    if _lib is None:
        build()
        _lib = _ffi.dlopen(str(_LIB))
    return _lib


STATUS = {0: "bp_converged", 1: "bp_failed", 2: "osd_ok", 3: "osd_failed"}


@dataclass
class DecodeResult:
    errors: np.ndarray        # (shots, n) uint8
    iters: np.ndarray         # (shots,) BP iterations used
    status: np.ndarray        # (shots,) see STATUS
    latency_ns: np.ndarray    # (shots,) latency (amortized for batch, per-shot for stream)
    bp_ns: np.ndarray | None = None


class FastDecoder:
    """Normalized min-sum BP (int16 fixed point, 16 shots per SIMD pass) + OSD-CS fallback."""

    def __init__(self, h, priors, max_iter: int = 20, llr_scale: float = 16.0, osd_lambda: int = 32):
        h = sp.csr_matrix(h)
        h.sum_duplicates()
        h.sort_indices()
        self.m, self.n = h.shape
        self._rp = np.ascontiguousarray(h.indptr, dtype=np.int32)
        self._ci = np.ascontiguousarray(h.indices, dtype=np.int32)
        self._pr = np.ascontiguousarray(np.broadcast_to(priors, (self.n,)), dtype=np.float64)
        self.lib = _load()
        self._lock = threading.Lock()   # the C++ scratch buffers are per-instance, not thread-safe
        self.lanes = self.lib.qd_lanes()
        self._h = self.lib.qd_create(self.m, self.n, _ffi.cast("int*", _ffi.from_buffer(self._rp)),
                                     _ffi.cast("int*", _ffi.from_buffer(self._ci)),
                                     _ffi.cast("double*", _ffi.from_buffer(self._pr)),
                                     max_iter, llr_scale, osd_lambda)
        self.H = h

    def __del__(self):
        try:
            self.lib.qd_destroy(self._h)
        except Exception:
            pass

    def _prep(self, synd):
        s = np.ascontiguousarray(np.atleast_2d(synd), dtype=np.uint8)
        if s.shape[1] != self.m:
            raise ValueError(f"syndrome length {s.shape[1]} != {self.m}")
        return s

    def decode_batch(self, synd, use_osd: bool = True) -> DecodeResult:
        s = self._prep(synd)
        B = s.shape[0]
        out = np.zeros((B, self.n), np.uint8)
        it = np.zeros(B, np.int32)
        st = np.zeros(B, np.uint8)
        lat = np.zeros(B, np.uint64)
        with self._lock:
          self.lib.qd_decode_batch(self._h, B, _ffi.from_buffer("uint8_t[]", s), _ffi.from_buffer("uint8_t[]", out),
                                 _ffi.from_buffer("int[]", it), _ffi.from_buffer("uint8_t[]", st),
                                 _ffi.from_buffer("uint64_t[]", lat), int(use_osd))
        return DecodeResult(out, it, st, lat)

    def decode_stream(self, synd, use_osd: bool = True) -> DecodeResult:
        """Decode shot-by-shot (lane 0 only) with per-shot timing taken inside C++."""
        s = self._prep(synd)
        B = s.shape[0]
        out = np.zeros((B, self.n), np.uint8)
        it = np.zeros(B, np.int32)
        st = np.zeros(B, np.uint8)
        bp = np.zeros(B, np.uint64)
        tot = np.zeros(B, np.uint64)
        with self._lock:
          self.lib.qd_decode_stream(self._h, B, _ffi.from_buffer("uint8_t[]", s), _ffi.from_buffer("uint8_t[]", out),
                                  _ffi.from_buffer("int[]", it), _ffi.from_buffer("uint8_t[]", st),
                                  _ffi.from_buffer("uint64_t[]", bp), _ffi.from_buffer("uint64_t[]", tot), int(use_osd))
        return DecodeResult(out, it, st, tot, bp)

    def bp_posterior(self, synd) -> tuple[np.ndarray, int, int]:
        """BP only on one syndrome: (final int16 posterior LLRs, iterations, status 0 converged / 1 not)."""
        s = np.ascontiguousarray(synd, dtype=np.uint8)
        if s.shape != (self.m,):
            raise ValueError(f"syndrome length {s.shape} != ({self.m},)")
        post = np.zeros(self.n, np.int16)
        it = np.zeros(1, np.int32)
        st = np.zeros(1, np.uint8)
        with self._lock:
            self.lib.qd_bp_posterior(self._h, _ffi.from_buffer("uint8_t[]", s), _ffi.from_buffer("int16_t[]", post),
                                     _ffi.from_buffer("int[]", it), _ffi.from_buffer("uint8_t[]", st))
        return post, int(it[0]), int(st[0])

    def channel_llr(self) -> np.ndarray:
        out = np.zeros(self.n, np.int16)
        self.lib.qd_channel_llr(self._h, _ffi.from_buffer("int16_t[]", out))
        return out

    def osd_external(self, synd, post_llr) -> np.ndarray | None:
        """OSD-CS with caller-supplied int16 posterior LLRs (lowest = most likely in error)."""
        s = np.ascontiguousarray(synd, dtype=np.uint8)
        post = np.ascontiguousarray(post_llr, dtype=np.int16)
        out = np.zeros(self.n, np.uint8)
        with self._lock:
          ok = self.lib.qd_osd_external(self._h, _ffi.from_buffer("uint8_t[]", s), _ffi.from_buffer("int16_t[]", post),
                                      _ffi.from_buffer("uint8_t[]", out))
        return out if ok else None

    def syndrome_ok(self, errors: np.ndarray, synd: np.ndarray) -> np.ndarray:
        return ((self.H @ errors.T.astype(np.int64)) % 2 == synd.T).all(axis=0)
