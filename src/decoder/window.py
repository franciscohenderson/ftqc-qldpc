"""Sliding-window (W, F) decoding of a circuit-level detector error model.

The spacetime check matrix of a memory experiment is partitioned by the measurement-round coordinate
of every detector.  Window k looks at the detector rounds [s, e) with s = k*F and e = min(s + W, L)
(L = number of detector layers).  It contains the DEM mechanisms whose *earliest* detector lies in the
window; detectors beyond e are truncated (open future boundary).  Truncation can make different
mechanisms identical inside the window, so identical columns are merged (priors combined by XOR
probability) before decoding; the decision of a merged column is attributed to its most probable member.
The window is solved with the C++ SIMD BP + OSD-CS kernel and the mechanisms of the commit region
[s, s + F) (all remaining mechanisms in the last window) are committed, but only if the window decoder
produced a syndrome-consistent answer.  Their full effect, including on detectors of later rounds, is
XORed into the running syndrome, which is what the next window decodes: corrections propagate across
the window boundary.

Decoding cost per step depends on W, not on the number of rounds of the experiment, and the only latency
that adds to the end of the experiment is that of the last window.  A window needs at least one round of
look-ahead (F < W) unless a single window covers the whole experiment.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .dem import DecodingProblem
from .fast import FastDecoder


def detector_rounds(circuit) -> np.ndarray:
    """Round index (second detector coordinate) of every detector of a Stim circuit."""
    coords = circuit.get_detector_coordinates()
    out = np.zeros(len(coords), dtype=np.int64)
    for d, c in coords.items():
        out[d] = int(round(c[1]))
    return out


@dataclass
class Window:
    s: int
    e: int
    last: bool
    rows: np.ndarray            # detector ids inside the window
    cols: np.ndarray            # mechanism ids inside the window
    n_unique: int               # decoder columns after merging identical (truncated) columns
    commit_groups: np.ndarray   # decoder columns whose decision is committed by this window
    commit_cols: np.ndarray     # global mechanism id attributed to each committed decoder column
    decoder: FastDecoder

    @property
    def shape(self) -> tuple[int, int]:
        """(detectors, decoder columns) of the problem the window actually solves."""
        return len(self.rows), self.n_unique


@dataclass
class WindowResult:
    errors: np.ndarray                 # (B, n_mechanisms) committed correction
    failed: np.ndarray                 # (B,) some window could not satisfy its syndrome
    residual: np.ndarray               # (B,) syndrome weight left after the last window (should be 0)
    shot_status: np.ndarray            # (B,) worst window status of each shot: 0 BP, 2 OSD, 3 failed
    latency_ns: np.ndarray | None = None          # (n_windows, B) batch-amortized per-shot time of each window step
    single_ns: np.ndarray | None = None           # (n_windows, B') single-shot time of each window step (first B' shots)
    status_counts: np.ndarray | None = None       # window decodes by status (summed over windows and shots)


@dataclass
class FusedResult:
    errors: np.ndarray                 # (B, n_mechanisms) committed correction
    failed: np.ndarray                 # (B,)
    residual: np.ndarray               # (B,) syndrome weight left after the last window
    shot_status: np.ndarray            # (B,) worst window status of each shot
    bp_ns: np.ndarray                  # (n_windows,) BP time per shot, batch-amortized
    post_ns: np.ndarray                # (n_windows,) OSD-CS time per shot, batch-amortized
    prop_ns: np.ndarray                # (n_windows,) propagation time per shot, batch-amortized
    status_counts: np.ndarray          # window decodes by status


class WindowDecoder:
    def __init__(self, prob: DecodingProblem, det_round: np.ndarray, W: int = 3, F: int = 1,
                 max_iter: int = 30, osd_lambda: int = 32):
        if not (1 <= F <= W):
            raise ValueError("need 1 <= F <= W")
        self.prob, self.W, self.F = prob, int(W), int(F)
        H = sp.csc_matrix(prob.h)
        self.H = H
        self.det_round = np.asarray(det_round)
        if H.shape[0] == 0 or len(self.det_round) != H.shape[0]:
            raise ValueError("det_round must have one entry per detector (and the problem must have detectors)")
        self.L = int(self.det_round.max()) + 1
        if self.F == self.W and self.W < self.L:
            raise ValueError("F must be smaller than W (at least one round of look-ahead) unless one window covers all layers")
        rr = self.det_round[H.indices]
        first = np.full(H.shape[1], self.L, dtype=np.int64)
        for j in range(H.shape[1]):
            seg = rr[H.indptr[j]:H.indptr[j + 1]]
            first[j] = seg.min() if len(seg) else self.L
        self.first = first
        self.windows: list[Window] = []
        s = 0
        while True:
            e = min(s + self.W, self.L)
            last = e >= self.L
            rows = np.flatnonzero((self.det_round >= s) & (self.det_round < e))
            cols = np.flatnonzero((first >= s) & (first < e))
            sub = sp.csc_matrix(H[rows][:, cols])
            groups: dict[bytes, list[int]] = {}
            for j in range(sub.shape[1]):
                groups.setdefault(sub.indices[sub.indptr[j]:sub.indptr[j + 1]].tobytes(), []).append(j)
            members = list(groups.values())
            pri = prob.priors[cols]
            rep = np.array([m[int(np.argmax(pri[m]))] for m in members], dtype=np.int64)
            merged = np.empty(len(members))
            for g, m in enumerate(members):
                p = 0.0
                for q in pri[m]:
                    p = p * (1 - q) + q * (1 - p)
                merged[g] = p
            dec = FastDecoder(sp.csr_matrix(sub[:, rep]), merged, max_iter=max_iter, osd_lambda=osd_lambda)
            commit_mask = np.ones(len(rep), bool) if last else first[cols[rep]] < s + F
            cg = np.flatnonzero(commit_mask)
            self.windows.append(Window(s, e, last, rows, cols, len(members), cg, cols[rep[cg]], dec))
            if last:
                break
            s += self.F
        self.Hc = sp.csr_matrix(H)

    @property
    def n_windows(self) -> int:
        return len(self.windows)

    def plan(self) -> dict:
        """Static summary used by the API: window geometry and sub-DEM sizes next to the full problem."""
        return {"W": self.W, "F": self.F, "detector_layers": self.L, "n_windows": self.n_windows,
                "full": {"detectors": int(self.H.shape[0]), "mechanisms": int(self.H.shape[1])},
                "windows": [{"rounds": [w.s, w.e], "detectors": int(len(w.rows)), "mechanisms": int(len(w.cols)),
                             "columns_after_merge": int(w.n_unique), "commits": int(len(w.commit_cols))} for w in self.windows]}

    def decode_batch(self, det: np.ndarray, timing: bool = False, single_shots: int = 0) -> WindowResult:
        det = np.ascontiguousarray(det, dtype=np.uint8)
        B = det.shape[0]
        syn = det.copy()
        err = np.zeros((B, self.H.shape[1]), dtype=np.uint8)
        failed = np.zeros(B, dtype=bool)
        shot_status = np.zeros(B, dtype=np.uint8)
        counts = np.zeros(4, dtype=np.int64)
        n_single = min(max(int(single_shots), 0), B)
        lat = np.zeros((self.n_windows, B), dtype=np.float64) if timing else None
        single = np.zeros((self.n_windows, n_single), dtype=np.float64) if n_single else None
        for k, w in enumerate(self.windows):
            sub = np.ascontiguousarray(syn[:, w.rows])
            res = w.decoder.decode_batch(sub, use_osd=True)
            bad = res.status == 3
            failed |= bad
            shot_status = np.maximum(shot_status, res.status.astype(np.uint8))
            counts += np.bincount(res.status, minlength=4)[:4]
            if timing:
                lat[k] = res.latency_ns
            if single is not None:
                single[k] = w.decoder.decode_stream(sub[:n_single], use_osd=True).latency_ns
            t0 = time.perf_counter_ns()
            committed = res.errors[:, w.commit_groups].copy()
            committed[bad] = 0                    # an unsatisfied window commits nothing
            err[:, w.commit_cols] ^= committed
            if not w.last:                        # propagate: committed mechanisms also flip detectors of later rounds
                delta = (self.Hc[:, w.commit_cols] @ committed.T.astype(np.int64)) % 2
                syn ^= delta.T.astype(np.uint8)
            if timing and not w.last:
                lat[k] += (time.perf_counter_ns() - t0) / B
        # residual = detectors still unexplained by the accumulated correction (0 for a fully consistent decode)
        residual = (((self.Hc @ err.T.astype(np.int64)) % 2).T.astype(np.uint8) ^ det).sum(axis=1)
        return WindowResult(err, failed, residual, shot_status, lat, single, counts)

    def decode_batch_fused(self, det: np.ndarray) -> "FusedResult":
        """Same decisions as ``decode_batch`` with the whole window loop inside one C++ call.

        Returns per-window batch-amortized per-shot times of the three costs: BP, post-processing (OSD-CS on the lanes
        BP left unsatisfied) and propagation of the committed correction into later rounds.
        """
        from .fast import _ffi, _load
        lib = _load()
        det = np.ascontiguousarray(det, dtype=np.uint8)
        B = det.shape[0]
        if not hasattr(self, "_fused"):
            keep, specs = [], _ffi.new("QdWin[]", self.n_windows)
            for k, win in enumerate(self.windows):
                rows = np.ascontiguousarray(win.rows, dtype=np.int32)
                groups = np.ascontiguousarray(win.commit_groups, dtype=np.int32)
                gcols = np.ascontiguousarray(win.commit_cols, dtype=np.int32)
                keep += [rows, groups, gcols]
                sp_ = specs[k]
                sp_.dec = win.decoder._h
                sp_.nrows = len(rows); sp_.rows = _ffi.cast("const int*", _ffi.from_buffer(rows))
                sp_.ngroups = len(groups); sp_.groups = _ffi.cast("const int*", _ffi.from_buffer(groups))
                sp_.gcols = _ffi.cast("const int*", _ffi.from_buffer(gcols)); sp_.last = int(win.last)
            hcc = sp.csc_matrix(self.H)
            colptr = np.ascontiguousarray(hcc.indptr, dtype=np.int32)
            rowidx = np.ascontiguousarray(hcc.indices, dtype=np.int32)
            self._fused = (specs, keep, colptr, rowidx)
        specs, keep, colptr, rowidx = self._fused
        syn = det.copy()
        err = np.zeros((B, self.H.shape[1]), dtype=np.uint8)
        failed = np.zeros(B, dtype=np.uint8)
        status = np.zeros(B, dtype=np.uint8)
        nw = self.n_windows
        bp, post, prop = (np.zeros(nw, dtype=np.uint64) for _ in range(3))
        counts = np.zeros(4, dtype=np.int64)
        lib.qd_window_decode(specs, nw, B, self.H.shape[0], self.H.shape[1],
                             _ffi.cast("const int*", _ffi.from_buffer(colptr)), _ffi.cast("const int*", _ffi.from_buffer(rowidx)),
                             _ffi.from_buffer("uint8_t[]", syn), _ffi.from_buffer("uint8_t[]", err),
                             _ffi.from_buffer("uint8_t[]", failed), _ffi.from_buffer("uint8_t[]", status),
                             _ffi.from_buffer("uint64_t[]", bp), _ffi.from_buffer("uint64_t[]", post),
                             _ffi.from_buffer("uint64_t[]", prop), _ffi.from_buffer("int64_t[]", counts))
        residual = (((self.Hc @ err.T.astype(np.int64)) % 2).T.astype(np.uint8) ^ det).sum(axis=1)
        return FusedResult(err, failed.astype(bool), residual, status, bp / B, post / B, prop / B, counts)

    def predict_observables(self, errors: np.ndarray) -> np.ndarray:
        return (self.prob.obs @ errors.T.astype(np.int64) % 2).T
