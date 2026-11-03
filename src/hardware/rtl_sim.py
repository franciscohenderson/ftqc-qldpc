"""Verification of the generated RTL against the C++ decoder.

Three independent implementations of the same arithmetic are compared bit by bit:

1. ``qdecoder.cpp`` (the production AVX2 kernel, via ``FastDecoder.bp_posterior``);
2. ``bp_fixed`` below, an int16 model written from the RTL's point of view (also gives the exact cycle count);
3. the generated SystemVerilog itself, elaborated by Yosys, translated to C++ by CXXRTL and compiled with ``zig c++``
   (so this really executes the ``.sv`` file, clock edge by clock edge).

Cycle counts are RTL clock cycles of the schedule in ``rtl_gen`` (``2*iters + 2``).  Times at a nominal clock are
just ``cycles / f``; no FPGA was synthesized, placed or timed, so no achievable clock frequency is claimed.
"""
from __future__ import annotations

import ctypes
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from .rtl_gen import LLR_MAX, edges_of


# ------------------------------------------------------------------ 1. bit-exact fixed-point model
def _sat(x):
    return np.clip(x, -32768, 32767)


def bp_fixed(h: np.ndarray, ch: np.ndarray, syndromes: np.ndarray, max_iter: int, trace: list | None = None):
    """int16 normalized min-sum, flooding schedule, as the RTL runs it.

    Returns (posterior (B,n) int16, iterations (B,), converged (B,), cycles (B,)).  If ``trace`` is a list, the number
    of unsatisfied checks of every shot is appended to it before each iteration (entry k = after k iterations).
    """
    h = (np.asarray(h) % 2).astype(np.uint8)
    chk, var_edges, E = edges_of(h)
    m, n = h.shape
    s = np.asarray(syndromes, dtype=np.int32).reshape(-1, m)
    B = s.shape[0]
    tot = np.tile(np.asarray(ch, dtype=np.int32), (B, 1))
    r = np.zeros((B, E), dtype=np.int32)
    done = np.zeros(B, bool)
    iters = np.zeros(B, np.int32)
    conv = np.zeros(B, bool)
    post = tot.copy()
    for it in range(max_iter + 1):
        neg = (tot < 0).astype(np.int32)
        ucount = np.zeros(B, np.int32)
        for c in range(m):
            par = s[:, c].copy()
            for _, v in chk[c]:
                par ^= neg[:, v]
            ucount += par
        unsat = ucount > 0
        if trace is not None:
            trace.append(ucount.copy())
        fin = ~done & ~unsat
        post[fin] = tot[fin]; iters[fin] = it; conv[fin] = True; done |= fin
        if it == max_iter:
            rest = ~done
            post[rest] = tot[rest]; iters[rest] = max_iter; done |= rest
            break
        rn = np.zeros_like(r)
        for c in range(m):
            m1 = np.full(B, 32767, np.int32); m2 = m1.copy(); idx = np.full(B, 15, np.int32); sg = s[:, c].copy()
            qs = []
            for k, (e, v) in enumerate(chk[c]):
                q = _sat(tot[:, v] - r[:, e])
                qs.append(q)
                a = np.where(q < 0, -q, q)
                sg ^= (q < 0).astype(np.int32)
                lt = a < m1
                m2 = np.where(lt, m1, np.where(a < m2, a, m2))
                idx = np.where(lt, k, idx)
                m1 = np.where(lt, a, m1)
            n1 = m1 - (m1 >> 2)
            n2 = m2 - (m2 >> 2)
            for k, (e, v) in enumerate(chk[c]):
                mag = np.where(idx == k, n2, n1)
                neg_q = (qs[k] < 0).astype(np.int32)
                rn[:, e] = np.where((sg ^ neg_q).astype(bool), -mag, mag)
        r = rn
        for v in range(n):
            t = np.full(B, int(ch[v]), np.int32)
            for e in var_edges[v]:
                t = _sat(t + r[:, e])
            tot[:, v] = np.clip(t, -LLR_MAX, LLR_MAX)
    return post.astype(np.int16), iters, conv, 2 * iters + 2


# ------------------------------------------------------------------ test vectors
def _hex(value: int, bits: int) -> str:
    return format(value, "x").zfill((bits + 3) // 4)


def syndrome_to_int(s: np.ndarray) -> int:
    return sum(int(b) << i for i, b in enumerate(np.asarray(s).ravel()))


def pack_output(post: np.ndarray, iters: int) -> int:
    n = len(post)
    flat = sum((int(v) & 0xFFFF) << (16 * j) for j, v in enumerate(post))
    hard = sum(int(v < 0) << j for j, v in enumerate(post))
    return (int(iters) << (n + 16 * n)) | (hard << (16 * n)) | flat


def write_vectors(path_in: Path, path_out: Path, syndromes: np.ndarray, post: np.ndarray, iters: np.ndarray) -> None:
    m, n = syndromes.shape[1], post.shape[1]
    path_in.write_text("\n".join(_hex(syndrome_to_int(s), m) for s in syndromes) + "\n")
    path_out.write_text("\n".join(_hex(pack_output(p, i), 8 + n + 16 * n) for p, i in zip(post, iters)) + "\n")


# ------------------------------------------------------------------ 3. the SystemVerilog, executed
def _yosys() -> list[str]:
    exe = shutil.which("yowasp-yosys") or str(Path(sys.executable).parent / ("yowasp-yosys.exe" if sys.platform == "win32" else "yowasp-yosys"))
    if not Path(exe).exists() and shutil.which("yowasp-yosys") is None:
        raise RuntimeError("yowasp-yosys is not installed (pip install yowasp-yosys)")
    return [exe]


def have_toolchain() -> bool:
    try:
        _yosys()
        import yowasp_yosys  # noqa: F401
        return True
    except Exception:
        return False


def _cxxrtl_include() -> Path:
    import yowasp_yosys
    return Path(yowasp_yosys.__file__).parent / "share" / "include" / "backends" / "cxxrtl" / "runtime"


def _mangle(name: str) -> str:
    return "p_" + name.replace("_", "__")


_DRIVER = r"""
#include <cxxrtl/cxxrtl.h>
#include <cstdint>
#include "design.cc"
using namespace cxxrtl_design;
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
// Runs `shots` syndromes through the RTL, one after another, exactly like the testbench does.
API int tars_rtl_run(const uint8_t* syn, int shots, int16_t* post, int* iters, uint8_t* conv, int* cycles) {
    constexpr int M = @M@, N = @N@;
    @CLS@ dut;
    auto tick = [&]() { dut.p_clk.set<bool>(false); dut.step(); dut.p_clk.set<bool>(true); dut.step(); };
    dut.p_start.set<bool>(false);
    dut.p_rst.set<bool>(true);
    tick(); tick();
    dut.p_rst.set<bool>(false);
    for (int s = 0; s < shots; s++) {
        cxxrtl::value<M> v;
        for (int c = 0; c < M; c++)
            if (syn[(size_t)s * M + c]) v.data[c / 32] |= (uint32_t)1 << (c % 32);
        dut.p_syndrome.next = v;
        dut.p_start.set<bool>(true);
        tick();                                  // the edge that samples `start` is cycle 1
        dut.p_start.set<bool>(false);
        int cyc = 1;
        while (!dut.p_done.get<bool>()) {
            if (cyc > 100000) return -1;
            tick(); cyc++;
        }
        cycles[s] = cyc;
        iters[s] = (int)dut.p_iters.get<uint8_t>();
        conv[s] = dut.p_converged.get<bool>();
        for (int j = 0; j < N; j++) {
            uint32_t w = dut.p_post__flat.curr.data[j / 2];
            post[(size_t)s * N + j] = (int16_t)((j & 1) ? (w >> 16) : (w & 0xFFFF));
        }
    }
    return 0;
}
"""


class RtlSim:
    """Compile once, run many syndromes through the real generated RTL."""

    def __init__(self, sv_path: Path, top: str, m: int, n: int, workdir: Path | None = None, opt: str = "-O1"):
        self.m, self.n, self.top = m, n, top
        self.dir = Path(workdir or tempfile.mkdtemp(prefix="tars_rtl_"))
        self.dir.mkdir(parents=True, exist_ok=True)
        design = self.dir / "design.cc"
        shutil.copy(sv_path, self.dir / "rtl.sv")       # the WebAssembly Yosys only sees its working directory
        script = f"read_verilog -sv rtl.sv; hierarchy -top {top}; proc; write_cxxrtl {opt} design.cc"
        r = subprocess.run(_yosys() + ["-q", "-p", script], capture_output=True, text=True, cwd=self.dir)
        if r.returncode != 0 or not design.exists():
            raise RuntimeError("yosys failed:\n" + (r.stdout + r.stderr)[-3000:])
        driver = self.dir / "driver.cc"
        driver.write_text(_DRIVER.replace("@M@", str(m)).replace("@N@", str(n)).replace("@CLS@", _mangle(top)))
        lib = self.dir / ("rtl.dll" if sys.platform == "win32" else "rtl.so")
        flags = ["-std=c++17", "-O2", "-shared", f"-I{_cxxrtl_include()}", f"-I{self.dir}"]
        if sys.platform != "win32":
            flags.append("-fPIC")
        err = ""
        for base in ([sys.executable, "-m", "ziglang", "c++"], ["clang++"], ["g++"]):
            try:
                c = subprocess.run(base + flags + ["-o", str(lib), str(driver)], capture_output=True, text=True)
            except FileNotFoundError:
                continue
            if c.returncode == 0 and lib.exists():
                break
            err = c.stderr[-3000:]
        else:
            raise RuntimeError("could not compile the CXXRTL model:\n" + err)
        self._lib = ctypes.CDLL(str(lib))

    def run(self, syndromes: np.ndarray):
        s = np.ascontiguousarray(syndromes, dtype=np.uint8).reshape(-1, self.m)
        B = s.shape[0]
        post = np.zeros((B, self.n), np.int16)
        it = np.zeros(B, np.int32)
        conv = np.zeros(B, np.uint8)
        cyc = np.zeros(B, np.int32)
        p = lambda a, t: a.ctypes.data_as(ctypes.POINTER(t))   # noqa: E731
        rc = self._lib.tars_rtl_run(p(s, ctypes.c_uint8), B, p(post, ctypes.c_int16), p(it, ctypes.c_int),
                                    p(conv, ctypes.c_uint8), p(cyc, ctypes.c_int))
        if rc != 0:
            raise RuntimeError("RTL simulation did not finish (done never rose)")
        return post, it, conv.astype(bool), cyc


# ------------------------------------------------------------------ size of the datapath (generic gates, no placement)
def _gate_stats(sv_text: str, top: str) -> dict:
    """Yosys generic synthesis (no ABC: it cannot run in the WebAssembly build) of one combinational block."""
    with tempfile.TemporaryDirectory(prefix="tars_syn_") as td:
        (Path(td) / "b.sv").write_text(sv_text)
        script = f"read_verilog -sv b.sv; hierarchy -top {top}; synth -noabc -top {top} -flatten; opt_clean; stat; ltp -noff"
        r = subprocess.run(_yosys() + ["-p", script], capture_output=True, text=True, cwd=td)
    out = r.stdout
    if r.returncode != 0 or "Printing statistics" not in out:
        raise RuntimeError(f"yosys synthesis of {top} failed:\n" + (out + r.stderr)[-2000:])
    tail = out[out.rindex("Printing statistics"):]
    gates = {k: int(v) for v, k in re.findall(r"^\s+(\d+)\s+(\$_\w+_)\s*$", tail, re.M)}
    depth = re.findall(r"Longest topological path in \S+ \(length=(\d+)\)", out)
    return {"gates": sum(gates.values()), "by_type": gates, "logic_depth_gates": int(depth[-1]) if depth else None}


def synth_report(h: np.ndarray, ch: np.ndarray, max_iter: int = 30) -> dict:
    """Size of the generated design: exact register bits, and generic-gate counts of its combinational blocks
    (one synthesis per distinct node degree, multiplied by the number of instances).

    This is NOT a LUT/FPGA estimate: ABC and LUT mapping are unavailable in the WebAssembly Yosys, so the gate
    counts are unoptimized 2-input-gate counts (an upper bound) and ``logic_depth_gates`` is the longest
    combinational path of each block in gates.  No placement, routing or timing was run.
    """
    from .rtl_gen import block_modules, flip_flop_bits
    blocks, total = {}, 0
    for name, (text, count) in block_modules(h, ch).items():
        st = _gate_stats(text, name)
        st["instances"] = count
        blocks[name] = st
        total += st["gates"] * count
    return {"tool": "yowasp-yosys (synth -noabc)", "registers": flip_flop_bits(h), "blocks": blocks,
            "combinational_gates_total": total,
            "critical_block_depth_gates": max(b["logic_depth_gates"] or 0 for b in blocks.values()),
            "caveat": "generic 2-input gates, no ABC optimization, no LUT mapping, no placement or timing: an upper bound on logic, not an FPGA resource report"}
