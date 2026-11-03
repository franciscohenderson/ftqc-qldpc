"""Generate the BP RTL, execute it and compare it bit by bit with qdecoder.cpp.

    python scripts/run_rtl.py [--code BB72] [--shots 1000] [--p 0.04] [--seed 2026] [--no-synth]

Writes exports/rtl/tars_bp_<code>_hz.sv, its test vectors (.mem), a self-checking testbench and
results/rtl_bitexact.json (which docs/FACTS.md is generated from).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algebra.qldpc import get_code                                                        # noqa: E402
from decoder.fast import FastDecoder                                                      # noqa: E402
from hardware.rtl_gen import generate_bp_sv, generate_testbench_sv                        # noqa: E402
from hardware.rtl_sim import RtlSim, bp_fixed, synth_report, write_vectors                # noqa: E402

CLOCKS_MHZ = (250, 400)       # nominal clocks for the conversion only; not an achieved frequency


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="BB72")
    ap.add_argument("--shots", type=int, default=1000)
    ap.add_argument("--p", type=float, default=0.04, help="iid X-error probability used to draw the syndromes")
    ap.add_argument("--max-iter", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--no-synth", action="store_true")
    a = ap.parse_args()

    code = get_code(a.code)
    H = code.hz
    m, n = H.shape
    name = f"tars_bp_{a.code.lower()}_hz"
    dec = FastDecoder(H, 0.02, max_iter=a.max_iter)            # prior 0.02 -> the channel LLR baked into the RTL
    ch = dec.channel_llr()
    out_dir = ROOT / "exports" / "rtl"
    out_dir.mkdir(parents=True, exist_ok=True)
    sv_path = out_dir / f"{name}.sv"
    sv_path.write_text(generate_bp_sv(H, ch, a.max_iter, name), encoding="utf-8", newline="\n")

    rng = np.random.default_rng(a.seed)
    err = (rng.random((a.shots, n)) < a.p).astype(np.uint8)
    syn = ((err @ H.T) % 2).astype(np.uint8)

    # 1) production kernel
    cpp_post = np.zeros((a.shots, n), np.int16); cpp_it = np.zeros(a.shots, int); cpp_conv = np.zeros(a.shots, bool)
    for i in range(a.shots):
        cpp_post[i], cpp_it[i], st = dec.bp_posterior(syn[i]); cpp_conv[i] = st == 0
    # 2) fixed-point model
    mp, mi, mc, mcyc = bp_fixed(H, ch, syn, a.max_iter)
    # 3) the SystemVerilog, executed
    wd = Path(tempfile.mkdtemp(prefix="tars_rtlrun_"))
    t0 = time.perf_counter()
    sim = RtlSim(sv_path, name, m, n, workdir=wd)
    build_s = time.perf_counter() - t0
    rp, ri, rc, rcyc = sim.run(syn)

    def same(p, i, c):
        return bool(np.array_equal(p, cpp_post) and np.array_equal(i, cpp_it) and np.array_equal(c, cpp_conv))
    rtl_hard = rp < 0
    cpp_hard = cpp_post < 0
    res = {
        "code": a.code, "matrix": "H_Z", "checks": m, "variables": n, "edges": int(H.sum()), "max_iter": a.max_iter,
        "shots": a.shots, "p_x": a.p, "seed": a.seed, "channel_llr": int(ch[0]), "sv_lines": len(sv_path.read_text().splitlines()),
        "bit_exact": {
            "rtl_vs_cpp": {"posterior_llr": bool(np.array_equal(rp, cpp_post)), "hard_decisions": bool(np.array_equal(rtl_hard, cpp_hard)),
                           "iterations": bool(np.array_equal(ri, cpp_it)), "converged": bool(np.array_equal(rc, cpp_conv)),
                           "all": same(rp, ri, rc), "vectors": a.shots,
                           "values_compared": int(a.shots * n) + 2 * a.shots},
            "model_vs_cpp": same(mp, mi, mc),
            "rtl_vs_model": bool(np.array_equal(rp, mp) and np.array_equal(ri, mi)),
        },
        "converged_shots": int(rc.sum()), "not_converged_shots": int((~rc).sum()),
        "cycles": {"formula": "2*iterations + 2", "formula_holds_on_every_vector": bool(np.array_equal(rcyc, 2 * ri + 2)),
                   "min": int(rcyc.min()), "p50": float(np.percentile(rcyc, 50)), "p95": float(np.percentile(rcyc, 95)),
                   "max": int(rcyc.max()), "mean": float(rcyc.mean()), "per_iteration": 2,
                   "iterations": {"mean": float(ri.mean()), "max": int(ri.max())}},
        "nominal_clock_conversion_us": {f"{f}MHz": {"p50": float(np.percentile(rcyc, 50)) / f, "p95": float(np.percentile(rcyc, 95)) / f,
                                                      "max": float(rcyc.max()) / f}
                                          for f in CLOCKS_MHZ},
        "note": "RTL clock cycles of the generated schedule; the clock frequencies are nominal, not achieved (no FPGA flow was run).",
        "rtl_elaboration_and_compile_s": round(build_s, 1),
    }
    # vectors + testbench for any external simulator (the testbench itself was NOT executed here)
    k = min(a.shots, 1000)
    write_vectors(out_dir / f"{name}_in.mem", out_dir / f"{name}_out.mem", syn[:k], cpp_post[:k], cpp_it[:k])
    (out_dir / f"tb_{name}.sv").write_text(
        generate_testbench_sv(m, n, name, k, f"{name}_in.mem", f"{name}_out.mem"), encoding="utf-8", newline="\n")
    res["files"] = [f"exports/rtl/{f}" for f in (f"{name}.sv", f"{name}_in.mem", f"{name}_out.mem", f"tb_{name}.sv")]
    res["testbench_executed"] = False
    if not a.no_synth:
        t0 = time.perf_counter()
        res["synthesis"] = synth_report(H, ch, a.max_iter)
        res["synthesis"]["seconds"] = round(time.perf_counter() - t0, 1)
    # context: BP-only time of the production CPU kernel on the same syndromes (a different thing from RTL cycles)
    cpu = dec.decode_stream(syn[:200], use_osd=False)
    res["cpu_bp_single_shot_us"] = {"p50": float(np.percentile(cpu.bp_ns, 50) / 1e3), "p95": float(np.percentile(cpu.bp_ns, 95) / 1e3)}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "rtl_bitexact.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    shutil.rmtree(wd, ignore_errors=True)
    b = res["bit_exact"]
    print(f'{a.code} H_Z {m}x{n}: RTL vs C++ bit-exact on {a.shots} vectors = {b["rtl_vs_cpp"]["all"]}; model vs C++ = {b["model_vs_cpp"]}; '
          f'cycles p50/p95/max = {res["cycles"]["p50"]:.0f}/{res["cycles"]["p95"]:.0f}/{res["cycles"]["max"]}')
    if "synthesis" in res:
        s = res["synthesis"]
        print(f'  size: {s["registers"]["total"]} register bits, {s["combinational_gates_total"]} generic gates (upper bound), '
              f'deepest block {s["critical_block_depth_gates"]} gates')
    if not (b["rtl_vs_cpp"]["all"] and res["cycles"]["formula_holds_on_every_vector"]):
        sys.exit(1)


if __name__ == "__main__":
    main()
