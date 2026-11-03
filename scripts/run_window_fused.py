"""Python window loop vs the fused C++ window loop, on the same Stim samples (seed 2026).

    python scripts/run_window_fused.py [--codes BB72 BB144] [--p 2e-3] [--rounds 6] [--shots 1024] [--single 96]

Reports, per (W,F), the step / reaction time and the exact split between BP, post-processing (OSD-CS) and
propagation.  Everything is written to results/window_fused.json; docs/FACTS.md is generated from it.
Run it on an otherwise idle machine: the numbers are wall-clock times.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algebra.qldpc import get_code                         # noqa: E402
from api.services import get_layers                        # noqa: E402
from circuits.builder import build_memory_experiment       # noqa: E402
from circuits.noise import NoiseModel                      # noqa: E402
from decoder.dem import dem_to_problem                     # noqa: E402
from decoder.window import WindowDecoder, detector_rounds  # noqa: E402

CONFIGS = [(3, 1), (4, 2)]
REPS = 3


def pc(a_ns) -> dict:
    a = np.asarray(a_ns, dtype=np.float64)
    if a.size == 0:
        return {"p50_us": None, "p95_us": None, "mean_us": None}
    return {"p50_us": float(np.percentile(a, 50) / 1e3), "p95_us": float(np.percentile(a, 95) / 1e3), "mean_us": float(a.mean() / 1e3)}


def run(name: str, p: float, rounds: int, shots: int, single: int, seed: int) -> dict:
    code = get_code(name)
    exp = build_memory_experiment(code, rounds, NoiseModel.uniform(p), layers=get_layers(name, "interleaved"))
    prob = dem_to_problem(exp.dem())
    rnd = detector_rounds(exp.circuit)
    det, obs = exp.circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    det = det.astype(np.uint8)
    out = {"code": name, "p": p, "rounds": rounds, "shots": shots, "single_shots": single, "seed": seed, "wall_time_best_of": REPS,
           "dem": {"detectors": int(prob.h.shape[0]), "mechanisms": int(prob.h.shape[1])}, "configs": {}}
    for W, F in CONFIGS:
        wd = WindowDecoder(prob, rnd, W=W, F=F)
        nw = wd.n_windows
        wd.decode_batch(det[:32]); wd.decode_batch_fused(det[:32])                       # warm-up
        wall_py = wall_fu = float("inf")
        for _ in range(REPS):                          # best of REPS, the same treatment for both loops
            t0 = time.perf_counter(); py_ = wd.decode_batch(det, timing=True, single_shots=single); w = time.perf_counter() - t0
            if w < wall_py:
                wall_py, py = w, py_
            t0 = time.perf_counter(); fu_ = wd.decode_batch_fused(det); w = time.perf_counter() - t0
            if w < wall_fu:
                wall_fu, fu = w, fu_
        same = bool(np.array_equal(py.errors, fu.errors) and np.array_equal(py.failed, fu.failed))
        fails = int(((wd.predict_observables(fu.errors) != obs).any(1) | fu.failed).sum())
        # single-shot fused calls: one shot per C++ call, per-window split
        sg_bp, sg_post, sg_prop = (np.zeros((nw, single)) for _ in range(3))
        for i in range(single):
            r = wd.decode_batch_fused(det[i:i + 1])
            sg_bp[:, i], sg_post[:, i], sg_prop[:, i] = r.bp_ns, r.post_ns, r.prop_ns
        sg_tot = sg_bp + sg_post + sg_prop
        stat = slice(0, -1) if nw > 1 else slice(None)
        bat_tot = (fu.bp_ns + fu.post_ns + fu.prop_ns)
        out["configs"][f"W{W}F{F}"] = {
            "W": W, "F": F, "n_windows": nw, "identical_decisions": same, "logical_failures": fails,
            "python_loop": {"wall_per_shot_us": wall_py / shots * 1e6, "step_batch_us": pc(py.latency_ns[stat].ravel()),
                            "step_single_us": pc(py.single_ns[stat].ravel()), "reaction_single_us": pc(py.single_ns[-1])},
            "fused_cpp": {"wall_per_shot_us": wall_fu / shots * 1e6,
                          "step_batch_per_shot_us": float(bat_tot[stat].mean() / 1e3), "reaction_batch_per_shot_us": float(bat_tot[-1] / 1e3),
                          "split_batch_per_shot_us": {"bp": float(fu.bp_ns[stat].mean() / 1e3), "post": float(fu.post_ns[stat].mean() / 1e3),
                                                      "prop": float(fu.prop_ns[stat].mean() / 1e3)},
                          "step_single_us": pc(sg_tot[stat].ravel()), "reaction_single_us": pc(sg_tot[-1]),
                          "split_single_p50_us": {"bp": pc(sg_bp[stat].ravel())["p50_us"], "post": pc(sg_post[stat].ravel())["p50_us"],
                                                  "prop": pc(sg_prop[stat].ravel())["p50_us"]},
                          "split_reaction_single_p50_us": {"bp": pc(sg_bp[-1])["p50_us"], "post": pc(sg_post[-1])["p50_us"], "prop": pc(sg_prop[-1])["p50_us"]}},
            "status_counts": fu.status_counts.tolist()}
        c = out["configs"][f"W{W}F{F}"]
        print(f'  {name} (W,F)=({W},{F}) identical={same} fails={fails}/{shots} | wall/shot python {c["python_loop"]["wall_per_shot_us"]:.1f} us '
              f'vs fused {c["fused_cpp"]["wall_per_shot_us"]:.1f} us | step single p50 python {c["python_loop"]["step_single_us"]["p50_us"]:.1f} '
              f'vs fused {c["fused_cpp"]["step_single_us"]["p50_us"]:.1f} us | split(single p50) {c["fused_cpp"]["split_single_p50_us"]}', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="*", default=["BB72", "BB144"])
    ap.add_argument("--p", type=float, default=2e-3)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--single", type=int, default=96)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default=str(ROOT / "results" / "window_fused.json"))
    a = ap.parse_args()
    res = {"runs": [run(n, a.p, a.rounds, a.shots, a.single, a.seed) for n in a.codes]}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
