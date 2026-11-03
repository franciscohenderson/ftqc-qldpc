"""Benchmark: sliding-window (W, F) decoding vs full-horizon decoding of the same Stim samples.

    python scripts/bench_window.py [--codes BB72 BB144] [--out results/window]

Every decoder sees exactly the same detection events.  Latencies are measured inside the C++ kernel
(batch-amortized over a 16-shot SIMD pass, and one shot at a time on the first ``--single`` shots).
For the windowed decoder the *step* latency is the time of one stationary window step (every window but the
last, including the propagation of the committed correction); dividing it by F gives the per-round cost; the
*reaction* latency (what is left to wait after the last syndrome round arrives) is the time of the last window.
Batch-amortized = chunk time / 16, so it includes the scalar OSD time of any lane of the chunk; single-shot is the
per-shot latency.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from algebra.qldpc import get_code  # noqa: E402
from api.services import _wilson, get_layers  # noqa: E402
from circuits.builder import build_memory_experiment  # noqa: E402
from circuits.noise import NoiseModel  # noqa: E402
from decoder.dem import dem_to_problem  # noqa: E402
from decoder.fast import FastDecoder  # noqa: E402
from decoder.window import WindowDecoder, detector_rounds  # noqa: E402

CONFIGS = [(2, 1), (3, 1), (4, 2)]
SHOTS = {("BB72", 2e-3): 20000, ("BB144", 2e-3): 30000, ("BB72", 3e-3): 10000, ("BB144", 3e-3): 10000}


def pc(a_ns):
    a = np.asarray(a_ns, dtype=np.float64).ravel() / 1e3
    p50, p95, p99 = np.percentile(a, [50, 95, 99])
    return {"p50_us": float(p50), "p95_us": float(p95), "p99_us": float(p99), "mean_us": float(a.mean()), "max_us": float(a.max())}


def run(code_name: str, p: float, rounds: int, shots: int, single: int, seed: int) -> dict:
    code = get_code(code_name)
    exp = build_memory_experiment(code, rounds, NoiseModel.uniform(p), layers=get_layers(code_name, "interleaved"))
    prob = dem_to_problem(exp.dem())
    rnd = detector_rounds(exp.circuit)
    det, obs = exp.circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    det = det.astype(np.uint8)
    out = {"code": code_name, "p": p, "rounds": rounds, "shots": shots, "seed": seed,
           "dem": {"detectors": int(prob.h.shape[0]), "mechanisms": int(prob.h.shape[1])}, "configs": {}}

    full = FastDecoder(prob.h, prob.priors, max_iter=30)
    t0 = time.perf_counter()
    rb = full.decode_batch(det)
    rs = full.decode_stream(det[:single])
    wrong = ((prob.obs @ rb.errors.T.astype(np.int64) % 2).T != obs).any(1) | (rb.status == 3)
    f = int(wrong.sum())
    lo, hi = _wilson(f, shots)
    out["full"] = {"failures": f, "p_logical": f / shots, "lo": lo, "hi": hi, "wall_s": time.perf_counter() - t0,
                   "batch_amortized": pc(rb.latency_ns), "single_shot": pc(rs.latency_ns)}
    print(f"  {code_name} p={p:g} full: PL={f / shots:.3e} ({f}/{shots}) lat p50={out['full']['batch_amortized']['p50_us']:.0f} us", flush=True)

    for W, F in CONFIGS:
        wd = WindowDecoder(prob, rnd, W=W, F=F)          # built outside the timer, like the full decoder
        t0 = time.perf_counter()
        r = wd.decode_batch(det, timing=True, single_shots=single)
        wrong = (wd.predict_observables(r.errors) != obs).any(1) | r.failed
        f = int(wrong.sum())
        lo, hi = _wilson(f, shots)
        lat = r.latency_ns.astype(np.float64)
        sg = r.single_ns.astype(np.float64)
        stat = slice(0, -1) if lat.shape[0] > 1 else slice(None)
        out["configs"][f"W{W}F{F}"] = {
            "W": W, "F": F, "n_windows": wd.n_windows, "failures": f, "p_logical": f / shots, "lo": lo, "hi": hi,
            "wall_s": time.perf_counter() - t0, "residual_nonzero": int((r.residual > 0).sum()), "window_failed": int(r.failed.sum()),
            "window_shape_max": [max(w.shape[0] for w in wd.windows), max(w.shape[1] for w in wd.windows)],
            "window_shape_max_raw": [max(len(w.rows) for w in wd.windows), max(len(w.cols) for w in wd.windows)],
            "step_batch_amortized": pc(lat[stat]), "step_single_shot": pc(sg[stat]),
            "per_round_batch_amortized": pc(lat[stat] / F), "per_round_single_shot": pc(sg[stat] / F),
            "reaction_batch_amortized": pc(lat[-1]), "reaction_single_shot": pc(sg[-1]),
            "total_per_shot_batch": pc(lat.sum(axis=0)),
        }
        c = out["configs"][f"W{W}F{F}"]
        print(f"  {code_name} p={p:g} (W,F)=({W},{F}): PL={f / shots:.3e} ({f}/{shots}) step p50={c['step_batch_amortized']['p50_us']:.1f} us "
              f"reaction p50={c['reaction_batch_amortized']['p50_us']:.1f} us", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", nargs="+", default=["BB72", "BB144"])
    ap.add_argument("--ps", nargs="+", type=float, default=[2e-3, 3e-3])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--single", type=int, default=400)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="results/window")
    a = ap.parse_args()
    res = []
    for code in a.codes:
        for p in a.ps:
            res.append(run(code, p, a.rounds, SHOTS.get((code, p), 10000), a.single, a.seed))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps({"runs": res, "machine": "desktop processor with AVX2, 16 x int16 lanes",
                                                  "schedule": "interleaved depth 7", "decoder_core": "bp30-osdcs32-int16"}, indent=2))
    print("wrote", out / "results.json")


if __name__ == "__main__":
    main()
