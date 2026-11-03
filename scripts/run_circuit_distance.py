"""Measure the circuit-level distance d_circ of a BB code for the 7-layer and 12-layer schedules.

    python scripts/run_circuit_distance.py [--code BB72] [--rounds 3] [--time 120] [--out results/circuit_distance.json]

Writes the bounds, solver status and timings; the result is the only source for the whitepaper/FACTS numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algebra.circuit_distance import compute_circuit_distance          # noqa: E402
from api.services import code_of, get_layers                           # noqa: E402
from circuits.builder import build_memory_experiment                   # noqa: E402
from circuits.noise import NoiseModel                                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="BB72")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--time", type=float, default=120.0, help="MILP time limit per schedule, seconds")
    ap.add_argument("--basis", default="Z")
    ap.add_argument("--max-weight", type=int, default=None, help="decision problem: is there an undetected logical fault of weight <= W?")
    ap.add_argument("--schedules", nargs="*", default=["interleaved", "sequential"])
    ap.add_argument("--out", default=str(ROOT / "results" / "circuit_distance.json"))
    a = ap.parse_args()
    code = code_of(a.code)
    out_path = Path(a.out)
    data = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"runs": []}
    for sched in a.schedules:
        layers = get_layers(a.code, sched)
        exp = build_memory_experiment(code, a.rounds, NoiseModel.uniform(1e-3), basis=a.basis, layers=layers)
        dem = exp.dem()
        t0 = time.perf_counter()
        r = compute_circuit_distance(dem, time_limit=a.time, max_weight=a.max_weight)
        r.update(code=a.code, schedule=sched, cnot_layers=len(layers), rounds=a.rounds, basis=a.basis,
                 distance_code=(code.meta.get("published_d") or code.meta.get("d")) if hasattr(code, "meta") else None, wall_seconds=time.perf_counter() - t0)
        r.pop("witness", None) if len(r.get("witness", [])) > 40 else None
        data["runs"] = [x for x in data["runs"] if not (x["code"] == a.code and x["schedule"] == sched and x["rounds"] == a.rounds and x["basis"] == a.basis and x.get("max_weight") == a.max_weight)]
        data["runs"].append(r)
        print(f'{a.code} {sched:11s} layers={len(layers):2d} rounds={a.rounds} mech={r["n_mechanisms"]} det={r["n_detectors"]} '
              f'd_circ in [{r.get("d_lower")}, {r.get("d_upper")}] certified={r.get("certified")} {r["seconds"]:.1f}s | {r["status"][:60]}', flush=True)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(data, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
