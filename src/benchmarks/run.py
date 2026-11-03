"""CLI: run the threshold sweep, store Parquet/HDF5, and render publication figures.

    python -m benchmarks.run sweep  --codes BB72 BB90 --ps 1e-3 2e-3 --rounds 4 --out results/run1
    python -m benchmarks.run figures --out results/run1
    python -m benchmarks.run depth   --out results/run1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from algebra.qldpc import BB_ZOO, get_code
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel

from .storage import latency_percentiles, load_latencies, load_results, save_latencies, save_results


def depth_table(codes: list[str], rounds: int = 1, layers_by_code: dict | None = None) -> pd.DataFrame:
    rows = []
    for name in codes:
        c = get_code(name)
        exp = build_memory_experiment(c, rounds, NoiseModel(), layers=(layers_by_code or {}).get(name))
        rows.append({"code": name, "n": c.n, "k": c.k, "depth": exp.depth_per_round,
                     "cnot_count": exp.cnot_per_round * rounds, "rounds": rounds,
                     "ancillas": c.hx.shape[0] + c.hz.shape[0]})
    return pd.DataFrame(rows)


def cmd_sweep(a):
    from .montecarlo import run_sweep, to_rows
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rounds = a.rounds if a.rounds > 0 else {c: get_code(c).meta["published_d"] for c in a.codes}
    pts = run_sweep(a.codes, a.ps, rounds, shots_per_task=a.shots_per_task, target_failures=a.target_failures,
                    max_shots=a.max_shots, workers=a.workers, noise=a.noise, layers_key=a.schedule,
                    max_iter=a.max_iter, osd_lambda=a.osd_lambda)
    rows = to_rows(pts, decoder=f"bp{a.max_iter}-osdcs{a.osd_lambda}-int16")
    save_results(rows, out / "results.parquet")
    save_results(rows, out / "results.h5")
    lat_path = out / "latency.h5"
    if lat_path.exists():
        lat_path.unlink()
    for s in pts:
        if s.batch_lat:
            save_latencies(lat_path, "batch-amortized", s.code, s.p, np.concatenate(s.batch_lat))
        if s.stream_lat and sum(len(x) for x in s.stream_lat):
            save_latencies(lat_path, "single-shot", s.code, s.p, np.concatenate(s.stream_lat))
    for r in rows:
        print(f"{r['code']:>6} p={r['p']:.1e} rounds={r['rounds']} PL={r['p_logical']:.3e} "
              f"({r['failures']}/{r['shots']})")


def cmd_depth(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = depth_table(list(BB_ZOO), rounds=1)
    df.to_parquet(out / "depth.parquet")
    print(df.to_string(index=False))


def cmd_figures(a):
    from .plotting import (estimate_threshold, plot_depth_scaling, plot_latency_hist, plot_logical_vs_p,
                           plot_threshold)
    out = Path(a.out)
    fig_dir = Path(a.fig_dir or out / "figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    df = load_results(out / "results.parquet")
    plot_logical_vs_p(df, fig_dir / "logical_vs_p", per_round=True)
    plot_logical_vs_p(df, fig_dir / "logical_total_vs_p", per_round=False)
    plot_threshold(df, fig_dir / "threshold")
    print("threshold:", estimate_threshold(df))
    lat = load_latencies(out / "latency.h5")
    if lat:
        for dec in sorted({k.split("/")[0] for k in lat}):
            sel = {k.split("/", 1)[1]: v for k, v in lat.items() if k.startswith(dec + "/")}
            # aggregate over p per code so each panel is one code
            by_code: dict[str, list] = {}
            for k, v in sel.items():
                by_code.setdefault(k.split("/")[0], []).append(v)
            merged = {f"{dec} {c}": np.concatenate(v) for c, v in by_code.items()}
            plot_latency_hist(merged, fig_dir / f"latency_{dec}")
            for k, v in merged.items():
                print(k, {n: round(x / 1e3, 3) for n, x in latency_percentiles(v).items()}, "us")
    depth = depth_table(sorted(set(df["code"]), key=lambda c: get_code(c).n))
    plot_depth_scaling(depth, fig_dir / "depth_scaling")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmarks.run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep")
    s.add_argument("--codes", nargs="+", default=["BB72", "BB90", "BB108"])
    s.add_argument("--ps", nargs="+", type=float, default=[5e-4, 1e-3, 2e-3, 3e-3, 5e-3])
    s.add_argument("--rounds", type=int, default=0, help="0 = use distance d rounds")
    s.add_argument("--shots-per-task", type=int, default=1000)
    s.add_argument("--target-failures", type=int, default=100)
    s.add_argument("--max-shots", type=int, default=50000)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--noise", choices=["uniform", "coherence"], default="uniform")
    s.add_argument("--schedule", choices=["sequential", "interleaved"], default="sequential")
    s.add_argument("--max-iter", type=int, default=30)
    s.add_argument("--osd-lambda", type=int, default=32)
    s.add_argument("--out", default="results/run")
    s.set_defaults(fn=cmd_sweep)
    f = sub.add_parser("figures")
    f.add_argument("--out", default="results/run")
    f.add_argument("--fig-dir", default=None)
    f.set_defaults(fn=cmd_figures)
    d = sub.add_parser("depth")
    d.add_argument("--out", default="results/run")
    d.set_defaults(fn=cmd_depth)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
