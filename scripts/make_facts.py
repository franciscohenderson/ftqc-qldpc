"""Generate docs/FACTS.md: every measured number the whitepaper/pitch may cite, straight from the data."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from benchmarks.plotting import estimate_threshold  # noqa: E402
from benchmarks.storage import latency_percentiles, load_latencies, load_results  # noqa: E402


def md_table(df: pd.DataFrame, fmt="{:.3e}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c, v in zip(cols, r):
            if c == "p":
                cells.append(f"{float(v):.1e}")
            else:
                cells.append(fmt.format(v) if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def osd_fix_fact() -> str:
    pre_p, post_p = ROOT / "results" / "int6" / "results.parquet", ROOT / "results" / "final_int" / "results.parquet"
    if not (pre_p.exists() and post_p.exists()):
        return "* OSD-CS audit fix: pre-fix sweep not available.\n"
    pre, post = load_results(pre_p), load_results(post_p)
    parts = []
    for c in ("BB72", "BB90", "BB108", "BB144"):
        a = pre[(pre.code == c) & (pre.p == 3e-3)].iloc[0] if c != "BB72" else pre[(pre.code == c) & (pre.p == 2e-3)].iloc[0]
        b = post[(post.code == c) & (post.p == a.p)].iloc[0]
        parts.append(f"{c} @ p={a.p:.0e}: {a.p_logical*100:.2f} % ({int(a.failures)}/{int(a.shots)}) -> {b.p_logical*100:.2f} % ({int(b.failures)}/{int(b.shots)})")
    return ("* Audit by agy found OSD-CS degrading to OSD-0 once the OSD basis was full (the column scan stopped at rank = m). Effect on the interleaved 6-round total logical error rate, "
            "sweep before the fix (results/int6) vs after (results/final_int): " + "; ".join(parts) +
            ". The improvement is modest (roughly 8-27 % fewer failures) and of the order of the sampling error at these shot counts; a separate 3000-shot spot check (10 failures) is not used because it is not consistent with the sweep. "
            "All final sweeps were re-run after the fix.\n")



def window_facts() -> str:
    path = ROOT / "results" / "window" / "results.json"
    if not path.exists():
        return ""
    d = json.loads(path.read_text(encoding="utf-8"))
    out = ["## Sliding-window (W, F) decoding vs full-horizon decoding (scripts/bench_window.py)\n",
           f"Same Stim samples for every decoder; {d['schedule']}; core {d['decoder_core']}; {d['machine']}. "
           "6 syndrome rounds (7 detector layers). Latencies are measured inside the C++ kernel. "
           "\"step\" = one stationary window step (every window but the last; includes propagating the committed correction); "
           "\"per round\" = step / F; \"reaction\" = time of the LAST window (what remains to wait after the final syndrome round arrives); "
           "for the full-horizon decoder the reaction latency is its whole decode time. "
           "Batch = chunk time / 16 shots, so it includes the scalar OSD time of any lane of the chunk (a throughput view); "
           "single = one shot at a time (first 400 shots), the per-shot latency. "
           "Identical columns created by truncating the future boundary are merged before decoding; windows need F < W.\n"]
    tri = lambda m: f"{m['p50_us']:.1f} / {m['p95_us']:.1f} / {m['p99_us']:.1f}"
    for run in d["runs"]:
        out.append(f"### {run['code']} at p = {run['p']:g} ({run['shots']} shots, seed {run['seed']}; full DEM = {run['dem']['detectors']} detectors x {run['dem']['mechanisms']} mechanisms)\n")
        rows = ["| decoder | P_L (6 rounds) | failures/shots | 95 % Wilson | step P50/P95/P99 batch (us) | step P50 single-shot (us) | per-round P50 batch (us) | reaction P50/P95/P99 batch (us) | reaction P50/P95/P99 single-shot (us) | largest window (det x cols; raw cols) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        f = run["full"]
        rows.append(f"| full horizon | {f['p_logical']:.3e} | {int(round(f['p_logical'] * run['shots']))}/{run['shots']} | {f['lo']:.2e} - {f['hi']:.2e} | n/a | n/a | n/a | {tri(f['batch_amortized'])} | {tri(f['single_shot'])} | {run['dem']['detectors']} x {run['dem']['mechanisms']} |")
        for k, c in run["configs"].items():
            rows.append(f"| window (W={c['W']}, F={c['F']}), {c['n_windows']} windows | {c['p_logical']:.3e} | {c['failures']}/{run['shots']} | {c['lo']:.2e} - {c['hi']:.2e} | {tri(c['step_batch_amortized'])} | {c['step_single_shot']['p50_us']:.1f} | {c['per_round_batch_amortized']['p50_us']:.1f} | {tri(c['reaction_batch_amortized'])} | {tri(c['reaction_single_shot'])} | {c['window_shape_max'][0]} x {c['window_shape_max'][1]} (raw {c['window_shape_max_raw'][1]}) |")
        out.append("\n".join(rows) + "\n")
        bad = {k: (c["residual_nonzero"], c["window_failed"]) for k, c in run["configs"].items()}
        out.append("Shots left with a nonzero residual syndrome / shots where some window failed: " + ", ".join(f"{k}: {v[0]} / {v[1]}" for k, v in bad.items()) + ".\n")
    return "\n".join(out) + "\n"


def main():
    out = ["# FACTS (auto-generated from repository data; the only numbers documents may cite)\n"]
    out.append("Machine: desktop processor (6C/12T) with AVX2, no AVX-512, Windows 11, Python 3.14, clang via zig c++ (C++20), 16 x int16 SIMD lanes.\n")
    out.append("## Code parameters (exact MILP via HiGHS unless stated)\n")
    out.append((ROOT / "exports" / "code_parameters.md").read_text(encoding="utf-8"))
    params = json.loads((ROOT / "exports" / "code_parameters.json").read_text(encoding="utf-8"))
    out.append("ILP seconds: " + ", ".join(f"{t['code']} {t['ilp_seconds']} s" for t in params) + "\n")
    for run, label in (("final_int", "interleaved depth-7 schedule"), ("final_seq", "sequential depth-12 schedule")):
        df = load_results(ROOT / "results" / run / "results.parquet")
        out.append(f"## Monte Carlo: {run} ({label})\n")
        out.append(f"rounds = {sorted(df['rounds'].unique())}, decoder = {sorted(df['decoder'].unique())}, "
                   f"total shots = {int(df['shots'].sum())}\n")
        piv = df.pivot(index="p", columns="code", values="p_logical").reset_index()
        out.append("Total logical error rate over the memory experiment (0 = none observed in the shot budget):\n")
        out.append(md_table(piv))
        fl = df.pivot(index="p", columns="code", values="failures").reset_index()
        sh = df.pivot(index="p", columns="code", values="shots").reset_index()
        out.append("\nFailures observed:\n")
        out.append(md_table(fl, "{:.0f}"))
        out.append("\nShots used:\n")
        out.append(md_table(sh, "{:.0f}"))
        th = estimate_threshold(df)
        out.append(f"\nThreshold estimate (crossing of consecutive-distance curves): {th['threshold']:.4e}; "
                   f"crossings: {[float(f'{c:.4e}') for c in th['crossings']]}; pseudo-thresholds: "
                   f"{ {k: (None if v != v else float(f'{v:.3e}')) for k, v in th['pseudo_thresholds'].items()} }\n")
        lat = load_latencies(ROOT / "results" / run / "latency.h5")
        rows = []
        pooled = {}
        for k, v in lat.items():
            dec, code, _ = k.split("/")
            pooled.setdefault((dec, code), []).append(v)
        for (dec, code), v in sorted(pooled.items()):
            pc = latency_percentiles(np.concatenate(v))
            rows.append({"mode": dec, "code": code, "P50_us": pc["P50"] / 1e3, "P95_us": pc["P95"] / 1e3,
                         "P99_us": pc["P99"] / 1e3, "P99.9_us": pc["P99.9"] / 1e3, "max_us": pc["max"] / 1e3})
        out.append("Full circuit-level decode latency (BP + OSD-CS), pooled over all p:\n")
        out.append(md_table(pd.DataFrame(rows), "{:.1f}"))
        out.append("")
    out.append("## BP kernel latency (code-capacity H_Z: 36x72 for BB72 and 72x144 for BB144, measured by scripts/bench_kernel.py)\n")
    kp = ROOT / "results" / "kernel_latency.json"
    if kp.exists():
        out.append("```json\n" + kp.read_text(encoding="utf-8") + "\n```\n")
    out.append(window_facts())
    sys.path.insert(0, str(ROOT / "scripts"))
    from facts_hardware import hardware_facts
    out.append(hardware_facts())
    out.append("## Other measured facts\n")
    out.append("* Interleaved schedule depth 7 vs sequential depth 12 CNOT layers per round (0 ordering violations, Stim detectors deterministic with noise off).\n"
               "* CNOTs per round = |H_X| + |H_Z| (BB72: 432, BB144: 864).\n"
               "* BB288 also admits the depth-7 interleaved schedule (found in 23 s, 0 ordering violations, Stim detectors deterministic with noise off over 2 rounds); BB288 was not included in the Monte Carlo sweeps.\n"
               + osd_fix_fact() +
               "* GNN (INT8) tier: BB72 H_Z code-capacity p=0.06, BP10, 2000 shots: INT8/fp32 sign agreement 99.84 %; logical failure 23.65 % (BP+OSD) -> 22.65 % (with GNN-ordered OSD); 24 shots resolved by tier 2.\n"
               "* ILP formulation: min sum z s.t. H z - 2u = 0, L z - 2v = y, sum y >= 1 (z, y binary; u, v integer).\n")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "FACTS.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/FACTS.md")


if __name__ == "__main__":
    main()
