"""Figure for the sliding-window study: logical error rate and reaction latency, full horizon vs (W, F).

    python scripts/plot_window.py        # reads results/window/results.json, writes figures/window_latency.{pdf,svg}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmarks.plotting import apply_publication_style  # noqa: E402


def main():
    data = json.loads((ROOT / "results" / "window" / "results.json").read_text())
    palette = apply_publication_style()
    runs = [r for r in data["runs"] if abs(r["p"] - 2e-3) < 1e-12]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    labels = ["full"] + [f"({c['W']},{c['F']})" for c in runs[0]["configs"].values()]
    x = np.arange(len(labels))
    w = 0.38
    for i, r in enumerate(runs):
        vals = [r["full"]] + list(r["configs"].values())
        y = np.array([max(v["p_logical"], 1e-6) for v in vals])
        lo = np.array([v["lo"] for v in vals]); hi = np.array([v["hi"] for v in vals])
        a1.bar(x + (i - 0.5) * w, y, w, color=palette[i % len(palette)], label=r["code"])
        a1.errorbar(x + (i - 0.5) * w, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt="none", ecolor="black", lw=0.8, capsize=2)
    a1.set_yscale("log"); a1.set_xticks(x); a1.set_xticklabels(labels)
    a1.set_ylabel(r"$P_L$ over 6 rounds ($p=2\times10^{-3}$)"); a1.set_xlabel("decoder: full horizon or window $(W,F)$")
    a1.legend(frameon=False)
    for i, r in enumerate(runs):
        vals = [r["full"]["batch_amortized"]] + [c["reaction_batch_amortized"] for c in r["configs"].values()]
        for j, (q, ls) in enumerate((("p50_us", "-"), ("p99_us", ":"))):
            a2.plot(x, [v[q] for v in vals], ls, marker="o", color=palette[i % len(palette)], label=f"{r['code']} {q[:3].upper()}" if True else None)
    a2.set_yscale("log"); a2.set_xticks(x); a2.set_xticklabels(labels)
    a2.set_ylabel(r"reaction latency after last round ($\mu$s)"); a2.set_xlabel("decoder: full horizon or window $(W,F)$")
    a2.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.tight_layout()
    out = ROOT / "figures"
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(out / f"window_latency.{ext}")
    print("wrote", out / "window_latency.svg")


if __name__ == "__main__":
    main()
