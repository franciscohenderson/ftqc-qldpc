"""Publication-quality plotting and threshold estimation for QEC benchmarks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from benchmarks.storage import latency_percentiles, load_latencies, load_results

matplotlib.use("Agg")

SINGLE_COL_SIZE: tuple[float, float] = (3.4, 2.6)
DOUBLE_COL_SIZE: tuple[float, float] = (7.0, 2.8)
PUBLICATION_DPI: int = 300


def apply_publication_style() -> list[Any]:
    """Configure matplotlib and seaborn for publication-grade figures."""
    sns.set_theme(style="ticks", palette="colorblind")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.2,
            "lines.markersize": 4.0,
        }
    )
    return sns.color_palette("colorblind")


def _save_figure(fig: Figure, out_stem: str | Path | None) -> None:
    """Save figure to both PDF and SVG formats at 300 DPI."""
    if out_stem is None:
        return
    stem_path = Path(out_stem)
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".pdf", ".svg"):
        fig.savefig(
            stem_path.with_suffix(ext),
            dpi=PUBLICATION_DPI,
            bbox_inches="tight",
        )


def wilson_interval(
    failures: np.ndarray | pd.Series | list[int] | int | float,
    shots: np.ndarray | pd.Series | list[int] | int | float,
    z: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute vectorized Wilson score confidence interval (lo, hi) for binomial proportions.

    Parameters
    ----------
    failures : array-like or scalar
        Number of observed logical failures.
    shots : array-like or scalar
        Number of Monte Carlo shots.
    z : float, default 1.96
        Normal quantile (1.96 corresponds to ~95% confidence).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Lower and upper confidence bounds ``(lo, hi)`` clipped to ``[0, 1]``.
    """
    k = np.asarray(failures, dtype=np.float64)
    n = np.asarray(shots, dtype=np.float64)
    k, n = np.broadcast_arrays(k, n)

    valid = (n > 0) & (k >= 0) & (k <= n)
    n_safe = np.where(valid, n, 1.0)
    k_safe = np.where(valid, k, 0.0)

    p_hat = k_safe / n_safe
    z2 = float(z) ** 2
    denom = 1.0 + z2 / n_safe
    center = (p_hat + z2 / (2.0 * n_safe)) / denom
    radicand = np.maximum(
        0.0, (p_hat * (1.0 - p_hat) + z2 / (4.0 * n_safe)) / n_safe
    )
    half_width = (float(z) * np.sqrt(radicand)) / denom

    lo = np.clip(center - half_width, 0.0, 1.0)
    hi = np.clip(center + half_width, 0.0, 1.0)

    lo = np.where(valid, lo, np.nan)
    hi = np.where(valid, hi, np.nan)
    return lo, hi


def _compute_yerr(
    y: np.ndarray,
    failures: np.ndarray,
    shots: np.ndarray,
    rounds: np.ndarray,
    per_round: bool,
) -> np.ndarray:
    """Compute asymmetric error bars ``(2, N)`` suitable for log-scale plotting."""
    lo, hi = wilson_interval(failures, shots, z=1.96)
    if per_round:
        r_safe = np.maximum(np.asarray(rounds, dtype=np.float64), 1.0)
        p_hat = np.where(shots > 0, failures / np.maximum(shots, 1), np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            scale = np.where(p_hat > 0, y / p_hat, 1.0 / r_safe)
        lo = np.clip(lo * scale, 0.0, 1.0)
        hi = np.clip(hi * scale, 0.0, 1.0)

    err_lo = np.maximum(0.0, y - lo)
    err_hi = np.maximum(0.0, hi - y)
    # Prevent lower error bar from reaching <= 0 on a logarithmic axis
    err_lo = np.minimum(err_lo, y * 0.99)
    err_lo = np.nan_to_num(err_lo, nan=0.0)
    err_hi = np.nan_to_num(err_hi, nan=0.0)
    return np.vstack([err_lo, err_hi])


def plot_logical_vs_p(
    df: pd.DataFrame,
    out_stem: str | Path,
    per_round: bool = True,
    show_break_even: bool = True,
) -> Figure:
    """Plot logical error rate vs physical error rate on log-log axes.

    Draws one curve per ``(code, d)`` with 95% Wilson error bars and an
    optional ``y = x`` break-even reference line.
    """
    palette = apply_publication_style()
    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)

    y_col = "p_logical_per_round" if per_round else "p_logical"
    if y_col not in df.columns:
        y_col = "p_logical"

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    grouped = list(df.groupby(["code", "d"], sort=True))

    all_p: list[float] = []
    for idx, ((code, d), sub) in enumerate(grouped):
        sub_sorted = sub.sort_values("p")
        mask = (sub_sorted["p"] > 0) & (sub_sorted[y_col] > 0)
        sub_valid = sub_sorted.loc[mask]
        if sub_valid.empty:
            continue

        p_vals = sub_valid["p"].to_numpy(dtype=np.float64)
        y_vals = sub_valid[y_col].to_numpy(dtype=np.float64)
        failures = (
            sub_valid["failures"].to_numpy(dtype=np.float64)
            if "failures" in sub_valid.columns
            else np.round(y_vals * 1000.0)
        )
        shots = (
            sub_valid["shots"].to_numpy(dtype=np.float64)
            if "shots" in sub_valid.columns
            else np.full_like(y_vals, 1000.0)
        )
        rounds = (
            sub_valid["rounds"].to_numpy(dtype=np.float64)
            if "rounds" in sub_valid.columns
            else np.ones_like(y_vals)
        )

        yerr = _compute_yerr(y_vals, failures, shots, rounds, per_round=per_round)
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]

        ax.errorbar(
            p_vals,
            y_vals,
            yerr=yerr,
            label=f"{code} (d={d})",
            color=color,
            marker=marker,
            capsize=2.0,
            elinewidth=0.8,
        )
        all_p.extend(p_vals.tolist())

    if show_break_even and all_p:
        p_min, p_max = min(all_p), max(all_p)
        p_ref = np.geomspace(p_min, p_max, 50)
        ax.plot(
            p_ref,
            p_ref,
            linestyle=":",
            color="0.35",
            linewidth=1.0,
            label=r"$P_L = p$",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(
        r"Logical error rate per round $p_{L,\mathrm{round}}$"
        if per_round
        else r"Logical error rate $P_L$"
    )
    if grouped:
        ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save_figure(fig, out_stem)
    return fig


def _find_log_crossings(
    p1: np.ndarray,
    y1: np.ndarray,
    p2: np.ndarray,
    y2: np.ndarray,
) -> list[float]:
    """Find crossings between two curves by linearly interpolating log(y) vs log(p)."""
    m1 = np.isfinite(p1) & np.isfinite(y1) & (p1 > 0) & (y1 > 0)
    m2 = np.isfinite(p2) & np.isfinite(y2) & (p2 > 0) & (y2 > 0)
    if np.count_nonzero(m1) < 2 or np.count_nonzero(m2) < 2:
        return []

    x1 = np.log(p1[m1])
    ly1 = np.log(y1[m1])
    ord1 = np.argsort(x1)
    x1, ly1 = x1[ord1], ly1[ord1]

    x2 = np.log(p2[m2])
    ly2 = np.log(y2[m2])
    ord2 = np.argsort(x2)
    x2, ly2 = x2[ord2], ly2[ord2]

    x_min = max(float(x1[0]), float(x2[0]))
    x_max = min(float(x1[-1]), float(x2[-1]))
    if not (x_min < x_max):
        return []

    grid = np.unique(
        np.concatenate(
            [
                x1[(x1 >= x_min) & (x1 <= x_max)],
                x2[(x2 >= x_min) & (x2 <= x_max)],
                np.linspace(x_min, x_max, 64),
            ]
        )
    )
    if grid.size < 2:
        return []

    diff = np.interp(grid, x1, ly1) - np.interp(grid, x2, ly2)
    roots: list[float] = []
    for i in range(len(grid) - 1):
        d0, d1 = diff[i], diff[i + 1]
        if d0 == 0.0 and d1 == 0.0:
            continue
        if d0 == 0.0:
            roots.append(float(np.exp(grid[i])))
        elif d0 * d1 < 0.0:
            x_root = grid[i] - d0 * (grid[i + 1] - grid[i]) / (d1 - d0)
            roots.append(float(np.exp(x_root)))
    if not roots and diff[-1] == 0.0 and not np.all(diff == 0.0):
        roots.append(float(np.exp(grid[-1])))

    if not roots:
        return []
    return [float(np.median(roots))]


def estimate_threshold(df: pd.DataFrame) -> dict[str, Any]:
    """Estimate code threshold and pseudo-thresholds by log-log interpolation.

    For consecutive distances ``(d_i, d_{i+1})``, interpolates ``log P_L(p)`` vs
    ``log p`` to find crossings. Also computes the pseudo-threshold for each
    code where total ``P_L(p) = p``.

    Returns
    -------
    dict
        ``{"threshold": float, "crossings": list[float], "pseudo_thresholds": dict[str, float]}``
        Uses ``NaN`` when no crossing exists without raising exceptions.
    """
    if df is None or df.empty or "p" not in df.columns or "p_logical" not in df.columns:
        return {
            "threshold": float("nan"),
            "crossings": [],
            "pseudo_thresholds": {},
        }

    crossings: list[float] = []

    # Determine whether each `code` has multiple distances or represents a single distance
    code_d_counts = df.groupby("code")["d"].nunique() if "code" in df.columns and "d" in df.columns else pd.Series(dtype=int)
    if not code_d_counts.empty and (code_d_counts >= 2).any():
        families = [sub for _, sub in df.groupby("code", sort=True)]
    else:
        families = [df]

    for fam_df in families:
        if "d" not in fam_df.columns:
            continue
        distances = sorted(fam_df["d"].dropna().unique())
        for d1, d2 in zip(distances[:-1], distances[1:]):
            s1 = (
                fam_df[fam_df["d"] == d1]
                .groupby("p", as_index=False)["p_logical"]
                .mean()
                .sort_values("p")
            )
            s2 = (
                fam_df[fam_df["d"] == d2]
                .groupby("p", as_index=False)["p_logical"]
                .mean()
                .sort_values("p")
            )
            pair_roots = _find_log_crossings(
                s1["p"].to_numpy(dtype=np.float64),
                s1["p_logical"].to_numpy(dtype=np.float64),
                s2["p"].to_numpy(dtype=np.float64),
                s2["p_logical"].to_numpy(dtype=np.float64),
            )
            crossings.extend(pair_roots)

    threshold = float(np.nanmedian(crossings)) if crossings else float("nan")

    pseudo_thresholds: dict[str, float] = {}
    if "code" in df.columns:
        for code, code_df in df.groupby("code", sort=True):
            if "d" in code_df.columns and code_df["d"].nunique() > 1:
                code_roots: list[float] = []
                for _, d_df in code_df.groupby("d", sort=True):
                    agg = (
                        d_df.groupby("p", as_index=False)["p_logical"]
                        .mean()
                        .sort_values("p")
                    )
                    p_arr = agg["p"].to_numpy(dtype=np.float64)
                    pl_arr = agg["p_logical"].to_numpy(dtype=np.float64)
                    roots = _find_log_crossings(p_arr, pl_arr, p_arr, p_arr)
                    code_roots.extend(roots)
                pseudo_thresholds[str(code)] = (
                    float(np.nanmedian(code_roots)) if code_roots else float("nan")
                )
            else:
                agg = (
                    code_df.groupby("p", as_index=False)["p_logical"]
                    .mean()
                    .sort_values("p")
                )
                p_arr = agg["p"].to_numpy(dtype=np.float64)
                pl_arr = agg["p_logical"].to_numpy(dtype=np.float64)
                roots = _find_log_crossings(p_arr, pl_arr, p_arr, p_arr)
                pseudo_thresholds[str(code)] = (
                    float(roots[0]) if roots else float("nan")
                )

    return {
        "threshold": threshold,
        "crossings": crossings,
        "pseudo_thresholds": pseudo_thresholds,
    }


def plot_threshold(df: pd.DataFrame, out_stem: str | Path) -> Figure:
    """Plot logical error rate curves with vertical crossing lines and threshold band."""
    palette = apply_publication_style()
    fig, ax = plt.subplots(figsize=SINGLE_COL_SIZE)

    est = estimate_threshold(df)
    threshold = est["threshold"]
    crossings = est["crossings"]

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    grouped = list(df.groupby(["code", "d"], sort=True))
    for idx, ((code, d), sub) in enumerate(grouped):
        sub_sorted = sub.sort_values("p")
        mask = (sub_sorted["p"] > 0) & (sub_sorted["p_logical"] > 0)
        sub_valid = sub_sorted.loc[mask]
        if sub_valid.empty:
            continue

        p_vals = sub_valid["p"].to_numpy(dtype=np.float64)
        y_vals = sub_valid["p_logical"].to_numpy(dtype=np.float64)
        failures = (
            sub_valid["failures"].to_numpy(dtype=np.float64)
            if "failures" in sub_valid.columns
            else np.round(y_vals * 1000.0)
        )
        shots = (
            sub_valid["shots"].to_numpy(dtype=np.float64)
            if "shots" in sub_valid.columns
            else np.full_like(y_vals, 1000.0)
        )
        rounds = (
            sub_valid["rounds"].to_numpy(dtype=np.float64)
            if "rounds" in sub_valid.columns
            else np.ones_like(y_vals)
        )
        yerr = _compute_yerr(y_vals, failures, shots, rounds, per_round=False)

        ax.errorbar(
            p_vals,
            y_vals,
            yerr=yerr,
            label=f"{code} (d={d})",
            color=palette[idx % len(palette)],
            marker=markers[idx % len(markers)],
            capsize=2.0,
            elinewidth=0.8,
        )

    for c_val in crossings:
        if np.isfinite(c_val) and c_val > 0:
            ax.axvline(
                c_val,
                color="0.45",
                linestyle="--",
                linewidth=0.8,
                alpha=0.7,
            )

    if np.isfinite(threshold) and threshold > 0:
        ax.axvline(
            threshold,
            color="black",
            linestyle="-.",
            linewidth=1.1,
            label=rf"$p_{{\mathrm{{th}}}} \approx {threshold:.4f}$",
        )
        valid_crossings = [c for c in crossings if np.isfinite(c) and c > 0]
        if len(valid_crossings) >= 2:
            lo_band, hi_band = min(valid_crossings), max(valid_crossings)
            if lo_band == hi_band:
                lo_band, hi_band = threshold * 0.95, threshold * 1.05
        else:
            lo_band, hi_band = threshold * 0.95, threshold * 1.05
        ax.axvspan(lo_band, hi_band, color="0.5", alpha=0.18)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Physical error rate $p$")
    ax.set_ylabel(r"Logical error rate $P_L$")
    if grouped:
        ax.legend(frameon=False, loc="best")
    sns.despine(ax=ax)
    fig.tight_layout()
    _save_figure(fig, out_stem)
    return fig


def plot_latency_hist(
    latencies_by_label: dict[str, np.ndarray],
    out_stem: str | Path,
) -> Figure:
    """Plot log-scale latency histograms with labeled P50, P95, and P99 vertical lines."""
    palette = apply_publication_style()
    n_panels = max(1, len(latencies_by_label))
    fig_size = SINGLE_COL_SIZE if n_panels == 1 else (DOUBLE_COL_SIZE[0], 2.6 * ((n_panels + 1) // 2))
    fig, axes = plt.subplots(
        1,
        n_panels,
        figsize=DOUBLE_COL_SIZE if n_panels > 1 else SINGLE_COL_SIZE,
        squeeze=False,
    )
    ax_list = axes.ravel()

    for idx, (label, raw_lat) in enumerate(latencies_by_label.items()):
        ax = ax_list[idx]
        arr = np.asarray(raw_lat, dtype=np.float64).ravel()
        arr_pos = arr[arr > 0]
        if arr_pos.size == 0:
            ax.set_title(str(label))
            continue

        vmin, vmax = float(np.min(arr_pos)), float(np.max(arr_pos))
        if vmin < vmax:
            bins = np.geomspace(vmin, vmax, 30)
            ax.set_xscale("log")
        else:
            bins = 15

        ax.hist(
            arr_pos,
            bins=bins,
            color=palette[idx % len(palette)],
            alpha=0.65,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_yscale("log")

        stats = latency_percentiles(arr_pos)
        line_specs = [
            ("P50", stats["P50"], "-", "black"),
            ("P95", stats["P95"], "--", "0.25"),
            ("P99", stats["P99"], ":", "0.4"),
        ]
        for pname, pval, lstyle, lcolor in line_specs:
            ax.axvline(
                pval,
                linestyle=lstyle,
                color=lcolor,
                linewidth=1.0,
                label=f"{pname}: {pval:.0f} ns",
            )

        ax.set_title(str(label))
        ax.set_xlabel("Latency (ns)")
        if idx == 0:
            ax.set_ylabel("Count")
        ax.legend(frameon=False, loc="best")
        sns.despine(ax=ax)

    fig.tight_layout()
    _save_figure(fig, out_stem)
    return fig


def _infer_logical_k(df_depth: pd.DataFrame) -> np.ndarray:
    """Extract logical qubit count k from column ``k`` or code name if available."""
    if "k" in df_depth.columns:
        return np.maximum(df_depth["k"].to_numpy(dtype=np.float64), 1.0)
    k_vals: list[float] = []
    for code_str in df_depth["code"].astype(str):
        nums = re.findall(r"\d+", code_str)
        if len(nums) >= 2:
            k_vals.append(max(float(nums[1]), 1.0))
        else:
            k_vals.append(1.0)
    return np.asarray(k_vals, dtype=np.float64)


def plot_depth_scaling(df_depth: pd.DataFrame, out_stem: str | Path) -> Figure:
    """Plot circuit depth vs n and syndrome extraction CNOT overhead vs n.

    Expects columns ``code, n, depth, cnot_count, rounds`` (and optionally ``k``).
    """
    required = {"code", "n", "depth", "cnot_count", "rounds"}
    missing = required - set(df_depth.columns)
    if missing:
        raise ValueError(f"df_depth is missing required columns: {sorted(missing)}")

    palette = apply_publication_style()
    fig, (ax_depth, ax_cnot) = plt.subplots(1, 2, figsize=DOUBLE_COL_SIZE)

    work = df_depth.copy()
    k_arr = _infer_logical_k(work)
    rounds_arr = np.maximum(work["rounds"].to_numpy(dtype=np.float64), 1.0)
    work["cnot_overhead"] = (
        work["cnot_count"].to_numpy(dtype=np.float64) / (rounds_arr * k_arr)
    )

    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    for idx, (code, sub) in enumerate(work.groupby("code", sort=True)):
        sub_sorted = sub.sort_values("n")
        color = palette[idx % len(palette)]
        marker = markers[idx % len(markers)]

        ax_depth.plot(
            sub_sorted["n"],
            sub_sorted["depth"],
            label=str(code),
            color=color,
            marker=marker,
        )
        ax_cnot.plot(
            sub_sorted["n"],
            sub_sorted["cnot_overhead"],
            label=str(code),
            color=color,
            marker=marker,
        )

    ax_depth.set_xlabel(r"Physical qubits $n$")
    ax_depth.set_ylabel("Circuit depth")
    ax_depth.legend(frameon=False, loc="best")
    sns.despine(ax=ax_depth)

    ax_cnot.set_xlabel(r"Physical qubits $n$")
    ax_cnot.set_ylabel("CNOTs / round / logical qubit")
    ax_cnot.legend(frameon=False, loc="best")
    sns.despine(ax=ax_cnot)

    fig.tight_layout()
    _save_figure(fig, out_stem)
    return fig


def make_all_figures(
    results_path: str | Path,
    latency_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Figure]:
    """Generate all standard benchmark figures from stored results and latencies."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df_results = load_results(results_path)
    latencies = load_latencies(latency_path)

    figures: dict[str, Figure] = {}
    figures["logical_vs_p"] = plot_logical_vs_p(
        df_results, out_path / "logical_vs_p", per_round=True
    )
    figures["threshold"] = plot_threshold(df_results, out_path / "threshold")
    if latencies:
        figures["latency_hist"] = plot_latency_hist(
            latencies, out_path / "latency_hist"
        )
    if {"depth", "cnot_count"}.issubset(df_results.columns):
        figures["depth_scaling"] = plot_depth_scaling(
            df_results, out_path / "depth_scaling"
        )
    return figures
