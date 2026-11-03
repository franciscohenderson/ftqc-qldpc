"""Tests for benchmark result storage and publication plotting utilities."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from benchmarks.plotting import (
    estimate_threshold,
    make_all_figures,
    plot_depth_scaling,
    plot_latency_hist,
    plot_logical_vs_p,
    plot_threshold,
    wilson_interval,
)
from benchmarks.storage import (
    RESULT_COLUMNS,
    SCHEMA_VERSION,
    latency_percentiles,
    load_latencies,
    load_results,
    save_latencies,
    save_results,
)


def _make_synthetic_rows(
    p_th: float = 0.007,
    amplitude: float = 0.1,
    distances: tuple[int, ...] = (3, 5, 7),
) -> list[dict]:
    """Generate synthetic Monte Carlo rows following P_L = A * (p / p_th)**((d + 1) // 2)."""
    p_grid = np.geomspace(0.002, 0.015, 9)
    shots = 100_000
    rows: list[dict] = []
    for idx_d, d in enumerate(distances):
        exponent = (d + 1) // 2
        n_qubits = 2 * d * d
        k_qubits = 1
        rounds = d
        code_name = f"surface_d{d}"
        for idx_p, p in enumerate(p_grid):
            p_logical = float(np.clip(amplitude * (p / p_th) ** exponent, 1e-7, 0.99))
            failures = int(round(p_logical * shots))
            p_logical_exact = failures / shots
            p_logical_per_round = float(1.0 - (1.0 - p_logical_exact) ** (1.0 / rounds))
            rows.append(
                {
                    "code": code_name,
                    "n": n_qubits,
                    "k": k_qubits,
                    "d": d,
                    "p": float(p),
                    "rounds": rounds,
                    "shots": shots,
                    "failures": failures,
                    "p_logical": float(p_logical_exact),
                    "p_logical_per_round": p_logical_per_round,
                    "decoder": "bp_osd",
                    "seed": 1000 + idx_d * 100 + idx_p,
                    "wall_s": 0.25 + 0.05 * idx_d,
                }
            )
    return rows


def test_parquet_and_hdf5_roundtrip_and_append(tmp_path: Path) -> None:
    """Verify Parquet and HDF5 save/load roundtrips, schema attributes, and append mode."""
    rows = _make_synthetic_rows()
    expected_df = pd.DataFrame(rows)[RESULT_COLUMNS]

    parquet_path = tmp_path / "results.parquet"
    save_results(rows, parquet_path)
    loaded_pq = load_results(parquet_path)
    assert list(loaded_pq.columns) == RESULT_COLUMNS
    pd.testing.assert_frame_equal(loaded_pq, expected_df, check_exact=False, rtol=1e-12)

    # Test append on Parquet
    save_results(rows[:3], parquet_path, append=True)
    loaded_pq_appended = load_results(parquet_path)
    assert len(loaded_pq_appended) == len(rows) + 3

    hdf5_path = tmp_path / "results.h5"
    save_results(rows, hdf5_path)
    with h5py.File(hdf5_path, "r") as h5f:
        assert h5f.attrs["schema_version"] == SCHEMA_VERSION
        for col in RESULT_COLUMNS:
            assert col in h5f

    loaded_h5 = load_results(hdf5_path)
    assert list(loaded_h5.columns) == RESULT_COLUMNS
    pd.testing.assert_frame_equal(loaded_h5, expected_df, check_exact=False, rtol=1e-12)

    # Test append on HDF5
    save_results(rows[:2], hdf5_path, append=True)
    loaded_h5_appended = load_results(hdf5_path)
    assert len(loaded_h5_appended) == len(rows) + 2


def test_storage_validation_missing_columns(tmp_path: Path) -> None:
    """Ensure missing columns raise ValueError when saving or loading."""
    incomplete_row = {"code": "BB_72_12_6", "n": 72, "k": 12}
    with pytest.raises(ValueError, match="missing required result columns"):
        save_results([incomplete_row], tmp_path / "bad.parquet")

    bad_df = pd.DataFrame([incomplete_row])
    with pytest.raises(ValueError, match="Missing required result columns"):
        save_results(bad_df, tmp_path / "bad.h5")

    bad_h5 = tmp_path / "corrupt.hdf5"
    with h5py.File(bad_h5, "w") as h5f:
        h5f.create_dataset("n", data=np.array([72], dtype=np.int64))
    with pytest.raises(ValueError, match="missing required result columns"):
        load_results(bad_h5)


def test_latencies_hdf5_and_percentiles(tmp_path: Path) -> None:
    """Verify HDF5 latency storage structure and known-array percentile calculations."""
    known = np.arange(1, 1001, dtype=np.uint64) * 100  # 100 ns .. 100_000 ns
    stats = latency_percentiles(known)
    assert set(stats.keys()) == {"P50", "P95", "P99", "P99.9", "mean", "max"}
    assert np.isclose(stats["P50"], np.percentile(known, 50.0))
    assert np.isclose(stats["P95"], np.percentile(known, 95.0))
    assert np.isclose(stats["P99"], np.percentile(known, 99.0))
    assert np.isclose(stats["P99.9"], np.percentile(known, 99.9))
    assert np.isclose(stats["mean"], 50_050.0)
    assert np.isclose(stats["max"], 100_000.0)

    lat_path = tmp_path / "latencies.h5"
    save_latencies(lat_path, decoder="bp_osd", code="BB_72_12_6", p=0.001, latencies_ns=known)
    save_latencies(lat_path, decoder="uf", code="BB_72_12_6", p=0.003, latencies_ns=known * 2)

    with h5py.File(lat_path, "r") as h5f:
        ds = h5f["bp_osd/BB_72_12_6/p=0.001/latency_ns"]
        assert ds.dtype == np.uint64
        assert ds.compression == "gzip"

    loaded = load_latencies(lat_path)
    assert "bp_osd/BB_72_12_6/p=0.001" in loaded
    assert "uf/BB_72_12_6/p=0.003" in loaded
    np.testing.assert_array_equal(loaded["bp_osd/BB_72_12_6/p=0.001"], known)
    np.testing.assert_array_equal(loaded["uf/BB_72_12_6/p=0.003"], known * 2)


def test_wilson_interval_contains_p_hat() -> None:
    """Verify vectorized Wilson interval contains empirical proportion p_hat = failures / shots."""
    failures = np.array([0, 1, 15, 250, 1000], dtype=np.int64)
    shots = np.array([1000, 1000, 1000, 1000, 1000], dtype=np.int64)
    p_hat = failures / shots

    lo, hi = wilson_interval(failures, shots, z=1.96)
    assert lo.shape == p_hat.shape
    assert hi.shape == p_hat.shape
    assert np.all(lo <= p_hat + 1e-12)
    assert np.all(p_hat <= hi + 1e-12)
    assert np.all((lo >= 0.0) & (hi <= 1.0))
    assert np.all(lo < hi)


def test_estimate_threshold_recovers_synthetic_pth() -> None:
    """Verify estimate_threshold recovers p_th ~ 0.007 within 15% and handles non-crossing curves."""
    p_th_true = 0.007
    df = pd.DataFrame(_make_synthetic_rows(p_th=p_th_true, amplitude=0.007))
    result = estimate_threshold(df)

    assert "threshold" in result
    assert "crossings" in result
    assert "pseudo_thresholds" in result
    assert len(result["crossings"]) == 2
    assert abs(result["threshold"] - p_th_true) / p_th_true <= 0.15

    for code_name in df["code"].unique():
        assert code_name in result["pseudo_thresholds"]
        assert abs(result["pseudo_thresholds"][code_name] - p_th_true) / p_th_true <= 0.15

    # Non-crossing parallel curves must return NaN without raising
    df_no_cross = df.copy()
    df_no_cross["p_logical"] = df_no_cross["d"] * 0.01 * df_no_cross["p"]
    res_no_cross = estimate_threshold(df_no_cross)
    assert np.isnan(res_no_cross["threshold"])
    assert res_no_cross["crossings"] == []
    for val in res_no_cross["pseudo_thresholds"].values():
        assert np.isnan(val)


def test_plotting_functions_generate_pdf_and_svg(tmp_path: Path) -> None:
    """Ensure all plotting functions generate non-empty PDF and SVG files in tmp_path."""
    rows = _make_synthetic_rows()
    df = pd.DataFrame(rows)

    results_path = tmp_path / "mc_results.parquet"
    save_results(df, results_path)

    latency_path = tmp_path / "latencies.h5"
    rng = np.random.default_rng(42)
    lat_bp = rng.lognormal(mean=9.0, sigma=0.4, size=500).astype(np.uint64)
    lat_uf = rng.lognormal(mean=7.5, sigma=0.3, size=500).astype(np.uint64)
    save_latencies(latency_path, "bp_osd", "surface_d5", 0.005, lat_bp)
    save_latencies(latency_path, "uf", "surface_d5", 0.005, lat_uf)

    stem_logical = tmp_path / "fig_logical"
    fig1 = plot_logical_vs_p(df, stem_logical, per_round=True)
    assert fig1 is not None
    assert stem_logical.with_suffix(".pdf").is_file()
    assert stem_logical.with_suffix(".svg").is_file()

    stem_thresh = tmp_path / "fig_threshold"
    fig2 = plot_threshold(df, stem_thresh)
    assert fig2 is not None
    assert stem_thresh.with_suffix(".pdf").is_file()
    assert stem_thresh.with_suffix(".svg").is_file()

    stem_lat = tmp_path / "fig_latency"
    fig3 = plot_latency_hist({"BP-OSD": lat_bp, "UF": lat_uf}, stem_lat)
    assert fig3 is not None
    assert stem_lat.with_suffix(".pdf").is_file()
    assert stem_lat.with_suffix(".svg").is_file()

    df_depth = pd.DataFrame(
        [
            {"code": "BB", "n": 72, "k": 12, "depth": 28, "cnot_count": 432, "rounds": 6},
            {"code": "BB", "n": 144, "k": 12, "depth": 32, "cnot_count": 1728, "rounds": 12},
            {"code": "Surface", "n": 50, "k": 1, "depth": 22, "cnot_count": 900, "rounds": 5},
            {"code": "Surface", "n": 98, "k": 1, "depth": 22, "cnot_count": 2548, "rounds": 7},
        ]
    )
    stem_depth = tmp_path / "fig_depth"
    fig4 = plot_depth_scaling(df_depth, stem_depth)
    assert fig4 is not None
    assert stem_depth.with_suffix(".pdf").is_file()
    assert stem_depth.with_suffix(".svg").is_file()

    all_out = tmp_path / "all_figs"
    figs = make_all_figures(results_path, latency_path, all_out)
    assert set(figs.keys()) == {"logical_vs_p", "threshold", "latency_hist"}
    for name in figs:
        assert (all_out / f"{name}.pdf").is_file()
        assert (all_out / f"{name}.svg").is_file()

    plt.close("all")
