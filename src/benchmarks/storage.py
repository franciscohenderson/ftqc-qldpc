"""Input/output utilities for benchmark Monte Carlo results and decoder latencies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA_VERSION: str = "1.0"

RESULT_COLUMNS: list[str] = [
    "code",
    "n",
    "k",
    "d",
    "p",
    "rounds",
    "shots",
    "failures",
    "p_logical",
    "p_logical_per_round",
    "decoder",
    "seed",
    "wall_s",
]

_STRING_COLUMNS: frozenset[str] = frozenset({"code", "decoder"})
_INT_COLUMNS: frozenset[str] = frozenset(
    {"n", "k", "d", "rounds", "shots", "failures", "seed"}
)
_FLOAT_COLUMNS: frozenset[str] = frozenset(
    {"p", "p_logical", "p_logical_per_round", "wall_s"}
)


def _validate_and_normalize_rows(
    rows: list[dict[str, Any]] | pd.DataFrame,
) -> pd.DataFrame:
    """Validate that all required columns exist and coerce dtypes."""
    required = set(RESULT_COLUMNS)
    if isinstance(rows, pd.DataFrame):
        missing = required - set(rows.columns)
        if missing:
            raise ValueError(
                f"Missing required result columns: {sorted(missing)}"
            )
        df = rows[RESULT_COLUMNS].copy()
    elif isinstance(rows, list):
        if len(rows) == 0:
            raise ValueError(
                "Cannot validate schema on an empty row list; required columns: "
                f"{RESULT_COLUMNS}"
            )
        for idx, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Row {idx} must be a dict, got {type(row)!r}")
            missing = required - set(row.keys())
            if missing:
                raise ValueError(
                    f"Row {idx} is missing required result columns: {sorted(missing)}"
                )
        df = pd.DataFrame(rows)[RESULT_COLUMNS].copy()
    else:
        raise ValueError(
            f"Expected list[dict] or pandas.DataFrame, got {type(rows)!r}"
        )

    for col in RESULT_COLUMNS:
        if col in _STRING_COLUMNS:
            df[col] = df[col].astype(str)
        elif col in _INT_COLUMNS:
            df[col] = df[col].astype(np.int64)
        elif col in _FLOAT_COLUMNS:
            df[col] = df[col].astype(np.float64)

    return df


def _save_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write validated results DataFrame to a Parquet file with schema metadata."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    existing_meta = table.schema.metadata or {}
    merged_meta = {
        **existing_meta,
        b"schema_version": SCHEMA_VERSION.encode("utf-8"),
    }
    table = table.replace_schema_metadata(merged_meta)
    pq.write_table(table, path)


def _load_parquet(path: Path) -> pd.DataFrame:
    """Read and validate results DataFrame from a Parquet file."""
    table = pq.read_table(path)
    df = table.to_pandas()
    return _validate_and_normalize_rows(df)


def _save_hdf5(df: pd.DataFrame, path: Path) -> None:
    """Write validated results DataFrame to an HDF5 file (one dataset per column)."""
    str_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(path, "w") as h5f:
        h5f.attrs["schema_version"] = SCHEMA_VERSION
        h5f.attrs["columns"] = RESULT_COLUMNS
        for col in RESULT_COLUMNS:
            if col in _STRING_COLUMNS:
                data = np.asarray(df[col].astype(str).tolist(), dtype=object)
                h5f.create_dataset(col, data=data, dtype=str_dtype)
            elif col in _INT_COLUMNS:
                data = df[col].to_numpy(dtype=np.int64)
                h5f.create_dataset(col, data=data, dtype=np.int64)
            else:
                data = df[col].to_numpy(dtype=np.float64)
                h5f.create_dataset(col, data=data, dtype=np.float64)


def _load_hdf5(path: Path) -> pd.DataFrame:
    """Read and validate results DataFrame from an HDF5 file."""
    data: dict[str, Any] = {}
    with h5py.File(path, "r") as h5f:
        missing = set(RESULT_COLUMNS) - set(h5f.keys())
        if missing:
            raise ValueError(
                f"HDF5 file is missing required result columns: {sorted(missing)}"
            )
        for col in RESULT_COLUMNS:
            raw = h5f[col][...]
            if col in _STRING_COLUMNS:
                data[col] = [
                    val.decode("utf-8")
                    if isinstance(val, (bytes, np.bytes_))
                    else str(val)
                    for val in raw
                ]
            elif col in _INT_COLUMNS:
                data[col] = np.asarray(raw, dtype=np.int64)
            else:
                data[col] = np.asarray(raw, dtype=np.float64)

    df = pd.DataFrame(data)[RESULT_COLUMNS]
    return _validate_and_normalize_rows(df)


def save_results(
    rows: list[dict[str, Any]] | pd.DataFrame,
    path: str | Path,
    append: bool = False,
) -> Path:
    """Save benchmark result rows to Parquet (.parquet) or HDF5 (.h5/.hdf5).

    Parameters
    ----------
    rows : list[dict] or pandas.DataFrame
        Monte Carlo result rows adhering to ``RESULT_COLUMNS``.
    path : str or Path
        Destination file path. Format is selected from the file extension.
    append : bool, default False
        If True and ``path`` exists, concatenate existing rows with ``rows``.

    Returns
    -------
    Path
        Resolved path to the written file.
    """
    target = Path(path)
    df_new = _validate_and_normalize_rows(rows)

    if append and target.exists():
        df_existing = load_results(target)
        df_to_write = pd.concat([df_existing, df_new], ignore_index=True)
        df_to_write = _validate_and_normalize_rows(df_to_write)
    else:
        df_to_write = df_new

    target.parent.mkdir(parents=True, exist_ok=True)
    suffix = target.suffix.lower()
    if suffix == ".parquet":
        _save_parquet(df_to_write, target)
    elif suffix in {".h5", ".hdf5"}:
        _save_hdf5(df_to_write, target)
    else:
        raise ValueError(
            f"Unsupported result file extension {suffix!r}; expected '.parquet', '.h5', or '.hdf5'."
        )
    return target


def load_results(path: str | Path) -> pd.DataFrame:
    """Load benchmark results from a Parquet (.parquet) or HDF5 (.h5/.hdf5) file."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".parquet":
        return _load_parquet(target)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5(target)
    raise ValueError(
        f"Unsupported result file extension {suffix!r}; expected '.parquet', '.h5', or '.hdf5'."
    )


def save_latencies(
    path: str | Path,
    decoder: str,
    code: str,
    p: float,
    latencies_ns: np.ndarray,
) -> Path:
    """Save per-shot decoder latencies in nanoseconds to an HDF5 file.

    Stores dataset ``latency_ns`` (uint64, gzip-compressed) under group
    ``{decoder}/{code}/p={p}``.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(latencies_ns, dtype=np.uint64).ravel()

    group_path = f"{decoder}/{code}/p={p}"
    with h5py.File(target, "a") as h5f:
        h5f.attrs["schema_version"] = SCHEMA_VERSION
        grp = h5f.require_group(group_path)
        grp.attrs["decoder"] = str(decoder)
        grp.attrs["code"] = str(code)
        grp.attrs["p"] = float(p)
        if "latency_ns" in grp:
            del grp["latency_ns"]
        grp.create_dataset(
            "latency_ns",
            data=arr,
            dtype=np.uint64,
            compression="gzip",
        )
    return target


def load_latencies(path: str | Path) -> dict[str, np.ndarray]:
    """Load all latency datasets from an HDF5 file.

    Returns a dictionary mapping ``"{decoder}/{code}/p={p}"`` to uint64 numpy
    arrays of latencies in nanoseconds.
    """
    target = Path(path)
    result: dict[str, np.ndarray] = {}
    with h5py.File(target, "r") as h5f:
        for decoder in h5f.keys():
            dec_grp = h5f[decoder]
            if not isinstance(dec_grp, h5py.Group):
                continue
            for code in dec_grp.keys():
                code_grp = dec_grp[code]
                if not isinstance(code_grp, h5py.Group):
                    continue
                for p_key in code_grp.keys():
                    p_grp = code_grp[p_key]
                    if isinstance(p_grp, h5py.Group) and "latency_ns" in p_grp:
                        key = f"{decoder}/{code}/{p_key}"
                        result[key] = np.asarray(
                            p_grp["latency_ns"][...], dtype=np.uint64
                        )
    return result


def latency_percentiles(latencies_ns: np.ndarray | list[int | float]) -> dict[str, float]:
    """Compute summary latency percentiles and statistics in nanoseconds.

    Returns a dictionary with keys ``P50``, ``P95``, ``P99``, ``P99.9``,
    ``mean``, and ``max``.
    """
    arr = np.asarray(latencies_ns, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("latencies_ns must not be empty.")
    p50, p95, p99, p999 = np.percentile(arr, [50.0, 95.0, 99.0, 99.9])
    return {
        "P50": float(p50),
        "P95": float(p95),
        "P99": float(p99),
        "P99.9": float(p999),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }
