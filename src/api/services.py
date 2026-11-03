"""Service layer for the API: code catalog, schedules, compilation, benchmark data. No FastAPI imports."""
from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from algebra.qldpc import BB_ZOO, CSSCode, get_code
from benchmarks.plotting import estimate_threshold
from benchmarks.storage import load_latencies, load_results
from circuits.builder import build_memory_experiment
from circuits.interleaved import find_interleaved_schedule
from circuits.noise import NoiseModel
from circuits.qasm import compile_openqasm3
from circuits.schedule import sequential_schedule

ROOT = Path(__file__).resolve().parents[2]
CODE_NAMES = list(BB_ZOO)
MACHINE = "AMD Ryzen 5 5600G (6C/12T), AVX2 (no AVX-512), Windows 11, Python 3.14, C++20 via clang (zig c++), 16 x int16 SIMD lanes"
_sched_lock = threading.Lock()


@lru_cache(maxsize=None)
def code_of(name: str) -> CSSCode:
    if name not in BB_ZOO:
        raise KeyError(name)
    return get_code(name)


def _layers_json(layers):
    return [[[k, int(c), int(q)] for k, c, q in layer] for layer in layers]


@lru_cache(maxsize=None)
def sequential_layers(name: str):
    c = code_of(name)
    return sequential_schedule(c.hx, c.hz)


def interleaved_layers(name: str, time_limit_s: float = 90.0):
    """Interleaved schedule from the on-disk cache; computed (and cached atomically) if absent."""
    cache = ROOT / "results" / "schedules" / f"{name}.json"
    try:
        return [[tuple(e) for e in layer] for layer in json.loads(cache.read_text())]
    except (OSError, ValueError):
        pass
    with _sched_lock:
        try:
            return [[tuple(e) for e in layer] for layer in json.loads(cache.read_text())]
        except (OSError, ValueError):
            pass
        c = code_of(name)
        layers = find_interleaved_schedule(c.hx, c.hz, seed=0, time_limit_s=time_limit_s)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(layers))
        os.replace(tmp, cache)
        return layers


def get_layers(name: str, schedule: str):
    return interleaved_layers(name) if schedule == "interleaved" else sequential_layers(name)


def _params() -> dict:
    rows = json.loads((ROOT / "exports" / "code_parameters.json").read_text(encoding="utf-8"))
    return {r["code"]: r for r in rows}


def tanner(name: str, with_layers: bool = True) -> dict:
    c = code_of(name)
    l, m = c.meta["l"], c.meta["m"]
    lm = l * m
    data = []
    for i in range(c.n):
        blk = "L" if i < lm else "R"
        j = i % lm
        a, b = j // m, j % m
        data.append({"id": i, "x": a + (0.0 if blk == "L" else 0.5), "y": b + (0.0 if blk == "L" else 0.5), "block": blk})
    xs = [{"id": r, "x": (r // m) + 0.5, "y": float(r % m)} for r in range(c.hx.shape[0])]
    zs = [{"id": r, "x": float(r // m), "y": (r % m) + 0.5} for r in range(c.hz.shape[0])]
    t = {
        "l": l, "m": m, "data": data, "x_checks": xs, "z_checks": zs,
        "x_edges": [[int(i), int(j)] for i, j in zip(*np.nonzero(c.hx))],
        "z_edges": [[int(i), int(j)] for i, j in zip(*np.nonzero(c.hz))],
    }
    if with_layers:
        t["layers"] = {"sequential": _layers_json(sequential_layers(name))}
        cache = ROOT / "results" / "schedules" / f"{name}.json"
        if cache.exists():
            t["layers"]["interleaved"] = _layers_json(interleaved_layers(name))
        else:
            t["layers"]["interleaved"] = None      # not precomputed yet (BB288 may take minutes)
    return t


def catalog(with_tanner: bool = True) -> dict:
    params = _params()
    out = []
    for name in CODE_NAMES:
        c = code_of(name)
        p = params.get(name, {})
        cache_exists = (ROOT / "results" / "schedules" / f"{name}.json").exists()
        out.append({
            "name": name, "n": c.n, "k": c.k,
            "d_x": p.get("dX_upper"), "d_z": p.get("dZ_upper"),
            "d_x_lower": p.get("dX_lower"), "d_z_lower": p.get("dZ_lower"),
            "exact": bool(p.get("dX_exact") and p.get("dZ_exact")),
            "published_d": c.meta["published_d"],
            "m_x": int(c.hx.shape[0]), "m_z": int(c.hz.shape[0]),
            "w_col": p.get("w_col"), "w_row": p.get("w_row"), "ilp_seconds": p.get("ilp_seconds"),
            "depth": {"sequential": len(sequential_layers(name)),
                      "interleaved": len(interleaved_layers(name)) if cache_exists else None},
            "cnot_per_round": int(c.hx.sum() + c.hz.sum()),
            **({"tanner": tanner(name)} if with_tanner else {}),
        })
    return {"codes": out}


def noise_from(p: float, t1_us, t2_us, layer_time_us: float) -> NoiseModel:
    if t1_us and t2_us:
        return NoiseModel.from_coherence(p, t1_us, t2_us, layer_time_us)
    return NoiseModel.uniform(p)


def compile_request(code: str, schedule: str, rounds: int, basis: str, p: float, t1_us, t2_us, layer_time_us: float,
                    decoder: str = "full", W: int = 3, F: int = 1) -> dict:
    t0 = time.perf_counter()
    c = code_of(code)
    layers = get_layers(code, schedule)
    noise = noise_from(p, t1_us, t2_us, layer_time_us)
    exp = build_memory_experiment(c, rounds, noise, basis=basis, layers=layers)
    windowed = decoder == "window"
    if windowed:
        if not (1 <= F <= W):
            raise ValueError("sliding-window decoding needs 1 <= F <= W")
        if F == W and W < rounds + 1:
            raise ValueError("sliding-window decoding needs F < W (one round of look-ahead) unless one window covers all rounds")
        if F > rounds:
            raise ValueError("F cannot exceed the number of rounds")
    # window mode: decoder call every F rounds (and after the last round); full horizon: one call after the last round
    qasm = compile_openqasm3(c, layers, rounds, basis=basis, feedback_every=F if windowed else rounds)
    dem_summary = None
    decoder_plan = None
    if c.n <= 144 or rounds <= 3:
        dem = exp.dem()
        if windowed:
            from decoder.dem import dem_to_problem
            from decoder.window import WindowDecoder, detector_rounds
            decoder_plan = WindowDecoder(dem_to_problem(dem), detector_rounds(exp.circuit), W=W, F=F).plan()
        mx = 0
        for inst in dem.flattened():
            if inst.type == "error":
                mx = max(mx, sum(1 for t in inst.targets_copy() if t.is_relative_detector_id()))
        dem_summary = {"num_detectors": dem.num_detectors, "num_errors": dem.num_errors,
                       "num_observables": dem.num_observables, "max_detectors_per_error": mx}
    ip = noise.idle_pauli()
    return {
        "code": code, "schedule": schedule, "rounds": rounds, "qasm": qasm,
        "stats": {
            "n": c.n, "ancillas": int(c.hx.shape[0] + c.hz.shape[0]), "qubits_total": int(c.n + c.hx.shape[0] + c.hz.shape[0]),
            "cnot_layers": len(layers), "depth_per_round": exp.depth_per_round, "cnot_per_round": exp.cnot_per_round,
            "cnot_total": exp.cnot_per_round * rounds, "qasm_lines": qasm.count("\n"),
            "stim_instructions": len(exp.circuit),
        },
        "dem": dem_summary,
        "decoder": {"mode": decoder, "W": W if windowed else None, "F": F if windowed else None,
                    "feedback_every_rounds": F if windowed else 1, "plan": decoder_plan},
        "noise": {"p_gate": noise.p_gate, "p_prep": noise.p_prep, "p_meas": noise.p_meas, "p_idle": noise.p_idle,
                  "t1_us": noise.t1_us, "t2_us": noise.t2_us, "idle_pauli": list(ip) if ip else None},
        "elapsed_ms": round((time.perf_counter() - t0) * 1e3, 1),
    }


def _wilson(f: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 1.0
    ph = f / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    lo, hi = float(max(0.0, c - h)), float(min(1.0, c + h))
    return (0.0 if f == 0 else min(lo, ph)), (1.0 if f == n else max(hi, ph))


_BENCH_LOCK = threading.Lock()
_BENCH_CACHE: dict | None = None


def _run_payload(run: str, label: str, schedule: str) -> dict:
    d = ROOT / "results" / run
    df = load_results(d / "results.parquet")
    pts = []
    for _, r in df.sort_values(["code", "p"]).iterrows():
        lo, hi = _wilson(int(r["failures"]), int(r["shots"]))
        pts.append({"code": r["code"], "n": int(r["n"]), "k": int(r["k"]), "d": int(r["d"]), "p": float(r["p"]),
                    "shots": int(r["shots"]), "failures": int(r["failures"]), "p_logical": float(r["p_logical"]),
                    "p_logical_per_round": float(r["p_logical_per_round"]), "lo": lo, "hi": hi})
    th = estimate_threshold(df)
    clean = lambda v: None if v is None or (isinstance(v, float) and v != v) else float(v)
    lat = load_latencies(d / "latency.h5")
    pooled: dict[tuple, list] = {}
    for k, v in lat.items():
        mode, code, _ = k.split("/")
        pooled.setdefault((mode, code), []).append(v)
    lat_out = []
    for (mode, code), v in sorted(pooled.items()):
        a = np.concatenate(v).astype(np.float64) / 1e3
        a = a[a > 0]
        edges = np.logspace(np.log10(a.min()), np.log10(a.max()), 41)
        counts, _ = np.histogram(a, bins=edges)
        pc = np.percentile(a, [50, 95, 99, 99.9])
        lat_out.append({"mode": mode, "code": code, "n_samples": int(a.size),
                        "percentiles_us": {"P50": float(pc[0]), "P95": float(pc[1]), "P99": float(pc[2]), "P99.9": float(pc[3]),
                                           "mean": float(a.mean()), "max": float(a.max())},
                        "hist": {"edges_us": edges.tolist(), "counts": counts.tolist()}})
    return {"label": label, "schedule": schedule, "rounds": int(df["rounds"].iloc[0]),
            "decoder": str(df["decoder"].iloc[0]), "total_shots": int(df["shots"].sum()),
            "threshold": {"estimate": clean(th["threshold"]), "crossings": [clean(x) for x in th["crossings"]],
                          "pseudo_thresholds": {k: clean(v) for k, v in th["pseudo_thresholds"].items()}},
            "points": pts, "latency": lat_out}


def benchmarks() -> dict:
    global _BENCH_CACHE
    with _BENCH_LOCK:
        if _BENCH_CACHE is None:
            kp = ROOT / "results" / "kernel_latency.json"
            _BENCH_CACHE = {
                "machine": MACHINE,
                "runs": {"final_int": _run_payload("final_int", "Interleaved (depth 7)", "interleaved"),
                         "final_seq": _run_payload("final_seq", "Sequential (depth 12)", "sequential")},
                "kernel": json.loads(kp.read_text()) if kp.exists() else {},
                "notes": [
                    "Logical error rates are totals over the whole memory experiment (6 rounds); zero failures means none observed in the shot budget, not a true zero.",
                    "Sub-microsecond latency applies to the BP kernel amortized over a 16-shot SIMD pass on the per-round code-capacity matrix; single-shot BP is ~0.6-2.4 us; full circuit-level BP+OSD decoding costs hundreds of microseconds to milliseconds.",
                    "Full-decode latency histograms pool all physical error rates, including p up to 6e-3 where most shots need OSD.",
                    "Thresholds are crossing estimates from finite-shot 6-round experiments on four small codes and depend on the noise model and decoder.",
                ]}
        return _BENCH_CACHE
