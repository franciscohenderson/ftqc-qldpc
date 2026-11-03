"""Distributed Monte Carlo sampler for circuit-level logical error rates.

Each work unit = (code, p, rounds, shots, seed).  Workers (spawned processes) cache the compiled
Stim circuit, the DEM check matrix and the C++ decoder per (code, p), sample detection events with
Stim, decode with the chosen decoder and return failure counts plus decoder latencies.  The driver
runs waves of work units for every (code, p) point until ``target_failures`` is reached or
``max_shots`` is exhausted (adaptive stopping, so low-p points get many shots and high-p few).
"""
from __future__ import annotations

import multiprocessing as mp
import time
import zlib
from dataclasses import dataclass, field

import numpy as np

from algebra.qldpc import get_code
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel
from decoder.dem import dem_to_problem
from decoder.fast import FastDecoder

_CACHE: dict = {}


@dataclass(frozen=True)
class Task:
    code: str
    p: float
    rounds: int
    shots: int
    seed: int
    max_iter: int = 30
    osd_lambda: int = 32
    stream_shots: int = 256          # shots re-decoded one-by-one for per-shot kernel latency
    noise: str = "uniform"           # "uniform" | "coherence"
    layers_key: str = "sequential"   # "sequential" | "interleaved"


def _noise(kind: str, p: float) -> NoiseModel:
    if kind == "coherence":
        return NoiseModel.from_coherence(p, t1_us=200.0, t2_us=150.0, layer_time_us=0.05)
    return NoiseModel.uniform(p)


def _layers(code, key: str):
    if key == "interleaved":
        import json
        from pathlib import Path

        from circuits.interleaved import find_interleaved_schedule
        import os
        cache = Path(__file__).resolve().parents[2] / "results" / "schedules" / f"{code.name}.json"
        try:
            return [[tuple(e) for e in layer] for layer in json.loads(cache.read_text())]
        except (OSError, ValueError):
            pass
        layers = find_interleaved_schedule(code.hx, code.hz, seed=0)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(layers))
        os.replace(tmp, cache)          # atomic: concurrent workers never see a partial file
        return layers
    return None


def _setup(t: Task):
    key = (t.code, t.p, t.rounds, t.max_iter, t.osd_lambda, t.noise, t.layers_key)
    if key not in _CACHE:
        if len(_CACHE) > 8:
            _CACHE.clear()
        code = get_code(t.code)
        exp = build_memory_experiment(code, t.rounds, _noise(t.noise, t.p), layers=_layers(code, t.layers_key))
        prob = dem_to_problem(exp.dem())
        dec = FastDecoder(prob.h, prob.priors, max_iter=t.max_iter, osd_lambda=t.osd_lambda)
        _CACHE[key] = (exp, prob, dec)
    return _CACHE[key]


def run_task(t: Task) -> dict:
    exp, prob, dec = _setup(t)
    sampler = exp.circuit.compile_detector_sampler(seed=t.seed)
    det, obs = sampler.sample(t.shots, separate_observables=True)
    det = det.astype(np.uint8)
    t0 = time.perf_counter()
    res = dec.decode_batch(det, use_osd=True)
    wall = time.perf_counter() - t0
    pred = (prob.obs @ res.errors.T.astype(np.int64) % 2).T
    wrong = (pred != obs).any(axis=1)
    # a detector-consistent but undecodable shot counts as a failure too
    wrong |= res.status == 3
    stream_lat = np.empty(0, np.uint64)
    if t.stream_shots:
        sres = dec.decode_stream(det[: t.stream_shots], use_osd=True)
        stream_lat = sres.latency_ns
    return {
        "task": t, "failures": int(wrong.sum()), "shots": int(t.shots), "wall_s": wall,
        "batch_lat": res.latency_ns, "stream_lat": stream_lat,
        "bp_conv": float((res.status == 0).mean()),
    }


@dataclass
class PointState:
    code: str
    p: float
    rounds: int
    failures: int = 0
    shots: int = 0
    wall_s: float = 0.0
    batch_lat: list = field(default_factory=list)
    stream_lat: list = field(default_factory=list)
    seed_counter: int = 0
    queued: int = 0
    n: int = 0
    k: int = 0
    d: int = 0


def run_sweep(codes: list[str], ps: list[float], rounds: dict[str, int] | int, shots_per_task: int = 2000,
              target_failures: int = 100, max_shots: int = 100_000, workers: int | None = None,
              base_seed: int = 12345, progress=print, **task_kw) -> list[PointState]:
    """Run the adaptive sweep.  Returns one PointState per (code, p)."""
    points = {}
    for c in codes:
        code = get_code(c)
        for p in ps:
            r = rounds[c] if isinstance(rounds, dict) else rounds
            points[(c, p)] = PointState(c, p, r, n=code.n, k=code.k, d=code.meta["published_d"])
    workers = workers or max(1, (mp.cpu_count() or 2) - 1)
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        while True:
            wave = []
            for (c, p), st in points.items():
                if st.failures >= target_failures or st.shots >= max_shots:
                    continue
                n_tasks = workers if st.failures < target_failures // 4 else 1
                for _ in range(n_tasks):
                    if st.shots + st.queued >= max_shots:
                        break
                    st.seed_counter += 1
                    st.queued += shots_per_task
                    ss = np.random.SeedSequence([base_seed, zlib.crc32(f"{c}|{p!r}|{st.rounds}".encode()), st.seed_counter])
                    wave.append(Task(c, p, st.rounds, shots_per_task, int(ss.generate_state(1, np.uint64)[0] & 0x7FFFFFFFFFFFFFFF), **task_kw))
            if not wave:
                break
            for out in pool.imap_unordered(run_task, wave):
                t = out["task"]
                st = points[(t.code, t.p)]
                st.failures += out["failures"]
                st.shots += out["shots"]
                st.queued -= out["shots"]
                st.wall_s += out["wall_s"]
                st.batch_lat.append(out["batch_lat"])
                st.stream_lat.append(out["stream_lat"])
            progress("  ".join(f"{c}@{p:.1e}:{s.failures}/{s.shots}" for (c, p), s in points.items()
                               if s.failures < target_failures and s.shots < max_shots) or "wave done")
    return list(points.values())


def to_rows(points: list[PointState], decoder: str = "bp-osd-int16", seed: int = 12345) -> list[dict]:
    rows = []
    for s in points:
        pl = s.failures / max(s.shots, 1)
        rows.append({
            "code": s.code, "n": s.n, "k": s.k, "d": s.d, "p": s.p, "rounds": s.rounds, "shots": s.shots,
            "failures": s.failures, "p_logical": pl,
            "p_logical_per_round": 1.0 - (1.0 - min(pl, 1.0 - 1e-12)) ** (1.0 / s.rounds),
            "decoder": decoder, "seed": seed, "wall_s": s.wall_s,
        })
    return rows
