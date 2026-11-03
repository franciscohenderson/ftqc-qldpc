"""Live decode session behind WS /api/v1/stream/decode (synchronous core; the server runs it in a thread)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel
from decoder.dem import dem_to_problem
from decoder.fast import FastDecoder
from decoder.hybrid import HybridDecoder
from decoder.window import WindowDecoder, detector_rounds

from .services import ROOT, _wilson, code_of, get_layers


def _load_gnn(code: str, tag: str):
    path = ROOT / "results" / "gnn" / f"{code}_{tag}.npz"
    if not path.exists():
        return None
    try:
        from decoder.gnn import QuantizedGNN
        return QuantizedGNN.load(path)
    except Exception:
        return None


def _pct(a, qs=(50, 95)):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return [0.0] * len(qs)
    return [float(x) for x in np.percentile(a, qs)]


@dataclass
class Config:
    code: str = "BB72"
    mode: str = "code_capacity"
    p: float = 0.01
    rate_hz: float = 10.0
    batch: int = 16
    schedule: str = "interleaved"
    rounds: int = 3
    decoder: str = "full"        # circuit mode only: "full" horizon or sliding "window" (W, F)
    W: int = 3
    F: int = 1

    @classmethod
    def from_msg(cls, msg: dict) -> "Config":
        from .services import CODE_NAMES
        c = cls()
        code = str(msg.get("code", c.code))
        if code not in CODE_NAMES:
            raise ValueError(f"unknown code {code!r}")
        mode = str(msg.get("mode", c.mode))
        if mode not in ("code_capacity", "circuit"):
            raise ValueError("mode must be code_capacity or circuit")
        p = float(msg.get("p", c.p))
        if not (1e-5 <= p <= 0.3):
            raise ValueError("p must be in [1e-5, 0.3]")
        schedule = str(msg.get("schedule", c.schedule))
        if schedule not in ("interleaved", "sequential"):
            raise ValueError("schedule must be interleaved or sequential")
        rounds = int(msg.get("rounds", c.rounds))
        if not (1 <= rounds <= 6):
            raise ValueError("rounds must be in 1..6 for streaming")
        decoder = str(msg.get("decoder", c.decoder))
        if decoder not in ("full", "window"):
            raise ValueError("decoder must be full or window")
        W, F = int(msg.get("W", c.W)), int(msg.get("F", c.F))
        if not (1 <= F < W <= 6):
            raise ValueError("need 1 <= F < W <= 6 (at least one round of look-ahead)")
        rate = float(msg.get("rate_hz", c.rate_hz))
        batch = int(msg.get("batch", c.batch))
        return cls(code, mode, p, min(max(rate, 0.5), 30.0), min(max(batch, 1), 64), schedule, rounds, decoder, W, F)


class DecodeSession:
    def __init__(self, cfg: Config, seed: int | None = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seq = 0
        self.t_start = time.time()
        self.counters = {"shots": 0, "bp_converged": 0, "osd": 0, "gnn": 0, "failed_decode": 0}
        self.fails = 0
        code = code_of(cfg.code)
        self.code = code
        if cfg.mode == "code_capacity":
            # Z checks detect X errors (decoded with H_Z); X checks detect Z errors (decoded with H_X)
            self.gnn_z, self.gnn_x = _load_gnn(cfg.code, "hz"), _load_gnn(cfg.code, "hx")
            self.hyb_z = HybridDecoder(code.hz, cfg.p, gnn=self.gnn_z, max_iter=20)
            self.hyb_x = HybridDecoder(code.hx, cfg.p, gnn=self.gnn_x, max_iter=20)
            self.lz, self.lx = code.logical_z().astype(np.int64), code.logical_x().astype(np.int64)
            self.gnn_available = self.gnn_z is not None and self.gnn_x is not None
        else:
            layers = get_layers(cfg.code, cfg.schedule)
            self.exp = build_memory_experiment(code, cfg.rounds, NoiseModel.uniform(cfg.p), layers=layers)
            self.prob = dem_to_problem(self.exp.dem())
            self.win = None
            if cfg.decoder == "window":
                self.win = WindowDecoder(self.prob, detector_rounds(self.exp.circuit), W=cfg.W, F=min(cfg.F, cfg.W))
            else:
                self.hyb = HybridDecoder(self.prob.h, self.prob.priors, max_iter=30)
            self.sampler = self.exp.circuit.compile_detector_sampler(seed=int(self.rng.integers(1 << 31)))
            self.coords = self.exp.circuit.get_detector_coordinates()
            self.gnn_available = False

    # ------------------------------------------------------------------ helpers
    def _tier_counts(self, tiers):
        for t in tiers:
            if t == 0:
                self.counters["bp_converged"] += 1
            elif t == 1:
                self.counters["osd"] += 1
            elif t == 2:
                self.counters["gnn"] += 1
            else:
                self.counters["failed_decode"] += 1

    def _decode_set(self, hyb: HybridDecoder, synd: np.ndarray):
        """Decode shots one by one with the hybrid decoder; returns errors, tiers, total_ns and kernel timings."""
        B = synd.shape[0]
        errs = np.zeros((B, hyb.fast.n), np.uint8)
        tiers = np.zeros(B, np.uint8)
        total = np.zeros(B, np.int64)
        for i in range(B):
            r = hyb.decode_batch(synd[i:i + 1])
            errs[i], tiers[i], total[i] = r.errors[0], r.tier[0], r.latency_ns[0]
        bp = hyb.fast.decode_stream(synd, use_osd=False).bp_ns.astype(np.int64)
        amort = hyb.fast.decode_batch(synd, use_osd=False).latency_ns.astype(np.int64)
        return errs, tiers, total, bp, amort

    # ------------------------------------------------------------------ ticks
    def tick(self) -> dict:
        self.seq += 1
        cfg, c = self.cfg, self.code
        B = cfg.batch
        if cfg.mode == "code_capacity":
            ex = (self.rng.random((B, c.n)) < cfg.p).astype(np.uint8)
            ez = (self.rng.random((B, c.n)) < cfg.p).astype(np.uint8)
            sz = (c.hz.astype(np.int64) @ ex.T % 2).T.astype(np.uint8)
            sx = (c.hx.astype(np.int64) @ ez.T % 2).T.astype(np.uint8)
            cx, tx, tot_x, bp_x, am_x = self._decode_set(self.hyb_z, sz)
            cz, tz, tot_z, bp_z, am_z = self._decode_set(self.hyb_x, sx)
            rx, rz = ex ^ cx, ez ^ cz
            ok_x = ((c.hz.astype(np.int64) @ rx.T % 2) == 0).all(0) & (tx != 255)
            ok_z = ((c.hx.astype(np.int64) @ rz.T % 2) == 0).all(0) & (tz != 255)
            lfail = (((self.lz @ rx.T) % 2).any(0)) | (((self.lx @ rz.T) % 2).any(0)) | ~ok_x | ~ok_z
            self._tier_counts(np.concatenate([tx, tz]))
            interesting = np.flatnonzero(sz.any(1) | sx.any(1))
            shot = int(interesting[0]) if interesting.size else 0
            display = {
                "shot": shot, "err_x": np.flatnonzero(ex[shot]).tolist(), "err_z": np.flatnonzero(ez[shot]).tolist(),
                "z_syndrome": np.flatnonzero(sz[shot]).tolist(), "x_syndrome": np.flatnonzero(sx[shot]).tolist(),
                "corr_x": np.flatnonzero(cx[shot]).tolist(), "corr_z": np.flatnonzero(cz[shot]).tolist(),
                "tier_x": int(tx[shot]), "tier_z": int(tz[shot]), "logical_fail": bool(lfail[shot]), "rounds": [],
            }
            total, bp, amort = np.concatenate([tot_x, tot_z]), np.concatenate([bp_x, bp_z]), np.concatenate([am_x, am_z])
        else:
            det, obs = self.sampler.sample(B, separate_observables=True)
            det = det.astype(np.uint8)
            if self.win is not None:
                r = self.win.decode_batch(det, timing=True, single_shots=min(B, 8))
                lat = r.latency_ns.astype(np.int64)
                sg = r.single_ns.astype(np.int64)
                lfail = (self.win.predict_observables(r.errors) != obs).any(1) | r.failed
                st = r.shot_status                                  # worst window status of each shot
                tiers = np.where(st == 3, 255, np.where(st == 2, 1, 0)).astype(np.uint8)
                stat = slice(0, -1) if lat.shape[0] > 1 else slice(None)      # stationary windows (all but the last)
                total = sg[-1]                                      # reaction after the last round, single shot
                bp = sg[stat].ravel()                               # one window step, single shot
                amort = lat[stat].ravel()                           # one window step, batch-amortized
                self.react_batch = lat[-1]
                self._tier_counts(tiers)                            # per shot, so counters sum to the shots
            else:
                errs, tiers, total, bp, amort = self._decode_set(self.hyb, det)
                pred = (self.prob.obs @ errs.T.astype(np.int64) % 2).T
                lfail = (pred != obs).any(1) | (tiers == 255)
                self._tier_counts(tiers)
            interesting = np.flatnonzero(det.any(1))
            shot = int(interesting[0]) if interesting.size else 0
            kind = "z_syndrome" if self.exp.basis == "Z" else "x_syndrome"
            by_round: dict[int, list] = {}
            for d in np.flatnonzero(det[shot]):
                chk, rnd = int(self.coords[int(d)][0]), int(self.coords[int(d)][1])
                by_round.setdefault(rnd, []).append(chk)
            display = {
                "shot": shot, "err_x": [], "err_z": [], "z_syndrome": [], "x_syndrome": [], "corr_x": [], "corr_z": [],
                "tier_x": int(tiers[shot]), "tier_z": int(tiers[shot]), "logical_fail": bool(lfail[shot]),
                "rounds": [{"round": r, kind: sorted(v), ("x_syndrome" if kind == "z_syndrome" else "z_syndrome"): []}
                           for r, v in sorted(by_round.items())],
            }
        self.counters["shots"] += B
        self.fails += int(lfail.sum())
        react = getattr(self, "react_batch", None) if (cfg.mode == "circuit" and self.win is not None) else None
        lo, hi = _wilson(self.fails, self.counters["shots"])
        p50, p95 = _pct(total)
        b50, b95 = _pct(bp)
        return {
            "type": "tick", "seq": self.seq, "mode": cfg.mode, "shots_in_tick": B, "display": display,
            "decoder": cfg.decoder if cfg.mode == "circuit" else "full",
            "window": ({"W": cfg.W, "F": cfg.F, "n_windows": self.win.n_windows} if cfg.mode == "circuit" and self.win is not None else None),
            "latency": {
                "bp_kernel_ns": {"p50": b50, "p95": b95, "max": float(bp.max())},
                "amortized_ns": {"p50": _pct(amort, (50,))[0]},
                "total_ns": {"p50": p50, "p95": p95, "max": float(total.max())},
                "last_total_ns": int(total[-1]),
                "reaction_batch_ns": ({"p50": _pct(react, (50,))[0]} if react is not None else None),
            },
            "counters": dict(self.counters),
            "ler": {"shots": self.counters["shots"], "failures": self.fails,
                    "rate": self.fails / max(self.counters["shots"], 1), "wilson_lo": lo, "wilson_hi": hi},
            "elapsed_s": round(time.time() - self.t_start, 2),
        }

    def hello(self) -> dict:
        from .services import CODE_NAMES
        return {"type": "hello", "config": cfg_dict(self.cfg), "gnn_available": bool(self.gnn_available),
                "lanes": 16, "codes": CODE_NAMES}


def cfg_dict(cfg: Config) -> dict:
    return {"code": cfg.code, "mode": cfg.mode, "p": cfg.p, "rate_hz": cfg.rate_hz, "batch": cfg.batch,
            "schedule": cfg.schedule, "rounds": cfg.rounds, "decoder": cfg.decoder, "W": cfg.W, "F": cfg.F}
