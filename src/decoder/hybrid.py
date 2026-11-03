"""Two-tier hybrid decoder.

Tier 0  fast path: INT16 normalized min-sum BP, 16 shots per SIMD pass, early exit (C++20).
Tier 1  BP failures: bit-packed OSD-CS on the BP posterior ordering (C++20).
Tier 2  ambiguous / degenerate cases: GNN posterior (INT8, numpy) -> OSD-CS on the GNN ordering;
        the minimum channel-weight syndrome-consistent candidate of tier 1 and tier 2 wins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fast import FastDecoder


@dataclass
class HybridResult:
    errors: np.ndarray
    tier: np.ndarray        # 0 BP, 1 OSD, 2 GNN+OSD, 255 failed
    latency_ns: np.ndarray  # per-shot (amortized over the BP chunk, plus per-shot OSD/GNN time)


class HybridDecoder:
    def __init__(self, h, priors, gnn=None, max_iter: int = 20, osd_lambda: int = 32):
        self.fast = FastDecoder(h, priors, max_iter=max_iter, osd_lambda=osd_lambda)
        self.gnn = gnn
        self.h = self.fast.H
        self._w = self.fast.channel_llr().astype(np.int64)

    def _cost(self, e: np.ndarray) -> int:
        return int(self._w[e.astype(bool)].sum())

    def decode_batch(self, synd) -> HybridResult:
        import time

        synd = np.ascontiguousarray(synd, dtype=np.uint8)
        res = self.fast.decode_batch(synd, use_osd=False)
        errors = res.errors.copy()
        tier = np.zeros(len(synd), np.uint8)
        lat = res.latency_ns.astype(np.int64).copy()
        failed = np.flatnonzero(res.status != 0)
        if failed.size == 0:
            return HybridResult(errors, tier, lat)
        gnn_llr = None
        if self.gnn is not None:
            t0 = time.perf_counter_ns()
            gnn_llr = self.gnn.llr(synd[failed])
            lat[failed] += (time.perf_counter_ns() - t0) // failed.size
        for i, s in enumerate(failed):
            t0 = time.perf_counter_ns()
            cands = []
            # tier 1: OSD on the BP posterior == internal ordering; recompute through a 1-shot stream decode
            r1 = self.fast.decode_stream(synd[s:s + 1], use_osd=True)
            if r1.status[0] in (0, 2):
                cands.append((1, r1.errors[0]))
            if gnn_llr is not None:
                e2 = self.fast.osd_external(synd[s], gnn_llr[i])
                if e2 is not None:
                    cands.append((2, e2))
            if cands:
                t, e = min(cands, key=lambda c: self._cost(c[1]))
                errors[s], tier[s] = e, t
            else:
                tier[s] = 255
            lat[s] += time.perf_counter_ns() - t0
        return HybridResult(errors, tier, lat)
