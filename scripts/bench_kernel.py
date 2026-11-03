"""Measure the BP kernel latency (batch-amortized and single-shot) and store results/kernel_latency.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from algebra.qldpc import get_code  # noqa: E402
from decoder.fast import FastDecoder  # noqa: E402


def measure(code_names=("BB72", "BB144"), ps=(1e-3, 3e-3, 1e-2), shots=4000, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for name in code_names:
        c = get_code(name)
        for p in ps:
            dec = FastDecoder(c.hz, p, max_iter=10)
            e = (rng.random((shots, c.n)) < p).astype(np.uint8)
            s = (c.hz.astype(int) @ e.T % 2).T.astype(np.uint8)
            dec.decode_batch(s[:64], use_osd=False)
            dec.decode_stream(s[:64], use_osd=False)
            b = dec.decode_batch(s, use_osd=False)
            st = dec.decode_stream(s, use_osd=False)
            q = lambda a: {k: float(np.percentile(a, v)) for k, v in (("p50", 50), ("p95", 95), ("p99", 99))}
            out[f"{name}@{p:g}"] = {
                "matrix": list(map(int, c.hz.shape)), "lanes": dec.lanes, "shots": shots,
                "batch_amortized_ns": q(b.latency_ns), "single_shot_bp_ns": q(st.bp_ns),
                "bp_converged_fraction": float((b.status == 0).mean()),
            }
    return out


if __name__ == "__main__":
    res = measure()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "kernel_latency.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
