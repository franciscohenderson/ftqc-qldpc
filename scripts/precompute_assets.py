"""Precompute server assets: interleaved schedules (incl. BB288) and small INT8 GNN models.

    python scripts/precompute_assets.py [--gnn BB72 BB144] [--steps 1500]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from algebra.qldpc import BB_ZOO, get_code  # noqa: E402


def schedules(names):
    from circuits.interleaved import find_interleaved_schedule, schedule_depth
    for name in names:
        out = ROOT / "results" / "schedules" / f"{name}.json"
        if out.exists():
            continue
        c = get_code(name)
        t = time.time()
        layers = find_interleaved_schedule(c.hx, c.hz, seed=0, time_limit_s=240.0)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(layers))
        os.replace(tmp, out)
        print(f"{name}: depth {schedule_depth(layers)} in {time.time() - t:.0f}s", flush=True)


def gnns(names, steps):
    if not names:
        return
    try:
        import torch
    except ImportError:
        print("torch is not installed: skipping the INT8 GNN models (the console works without them).")
        return

    from decoder.gnn import quantize_int8, train_gnn
    torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
    for name in names:
        c = get_code(name)
        for tag, h in (("hz", c.hz), ("hx", c.hx)):
            out = ROOT / "results" / "gnn" / f"{name}_{tag}.npz"
            if out.exists():
                continue
            t = time.time()
            m = train_gnn(h, p_range=(0.03, 0.09), steps=steps, batch=64, hidden=16, layers=4, seed=0)
            rng = np.random.default_rng(0)
            e = (rng.random((1000, c.n)) < 0.06).astype(np.uint8)
            s = (h.astype(int) @ e.T % 2).T.astype(np.uint8)
            quantize_int8(m, s).save(out)
            print(f"{name}_{tag}: GNN trained+quantized in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sched", nargs="*", default=list(BB_ZOO))
    ap.add_argument("--gnn", nargs="*", default=["BB72", "BB144"])
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    schedules(a.sched)
    gnns(a.gnn, a.steps)
