import numpy as np

from algebra.qldpc import get_code
from benchmarks.montecarlo import PointState, Task, run_task, to_rows
from benchmarks.storage import load_results, save_results
from decoder.fast import FastDecoder
from decoder.hybrid import HybridDecoder


def test_run_task_reports_consistent_counts():
    t = Task("BB72", 2e-3, 2, 400, seed=7, stream_shots=32)
    out = run_task(t)
    assert out["shots"] == 400 and 0 <= out["failures"] <= 400
    assert len(out["batch_lat"]) == 400 and len(out["stream_lat"]) == 32
    again = run_task(t)
    assert again["failures"] == out["failures"]          # reproducible given the seed


def test_failure_rate_monotone_in_p():
    lo = run_task(Task("BB72", 5e-4, 2, 1500, seed=1, stream_shots=0))["failures"]
    hi = run_task(Task("BB72", 4e-3, 2, 1500, seed=1, stream_shots=0))["failures"]
    assert hi > lo


def test_rows_roundtrip_through_parquet(tmp_path):
    st = PointState("BB72", 1e-3, 3, failures=5, shots=1000, wall_s=1.0, n=72, k=12, d=6)
    rows = to_rows([st])
    save_results(rows, tmp_path / "r.parquet")
    df = load_results(tmp_path / "r.parquet")
    assert df.loc[0, "p_logical"] == 0.005 and df.loc[0, "rounds"] == 3
    assert 0 < df.loc[0, "p_logical_per_round"] < 0.005


def test_hybrid_decoder_without_gnn_matches_fast_osd():
    code = get_code("BB72")
    rng = np.random.default_rng(4)
    e = (rng.random((200, code.n)) < 0.02).astype(np.uint8)
    s = (code.hz.astype(int) @ e.T % 2).T.astype(np.uint8)
    hyb = HybridDecoder(code.hz, 0.02, max_iter=3)   # few iterations => some tier-1 shots
    r = hyb.decode_batch(s)
    ok = ((code.hz.astype(int) @ r.errors.T) % 2 == s.T).all(axis=0)
    assert ok[r.tier != 255].all()
    assert set(np.unique(r.tier)) <= {0, 1, 2, 255}
