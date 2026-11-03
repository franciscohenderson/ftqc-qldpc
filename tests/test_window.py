import numpy as np
import pytest

from algebra.qldpc import get_code
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel
from decoder.dem import dem_to_problem
from decoder.fast import FastDecoder
from decoder.window import WindowDecoder, detector_rounds


@pytest.fixture(scope="module")
def setup():
    code = get_code("BB72")
    exp = build_memory_experiment(code, 5, NoiseModel.uniform(2e-3))
    prob = dem_to_problem(exp.dem())
    return exp, prob, detector_rounds(exp.circuit)


def test_detector_rounds_cover_all_layers(setup):
    exp, prob, rounds = setup
    assert len(rounds) == prob.h.shape[0]
    assert sorted(set(rounds.tolist())) == list(range(6))      # 5 syndrome rounds + final readout layer


def test_window_plan_geometry(setup):
    _, prob, rounds = setup
    wd = WindowDecoder(prob, rounds, W=3, F=1)
    plan = wd.plan()
    assert plan["n_windows"] == 4 and plan["windows"][-1]["rounds"] == [3, 6]
    # every mechanism is committed exactly once
    seen = np.zeros(prob.h.shape[1], int)
    for w in wd.windows:
        seen[w.commit_cols] += 1
    assert seen.max() == 1 and seen.sum() == sum(len(w.commit_cols) for w in wd.windows)   # no mechanism committed twice
    # truncating the future boundary makes some columns identical: they are merged, never duplicated
    assert any(w.n_unique < len(w.cols) for w in wd.windows[:-1])
    # windows are strictly smaller than the full problem
    assert all(w["detectors"] < plan["full"]["detectors"] and w["mechanisms"] < plan["full"]["mechanisms"] for w in plan["windows"])


def test_invalid_parameters(setup):
    _, prob, rounds = setup
    with pytest.raises(ValueError):
        WindowDecoder(prob, rounds, W=1, F=2)


def test_window_corrections_clear_the_whole_syndrome(setup):
    exp, prob, rounds = setup
    wd = WindowDecoder(prob, rounds, W=3, F=1)
    det, obs = exp.circuit.compile_detector_sampler(seed=3).sample(300, separate_observables=True)
    res = wd.decode_batch(det.astype(np.uint8))
    ok = ~res.failed
    assert (res.residual[ok] == 0).mean() > 0.95
    # the committed correction reproduces the syndrome up to what windows could not satisfy
    full = (prob.h @ res.errors.T.astype(np.int64) % 2).T
    assert ((full == det).all(axis=1) | res.failed | (res.residual > 0)).all()


def test_window_matches_full_horizon_quality(setup):
    exp, prob, rounds = setup
    det, obs = exp.circuit.compile_detector_sampler(seed=11).sample(3000, separate_observables=True)
    det = det.astype(np.uint8)
    full = FastDecoder(prob.h, prob.priors, max_iter=30).decode_batch(det)
    f_full = ((prob.obs @ full.errors.T.astype(np.int64) % 2).T != obs).any(1).mean()
    wd = WindowDecoder(prob, rounds, W=3, F=1)
    r = wd.decode_batch(det)
    f_win = ((wd.predict_observables(r.errors) != obs).any(1) | r.failed).mean()
    raw = obs.any(1).mean()
    assert f_win < raw or raw == 0
    assert f_win <= max(3 * f_full, f_full + 0.02)           # same order as full horizon


def test_timing_shapes(setup):
    exp, prob, rounds = setup
    wd = WindowDecoder(prob, rounds, W=2, F=1)
    det, _ = exp.circuit.compile_detector_sampler(seed=5).sample(64, separate_observables=True)
    r = wd.decode_batch(det.astype(np.uint8), timing=True, single_shots=8)
    assert r.latency_ns.shape == (wd.n_windows, 64) and r.single_ns.shape == (wd.n_windows, 8)
    assert (r.latency_ns > 0).all() and (r.single_ns > 0).all()


def test_look_ahead_is_required_and_last_window_covers_everything(setup):
    _, prob, rounds = setup
    with pytest.raises(ValueError):
        WindowDecoder(prob, rounds, W=2, F=2)        # no look-ahead round while more layers follow
    one = WindowDecoder(prob, rounds, W=6, F=6)       # a single window that covers all 6 layers is allowed
    assert one.n_windows == 1


def test_shot_status_and_counts_are_per_shot(setup):
    exp, prob, rounds = setup
    wd = WindowDecoder(prob, rounds, W=3, F=1)
    det, _ = exp.circuit.compile_detector_sampler(seed=9).sample(40, separate_observables=True)
    r = wd.decode_batch(det.astype(np.uint8), timing=True)
    assert r.shot_status.shape == (40,) and set(np.unique(r.shot_status)) <= {0, 1, 2, 3}
    assert r.status_counts.sum() == 40 * wd.n_windows           # per window decode
    assert (r.latency_ns[:-1] > 0).all()                        # includes the propagation step
