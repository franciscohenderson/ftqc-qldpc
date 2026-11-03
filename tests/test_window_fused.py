"""The fused C++ window loop must make exactly the same decisions as the Python loop."""
import numpy as np
import pytest

from algebra.qldpc import get_code
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel
from decoder.dem import dem_to_problem
from decoder.window import WindowDecoder, detector_rounds


@pytest.fixture(scope="module")
def setup():
    code = get_code("BB72")
    exp = build_memory_experiment(code, 5, NoiseModel.uniform(3e-3))
    prob = dem_to_problem(exp.dem())
    det, obs = exp.circuit.compile_detector_sampler(seed=2026).sample(400, separate_observables=True)
    return prob, detector_rounds(exp.circuit), det.astype(np.uint8), obs


@pytest.mark.parametrize("W,F", [(3, 1), (4, 2), (6, 6)])
def test_fused_equals_python_loop(setup, W, F):
    prob, rnd, det, _ = setup
    wd = WindowDecoder(prob, rnd, W=W, F=F)
    a = wd.decode_batch(det)
    b = wd.decode_batch_fused(det)
    assert np.array_equal(a.errors, b.errors)
    assert np.array_equal(a.failed, b.failed)
    assert np.array_equal(a.residual, b.residual)
    assert np.array_equal(a.shot_status, b.shot_status)
    assert np.array_equal(a.status_counts, b.status_counts)


def test_fused_timings_are_separated_and_positive(setup):
    prob, rnd, det, obs = setup
    wd = WindowDecoder(prob, rnd, W=3, F=1)
    r = wd.decode_batch_fused(det)
    assert r.bp_ns.shape == r.post_ns.shape == r.prop_ns.shape == (wd.n_windows,)
    assert (r.bp_ns > 0).all() and (r.prop_ns >= 0).all() and (r.post_ns >= 0).all()
    ref = wd.decode_batch(det)
    fail_fused = ((wd.predict_observables(r.errors) != obs).any(1) | r.failed).sum()
    fail_py = ((wd.predict_observables(ref.errors) != obs).any(1) | ref.failed).sum()
    assert fail_fused == fail_py                      # same logical failures as the Python loop


def test_single_shot_fused_call_matches_batch_row(setup):
    prob, rnd, det, _ = setup
    wd = WindowDecoder(prob, rnd, W=3, F=1)
    full = wd.decode_batch_fused(det[:8])
    for i in range(8):
        assert np.array_equal(wd.decode_batch_fused(det[i:i + 1]).errors[0], full.errors[i])


def test_empty_batch(setup):
    prob, rnd, det, _ = setup
    wd = WindowDecoder(prob, rnd, W=3, F=1)
    r = wd.decode_batch_fused(det[:0])
    assert r.errors.shape == (0, prob.h.shape[1]) and r.failed.shape == (0,) and r.bp_ns.shape == (wd.n_windows,)


def test_concurrent_calls_do_not_corrupt_each_other(setup):
    from concurrent.futures import ThreadPoolExecutor
    prob, rnd, det, _ = setup
    wd = WindowDecoder(prob, rnd, W=3, F=1)
    ref = wd.decode_batch_fused(det[:96]).errors
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda _: wd.decode_batch_fused(det[:96]).errors, range(8)))
    assert all(np.array_equal(o, ref) for o in outs)      # the per-window locks serialise the shared C++ scratch buffers
