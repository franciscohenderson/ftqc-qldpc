import numpy as np
import pytest

from algebra.qldpc import get_code, hypergraph_product, repetition_check
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel
from decoder.dem import dem_to_problem
from decoder.fast import FastDecoder


def test_code_capacity_corrects_single_errors():
    code = get_code("BB72")
    dec = FastDecoder(code.hz, 0.01, max_iter=30)
    rng = np.random.default_rng(1)
    errs = np.zeros((64, code.n), np.uint8)
    errs[np.arange(64), rng.integers(0, code.n, 64)] = 1
    synd = (code.hz.astype(int) @ errs.T % 2).T.astype(np.uint8)
    res = dec.decode_batch(synd)
    assert dec.syndrome_ok(res.errors, synd).all()
    # a single qubit error is corrected exactly or up to a stabilizer
    resid = (res.errors ^ errs)
    assert not ((code.logical_x().astype(int) @ resid.T) % 2).any()


def test_zero_syndrome_is_instant():
    code = get_code("BB72")
    dec = FastDecoder(code.hz, 0.01)
    res = dec.decode_batch(np.zeros((33, code.hz.shape[0]), np.uint8))
    assert (res.status == 0).all() and (res.iters == 0).all() and not res.errors.any()


def test_stream_matches_batch_syndromes():
    code = get_code("BB72")
    dec = FastDecoder(code.hz, 0.02)
    rng = np.random.default_rng(2)
    errs = (rng.random((40, code.n)) < 0.02).astype(np.uint8)
    synd = (code.hz.astype(int) @ errs.T % 2).T.astype(np.uint8)
    a, b = dec.decode_batch(synd), dec.decode_stream(synd)
    assert dec.syndrome_ok(a.errors[a.status != 3], synd[a.status != 3]).all()
    assert dec.syndrome_ok(b.errors[b.status != 3], synd[b.status != 3]).all()
    assert (b.latency_ns > 0).all()


def test_osd_solves_what_bp_cannot():
    # hypergraph-product code has short cycles that stall plain BP on some patterns
    code = hypergraph_product(repetition_check(5), repetition_check(5))
    dec = FastDecoder(code.hz, 0.05, max_iter=5)
    rng = np.random.default_rng(3)
    errs = (rng.random((200, code.n)) < 0.08).astype(np.uint8)
    synd = (code.hz.astype(int) @ errs.T % 2).T.astype(np.uint8)
    res = dec.decode_batch(synd, use_osd=True)
    assert dec.syndrome_ok(res.errors, synd).all()
    assert (res.status == 2).any() or (res.status == 0).all()


def test_circuit_level_dem_decoding_beats_no_correction():
    code = get_code("BB72")
    exp = build_memory_experiment(code, 3, NoiseModel.uniform(1e-3))
    prob = dem_to_problem(exp.dem())
    dec = FastDecoder(prob.h, prob.priors, max_iter=30)
    det, obs = exp.circuit.compile_detector_sampler(seed=5).sample(300, separate_observables=True)
    res = dec.decode_batch(det.astype(np.uint8))
    pred = (prob.obs @ res.errors.T.astype(np.int64) % 2).T
    fails = (pred != obs).any(axis=1).mean()
    raw = obs.any(axis=1).mean()
    assert fails < raw or raw == 0
