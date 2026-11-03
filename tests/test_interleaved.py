import numpy as np
import pytest

from algebra.qldpc import get_code, hypergraph_product, repetition_check
from circuits.builder import build_memory_experiment
from circuits.interleaved import (
    bb_translation_schedule,
    find_interleaved_schedule,
    schedule_depth,
)
from circuits.noise import NoiseModel
from circuits.schedule import commutation_violations, validate_schedule


def test_bb72_interleaved_schedule_and_noiseless_determinism():
    code = get_code("BB72")
    layers = find_interleaved_schedule(code.hx, code.hz, max_tries=5000, seed=0, time_limit_s=30.0)
    assert layers is not None
    validate_schedule(layers, code.hx, code.hz)
    assert commutation_violations(layers, code.hx, code.hz) == 0
    assert schedule_depth(layers) == len(layers)
    assert schedule_depth(layers) <= 12

    exp = build_memory_experiment(code, 2, NoiseModel(), layers=layers, detectors="all")
    sampler = exp.circuit.compile_detector_sampler(seed=0)
    det_samples = sampler.sample(20)
    assert not np.any(det_samples)


@pytest.mark.parametrize("name", ["BB72", "BB90"])
def test_bb_translation_schedule_helper(name):
    code = get_code(name)
    layers = bb_translation_schedule(code, depth=8, seed=0, max_tries=2000)
    assert layers is not None
    validate_schedule(layers, code.hx, code.hz)
    assert commutation_violations(layers, code.hx, code.hz) == 0
    assert schedule_depth(layers) <= 8


def test_general_hgp_interleaved_schedule():
    h = repetition_check(3, ring=True)
    code = hypergraph_product(h, h)
    layers = find_interleaved_schedule(code.hx, code.hz, max_tries=2000, seed=42, time_limit_s=10.0)
    assert layers is not None
    validate_schedule(layers, code.hx, code.hz)
    assert commutation_violations(layers, code.hx, code.hz) == 0
    assert schedule_depth(layers) <= 8

    exp = build_memory_experiment(code, 2, NoiseModel(), layers=layers, detectors="all")
    det_samples = exp.circuit.compile_detector_sampler(seed=1).sample(20)
    assert not np.any(det_samples)
