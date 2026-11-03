import numpy as np
import pytest

from algebra.qldpc import get_code
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel
from circuits.schedule import (commutation_violations, edge_colour_bipartite, sequential_schedule,
                               validate_schedule)


@pytest.mark.parametrize("name", ["BB72", "BB90", "BB144"])
def test_edge_colouring_is_optimal_and_proper(name):
    c = get_code(name)
    for h in (c.hx, c.hz):
        cls = edge_colour_bipartite(h)
        assert len(cls) == 6                                    # Delta colours (Konig)
        assert sum(len(x) for x in cls) == h.sum()
        for m in cls:
            assert len({i for i, _ in m}) == len(m) and len({j for _, j in m}) == len(m)
    lay = sequential_schedule(c.hx, c.hz)
    validate_schedule(lay, c.hx, c.hz)
    assert len(lay) == 12


@pytest.mark.parametrize("basis", ["Z", "X"])
@pytest.mark.parametrize("det", ["matched", "all"])
def test_noiseless_circuit_is_deterministic(basis, det):
    c = get_code("BB72")
    e = build_memory_experiment(c, 3, NoiseModel(), basis=basis, detectors=det)
    d, o = e.circuit.compile_detector_sampler(seed=1).sample(64, separate_observables=True)
    assert not d.any() and not o.any()
    assert e.circuit.num_observables == c.k


def test_dem_is_graphlike_free_and_has_expected_structure():
    c = get_code("BB72")
    e = build_memory_experiment(c, 4, NoiseModel.uniform(1e-3))
    dem = e.dem()
    assert dem.num_detectors == 5 * c.hz.shape[0] and dem.num_observables == c.k
    assert e.depth_per_round == 14 and e.cnot_per_round == int(c.hx.sum() + c.hz.sum())


def test_t1_t2_idle_channel():
    nm = NoiseModel.from_coherence(1e-3, t1_us=100, t2_us=80, layer_time_us=0.1)
    px, py, pz = nm.idle_pauli()
    assert px == py > 0 and pz > 0 and px + py + pz < 0.01
    c = get_code("BB72")
    e = build_memory_experiment(c, 2, nm)
    assert "PAULI_CHANNEL_1" in str(e.circuit)
    assert e.dem().num_errors > 0


def test_noise_actually_produces_detection_events():
    c = get_code("BB72")
    e = build_memory_experiment(c, 3, NoiseModel.uniform(5e-3))
    d, _ = e.circuit.compile_detector_sampler(seed=2).sample(200, separate_observables=True)
    assert d.mean() > 0.01
