"""The translation symmetry used to shrink the covering sweep must be verified from the data, and refused when it does not hold."""
import numpy as np
import pytest

from algebra.circuit_symmetry import _gf2_invertible, _rank_and_solve, orbits, verify_translation
from api.services import code_of, get_layers
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel


def _dem(sched):
    code = code_of("BB72")
    return code, build_memory_experiment(code, 4, NoiseModel.uniform(1e-3), basis="Z", layers=get_layers("BB72", sched)).dem()


def test_interleaved_schedule_is_translation_invariant_with_equal_orbits():
    code, dem = _dem("interleaved")
    o = orbits(code, dem, "Z")
    assert len(o["label"]) == sum(o["sizes"]) and set(o["sizes"]) == {36}           # 6 x 6 translations act freely on the mechanisms
    assert o["representatives"] == sorted(o["representatives"])                     # orbits numbered by smallest member
    for s in [(1, 0), (0, 1), (2, 3)]:
        assert verify_translation(code, dem, "Z", s)["ok"]


def test_sequential_schedule_is_refused():
    code, dem = _dem("sequential")
    assert not verify_translation(code, dem, "Z", (1, 0))["ok"]
    with pytest.raises(ValueError):
        orbits(code, dem, "Z")


def test_gf2_helpers():
    A = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.uint8)               # rank 2: rows sum to zero
    assert _rank_and_solve(A, np.array([[1], [1], [0]], dtype=np.uint8)) is not None
    assert _rank_and_solve(A, np.array([[1], [0], [0]], dtype=np.uint8)) is None    # inconsistent right-hand side
    assert _gf2_invertible(np.eye(4, dtype=np.uint8)) and not _gf2_invertible(A)
