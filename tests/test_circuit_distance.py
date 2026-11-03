"""Circuit-level distance by MILP: known answers first, then the BB code experiment builder."""
from __future__ import annotations

import stim

from algebra.circuit_distance import compute_circuit_distance, dem_columns


def _dem(circuit):
    return circuit.detector_error_model(decompose_errors=False, allow_gauge_detectors=False,
                                        approximate_disjoint_errors=True)


def test_repetition_code_circuit_distance_is_3():
    c = stim.Circuit.generated("repetition_code:memory", distance=3, rounds=3, after_clifford_depolarization=0.01,
                               before_round_data_depolarization=0.01, before_measure_flip_probability=0.01)
    r = compute_circuit_distance(_dem(c), time_limit=30)
    assert r["certified"] and r["d_upper"] == r["d_lower"] == 3


def test_rotated_surface_code_circuit_distance_is_3():
    c = stim.Circuit.generated("surface_code:rotated_memory_z", distance=3, rounds=3, after_clifford_depolarization=0.001,
                               before_round_data_depolarization=0.001, before_measure_flip_probability=0.001,
                               after_reset_flip_probability=0.001)
    r = compute_circuit_distance(_dem(c), time_limit=60)
    assert r["certified"] and r["d_upper"] == 3
    assert len(r["witness"]) == 3 and r["witness_observables"] == [0]


def test_undetectable_logical_fault_gives_distance_one():
    dem = stim.DetectorErrorModel("error(0.1) D0 D1\nerror(0.1) L0\n")
    r = compute_circuit_distance(dem)
    assert r["d_upper"] == 1 and r["certified"] and r["status"] == "undetectable_logical_fault"


def test_columns_keep_undetectable_mechanisms_and_merge_duplicates():
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D0 L0\nerror(0.1) L1\nerror(0.1) D1\n")
    cols = dem_columns(dem)
    assert ((0,), (0,)) in cols and ((), (1,)) in cols and ((1,), ()) in cols
    assert len(cols) == 3        # the two identical (D0, L0) mechanisms merge into one variable
