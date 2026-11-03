"""Translation automorphisms of BB codes as zero-overhead logical gates, checked against Stim."""
from __future__ import annotations

import numpy as np

from algebra import logical_gates as lg
from algebra.qldpc import get_code


def test_translations_are_stabilizer_automorphisms_and_induce_invertible_logicals():
    code = get_code("BB72")
    perm = lg.translation_perm(code, 1, 2)
    assert sorted(perm) == list(range(code.n)) and lg.preserves_stabilizers(code, perm)
    uz, ux = lg.induced_logical_matrices(code, perm)
    assert uz.shape == (code.k, code.k)
    lg.gf2_inverse(uz)                               # invertible over GF(2)
    lz, lx = code.logical_z(), code.logical_x()
    pair = (lz.astype(int) @ lx.T.astype(int)) & 1
    assert np.array_equal(((uz.astype(int) @ pair @ ux.T.astype(int)) & 1), pair)   # the pairing is preserved


def test_a_non_symmetry_is_rejected():
    code = get_code("BB72")
    bad = np.arange(code.n)
    bad[0], bad[1] = bad[1], bad[0]                  # swap two data qubits: not a symmetry
    assert not lg.preserves_stabilizers(code, bad)


def test_stim_agrees_with_the_predicted_logical_action():
    code = get_code("BB72")
    res = lg.enumerate_translation_gates(code, stim_trials=4)
    gates = [r for r in res if r.get("nontrivial_logical") and "stim" in r]
    assert gates, "BB72 should have at least one non-trivial logical translation"
    for g in gates:
        s = g["stim"]
        assert s["effective_perm_matches"] and s["code_space_preserved"] and s["logical_prediction_mismatches"] == 0, g["shift"]


def test_gate_between_rounds_keeps_detectors_deterministic():
    """The translation gates, inserted between syndrome rounds of the depth-7 schedule, break no detector."""
    import numpy as np
    from algebra.logical_gates import translation_perm, verify_in_memory_circuit
    from api.services import get_layers
    code = get_code("BB72")
    layers = get_layers("BB72", "interleaved")
    assert len(layers) == 7
    for st in [(1, 0), (0, 1), (2, 3), (5, 4)]:
        r = verify_in_memory_circuit(code, layers, {1: translation_perm(code, *st), 2: translation_perm(code, 1, 1)}, rounds=4)
        assert r["violations"] == 0 and r["gauge_detectors"] == 0


def test_non_automorphism_is_rejected_by_the_builder():
    import numpy as np
    import pytest
    from circuits.builder import build_memory_experiment
    code = get_code("BB72")
    perm = np.arange(code.n)
    perm[[0, 1]] = perm[[1, 0]]                    # one transposition is not a symmetry of the code
    with pytest.raises(ValueError):
        build_memory_experiment(code, 3, data_perms={0: perm})
