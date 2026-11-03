"""Zero-overhead logical gates of bivariate bicycle codes from translation automorphisms of Z_l x Z_m.

A BB code has H_X = [A|B], H_Z = [B^T|A^T] with A, B polynomials in the commuting shifts x, y.  Shifting every
data qubit (in both blocks) by (s, t) is a permutation P with H_X P = S H_X and H_Z P = S' H_Z, so it maps the
stabiliser group onto itself and acts on the k logical qubits as a matrix U in GL(k, F_2):  a logical gate that costs
no physical gate (a relabelling of qubits, or SWAPs).  This module finds those permutations, the logical matrices
they induce, and checks them against Stim's stabiliser simulator.
"""
from __future__ import annotations

import numpy as np
import stim

from algebra import gf2
from algebra.qldpc import CSSCode


def translation_perm(code: CSSCode, s: int, t: int) -> np.ndarray:
    """perm[i] = new position of data qubit i after shifting by (s, t) on the l x m torus (same shift in both blocks)."""
    l, m = code.meta["l"], code.meta["m"]
    lm = l * m
    perm = np.empty(code.n, dtype=np.int64)
    for i in range(code.n):
        blk, j = divmod(i, lm)
        a, b = divmod(j, m)
        perm[i] = blk * lm + ((a + s) % l) * m + (b + t) % m
    return perm


def apply_perm(v: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """Move the content of qubit i to qubit perm[i]."""
    out = np.zeros_like(v)
    out[..., perm] = v
    return out


def preserves_stabilizers(code: CSSCode, perm: np.ndarray) -> bool:
    """True if the permuted rows of H_X and of H_Z stay inside the original row spaces."""
    for h in (code.hx, code.hz):
        r = gf2.rank(h)
        if gf2.rank(np.vstack([h, apply_perm(h, perm)])) != r:
            return False
    return True


def gf2_inverse(m: np.ndarray) -> np.ndarray:
    k = m.shape[0]
    r, piv = gf2.rref(np.hstack([gf2.as_gf2(m), np.eye(k, dtype=np.uint8)]))
    if piv[:k] != list(range(k)):
        raise ValueError("matrix is singular over GF(2)")
    return r[:, k:]


def induced_logical_matrices(code: CSSCode, perm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(U_Z, U_X): row i holds the coordinates, in the logical Z (resp. X) basis, of the image of logical i."""
    if not preserves_stabilizers(code, perm):
        raise ValueError("perm is not an automorphism of the stabiliser group")
    lz, lx = code.logical_z(), code.logical_x()
    pair = (lz.astype(np.int64) @ lx.T.astype(np.int64) & 1).astype(np.uint8)     # pairing matrix, invertible
    inv = gf2_inverse(pair)
    inv_t = gf2_inverse(pair.T)
    img_z = apply_perm(lz, perm)
    img_x = apply_perm(lx, perm)
    uz = ((img_z.astype(np.int64) @ lx.T.astype(np.int64)) & 1).astype(np.uint8)
    uz = ((uz.astype(np.int64) @ inv.astype(np.int64)) & 1).astype(np.uint8)
    ux = ((img_x.astype(np.int64) @ lz.T.astype(np.int64)) & 1).astype(np.uint8)
    ux = ((ux.astype(np.int64) @ inv_t.astype(np.int64)) & 1).astype(np.uint8)
    return uz, ux


def swap_circuit(perm: np.ndarray) -> stim.Circuit:
    """A circuit of SWAP gates realising ``perm`` (built from the cycle decomposition)."""
    c = stim.Circuit()
    seen = np.zeros(len(perm), bool)
    for i in range(len(perm)):
        if seen[i] or perm[i] == i:
            continue
        cyc = [i]
        seen[i] = True
        j = perm[i]
        while j != i:
            cyc.append(int(j)); seen[j] = True; j = perm[j]
        rev = cyc[::-1]                       # a chain of SWAPs moves contents backwards along the chain
        for a, b in zip(rev[:-1], rev[1:]):
            c.append("SWAP", [a, b])
    return c


def effective_perm(circuit: stim.Circuit, n: int) -> np.ndarray:
    """Where Stim actually sends each qubit: image of X_i is X_{eff[i]}."""
    full = circuit.copy()
    full.append("I", [n - 1])                       # make the tableau span all n qubits
    tab = stim.Tableau.from_circuit(full)
    eff = np.empty(n, dtype=np.int64)
    for i in range(n):
        img = tab.x_output(i)
        support = [q for q in range(n) if img[q] != 0]
        assert len(support) == 1 and img[support[0]] == 1, "SWAP network is not a pure qubit permutation"
        eff[i] = support[0]
    return eff


def _pauli(row: np.ndarray, kind: str) -> stim.PauliString:
    return stim.PauliString("".join(kind if b else "_" for b in row))


def verify_with_stim(code: CSSCode, perm: np.ndarray, uz: np.ndarray, trials: int = 12, seed: int = 0) -> dict:
    """Check, with Stim's stabiliser simulator, that the SWAP network keeps the code space and acts as U_Z.

    1. Project |+...+> onto the code (measure every stabiliser) and measure a random subset of logical Z operators.
    2. Apply the SWAP network.  Every stabiliser must still have a definite sign (code space preserved).
    3. Logical Z_i is predicted to stay definite exactly when the support of row i of U_Z^{-1} lies in the measured
       subset; Stim must agree on every i, for every random subset.
    """
    n, k = code.n, uz.shape[0]
    circ = swap_circuit(perm)
    eff = effective_perm(circ, n)
    stabs = [_pauli(r, "X") for r in code.hx] + [_pauli(r, "Z") for r in code.hz]
    lz = [_pauli(r, "Z") for r in code.logical_z()]
    uinv = gf2_inverse(uz)
    rng = np.random.default_rng(seed)
    ok_code, mismatches, checked = True, 0, 0
    subsets_done: list[set] = []          # the first k subsets are all-but-one: they pin down every entry of U^-1
    for _ in range(max(trials, 0) + k):
        sim = stim.TableauSimulator()
        sim.h(*range(n))                                # |+...+>: no Z-type logical is definite until measured
        for p in stabs:
            sim.measure_observable(p)
        t = len(subsets_done)
        subset = set(range(k)) - {t} if t < k else set(np.flatnonzero(rng.integers(0, 2, k)).tolist())
        subsets_done.append(subset)
        for i in subset:
            sim.measure_observable(lz[i])
        sim.do_circuit(circ)
        if any(sim.peek_observable_expectation(p) == 0 for p in stabs):
            ok_code = False
        for i in range(k):
            predicted = set(np.flatnonzero(uinv[i]).tolist()) <= subset
            got = sim.peek_observable_expectation(lz[i]) != 0
            mismatches += int(predicted != got)
            checked += 1
    return {"effective_perm_matches": bool(np.array_equal(eff, perm)), "code_space_preserved": ok_code,
            "logical_predictions_checked": checked, "logical_prediction_mismatches": mismatches}


def enumerate_translation_gates(code: CSSCode, stim_trials: int = 6) -> list[dict]:
    """Every translation (s, t) != (0, 0): is it an automorphism, what logical matrix does it induce?"""
    l, m = code.meta["l"], code.meta["m"]
    out, seen = [], {}
    eye = np.eye(code.k, dtype=np.uint8)
    for s in range(l):
        for t in range(m):
            if (s, t) == (0, 0):
                continue
            perm = translation_perm(code, s, t)
            if not preserves_stabilizers(code, perm):
                out.append({"shift": [s, t], "automorphism": False})
                continue
            uz, ux = induced_logical_matrices(code, perm)
            nontrivial = not np.array_equal(uz, eye)
            key = uz.tobytes()
            rec = {"shift": [s, t], "automorphism": True, "nontrivial_logical": nontrivial, "same_logical_as": seen.get(key)}
            seen.setdefault(key, [s, t])
            if nontrivial and rec["same_logical_as"] is None:
                rec["stim"] = verify_with_stim(code, perm, uz, trials=stim_trials)
                rec["u_z"] = uz.tolist()
                rec["u_x"] = ux.tolist()
                n_ones = int(uz.sum())
                rec["u_z_ones"] = n_ones
            out.append(rec)
    return out


def verify_in_memory_circuit(code: CSSCode, layers, perms: dict, rounds: int = 4, basis: str = "Z", shots: int = 64, noise_p: float = 1e-3) -> dict:
    """Insert the gate(s) ``perms`` ({after_round: permutation}) between syndrome rounds of the real extraction circuit.

    Checks, with Stim, that (1) the detector error model builds with ``allow_gauge_detectors=False`` (every detector
    deterministic), (2) the noiseless circuit never fires a detector or flips an observable, and (3) the same circuit
    with noise still yields a detector error model without gauge detectors.
    """
    from circuits.builder import build_memory_experiment
    from circuits.noise import NoiseModel
    clean = build_memory_experiment(code, rounds, NoiseModel(), basis=basis, layers=layers, data_perms=perms)
    det, obs = clean.circuit.compile_detector_sampler(seed=7).sample(shots, separate_observables=True)
    violations = int(det.sum() + obs.sum())
    noisy = build_memory_experiment(code, rounds, NoiseModel.uniform(noise_p), basis=basis, layers=layers, data_perms=perms)
    dem = noisy.dem()                               # raises if some detector is not deterministic
    return {"detectors": clean.circuit.num_detectors, "observables": clean.circuit.num_observables, "noiseless_shots": shots,
            "violations": violations, "dem_mechanisms": dem.num_errors, "gauge_detectors": 0}
