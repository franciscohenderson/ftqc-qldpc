"""Translation symmetry of a bivariate-bicycle memory-experiment DEM, verified from the data itself.

A translation T of the torus Z_l x Z_m maps the data qubits (both blocks) and the checks onto themselves.  If the circuit is
translation invariant, T permutes the detectors (check i, round r) -> (T(i), r) and therefore the DEM mechanisms.  We do not
assume that: for every mechanism we look up the image (it must exist, and be unique because a mechanism is identified by the
detectors it fires), and we fit the action on the observables.

What the covering sweep needs from a symmetry T:  for any set e of mechanisms with zero total syndrome, T(e) is again a set of
mechanisms with zero total syndrome, of the same weight, and T(e) flips some observable iff e does.  The first two follow from
T permuting the mechanisms.  For the third we check that the observable part of the image is  ko' = U ko  xor  L kd  for fixed
matrices U (k x k) and L (k x n_det) valid for EVERY mechanism, and that U is invertible: for zero total syndrome the L term
sums to zero, so the total observable flip of T(e) is U (total flip of e), which is non-zero iff the flip of e is.
"""
from __future__ import annotations

import numpy as np

from algebra.circuit_distance import dem_columns
from algebra.logical_gates import translation_perm


def _rank_and_solve(A: np.ndarray, B: np.ndarray):
    """Solve A X = B over GF(2).  Returns X (or None if inconsistent)."""
    A = (A % 2).astype(np.uint8); B = (B % 2).astype(np.uint8)
    n, m = A.shape
    M = np.concatenate([A, B], axis=1)
    piv, r = [], 0
    for c in range(m):
        rows = np.flatnonzero(M[r:, c]) + r
        if len(rows) == 0:
            continue
        p = rows[0]
        M[[r, p]] = M[[p, r]]
        others = np.flatnonzero(M[:, c]); others = others[others != r]
        M[others] ^= M[r]
        piv.append(c); r += 1
        if r == n:
            break
    if M[r:, m:].any():
        return None                                   # inconsistent: the action is not of the assumed form
    X = np.zeros((m, B.shape[1]), dtype=np.uint8)
    for row, c in enumerate(piv):
        X[c] = M[row, m:]
    return X


def _gf2_invertible(U: np.ndarray) -> bool:
    k = U.shape[0]
    M = np.concatenate([U % 2, np.eye(k, dtype=np.uint8)], axis=1).astype(np.uint8)
    for c in range(k):
        rows = np.flatnonzero(M[c:, c]) + c
        if len(rows) == 0:
            return False
        M[[c, rows[0]]] = M[[rows[0], c]]
        others = np.flatnonzero(M[:, c]); others = others[others != c]
        M[others] ^= M[c]
    return True


def verify_translation(code, dem, basis: str, shift: tuple[int, int], cols=None) -> dict:
    """Check that the translation ``shift`` is a symmetry of ``dem`` in the sense above.  Returns a report with the column permutation."""
    cols = cols or dem_columns(dem)
    perm = translation_perm(code, *shift)
    h = code.hz if basis == "Z" else code.hx
    where = {frozenset(np.flatnonzero(row).tolist()): i for i, row in enumerate(h)}
    chk_img = []
    for row in h:
        img = frozenset(int(perm[q]) for q in np.flatnonzero(row))
        if img not in where:
            return {"ok": False, "reason": "translation does not map the checks onto checks"}
        chk_img.append(where[img])
    coords = dem.get_detector_coordinates()
    by_coord = {(int(c[0]), int(c[1])): d for d, c in coords.items()}
    det_img = {}
    for d, c in coords.items():
        key = (chk_img[int(c[0])], int(c[1]))
        if key not in by_coord:
            return {"ok": False, "reason": "a detector has no image"}
        det_img[d] = by_coord[key]
    index = {kd: j for j, (kd, ko) in enumerate(cols)}
    if len(index) != len(cols):
        return {"ok": False, "reason": "detector sets do not identify mechanisms uniquely"}
    img = np.empty(len(cols), dtype=np.int64)
    for j, (kd, ko) in enumerate(cols):
        k2 = tuple(sorted(det_img[d] for d in kd))
        if k2 not in index:
            return {"ok": False, "reason": f"mechanism {j} has no image under the translation"}
        img[j] = index[k2]
    if len(set(img.tolist())) != len(cols):
        return {"ok": False, "reason": "the image map is not a bijection"}
    nd, no, m = dem.num_detectors, dem.num_observables, len(cols)
    A = np.zeros((m, nd + no), dtype=np.uint8)
    B = np.zeros((m, no), dtype=np.uint8)
    for j, (kd, ko) in enumerate(cols):
        A[j, list(kd)] = 1
        A[j, [nd + o for o in ko]] = 1
        B[j, list(cols[img[j]][1])] = 1
    X = _rank_and_solve(A, B)
    if X is None:
        return {"ok": False, "reason": "the action on the observables is not of the form U ko + L kd"}
    U = X[nd:, :].T                                    # ko' = U ko  (+ L kd)
    return {"ok": bool(_gf2_invertible(U)), "reason": "" if _gf2_invertible(U) else "U is not invertible", "image": img, "k": no}


def orbits(code, dem, basis: str, shifts=((1, 0), (0, 1))) -> dict:
    """Orbits of the mechanisms under the group generated by ``shifts``; every generator is verified first."""
    cols = dem_columns(dem)
    parent = list(range(len(cols)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for s in shifts:
        rep = verify_translation(code, dem, basis, s, cols)
        if not rep["ok"]:
            raise ValueError(f"translation {s} is not a verified symmetry: {rep['reason']}")
        for j, t in enumerate(rep["image"]):
            a, b = find(j), find(int(t))
            if a != b:
                parent[max(a, b)] = min(a, b)
    root = np.array([find(j) for j in range(len(cols))])
    ids = {r: i for i, r in enumerate(sorted(set(root.tolist())))}      # orbits numbered by their smallest mechanism index
    label = np.array([ids[r] for r in root])
    reps = [int(np.flatnonzero(label == i)[0]) for i in range(len(ids))]
    return {"label": label, "representatives": reps, "n_orbits": len(ids), "sizes": np.bincount(label).tolist()}
