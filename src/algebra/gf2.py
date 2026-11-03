"""Dense GF(2) linear algebra on uint8 numpy arrays (rank, RREF, nullspace, solve)."""
from __future__ import annotations

import numpy as np


def as_gf2(a) -> np.ndarray:
    return (np.asarray(a, dtype=np.int64) & 1).astype(np.uint8)


def rref(a: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Reduced row echelon form over GF(2). Returns (R, pivot_columns)."""
    m = as_gf2(a).copy()
    rows, cols = m.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        nz = np.flatnonzero(m[r:, c])
        if nz.size == 0:
            continue
        p = r + nz[0]
        if p != r:
            m[[r, p]] = m[[p, r]]
        mask = m[:, c].astype(bool)
        mask[r] = False
        m[mask] ^= m[r]
        pivots.append(c)
        r += 1
    return m[:r], pivots


def rank(a: np.ndarray) -> int:
    return len(rref(a)[1])


def nullspace(a: np.ndarray) -> np.ndarray:
    """Basis of {x : a x = 0 mod 2} as rows of a (dim, ncols) matrix."""
    a = as_gf2(a)
    cols = a.shape[1]
    r, piv = rref(a)
    pivset = set(piv)
    free = [c for c in range(cols) if c not in pivset]
    basis = np.zeros((len(free), cols), dtype=np.uint8)
    for i, f in enumerate(free):
        basis[i, f] = 1
        for row, pc in enumerate(piv):
            basis[i, pc] = r[row, f]
    return basis


def rowspace_complement(stab: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """Rows of `ker` extending rowspace(stab) to rowspace(ker) (logical representatives)."""
    stab, ker = as_gf2(stab), as_gf2(ker)
    cur = rref(stab)[0] if stab.size else np.zeros((0, ker.shape[1]), np.uint8)
    cur_rank = cur.shape[0]
    chosen = []
    for v in ker:
        trial = np.vstack([cur, v[None, :]])
        r, piv = rref(trial)
        if len(piv) > cur_rank:
            chosen.append(v)
            cur, cur_rank = r, len(piv)
    return np.array(chosen, dtype=np.uint8).reshape(len(chosen), ker.shape[1])


def solve(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """One solution of a x = b over GF(2), or None if inconsistent."""
    a, b = as_gf2(a), as_gf2(b).reshape(-1, 1)
    r, piv = rref(np.hstack([a, b]))
    if a.shape[1] in piv:
        return None
    x = np.zeros(a.shape[1], dtype=np.uint8)
    for row, c in enumerate(piv):
        x[c] = r[row, -1]
    return x
