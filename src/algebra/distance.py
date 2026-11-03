"""Code distance via integer linear programming (HiGHS branch-and-bound through scipy.optimize.milp).

For the Z-type distance d_Z we minimise |z| subject to
    H_X z = 0 (mod 2)           (z commutes with every X check)
    at least one X-logical anticommutes with z  (z is not a stabilizer).
The parity constraints are linearised with integer slack variables:  H z - 2u = 0,  L z - 2v = y,
y binary, sum(y) >= 1.  HiGHS returns the incumbent (upper bound) and the dual bound (lower bound);
``exact`` is True when they meet.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import bmat, csr_matrix, identity

from .qldpc import CSSCode


@dataclass
class DistanceResult:
    lower: int
    upper: int
    exact: bool
    witness: np.ndarray | None


def _min_weight(h: np.ndarray, logicals: np.ndarray, time_limit: float) -> DistanceResult:
    m, n = h.shape
    k = logicals.shape[0]
    # variable layout: z (n, binary) | u (m, int) | v (k, int) | y (k, binary)
    nv = n + m + k + k
    H = csr_matrix(h.astype(float))
    L = csr_matrix(logicals.astype(float))
    Z = lambda r, c_: csr_matrix((r, c_))
    A_h = bmat([[H, -2 * identity(m), Z(m, k), Z(m, k)]], format="csr")
    A_l = bmat([[L, Z(k, m), -2 * identity(k), -identity(k)]], format="csr")
    A_y = csr_matrix(np.concatenate([np.zeros(n + m + k), np.ones(k)])[None, :])
    cons = [
        LinearConstraint(A_h, 0, 0),
        LinearConstraint(A_l, 0, 0),
        LinearConstraint(A_y, 1, np.inf),
    ]
    c = np.concatenate([np.ones(n), np.zeros(m + k + k)])
    ub = np.concatenate([np.ones(n), np.full(m, np.floor(h.sum(1).max() / 2)),
                         np.full(k, np.floor(logicals.sum(1).max() / 2) + n), np.ones(k)])
    res = milp(c, constraints=cons, integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub),
               options={"time_limit": time_limit, "mip_rel_gap": 0.0, "presolve": True})
    if res.x is None:
        return DistanceResult(1, n, False, None)
    z = np.round(res.x[:n]).astype(np.uint8)
    upper = int(z.sum())
    lower = int(np.ceil(res.mip_dual_bound - 1e-6)) if res.mip_dual_bound is not None else 1
    lower = max(1, min(lower, upper))
    return DistanceResult(lower, upper, lower == upper, z)


def distance_z(code: CSSCode, time_limit: float = 60.0) -> DistanceResult:
    """Minimum weight of a Z-type logical operator (undetected by X checks)."""
    return _min_weight(code.hx, code.logical_x(), time_limit)


def distance_x(code: CSSCode, time_limit: float = 60.0) -> DistanceResult:
    """Minimum weight of an X-type logical operator (undetected by Z checks)."""
    return _min_weight(code.hz, code.logical_z(), time_limit)


def verify_logical(code: CSSCode, z: np.ndarray, kind: str = "Z") -> bool:
    """Check that z is a genuine nontrivial logical of the given Pauli type."""
    h, lo = (code.hx, code.logical_x()) if kind == "Z" else (code.hz, code.logical_z())
    z = z.astype(np.int64)
    return bool(not np.any((h.astype(np.int64) @ z) & 1) and np.any((lo.astype(np.int64) @ z) & 1))
