"""Circuit-level fault distance by MILP over a detector error model.

d_circ is the minimum number of independent circuit faults (DEM mechanisms) that fire NO detector yet flip a
logical observable.  Formulation over GF(2), linearised with integer slack variables:

    minimise    sum_j e_j
    subject to  H e - 2 k = 0          (every detector fires an even number of times)
                O e - 2 u - y = 0      (y_o = parity of observable o)
                sum_o y_o >= 1         (some logical observable is flipped)
                e, y binary;  k, u integer >= 0

HiGHS (scipy.optimize.milp) returns the best feasible fault set (an UPPER bound on d_circ) and the dual bound
(a LOWER bound).  When the time limit is hit the two differ and both are reported; when they meet the distance is
certified.  Merging mechanisms with identical symptoms does not change the minimum, so each distinct (detectors,
observables) pair is one variable.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import stim
from scipy.optimize import Bounds, LinearConstraint, milp


def dem_columns(dem: stim.DetectorErrorModel):
    """Distinct fault symptoms of a DEM as (detector tuple, observable tuple), keeping undetectable ones.

    Unlike decoder.dem.dem_to_problem this does NOT drop mechanisms with no detectors: an undetectable
    mechanism that flips an observable means d_circ = 1 and must not be hidden.
    """
    seen: dict[tuple, int] = {}
    cols: list[tuple] = []
    for inst in dem.flattened():
        if inst.type != "error" or inst.args_copy()[0] <= 0:
            continue                      # not a fault, or one that can never happen
        dets, obss = [], []
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obss.append(t.val)
        kd = tuple(sorted(x for x in set(dets) if dets.count(x) % 2))
        ko = tuple(sorted(x for x in set(obss) if obss.count(x) % 2))
        if not kd and not ko:
            continue                      # harmless: no detector, no logical flip
        if (kd, ko) not in seen:
            seen[(kd, ko)] = len(cols)
            cols.append((kd, ko))
    return cols


def compute_circuit_distance(dem: stim.DetectorErrorModel, time_limit: float = 60.0, max_weight: int | None = None) -> dict:
    """Exact (or bounded) circuit-level distance of ``dem``.  See the module docstring.

    With ``max_weight`` the problem becomes a decision problem ("is there an undetected logical fault of weight <=
    max_weight?"): infeasible proves ``d >= max_weight + 1`` and is usually much easier for the solver.
    """
    t0 = time.perf_counter()
    nd, no = dem.num_detectors, dem.num_observables
    cols = dem_columns(dem)
    m = len(cols)
    undetectable = [j for j, (kd, ko) in enumerate(cols) if not kd and ko]
    if undetectable:                      # a single fault flips a logical with no detector clicking
        return {"d_upper": 1, "d_lower": 1, "certified": True, "status": "undetectable_logical_fault", "n_mechanisms": m,
                "n_detectors": nd, "n_observables": no, "witness": undetectable[:1],
                "witness_observables": list(cols[undetectable[0]][1]), "seconds": time.perf_counter() - t0, "time_limit": time_limit}
    if not any(ko for _, ko in cols):    # no fault flips any observable: there is no logical failure to measure
        return {"d_upper": None, "d_lower": None, "certified": False, "status": "no_logical_fault", "n_mechanisms": m,
                "n_detectors": nd, "n_observables": no, "seconds": time.perf_counter() - t0, "time_limit": time_limit}
    rows_h, cols_h, rows_o, cols_o = [], [], [], []
    for j, (kd, ko) in enumerate(cols):
        rows_h += kd; cols_h += [j] * len(kd)
        rows_o += ko; cols_o += [j] * len(ko)
    H = sp.csr_matrix((np.ones(len(rows_h)), (rows_h, cols_h)), shape=(nd, m))
    O = sp.csr_matrix((np.ones(len(rows_o)), (rows_o, cols_o)), shape=(no, m))
    degh = np.asarray(H.sum(axis=1)).ravel().astype(int)
    dego = np.asarray(O.sum(axis=1)).ravel().astype(int)
    # variable order: e (m) | k (nd) | u (no) | y (no)
    n = m + nd + no + no
    A1 = sp.hstack([H, -2 * sp.identity(nd), sp.csr_matrix((nd, no)), sp.csr_matrix((nd, no))]).tocsr()
    A2 = sp.hstack([O, sp.csr_matrix((no, nd)), -2 * sp.identity(no), -sp.identity(no)]).tocsr()
    A3 = sp.csr_matrix((np.ones(no), (np.zeros(no, int), np.arange(m + nd + no, n))), shape=(1, n))
    cons = [LinearConstraint(A1, 0, 0), LinearConstraint(A2, 0, 0), LinearConstraint(A3, 1, np.inf)]
    if max_weight is not None:
        cw = np.zeros((1, n)); cw[0, :m] = 1
        cons.append(LinearConstraint(sp.csr_matrix(cw), 0, max_weight))
    lb = np.zeros(n)
    ub = np.concatenate([np.ones(m), degh // 2, dego // 2, np.ones(no)]).astype(float)
    c = np.concatenate([np.ones(m), np.zeros(nd + no + no)])
    res = milp(c, constraints=cons, integrality=np.ones(n), bounds=Bounds(lb, ub),
               options={"time_limit": float(time_limit), "disp": False, "mip_rel_gap": 0.0})
    secs = time.perf_counter() - t0
    out = {"max_weight": max_weight, "n_mechanisms": m, "n_detectors": nd, "n_observables": no, "seconds": secs, "time_limit": time_limit,
           "status": str(res.message).strip()}
    dual = getattr(res, "mip_dual_bound", None)
    if dual is not None and not np.isfinite(dual):
        dual = None
    floor = 2                             # no single fault is undetectable-and-logical (handled above)
    if res.x is not None:
        e = np.round(res.x[:m]).astype(int)
        w = int(e.sum())
        ones = np.flatnonzero(e)
        # independent check of the witness in GF(2): zero syndrome, some observable flipped
        syn = np.asarray(H[:, ones].sum(axis=1)).ravel() % 2 if len(ones) else np.zeros(nd)
        flip = np.asarray(O[:, ones].sum(axis=1)).ravel() % 2 if len(ones) else np.zeros(no)
        assert not syn.any() and flip.any(), "MILP witness failed the GF(2) check"
        out.update(d_upper=w, witness=ones.tolist(), witness_observables=np.flatnonzero(flip).tolist())
        lower = max(floor, int(np.ceil(dual - 1e-6))) if dual is not None else (w if res.status == 0 else floor)
        out["d_lower"] = min(lower, w)
        out["certified"] = bool(res.status == 0 and out["d_lower"] == w)
    elif max_weight is not None and res.status == 2:      # infeasible: no undetected logical fault of weight <= max_weight
        out.update(d_upper=None, d_lower=max_weight + 1, certified=False, proved_no_fault_up_to=max_weight)
    else:
        out.update(d_upper=None, d_lower=max(floor, int(np.ceil(dual - 1e-6))) if dual is not None else None, certified=False)
    return out
