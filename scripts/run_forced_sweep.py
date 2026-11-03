"""Exact circuit-distance bound by a covering sweep: is there an undetected logical fault of weight <= W?

    python scripts/run_forced_sweep.py --code BB72 --schedule interleaved --rounds 6 --weight 4 --workers 6

Subproblem j forces mechanism j into the fault and excludes every mechanism with a smaller index.  Every fault of weight <= W has a
smallest mechanism index j, so it is found in exactly subproblem j; if all subproblems are infeasible, NO such fault exists and
d_circ >= W + 1.  A subproblem that hits its time limit is reported, and the claim is then NOT made (the covering has a hole).
Progress is saved after every batch (results/forced_sweep_<code>_<schedule>_r<rounds>_w<W>.json) and the sweep resumes from it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_S: dict = {}


def _build(code_name: str, schedule: str, rounds: int, basis: str, W: int, sym: bool = False):
    from algebra.circuit_distance import dem_columns
    from api.services import code_of, get_layers
    from circuits.builder import build_memory_experiment
    from circuits.noise import NoiseModel
    code = code_of(code_name)
    dem = build_memory_experiment(code, rounds, NoiseModel.uniform(1e-3), basis=basis, layers=get_layers(code_name, schedule)).dem()
    cols = dem_columns(dem)
    m, nd, no = len(cols), dem.num_detectors, dem.num_observables
    rh, ch, ro, co = [], [], [], []
    for j, (kd, ko) in enumerate(cols):
        rh += kd; ch += [j] * len(kd); ro += ko; co += [j] * len(ko)
    H = sp.csr_matrix((np.ones(len(rh)), (rh, ch)), shape=(nd, m))
    O = sp.csr_matrix((np.ones(len(ro)), (ro, co)), shape=(no, m))
    degh = np.asarray(H.sum(1)).ravel().astype(int); dego = np.asarray(O.sum(1)).ravel().astype(int)
    n = m + nd + no + no
    A1 = sp.hstack([H, -2 * sp.identity(nd), sp.csr_matrix((nd, no)), sp.csr_matrix((nd, no))]).tocsr()
    A2 = sp.hstack([O, sp.csr_matrix((no, nd)), -2 * sp.identity(no), -sp.identity(no)]).tocsr()
    A3 = sp.csr_matrix((np.ones(no), (np.zeros(no, int), np.arange(m + nd + no, n))), shape=(1, n))
    cw = np.zeros((1, n)); cw[0, :m] = 1
    cons = [LinearConstraint(A1, 0, 0), LinearConstraint(A2, 0, 0), LinearConstraint(A3, 1, np.inf), LinearConstraint(sp.csr_matrix(cw), 0, W)]
    ub0 = np.concatenate([np.ones(m), degh // 2, dego // 2, np.ones(no)]).astype(float)
    undetectable = [j for j, (kd, ko) in enumerate(cols) if not kd and ko]
    label = reps = None
    if sym:                                             # orbits of the verified translation symmetry (raises if it is not a symmetry)
        from algebra.circuit_symmetry import orbits
        o = orbits(code, dem, basis)
        label, reps = o["label"], o["representatives"]
    return {"m": m, "n": n, "cons": cons, "ub0": ub0, "undetectable": undetectable, "label": label, "reps": reps}


def _init(args):
    _S.update(_build(*args))


def _solve(task):
    j, tlim = task
    n, m = _S["n"], _S["m"]
    lb = np.zeros(n); ub = _S["ub0"].copy()
    if _S["label"] is not None:
        ub[:m][_S["label"] < _S["label"][j]] = 0         # exclude every mechanism of an earlier orbit; j is its orbit's representative
    else:
        ub[:j] = 0                                       # exclude smaller indices
    lb[j] = 1                                            # force j
    t0 = time.perf_counter()
    r = milp(np.zeros(n), constraints=_S["cons"], integrality=np.ones(n), bounds=Bounds(lb, ub), options={"time_limit": float(tlim)})
    retried = False
    if r.status == 4:                                   # HiGHS "solve error" (seen in its presolve): ask again without presolve, record that we did
        retried = True
        r = milp(np.zeros(n), constraints=_S["cons"], integrality=np.ones(n), bounds=Bounds(lb, ub), options={"time_limit": float(tlim), "presolve": False})
    out = {"j": j, "status": int(r.status), "seconds": round(time.perf_counter() - t0, 1), **({"retried_without_presolve": True} if retried else {})}
    if r.x is not None:
        out["witness"] = np.flatnonzero(np.round(r.x[:m]) > 0.5).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="BB72"); ap.add_argument("--schedule", default="interleaved")
    ap.add_argument("--rounds", type=int, default=6); ap.add_argument("--basis", default="Z")
    ap.add_argument("--weight", type=int, default=4); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--task-limit", type=float, default=300.0)
    ap.add_argument("--symmetry", action="store_true", help="one subproblem per orbit of the verified translation symmetry")
    a = ap.parse_args()
    args = (a.code, a.schedule, a.rounds, a.basis, a.weight, a.symmetry)
    out_path = ROOT / "results" / f"forced_sweep_{a.code}_{a.schedule}_{a.basis}_r{a.rounds}_w{a.weight}{"_sym" if a.symmetry else ""}.json"
    info = _build(*args)
    m = info["m"]
    data = json.loads(out_path.read_text()) if out_path.exists() else {"code": a.code, "schedule": a.schedule, "rounds": a.rounds, "basis": a.basis,
                                                                         "max_weight": a.weight, "mechanisms": m, "done": {}}
    for k, v in (("code", a.code), ("schedule", a.schedule), ("rounds", a.rounds), ("basis", a.basis), ("max_weight", a.weight), ("mechanisms", m)):
        if data[k] != v:                                  # never resume from a file that belongs to a different problem
            raise SystemExit(f"progress file mismatch on {k}: {data[k]!r} != {v!r}")
    if info["undetectable"]:
        print("an undetectable logical mechanism exists: d_circ = 1"); return
    universe = info["reps"] if a.symmetry else list(range(m))
    todo = [j for j in universe if str(j) not in data["done"] or data["done"][str(j)]["status"] not in (2, 0)]
    print(f"{a.code} {a.schedule}: {m} mechanisms, {len(data['done'])} already done, {len(todo)} to go, weight <= {a.weight}", flush=True)
    t0 = time.perf_counter(); found = []
    with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(args,)) as ex:
        for k, r in enumerate(ex.map(_solve, [(j, a.task_limit) for j in todo], chunksize=1)):
            data["done"][str(r["j"])] = r
            if r["status"] == 0:
                found.append(r)
            if k % 20 == 19 or k == len(todo) - 1:
                out_path.write_text(json.dumps(data), encoding="utf-8")
                print(f"  {k + 1}/{len(todo)} subproblems, {time.perf_counter() - t0:.0f}s elapsed, found {len(found)}, latest j={r['j']} status {r['status']} {r['seconds']}s", flush=True)
    st = [v["status"] for v in data["done"].values()]
    n_inf, n_feas, n_other = st.count(2), st.count(0), len(st) - st.count(2) - st.count(0)
    complete = len(st) == len(universe)
    data["summary"] = {"complete": complete, "infeasible": n_inf, "feasible": n_feas, "unresolved": n_other,
                       "claim": (f"no undetected logical fault of weight <= {a.weight}: d_circ >= {a.weight + 1}" if complete and n_feas == 0 and n_other == 0 else
                                 f"found a fault of weight <= {a.weight}" if n_feas else "INCOMPLETE: no claim"),
                       "wall_seconds": round(time.perf_counter() - t0, 1)}
    out_path.write_text(json.dumps(data), encoding="utf-8")
    print(json.dumps(data["summary"]))


if __name__ == "__main__":
    main()
