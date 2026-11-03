"""Timing probe: is 'weight <= W with mechanism j forced into the fault' fast for HiGHS?  (BB72, 7 layers, 6 rounds)"""
import sys, time
from pathlib import Path
import numpy as np, scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from algebra.circuit_distance import dem_columns
from api.services import code_of, get_layers
from circuits.builder import build_memory_experiment
from circuits.noise import NoiseModel

CODE = sys.argv[3] if len(sys.argv) > 3 else "BB72"
code = code_of(CODE)
dem = build_memory_experiment(code, 6, NoiseModel.uniform(1e-3), basis="Z", layers=get_layers(CODE, "interleaved")).dem()
cols = dem_columns(dem); m = len(cols); nd, no = dem.num_detectors, dem.num_observables
rows_h, cols_h, rows_o, cols_o = [], [], [], []
for j, (kd, ko) in enumerate(cols):
    rows_h += kd; cols_h += [j] * len(kd); rows_o += ko; cols_o += [j] * len(ko)
H = sp.csr_matrix((np.ones(len(rows_h)), (rows_h, cols_h)), shape=(nd, m)); O = sp.csr_matrix((np.ones(len(rows_o)), (rows_o, cols_o)), shape=(no, m))
degh = np.asarray(H.sum(1)).ravel().astype(int); dego = np.asarray(O.sum(1)).ravel().astype(int)
n = m + nd + no + no
A1 = sp.hstack([H, -2 * sp.identity(nd), sp.csr_matrix((nd, no)), sp.csr_matrix((nd, no))]).tocsr()
A2 = sp.hstack([O, sp.csr_matrix((no, nd)), -2 * sp.identity(no), -sp.identity(no)]).tocsr()
A3 = sp.csr_matrix((np.ones(no), (np.zeros(no, int), np.arange(m + nd + no, n))), shape=(1, n))
cw = np.zeros((1, n)); cw[0, :m] = 1
W = int(sys.argv[1]) if len(sys.argv) > 1 else 4
rng = np.random.default_rng(int(sys.argv[2]) if len(sys.argv) > 2 else 0)
for j in rng.choice(m, 3, replace=False):
    lb = np.zeros(n); ub = np.concatenate([np.ones(m), degh // 2, dego // 2, np.ones(no)]).astype(float); lb[j] = 1
    t0 = time.perf_counter()
    r = milp(np.zeros(n), constraints=[LinearConstraint(A1, 0, 0), LinearConstraint(A2, 0, 0), LinearConstraint(A3, 1, np.inf), LinearConstraint(sp.csr_matrix(cw), 0, W)],
             integrality=np.ones(n), bounds=Bounds(lb, ub), options={"time_limit": 120.0})
    print(f"mech {j:5d} W<={W}: status {r.status} ({'infeasible' if r.status == 2 else 'feasible' if r.x is not None else 'timeout'}) {time.perf_counter() - t0:.1f}s", flush=True)
