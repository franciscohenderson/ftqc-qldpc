"""The covering-sweep logic must agree with the certified circuit distance of BB72 (known witnesses)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
fs = pytest.importorskip("run_forced_sweep")


@pytest.mark.skipif(not (ROOT / "results" / "circuit_distance.json").exists(), reason="needs the saved witnesses")
def test_forcing_a_known_fault_is_feasible_at_its_weight_and_infeasible_below():
    runs = {r["schedule"]: r for r in json.loads((ROOT / "results" / "circuit_distance.json").read_text())["runs"] if r["code"] == "BB72" and r["rounds"] == 6}
    wit = runs["sequential"]["witness"]                       # certified d_circ = 4
    j0 = min(wit)
    fs._init(("BB72", "sequential", 6, "Z", 4))
    assert fs._solve((j0, 300))["status"] == 0
    fs._init(("BB72", "sequential", 6, "Z", 3))
    assert fs._solve((j0, 300))["status"] == 2
