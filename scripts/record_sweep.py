"""Turn a finished covering sweep into the decision entry that FACTS, the whitepaper and Tars read."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sweep = json.load(open(sys.argv[1]))
s = sweep["summary"]
assert s["complete"] and s["feasible"] == 0 and s["unresolved"] == 0, "the sweep does not prove the claim"
dec_path = ROOT / "results" / "circuit_distance_decision.json"
data = json.load(open(dec_path)) if dec_path.exists() else {"runs": []}
data["runs"] = [r for r in data["runs"] if not (r["code"] == sweep["code"] and r["schedule"] == sweep["schedule"] and r["rounds"] == sweep["rounds"] and r.get("basis") == sweep["basis"] and r.get("max_weight") == sweep["max_weight"])]
layers = {"interleaved": 7, "sequential": 12}[sweep["schedule"]]
data["runs"].append({"code": sweep["code"], "schedule": sweep["schedule"], "cnot_layers": layers, "rounds": sweep["rounds"], "basis": sweep["basis"],
                     "max_weight": sweep["max_weight"], "d_lower": sweep["max_weight"] + 1, "d_upper": None, "certified": False,
                     "proved_no_fault_up_to": sweep["max_weight"], "seconds": s["wall_seconds"], "time_limit": 300,
                     "status": f"covering sweep: {s['infeasible']} of {sweep['mechanisms']} subproblems infeasible, 0 unresolved (6 workers; per-subproblem limit 300 s)",
                     "method": "forced_sweep", "n_mechanisms": sweep["mechanisms"]})
json.dump(data, open(dec_path, "w"), indent=1)
print("recorded:", s["claim"])
