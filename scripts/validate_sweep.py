"""Validate the forcing/exclusion logic of run_forced_sweep on faults whose weight is already known from certified/verified witnesses."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
import run_forced_sweep as fs
runs = {(r["schedule"]): r for r in json.load(open(ROOT / "results" / "circuit_distance.json"))["runs"] if r["code"] == "BB72" and r["rounds"] == 6 and r["basis"] == "Z"}
ok = True
for sched, w in (("sequential", 4), ("interleaved", 5)):
    wit = runs[sched]["witness"]; j0 = min(wit)
    for W, expect in ((w, 0), (w - 1, 2)):
        fs._init(("BB72", sched, 6, "Z", W))
        r = fs._solve((j0, 600))
        good = r["status"] == expect
        ok &= good
        print(f"{sched:11s} known weight {w}: forced j={j0}, W<={W}: status {r['status']} ({'feasible' if r['status']==0 else 'infeasible' if r['status']==2 else 'other'}) in {r['seconds']}s -> {'as expected' if good else 'UNEXPECTED'}", flush=True)
print("VALIDATION", "OK" if ok else "FAILED"); sys.exit(0 if ok else 1)
