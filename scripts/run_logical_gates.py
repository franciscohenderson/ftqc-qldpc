"""Enumerate the translation-automorphism logical gates of the BB codes and verify them with Stim.

    python scripts/run_logical_gates.py [--out results/logical_gates.json] [CODE ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algebra.logical_gates import enumerate_translation_gates   # noqa: E402
from algebra.qldpc import get_code                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", default=["BB72", "BB90", "BB108", "BB144"])
    ap.add_argument("--out", default=str(ROOT / "results" / "logical_gates.json"))
    a = ap.parse_args()
    out = {}
    for name in a.codes:
        code = get_code(name)
        t0 = time.perf_counter()
        res = enumerate_translation_gates(code, stim_trials=6)
        nontriv = [r for r in res if r.get("nontrivial_logical")]
        classes = [r for r in nontriv if r["same_logical_as"] is None]
        checked = sum(r["stim"]["logical_predictions_checked"] for r in classes)
        bad = sum(r["stim"]["logical_prediction_mismatches"] for r in classes)
        all_ok = all(r["stim"]["effective_perm_matches"] and r["stim"]["code_space_preserved"] for r in classes)
        out[name] = {"n": code.n, "k": code.k, "l": code.meta["l"], "m": code.meta["m"], "translations_tested": len(res),
                     "automorphisms": sum(1 for r in res if r["automorphism"]),
                     "nontrivial_logical_gates": len(nontriv), "distinct_logical_matrices": len(classes),
                     "stim_predictions_checked": checked, "stim_mismatches": bad, "stim_code_space_preserved": all_ok,
                     "seconds": round(time.perf_counter() - t0, 2),
                     "examples": [{"shift": r["shift"], "u_z_ones": r["u_z_ones"], "u_z": r["u_z"]} for r in classes[:3]]}
        print(f'{name}: {out[name]["automorphisms"]}/{len(res)} automorphisms, {len(nontriv)} nontrivial, {len(classes)} distinct U, '
              f'stim {checked} checks / {bad} mismatches, code space ok={all_ok}, {out[name]["seconds"]} s', flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
