"""Compute [[n, k, d_X, d_Z]] for the BB zoo (ILP with dual bounds), export matrices, write a table.

    python -m algebra.report --out exports --time-limit 300
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .distance import distance_x, distance_z
from .export import export_code
from .qldpc import BB_ZOO, get_code


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="exports")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--codes", nargs="+", default=list(BB_ZOO))
    a = ap.parse_args(argv)
    table = []
    for name in a.codes:
        c = get_code(name)
        t0 = time.time()
        dz, dx = distance_z(c, a.time_limit), distance_x(c, a.time_limit)
        info = {
            "code": name, "n": c.n, "k": c.k, "published_d": c.meta["published_d"],
            "dZ_lower": dz.lower, "dZ_upper": dz.upper, "dZ_exact": dz.exact,
            "dX_lower": dx.lower, "dX_upper": dx.upper, "dX_exact": dx.exact,
            "w_col": max(c.weights["col_x"], c.weights["col_z"]), "w_row": max(c.weights["row_x"], c.weights["row_z"]),
            "m_x": int(c.hx.shape[0]), "m_z": int(c.hz.shape[0]), "ilp_seconds": round(time.time() - t0, 1),
        }
        export_code(c, a.out, info)
        table.append(info)
        print(json.dumps(info), flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "code_parameters.json").write_text(json.dumps(table, indent=2), encoding="utf-8")
    lines = ["| Code | [[n,k,d]] (ILP) | d_X | d_Z | exact | w_c / w_r | checks (X+Z) |", "|---|---|---|---|---|---|---|"]
    for t in table:
        fx = f"{t['dX_lower']}" if t["dX_exact"] else f"{t['dX_lower']}-{t['dX_upper']}"
        fz = f"{t['dZ_lower']}" if t["dZ_exact"] else f"{t['dZ_lower']}-{t['dZ_upper']}"
        ub = min(t["dX_upper"], t["dZ_upper"])
        lines.append(f"| {t['code']} | [[{t['n']},{t['k']},{'≤' if not (t['dX_exact'] and t['dZ_exact']) else ''}{ub}]] | {fx} | {fz} "
                     f"| {'yes' if t['dX_exact'] and t['dZ_exact'] else 'bounds'} | {t['w_col']} / {t['w_row']} | {t['m_x'] + t['m_z']} |")
    (Path(a.out) / "code_parameters.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
