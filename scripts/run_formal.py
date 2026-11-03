"""Prove the generated RTL blocks equivalent to the C++ semantics for all inputs (z3) and save results/rtl_formal.json.

    python scripts/run_formal.py [--code BB72] [--timeout 1800]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from algebra.qldpc import get_code            # noqa: E402
from decoder.fast import FastDecoder          # noqa: E402
from hardware.formal import prove_all         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="BB72")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    code = get_code(a.code)
    ch = FastDecoder(code.hz, 0.02, max_iter=30).channel_llr()
    r = prove_all(code.hz, ch, a.timeout)
    r.update(code=a.code, matrix="H_Z", z3="z3-solver", scope=("check node (per degree), variable node (per degree and channel LLR) and stop test, as combinational functions, for every 16-bit input; "
                                                               "range lemmas for the check node; NOT the register/FSM sequencing or the wiring between blocks (covered by executing the whole RTL)"))
    (ROOT / "results" / "rtl_formal.json").write_text(json.dumps(r, indent=1), encoding="utf-8")
    print("all proved:", r["all_proved"])
    sys.exit(0 if r["all_proved"] else 1)


if __name__ == "__main__":
    main()
