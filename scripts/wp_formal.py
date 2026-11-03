"""Section 9A.5 of the whitepaper (formal equivalence of the RTL blocks), written from results/rtl_formal.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def formal_block() -> str:
    p = ROOT / "results" / "rtl_formal.json"
    if not p.exists():
        return ""
    f = json.loads(p.read_text(encoding="utf-8"))
    cn = f["checks"][0]
    vn = "; ".join(f"degree {v['degree']} with channel LLR {v['channel_llr']}" for v in f["variables"])
    t = ["### 9A.5 Formal Equivalence of the RTL Blocks", "",
         "Executing the RTL on 1000 vectors shows agreement on those vectors only. We therefore also proved the blocks equivalent to the C++ semantics for **every** input value. "
         "Yosys exports each combinational block of the generated SystemVerilog to SMT-LIB; the specification is written as bit-vector terms that follow `qdecoder.cpp` operation by operation; "
         "the z3 solver is asked for an input on which they differ, and `unsat` is the proof (`src/hardware/formal.py`, `scripts/run_formal.py`). "
         f"The check node of degree {cn['degree']} ({cn['inputs_bits']} input bits) is proved in three stages (saturating difference; running first and second minimum with index and sign; normalisation and output) "
         f"in {cn['equivalence_seconds']:.1f} s of solver time. The variable nodes ({vn}) and the stop test ({f['stop_test']['checks']} checks) are proved the same way. "
         "A range lemma shows that, whenever $|\\mathrm{tot}| \\le 8000$ and $|r| \\le 24576$, the difference $q$ never saturates, $q$ is never $-32768$ (so the absolute value never wraps), "
         "and every new message again satisfies $|r| \\le 24576$; posteriors are clamped to $\\pm 8000$, so these bounds are an inductive invariant.", "",
         "To check that the proof is not vacuous, `tests/test_formal.py` plants seven bugs in the RTL (a non-strict comparison, $\\alpha = 0.875$, no second minimum, a dropped syndrome sign, a clamp of 8001, "
         "an unsaturated addition, a dropped term in the stop test) and requires a counterexample for each; the solver finds all seven.", "",
         "**Scope.** Proved: the combinational blocks, as functions, for all inputs. Not proved: the register and control sequencing and the wiring between blocks, which rest on the execution of the whole design "
         "against the C++ kernel (Section 9A.4). The specification is itself a transcription of the C++ and is validated by that same execution. "
         "We claim formally verified datapath blocks of one decoder, not a formally verified decoder."]
    return "\n".join(t) + "\n"
