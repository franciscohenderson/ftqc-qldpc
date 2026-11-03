"""OpenQASM 3.0 compilation of the syndrome-extraction schedule with real-time feedback.

The emitted program contains:
  * mid-circuit measurement + reset of ancillas (`measure`, `reset`) every round,
  * the entangling schedule as parallel layers separated by `barrier`,
  * a classical syndrome history register and an `extern` real-time decoder call per round,
  * dynamic feedback: Pauli corrections applied with `if` on the decoder's return registers.

The decoder is declared as an extern function (OpenQASM 3 `extern`), the standard hook through which
control hardware binds a real-time classical co-processor; its latency budget is what the
low-latency decoder tier must meet.
"""
from __future__ import annotations

from algebra.qldpc import CSSCode

from .schedule import Layer, validate_schedule


def compile_openqasm3(code: CSSCode, layers: list[Layer], rounds: int, feedback_every: int = 1,
                      basis: str = "Z") -> str:
    validate_schedule(layers, code.hx, code.hz)
    n, mx, mz = code.n, code.hx.shape[0], code.hz.shape[0]
    L = []
    w = L.append
    w("OPENQASM 3.0;")
    w('include "stdgates.inc";')
    w(f"// {code.name}: [[{n},{code.k}]] CSS qLDPC syndrome extraction, {rounds} rounds, depth {len(layers)} CNOT layers/round")
    w(f"// Real-time decoder binding: decode_x returns X corrections, decode_z returns Z corrections.")
    w(f"extern decode_x(bit[{mz * rounds}]) -> bit[{n}];")
    w(f"extern decode_z(bit[{mx * rounds}]) -> bit[{n}];")
    w("")
    w(f"qubit[{n}] data;")
    w(f"qubit[{mx}] anc_x;")
    w(f"qubit[{mz}] anc_z;")
    w(f"bit[{mx}] syn_x;")
    w(f"bit[{mz}] syn_z;")
    w(f"bit[{mx * rounds}] hist_x;")
    w(f"bit[{mz * rounds}] hist_z;")
    w(f"bit[{n}] corr_x;")
    w(f"bit[{n}] corr_z;")
    w(f"bit[{n}] readout;")
    w("")
    w("// ---- logical state preparation (transversal)")
    w("reset data;")
    if basis == "X":
        w("h data;")
    w("")
    w(f"for int r in [0:{rounds - 1}] {{")
    w("    // ancilla preparation: X-check ancillas in |+>, Z-check ancillas in |0>")
    w("    reset anc_x;")
    w("    reset anc_z;")
    w("    h anc_x;")
    w("    barrier data, anc_x, anc_z;")
    for t, layer in enumerate(layers):
        w(f"    // CNOT layer {t}")
        for kind, c, q in layer:
            if kind == "X":
                w(f"    cx anc_x[{c}], data[{q}];")
            else:
                w(f"    cx data[{q}], anc_z[{c}];")
        w("    barrier data, anc_x, anc_z;")
    w("    // mid-circuit measurement")
    w("    h anc_x;")
    w("    syn_x = measure anc_x;")
    w("    syn_z = measure anc_z;")
    w(f"    for int i in [0:{mx - 1}] {{ hist_x[r * {mx} + i] = syn_x[i]; }}")
    w(f"    for int i in [0:{mz - 1}] {{ hist_z[r * {mz} + i] = syn_z[i]; }}")
    if feedback_every and feedback_every > 0:
        w(f"    // dynamic feedback every {feedback_every} round(s): real-time decode + conditional Paulis")
        w(f"    if (((r + 1) % {feedback_every} == 0) || (r == {rounds - 1})) {{")
        w("        corr_x = decode_x(hist_z);")
        w("        corr_z = decode_z(hist_x);")
        w(f"        for int i in [0:{n - 1}] {{")
        w("            if (corr_x[i] == 1) { x data[i]; }")
        w("            if (corr_z[i] == 1) { z data[i]; }")
        w("        }")
        w("    }")
    w("}")
    w("")
    w("// ---- transversal logical readout")
    if basis == "X":
        w("h data;")
    w("readout = measure data;")
    return "\n".join(L) + "\n"


def parse_check(program: str):
    """Parse with the reference openqasm3 parser (raises on syntax errors). Returns the AST."""
    import openqasm3
    return openqasm3.parse(program)
