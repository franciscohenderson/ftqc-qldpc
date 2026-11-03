"""Formal equivalence of the generated RTL datapath blocks with the semantics of ``qdecoder.cpp``, for ALL inputs.

How it works.  Yosys exports each combinational block of the generated SystemVerilog (the real generated text, as
elaborated by Yosys) to SMT-LIB.  A specification is written next to it as SMT-LIB bit-vector terms that follow the
C++ kernel operation by operation (``vsat_sub``, ``abs``, ``a < m1`` with ``vsel``, ``m - (m >> 2)``,
``(mag ^ sign) - sign``, ``vsat_add`` chains, ``max(min(t, +8000), -8000)``).  z3 is asked for an input on which the RTL block and
the specification disagree; ``unsat`` is a proof that none exists, over every 16-bit value of every input (not only the
values reachable in a decode).

Scope, stated plainly:
* proved: the check-node block of every distinct degree, the variable-node block of every distinct (degree, channel
  LLR), and the stop test, as combinational functions;
* proved: the range lemmas that make the arithmetic safe (no saturation or absolute-value wrap inside a check node,
  messages stay within +-24576, posteriors within +-8000) *given* the bounds on their inputs, and that those bounds are
  preserved by one iteration (an inductive invariant);
* not proved here: the register/FSM sequencing and the wiring between blocks.  That part is covered by executing the
  whole RTL against the C++ kernel (``scripts/run_rtl.py``).
The specification itself is a transcription of the C++; it is tested against the C++ kernel by the same executions.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from .rtl_gen import LLR_MAX, _emit_cn, _emit_funcs, _emit_vn, block_modules, edges_of
from .rtl_sim import _yosys

R_MAX = 24576                      # |message| bound: 32767 - (32767 >> 2), the largest normalised minimum


def _bv(x: int, w: int = 16) -> str:
    return f"(_ bv{x & ((1 << w) - 1)} {w})"


def export_smt2(sv_text: str, top: str) -> str:
    with tempfile.TemporaryDirectory(prefix="tars_fv_") as td:
        (Path(td) / "b.sv").write_text(sv_text)
        r = subprocess.run(_yosys() + ["-q", "-p", f"read_verilog -sv b.sv; hierarchy -top {top}; proc; flatten; opt_clean; write_smt2 -wires o.smt2"],
                           capture_output=True, text=True, cwd=td)
        out = Path(td) / "o.smt2"
        if r.returncode != 0 or not out.exists():
            raise RuntimeError("yosys smt2 export failed:\n" + (r.stdout + r.stderr)[-2000:])
        return out.read_text()


_PRELUDE = f"""
(define-fun satsub ((x (_ BitVec 16)) (y (_ BitVec 16))) (_ BitVec 16)
  (let ((s (bvsub ((_ sign_extend 1) x) ((_ sign_extend 1) y))))
    (ite (bvsgt s {_bv(32767, 17)}) #x7FFF (ite (bvslt s {_bv(-32768, 17)}) #x8000 ((_ extract 15 0) s)))))
(define-fun satadd ((x (_ BitVec 16)) (y (_ BitVec 16))) (_ BitVec 16)
  (let ((s (bvadd ((_ sign_extend 1) x) ((_ sign_extend 1) y))))
    (ite (bvsgt s {_bv(32767, 17)}) #x7FFF (ite (bvslt s {_bv(-32768, 17)}) #x8000 ((_ extract 15 0) s)))))
(define-fun smin ((x (_ BitVec 16)) (y (_ BitVec 16))) (_ BitVec 16) (ite (bvslt x y) x y))
(define-fun smax ((x (_ BitVec 16)) (y (_ BitVec 16))) (_ BitVec 16) (ite (bvslt x y) y x))
(define-fun abs16 ((x (_ BitVec 16))) (_ BitVec 16) (ite (bvslt x #x0000) (bvneg x) x))
(define-fun inrange ((x (_ BitVec 16)) (b Int)) Bool (and (bvsle x ((_ int2bv 16) b)) (bvsge x (bvneg ((_ int2bv 16) b)))))
"""


def _D(L, n, e):
    L.append(f"(define-fun {n} () (_ BitVec 16) {e})")


def _cn_chain(d: int, q: list[str], syn: str) -> list[str]:
    """C++ first pass: signs, |q|, running first/second minimum and its index (strict <, first wins)."""
    L: list[str] = []
    for k in range(d):
        _D(L, f"q{k}", q[k]); _D(L, f"sq{k}", f"(bvashr q{k} #x000F)"); _D(L, f"a{k}", f"(abs16 q{k})")
    _D(L, "sg0", f"(ite {syn} #xFFFF #x0000)"); _D(L, "m1_0", "#x7FFF"); _D(L, "m2_0", "#x7FFF"); _D(L, "ix0", "#xFFFF")
    for k in range(d):
        _D(L, f"sg{k + 1}", f"(bvxor sg{k} sq{k})")
        L.append(f"(define-fun lt{k} () Bool (bvslt a{k} m1_{k}))")
        _D(L, f"m2_{k + 1}", f"(ite lt{k} m1_{k} (smin m2_{k} a{k}))")
        _D(L, f"ix{k + 1}", f"(ite lt{k} {_bv(k)} ix{k})")
        _D(L, f"m1_{k + 1}", f"(ite lt{k} a{k} m1_{k})")
    return L


def _cn_out(d: int, m1: str, m2: str, is_min: list[str], sg: str, sq: list[str]) -> tuple[list[str], list[str]]:
    """C++ second pass: normalise (m - (m >> 2)) and apply (mag ^ sign) - sign.  ``is_min[k]`` is a Bool: edge k holds the minimum."""
    L: list[str] = []
    _D(L, "n1", f"(bvsub {m1} (bvashr {m1} #x0002))"); _D(L, "n2", f"(bvsub {m2} (bvashr {m2} #x0002))")
    outs = []
    for k in range(d):
        _D(L, f"mag{k}", f"(ite {is_min[k]} n2 n1)"); _D(L, f"sn{k}", f"(bvxor {sg} {sq[k]})")
        _D(L, f"rn{k}", f"(bvsub (bvxor mag{k} sn{k}) sn{k})")
        outs.append(f"rn{k}")
    return L, outs


def _solve(smt: str, timeout_s: int) -> tuple[str, float, str | None]:
    import z3
    s = z3.Solver()
    s.set("timeout", int(timeout_s * 1000))
    t0 = time.perf_counter()
    s.from_string(smt)
    res = s.check()
    model = str(s.model())[:1500] if res == z3.sat else None
    return str(res), time.perf_counter() - t0, model


def _cn_module(d: int) -> str:
    L: list[str] = []
    w = L.append
    ports = ", ".join([f"input logic signed [15:0] tot_{k}" for k in range(d)] + [f"input logic signed [15:0] r_{k}" for k in range(d)]
                      + ["input logic syn"] + [f"output logic signed [15:0] rn_{k}" for k in range(d)])
    w(f"module cn_deg{d} ({ports});")
    _emit_funcs(w)
    w("  logic [0:0] syn_q; assign syn_q[0] = syn;")
    for k in range(d):
        w(f"  logic signed [15:0] q_{k};")
    _emit_cn(w, 0, [(k, k) for k in range(d)])
    w("endmodule")
    return "\n".join(L) + "\n"


def prove_check_node(d: int, timeout_s: int = 3600, mutate=None) -> dict:
    """RTL check node of degree d == C++ semantics for all 16-bit inputs, in three stages, plus the range lemmas.

    A: q_k = vsat_sub(tot_k, r_k).
    B: the running minimum chain (m1, m2, index, sign) computed from the RTL's own q_k equals the C++ chain.
    C: from the RTL's own (m1, m2, index, sign, q), the outputs equal the C++ second pass.
    Each stage quantifies over every value of the RTL state; chaining A, B, C gives the whole block.
    """
    top = f"cn_deg{d}"
    smt = export_smt2((mutate or (lambda t: t))(_cn_module(d)), top)
    W = lambda n: f"(|{top}_n {n}| st)"                      # noqa: E731
    head0 = smt + _PRELUDE + f"(declare-const st |{top}_s|)\n"
    res: dict = {"block": top, "degree": d, "inputs_bits": 16 * 2 * d + 1, "stages": {}}

    def run(name, extra, claim):
        status, secs, mdl = _solve(head0 + extra + f"(assert (not {claim}))\n", timeout_s)
        res["stages"][name] = {"status": status, "seconds": round(secs, 2), "proved": status == "unsat"}
        if mdl:
            res["stages"][name]["counterexample"] = mdl
    # A
    run("A_q_is_saturating_difference", "", "(and " + " ".join(f"(= {W(f'q_{k}')} (satsub {W(f'tot_{k}')} {W(f'r_{k}')}))" for k in range(d)) + ")")
    # B
    chain = "\n".join(_cn_chain(d, [W(f"q_{k}") for k in range(d)], W("syn"))) + "\n"
    run("B_minimum_chain", chain,
        f"(and (= {W('m1_0')} m1_{d}) (= {W('m2_0')} m2_{d}) (= {W('idx_0')} ((_ extract 3 0) ix{d})) (= {W('sg_0')} (distinct ((_ extract 15 15) sg{d}) #b0)))")
    # C
    defs = []
    sq = [f"(bvashr {W(f'q_{k}')} #x000F)" for k in range(d)]
    sgm = f"(ite {W('sg_0')} #xFFFF #x0000)"
    od, outs = _cn_out(d, W("m1_0"), W("m2_0"), [f"(= {W('idx_0')} {_bv(k, 4)})" for k in range(d)], sgm, sq)
    run("C_output_stage", "\n".join(defs + od) + "\n", "(and " + " ".join(f"(= {W(f'rn_{k}')} {outs[k]})" for k in range(d)) + ")")
    res["equivalent_for_all_inputs"] = all(s["proved"] for s in res["stages"].values())
    # range lemmas on the full specification (A+B+C proved it equals the RTL), under |tot| <= LLR_MAX, |r| <= R_MAX
    tot = [W(f"tot_{k}") for k in range(d)]
    r = [W(f"r_{k}") for k in range(d)]
    full = ["(define-fun _unused () Bool true)"] + _cn_chain(d, [f"(satsub {tot[k]} {r[k]})" for k in range(d)], W("syn"))
    od, outs = _cn_out(d, f"m1_{d}", f"m2_{d}", [f"(= ix{d} {_bv(k)})" for k in range(d)], f"sg{d}", [f"sq{k}" for k in range(d)])
    pre = " ".join(f"(inrange {tot[k]} {LLR_MAX}) (inrange {r[k]} {R_MAX})" for k in range(d))
    no_sat = " ".join(f"(and (bvsle (bvsub ((_ sign_extend 1) {tot[k]}) ((_ sign_extend 1) {r[k]})) {_bv(32767, 17)}) "
                      f"(bvsge (bvsub ((_ sign_extend 1) {tot[k]}) ((_ sign_extend 1) {r[k]})) {_bv(-32768, 17)}))" for k in range(d))
    bounded = " ".join(f"(inrange {o} {R_MAX})" for o in outs)
    nowrap = " ".join(f"(not (= q{k} #x8000))" for k in range(d))
    status, secs, mdl = _solve(head0 + "\n".join(full + od) + f"\n(assert (and {pre}))\n(assert (not (and {no_sat} {nowrap} {bounded})))\n", timeout_s)
    res["range_lemmas_hold"] = status == "unsat"; res["range_seconds"] = round(secs, 2)
    res["range_lemmas"] = f"with |tot| <= {LLR_MAX} and |r| <= {R_MAX}: q = tot - r never saturates, q is never -32768 (so |q| never wraps), and every output has |r_new| <= {R_MAX}"
    res["equivalence_seconds"] = round(sum(s["seconds"] for s in res["stages"].values()), 2)
    return res


def prove_variable_node(d: int, ch: int, timeout_s: int = 3600, mutate=None) -> dict:
    L: list[str] = []
    w = L.append
    ports = ", ".join([f"input logic signed [15:0] r_{k}" for k in range(d)] + ["output logic signed [15:0] tn_0"])
    w(f"module vn_deg{d} ({ports});")
    _emit_funcs(w)
    _emit_vn(w, 0, list(range(d)), int(ch))
    w("endmodule")
    top = f"vn_deg{d}"
    smt = export_smt2((mutate or (lambda t: t))("\n".join(L) + "\n"), top)
    r = [f"(|{top}_n r_{k}| st)" for k in range(d)]
    rtl = f"(|{top}_n tn_0| st)"
    defs = [f"(define-fun t0 () (_ BitVec 16) {_bv(ch)})"]
    for k in range(d):
        defs.append(f"(define-fun t{k + 1} () (_ BitVec 16) (satadd t{k} {r[k]}))")
    defs.append(f"(define-fun out () (_ BitVec 16) (smax (smin t{d} {_bv(LLR_MAX)}) {_bv(-LLR_MAX)}))")
    head = smt + _PRELUDE + f"(declare-const st |{top}_s|)\n" + "\n".join(defs) + "\n"
    status, secs, mdl = _solve(head + f"(assert (not (= {rtl} out)))\n", timeout_s)
    res = {"block": top, "degree": d, "channel_llr": int(ch), "equivalent_for_all_inputs": status == "unsat", "equivalence_status": status,
           "equivalence_seconds": round(secs, 2)}
    status, secs, _ = _solve(head + f"(assert (not (inrange {rtl} {LLR_MAX})))\n", timeout_s)
    res["posterior_always_within_llr_max"] = status == "unsat"
    if mdl:
        res["counterexample"] = mdl
    return res


def prove_stop_test(h: np.ndarray, timeout_s: int = 3600, mutate=None) -> dict:
    h = (np.asarray(h) % 2).astype(np.uint8)
    chk, _, _ = edges_of(h)
    m, n = h.shape
    top = "stop_test"
    text = block_modules(h, np.zeros(n, dtype=np.int64))["stop_test"][0]
    smt = export_smt2((mutate or (lambda t: t))(text), top)
    syn = f"(|{top}_n syn_q| st)"
    sign = f"(|{top}_n sign| st)"
    # spec: unsat = OR over checks of (syndrome bit xor hard decisions of the check's variables)
    xor_chain = lambda bits: bits[0] if len(bits) == 1 else "(bvxor " + " ".join(bits) + ")"   # noqa: E731
    par = [xor_chain([f"((_ extract {c} {c}) {syn})"] + [f"((_ extract {v} {v}) {sign})" for _, v in chk[c]]) for c in range(m)]
    spec = f"(distinct (bvor {' '.join(par)}) #b0)" if m > 1 else f"(distinct {par[0]} #b0)"
    out = f"(|{top}_n unsat| st)"
    head = smt + f"(declare-const st |{top}_s|)\n"
    status, secs, mdl = _solve(head + f"(assert (not (= {out} {spec})))\n", timeout_s)
    return {"block": top, "checks": m, "variables": n, "equivalent_for_all_inputs": status == "unsat", "equivalence_status": status,
            "equivalence_seconds": round(secs, 2), **({"counterexample": mdl} if mdl else {})}


def prove_all(h: np.ndarray, ch: np.ndarray, timeout_s: int = 3600, log=print) -> dict:
    """Every distinct block of the design generated for (h, ch)."""
    h = (np.asarray(h) % 2).astype(np.uint8)
    chk, var_edges, _ = edges_of(h)
    out = {"checks": [], "variables": [], "stop_test": None}
    for d in sorted({len(r) for r in chk}):
        log(f"check node degree {d} ..."); out["checks"].append(prove_check_node(d, timeout_s))
    for d, c in sorted({(len(var_edges[v]), int(ch[v])) for v in range(h.shape[1])}):
        log(f"variable node degree {d}, channel LLR {c} ..."); out["variables"].append(prove_variable_node(d, c, timeout_s))
    log("stop test ..."); out["stop_test"] = prove_stop_test(h, timeout_s)
    allok = all(x["equivalent_for_all_inputs"] and x["range_lemmas_hold"] for x in out["checks"]) \
        and all(x["equivalent_for_all_inputs"] and x["posterior_always_within_llr_max"] for x in out["variables"]) \
        and out["stop_test"]["equivalent_for_all_inputs"]
    out["all_proved"] = bool(allok)
    return out
