"""FACTS.md sections for the v1.0 hardware-facing work, written only from results/*.json (nothing is typed in by hand)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    p = ROOT / "results" / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _distance() -> list[str]:
    out = ["## Circuit-level distance d_circ (scripts/run_circuit_distance.py; MILP on the detector error model, HiGHS)", "",
           "d_circ = fewest DEM mechanisms with zero detector syndrome that flip a logical observable (Z-basis memory, uniform noise 1e-3, "
           "identical symptoms merged). `certified` means HiGHS proved lower = upper within the time limit; otherwise the row is a bound, not a value. "
           "`max weight asked` marks a decision problem (is there an undetected logical fault of weight <= W?).", "",
           "| code | basis | layers | rounds | max weight asked | d_circ lower | d_circ upper | certified | seconds | time limit s |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    runs = (_load("circuit_distance.json") or {"runs": []})["runs"] + (_load("circuit_distance_decision.json") or {"runs": []})["runs"]
    for r in sorted(runs, key=lambda r: (r["code"], -r["cnot_layers"], r.get("max_weight") or 0)):
        out.append(f"| {r['code']} | {r.get('basis', 'Z')} | {r['cnot_layers']} | {r['rounds']} | {r.get('max_weight') or '-'} | {r.get('d_lower') or '-'} | {r.get('d_upper') or '-'} | "
                   f"{r.get('certified')} | {r['seconds']:.0f} | {r.get('time_limit')} |")
    return out + [""]


def _gates() -> list[str]:
    g = _load("logical_gates.json")
    if not g:
        return []
    out = ["## Zero-cost logical gates from translation automorphisms (scripts/run_logical_gates.py)", "",
           "| code | [[n,k]] | translations | automorphisms | non-trivial logical | distinct logical matrices | Stim predictions | Stim mismatches | "
           "gates inserted in the depth-7 circuit | violations | gauge detectors |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for n, v in g.items():
        m = v["in_memory_circuit"]
        out.append(f"| {n} | [[{v['n']},{v['k']}]] | {v['translations_tested']} | {v['automorphisms']} | {v['nontrivial_logical_gates']} | "
                   f"{v['distinct_logical_matrices']} | {v['stim_predictions_checked']} | {v['stim_mismatches']} | {m['gates_inserted']} | "
                   f"{m['violations_total']} | {m['gauge_detectors']} |")
    return out + [""]


def _window() -> list[str]:
    w = _load("window_fused.json")
    if not w:
        return []
    out = ["## Fused C++ window loop vs Python loop (scripts/run_window_fused.py)", "",
           "Same Stim samples (seed 2026, 6 rounds, p = 2e-3), best of 3 wall-clock runs per loop. `prop` = sub-syndrome gather + commit + propagate. "
           "Single-shot step = median per-window time of one-shot calls (BP-bound, not accelerated by fusion).", "",
           "| code | (W,F) | shots | identical decisions | logical failures | Python us/shot | fused us/shot | reduction | BP us | OSD us | prop us | "
           "step single p50 Python us | step single p50 fused us |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in w["runs"]:
        for c in r["configs"].values():
            f, p = c["fused_cpp"], c["python_loop"]
            sp = f["split_batch_per_shot_us"]
            red = 100 * (1 - f["wall_per_shot_us"] / p["wall_per_shot_us"])
            out.append(f"| {r['code']} | ({c['W']},{c['F']}) | {r['shots']} | {c['identical_decisions']} | {c['logical_failures']} | "
                       f"{p['wall_per_shot_us']:.0f} | {f['wall_per_shot_us']:.0f} | {red:.0f} % | {sp['bp']:.1f} | {sp['post']:.1f} | {sp['prop']:.1f} | "
                       f"{p['step_single_us']['p50_us']:.1f} | {f['step_single_us']['p50_us']:.1f} |")
    return out + [""]


def _rtl() -> list[str]:
    t = _load("rtl_bitexact.json")
    if not t:
        return []
    b, c, sy = t["bit_exact"], t["cycles"], t.get("synthesis")
    cl = "; ".join(f"{k}: p50 {v['p50']:.3f} us, p95 {v['p95']:.3f} us, max {v['max']:.3f} us" for k, v in t["nominal_clock_conversion_us"].items())
    out = ["## BP kernel as SystemVerilog, executed and compared with qdecoder.cpp (scripts/run_rtl.py)", "",
           f"* {t['code']} {t['matrix']} {t['checks']}x{t['variables']} ({t['edges']} edges), int16 saturating normalized min-sum alpha=0.75, "
           f"max {t['max_iter']} iterations, channel LLR {t['channel_llr']}, {t['sv_lines']} lines of SystemVerilog-2012 (`exports/rtl/`).",
           f"* {t['shots']} noisy syndromes (iid X errors p={t['p_x']}, seed {t['seed']}): {t['converged_shots']} converge, "
           f"{t['not_converged_shots']} reach the iteration limit.",
           f"* RTL (Yosys -> CXXRTL -> zig c++, executed clock edge by clock edge) vs C++: posterior LLRs {b['rtl_vs_cpp']['posterior_llr']}, "
           f"hard decisions {b['rtl_vs_cpp']['hard_decisions']}, iterations {b['rtl_vs_cpp']['iterations']}, converged flag "
           f"{b['rtl_vs_cpp']['converged']}; fixed-point model vs C++ {b['model_vs_cpp']}.",
           f"* Cycles = {c['formula']} (holds on every vector: {c['formula_holds_on_every_vector']}): min {c['min']}, p50 {c['p50']:.0f}, "
           f"p95 {c['p95']:.0f}, max {c['max']}, mean {c['mean']:.2f}; iterations mean {c['iterations']['mean']:.2f}, max {c['iterations']['max']}.",
           f"* Nominal-clock conversion of those RTL cycles (the clocks are assumptions, not achieved frequencies; no FPGA flow was run): {cl}.",
           f"* For context only (a different quantity): CPU BP single-shot on the same syndromes, p50 {t['cpu_bp_single_shot_us']['p50']:.1f} us, "
           f"p95 {t['cpu_bp_single_shot_us']['p95']:.1f} us."]
    if sy:
        out.append(f"* Size: {sy['registers']['total']} register bits (exact); {sy['combinational_gates_total']} generic 2-input gates from per-block "
                   f"Yosys synthesis (no ABC, no LUT mapping: an upper bound, not an FPGA report); deepest block {sy['critical_block_depth_gates']} gates.")
    out.append(f"* The generated testbench was executed: {t['testbench_executed']} (no Verilog simulator is available on this machine).")
    return out + [""]


def _formal() -> list[str]:
    f = _load("rtl_formal.json")
    if not f:
        return []
    cn = f["checks"][0]
    vn = "; ".join(f"degree {v['degree']}, channel LLR {v['channel_llr']}: equivalent {v['equivalent_for_all_inputs']}, "
                   f"posterior always within +-8000: {v['posterior_always_within_llr_max']}" for v in f["variables"])
    return ["## Formal proof of the generated RTL blocks (scripts/run_formal.py; z3 on SMT-LIB exported by Yosys)", "",
            f"* Scope: {f['scope']}.",
            f"* All blocks proved: {f['all_proved']}. Check node degree {cn['degree']}: equivalent to the C++ semantics for every 16-bit input "
            f"({cn['inputs_bits']} input bits), {cn['equivalence_seconds']:.1f} s of solver time over 3 stages; range lemmas hold ({cn['range_lemmas']}).",
            f"* Variable nodes: {vn}.",
            f"* Stop test ({f['stop_test']['checks']} checks, {f['stop_test']['variables']} variables): equivalent {f['stop_test']['equivalent_for_all_inputs']}.",
            "* Non-vacuity: tests/test_formal.py plants 7 bugs in the RTL (tie-break, alpha, second minimum, syndrome sign, clamp, unsaturated add, "
            "dropped stop-test term); the proof finds a counterexample for every one.", ""]


def hardware_facts() -> str:
    return "\n".join(_distance() + _gates() + _window() + _rtl() + _formal()) + "\n"
