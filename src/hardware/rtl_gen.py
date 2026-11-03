"""Synthesizable SystemVerilog-2012 generator for the int16 normalized min-sum BP kernel of ``qdecoder.cpp``.

One module decodes one syndrome (one SIMD lane of the C++ kernel).  The arithmetic is the C++ arithmetic, operation
by operation, so the posteriors are bit-identical:

* messages and posteriors are int16, saturating add/sub (``__builtin_elementwise_add_sat``);
* check node: first minimum ``m1`` (strict ``<``, first index wins), second minimum ``m2``, normalised
  ``alpha = 0.75`` as ``m - (m >>> 2)``, sign = parity of the other signs xor the syndrome bit;
* variable node: ``ch + sum(r)`` accumulated in edge order with saturation, then clamped to [-8000, +8000];
* stop test: every check satisfied by the hard decisions, evaluated before each iteration (0 iterations possible).

Schedule (flooding, fully parallel: all checks in one cycle, all variables in the next)::

    IDLE --start--> load (1 cycle) --> [ CN (1) -> VN (1) ] x iters --> CN (1, finds the syndrome satisfied)

so a syndrome that converges after k iterations takes ``2k + 2`` clock cycles; one that does not converge takes
``2*MAX_ITER + 2``.  These are RTL cycles, not CPU time.
"""
from __future__ import annotations

import numpy as np

LLR_MAX = 8000


def _check_inputs(h: np.ndarray, ch: np.ndarray, max_iter: int):
    h = (np.asarray(h) % 2).astype(np.uint8)
    ch = np.asarray(ch).astype(np.int64)
    m, n = h.shape
    if ch.shape != (n,):
        raise ValueError("one channel LLR per column is required")
    if (np.abs(ch) > LLR_MAX).any():
        raise ValueError("channel LLRs must fit the saturation range")
    if h.sum(axis=1).max() > 15:
        raise ValueError("check degree above 15 does not fit the 4-bit minimum index")
    if (h.sum(axis=0) == 0).any() or (h.sum(axis=1) == 0).any():
        raise ValueError("empty rows/columns are not supported")
    if not 1 <= max_iter <= 255:
        raise ValueError("max_iter must be in 1..255")
    return h, ch, m, n


def edges_of(h: np.ndarray):
    """Edge numbering identical to qdecoder.cpp: CSR by check, then per-variable lists in edge order."""
    chk, e = [], 0
    var_edges: dict[int, list[int]] = {}
    for c in range(h.shape[0]):
        row = []
        for v in np.flatnonzero(h[c]):
            row.append((e, int(v)))
            var_edges.setdefault(int(v), []).append(e)
            e += 1
        chk.append(row)
    return chk, var_edges, e


# ------------------------------------------------------------------ emitters shared by the full module and the sizing blocks
def _emit_funcs(w) -> None:
    for op, sym in (("add", "+"), ("sub", "-")):
        w(f"  function automatic logic signed [15:0] sat_{op}(input logic signed [15:0] x, input logic signed [15:0] y);")
        w("    logic signed [16:0] s;")
        w("    begin")
        w(f"      s = 17'(x) {sym} 17'(y);")
        w(f"      if (s > 17'sd32767) sat_{op} = 16'sh7FFF;")
        w(f"      else if (s < -17'sd32768) sat_{op} = 16'sh8000;")
        w(f"      else sat_{op} = s[15:0];")
        w("    end")
        w("  endfunction")


def _emit_stop(w, chk, m: int) -> None:
    parts = []
    for c in range(m):
        terms = " ^ ".join([f"syn_q[{c}]"] + [f"tot_{v}[15]" for _, v in chk[c]])
        parts.append(f"({terms})")
    w("  logic unsat;")
    w("  assign unsat = " + "\n               | ".join(parts) + ";")


def _emit_cn(w, c: int, edges) -> None:
    """Check node ``c``: q = tot - r, first/second minimum, normalisation, sign.  ``edges`` = [(edge id, variable)]."""
    for e, v in edges:
        w(f"  assign q_{e} = sat_sub(tot_{v}, r_{e});")
    w(f"  logic signed [15:0] m1_{c}, m2_{c}, a_{c}, n1_{c}, n2_{c};")
    w(f"  logic [3:0] idx_{c};")
    w(f"  logic sg_{c};")
    w(f"  always_comb begin : cn_{c}")
    w(f"    m1_{c} = 16'sh7FFF; m2_{c} = 16'sh7FFF; idx_{c} = 4'hF; sg_{c} = syn_q[{c}];")
    for k, (e, _) in enumerate(edges):
        w(f"    a_{c} = q_{e}[15] ? -q_{e} : q_{e}; sg_{c} = sg_{c} ^ q_{e}[15];")
        w(f"    if (a_{c} < m1_{c}) begin m2_{c} = m1_{c}; idx_{c} = 4'd{k}; m1_{c} = a_{c}; end")
        w(f"    else if (a_{c} < m2_{c}) m2_{c} = a_{c};")
    w(f"    n1_{c} = m1_{c} - (m1_{c} >>> 2);")
    w(f"    n2_{c} = m2_{c} - (m2_{c} >>> 2);")
    w("  end")
    for k, (e, _) in enumerate(edges):
        w(f"  assign rn_{e} = (sg_{c} ^ q_{e}[15]) ? -((idx_{c} == 4'd{k}) ? n2_{c} : n1_{c}) : ((idx_{c} == 4'd{k}) ? n2_{c} : n1_{c});")


def _emit_vn(w, v: int, es, chv: int) -> None:
    """Variable node ``v``: channel LLR + saturating sum of its messages (in edge order), clamped."""
    w(f"  logic signed [15:0] t_{v};")
    w(f"  always_comb begin : vn_{v}")
    w(f"    t_{v} = 16'sd{chv};" if chv >= 0 else f"    t_{v} = -16'sd{-chv};")
    for e in es:
        w(f"    t_{v} = sat_add(t_{v}, r_{e});")
    w(f"    tn_{v} = (t_{v} > 16'sd{LLR_MAX}) ? 16'sd{LLR_MAX} : ((t_{v} < -16'sd{LLR_MAX}) ? -16'sd{LLR_MAX} : t_{v});")
    w("  end")


def _lit(x: int) -> str:
    return f"16'sd{x}" if x >= 0 else f"-16'sd{-x}"


def generate_bp_sv(h: np.ndarray, ch: np.ndarray, max_iter: int = 30, name: str = "tars_bp") -> str:
    h, ch, m, n = _check_inputs(h, ch, max_iter)
    chk, var_edges, E = edges_of(h)
    L: list[str] = []
    w = L.append
    w(f"// {name}.sv - generated by src/hardware/rtl_gen.py; do not edit by hand.")
    w(f"// Normalized min-sum BP (alpha = 0.75), int16 saturating, [-{LLR_MAX}, +{LLR_MAX}], flooding schedule.")
    w(f"// H: {m} checks x {n} variables, {E} edges, max {max_iter} iterations. One syndrome per run.")
    w("// Cycles from the start edge to done: 2*iters + 2.  Bit-exact with src/decoder/cpp/qdecoder.cpp (BP part).")
    w("`default_nettype none")
    w(f"module {name} (")
    w("  input  logic clk,")
    w("  input  logic rst,")
    w("  input  logic start,                      // pulse while idle: latches `syndrome`")
    w(f"  input  logic [{m - 1}:0] syndrome,")
    w("  output logic busy,")
    w("  output logic done,                       // stays high until the next start")
    w("  output logic converged,                  // every check satisfied by `hard`")
    w("  output logic [7:0] iters,                // iterations performed")
    w(f"  output logic [{n - 1}:0] hard,            // hard decisions (posterior < 0)")
    w(f"  output logic [{n * 16 - 1}:0] post_flat   // posterior LLRs, int16, variable j in bits [16j+15:16j]")
    w(");")
    w(f"  localparam logic [7:0] MAX_ITER = 8'd{max_iter};")
    w("")
    _emit_funcs(w)
    w("")
    w("  typedef enum logic [1:0] {S_IDLE, S_CN, S_VN} state_t;")
    w("  state_t state;")
    w(f"  logic [{m - 1}:0] syn_q;")
    w("  logic [7:0] it_q;")
    for v in range(n):
        w(f"  logic signed [15:0] tot_{v}, tn_{v};")
    for e in range(E):
        w(f"  logic signed [15:0] r_{e}, rn_{e}, q_{e};")
    w("")
    w("  // ---- stop test: parity of the hard decisions on every check, xor the syndrome bit")
    _emit_stop(w, chk, m)
    w("")
    w("  // ---- check nodes (all in parallel): q = tot - r, two minima, normalisation, sign")
    for c in range(m):
        _emit_cn(w, c, chk[c])
        w("")
    w("  // ---- variable nodes (all in parallel): channel + saturating sum of the messages, clamped")
    for v in range(n):
        _emit_vn(w, v, var_edges[v], int(ch[v]))
    w("")
    w("  // ---- control")
    w("  always_ff @(posedge clk) begin")
    w("    if (rst) begin")
    w("      state <= S_IDLE; busy <= 1'b0; done <= 1'b0; converged <= 1'b0; iters <= 8'd0; it_q <= 8'd0;")
    w("      syn_q <= '0;")
    for v in range(n):
        w(f"      tot_{v} <= 16'sd0;")
    for e in range(E):
        w(f"      r_{e} <= 16'sd0;")
    w("    end else begin")
    w("      case (state)")
    w("        S_IDLE: if (start) begin")
    w("          syn_q <= syndrome; it_q <= 8'd0; busy <= 1'b1; done <= 1'b0; converged <= 1'b0; state <= S_CN;")
    for v in range(n):
        w(f"          tot_{v} <= {_lit(int(ch[v]))};")
    for e in range(E):
        w(f"          r_{e} <= 16'sd0;")
    w("        end")
    w("        S_CN: begin")
    w("          if (!unsat) begin")
    w("            converged <= 1'b1; iters <= it_q; done <= 1'b1; busy <= 1'b0; state <= S_IDLE;")
    w("          end else if (it_q == MAX_ITER) begin")
    w("            converged <= 1'b0; iters <= it_q; done <= 1'b1; busy <= 1'b0; state <= S_IDLE;")
    w("          end else begin")
    for e in range(E):
        w(f"            r_{e} <= rn_{e};")
    w("            state <= S_VN;")
    w("          end")
    w("        end")
    w("        S_VN: begin")
    for v in range(n):
        w(f"          tot_{v} <= tn_{v};")
    w("          it_q <= it_q + 8'd1; state <= S_CN;")
    w("        end")
    w("        default: state <= S_IDLE;")
    w("      endcase")
    w("    end")
    w("  end")
    w("")
    for v in range(n):
        w(f"  assign hard[{v}] = tot_{v}[15];")
        w(f"  assign post_flat[{16 * v + 15}:{16 * v}] = tot_{v};")
    w("endmodule")
    w("`default_nettype wire")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ sizing blocks (the full design is too big for the WASM Yosys)
def block_modules(h: np.ndarray, ch: np.ndarray) -> dict:
    """Stand-alone combinational modules built from the same emitters, used to size the datapath without synthesizing
    the whole ~200k-cell design at once.  Returns {module name: (sv text, instances in the full design)}."""
    h = (np.asarray(h) % 2).astype(np.uint8)
    chk, var_edges, _ = edges_of(h)
    m, n = h.shape
    out: dict = {}
    for d in sorted({len(r) for r in chk}):
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
        out[f"cn_deg{d}"] = ("\n".join(L) + "\n", sum(1 for r in chk if len(r) == d))
    for d in sorted({len(x) for x in var_edges.values()}):
        L = []
        w = L.append
        ports = ", ".join([f"input logic signed [15:0] r_{k}" for k in range(d)] + ["output logic signed [15:0] tn_0"])
        w(f"module vn_deg{d} ({ports});")
        _emit_funcs(w)
        _emit_vn(w, 0, list(range(d)), int(np.median(ch)))
        w("endmodule")
        out[f"vn_deg{d}"] = ("\n".join(L) + "\n", sum(1 for x in var_edges.values() if len(x) == d))
    L = []
    w = L.append
    w(f"module stop_test (input logic [{m - 1}:0] syn_q, input logic [{n - 1}:0] sign, output logic unsat);")
    for v in range(n):
        w(f"  wire tot_{v}_15 = sign[{v}];")
    parts = ["(" + " ^ ".join([f"syn_q[{c}]"] + [f"tot_{v}_15" for _, v in chk[c]]) + ")" for c in range(m)]
    w("  assign unsat = " + " | ".join(parts) + ";")
    w("endmodule")
    out["stop_test"] = ("\n".join(L) + "\n", 1)
    return out


def flip_flop_bits(h: np.ndarray) -> dict:
    """Exact register bits of the generated module."""
    h = np.asarray(h) % 2
    m, n = h.shape
    e = int(h.sum())
    parts = {"posterior_tot": 16 * n, "messages_r": 16 * e, "syndrome": m, "iteration_counter": 8, "state": 2,
             "flags_busy_done_converged": 3, "iters_out": 8}
    parts["total"] = sum(parts.values())
    return parts


def generate_testbench_sv(m: int, n: int, name: str, vectors: int, in_mem: str, out_mem: str) -> str:
    """Self-checking testbench reading ``in_mem`` (syndromes) and ``out_mem`` ({iters, hard, post}) with $readmemh."""
    ow = 8 + n + n * 16
    return f"""// tb_{name}.sv - self-checking testbench for {name}; run with any SystemVerilog-2012 simulator.
`timescale 1ns/1ps
module tb_{name};
  logic clk = 0, rst = 1, start = 0;
  logic [{m - 1}:0] syndrome;
  logic busy, done, converged;
  logic [7:0] iters;
  logic [{n - 1}:0] hard;
  logic [{n * 16 - 1}:0] post_flat;
  logic [{m - 1}:0] vin [0:{vectors - 1}];
  logic [{ow - 1}:0] vout [0:{vectors - 1}];
  {name} dut (.*);
  always #2 clk = ~clk;
  integer i, errors = 0, cycles;
  initial begin
    $readmemh("{in_mem}", vin);
    $readmemh("{out_mem}", vout);
    repeat (3) @(posedge clk);
    rst = 0;
    for (i = 0; i < {vectors}; i = i + 1) begin
      @(negedge clk); syndrome = vin[i]; start = 1;
      @(negedge clk); start = 0; cycles = 1;
      while (!done) begin @(negedge clk); cycles = cycles + 1; end
      if ({{iters, hard, post_flat}} !== vout[i]) begin errors = errors + 1; $display("MISMATCH vector %0d", i); end
      if (cycles !== 2 * iters + 2) begin errors = errors + 1; $display("CYCLES vector %0d: %0d", i, cycles); end
    end
    $display("%0d vectors, %0d errors", {vectors}, errors);
    $finish;
  end
endmodule
"""
