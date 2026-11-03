"""RTL generator: the generated SystemVerilog, run clock edge by clock edge, must equal qdecoder.cpp bit for bit."""
import json
from pathlib import Path

import numpy as np
import pytest

from algebra.qldpc import get_code
from decoder.fast import FastDecoder
from hardware.rtl_gen import generate_bp_sv, generate_testbench_sv
from hardware.rtl_sim import RtlSim, bp_fixed, have_toolchain, pack_output, syndrome_to_int

ROOT = Path(__file__).resolve().parents[1]

HAMMING = np.array([[1, 0, 1, 0, 1, 0, 1],
                    [0, 1, 1, 0, 0, 1, 1],
                    [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
# a denser toy graph with a degree-5 check, to exercise the two-minima tie-breaking
TOY = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                [0, 1, 0, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1, 0, 1],
                [0, 0, 1, 0, 0, 0, 1, 1]], dtype=np.uint8)


def _syndromes(h, shots, p, seed):
    e = (np.random.default_rng(seed).random((shots, h.shape[1])) < p).astype(np.uint8)
    return (e @ h.T) % 2


@pytest.mark.parametrize("h", [HAMMING, TOY], ids=["hamming", "toy"])
def test_fixed_point_model_equals_cpp(h):
    dec = FastDecoder(h, 0.05, max_iter=12)
    ch = dec.channel_llr()
    syn = _syndromes(h, 400, 0.15, 7)
    post, it, conv, cyc = bp_fixed(h, ch, syn, 12)
    for i in range(len(syn)):
        p, k, st = dec.bp_posterior(syn[i])
        assert np.array_equal(p, post[i]) and k == it[i] and (st == 0) == conv[i]
    assert np.array_equal(cyc, 2 * it + 2)


def test_bb72_model_equals_cpp():
    code = get_code("BB72")
    dec = FastDecoder(code.hz, 0.02, max_iter=30)
    ch = dec.channel_llr()
    syn = _syndromes(code.hz, 120, 0.04, 2026)
    post, it, conv, _ = bp_fixed(code.hz, ch, syn, 30)
    for i in range(len(syn)):
        p, k, st = dec.bp_posterior(syn[i])
        assert np.array_equal(p, post[i]) and k == it[i] and (st == 0) == conv[i]
    assert conv.any() and not conv.all()                     # both outcomes are exercised


@pytest.mark.skipif(not have_toolchain(), reason="yowasp-yosys not installed")
@pytest.mark.parametrize("h,name,mixed", [(HAMMING, "t_hamming", False), (TOY, "t_toy", True)])
def test_generated_sv_executes_bit_exact(tmp_path, h, name, mixed):
    dec = FastDecoder(h, 0.05, max_iter=12)
    ch = dec.channel_llr()
    sv = tmp_path / f"{name}.sv"
    sv.write_text(generate_bp_sv(h, ch, 12, name))
    sim = RtlSim(sv, name, h.shape[0], h.shape[1], workdir=tmp_path / "w")
    syn = _syndromes(h, 300, 0.15, 11)
    post, it, conv, cyc = sim.run(syn)
    assert conv.any() and len(set(it.tolist())) > 1           # different iteration counts are exercised
    assert (not conv.all()) or not mixed                     # the toy graph also hits the max-iteration exit
    for i in range(len(syn)):
        p, k, st = dec.bp_posterior(syn[i])
        assert np.array_equal(p, post[i]), i
        assert k == it[i] and (st == 0) == conv[i]
    assert np.array_equal(cyc, 2 * it + 2)                   # cycle-accurate: exactly 2 per iteration + load + final check
    assert cyc.min() == 2                                    # an all-zero syndrome stops without iterating


def test_generator_rejects_bad_input():
    ch = np.full(7, 40)
    with pytest.raises(ValueError):
        generate_bp_sv(HAMMING, ch[:5], 10)                  # wrong LLR length
    with pytest.raises(ValueError):
        generate_bp_sv(HAMMING, np.full(7, 9000), 10)        # outside the saturation range
    with pytest.raises(ValueError):
        generate_bp_sv(HAMMING, ch, 0)
    with pytest.raises(ValueError):
        generate_bp_sv(np.zeros((2, 7), np.uint8), ch, 10)   # empty rows


def test_sv_is_synthesizable_style_and_complete():
    sv = generate_bp_sv(HAMMING, np.full(7, 40), 12, "dut")
    assert "module dut" in sv and "endmodule" in sv and "always_ff @(posedge clk)" in sv
    for banned in ("#", "$display", "initial", "real ", "$random"):
        assert banned not in sv.replace("`default_nettype", "")         # nothing simulation-only in the design
    assert sv.count("sat_add(") >= HAMMING.sum()                        # one saturating accumulation per edge
    tb = generate_testbench_sv(3, 7, "dut", 5, "in.mem", "out.mem")
    assert "$readmemh" in tb and "dut dut" in tb


def test_vector_packing_matches_port_order():
    post = np.array([5, -1, 300, -32768], dtype=np.int16)
    v = pack_output(post, 3)
    assert v >> (4 + 64) == 3                                             # iters on top
    assert (v >> 64) & 0xF == 0b1010                                      # hard bits: variables 1 and 3 are negative
    assert (v & 0xFFFF) == 5 and ((v >> 48) & 0xFFFF) == 0x8000          # variable 0 at bits 15:0, variable 3 at 63:48
    assert syndrome_to_int(np.array([1, 0, 1])) == 5


def test_bb72_results_file_is_consistent():
    p = ROOT / "results" / "rtl_bitexact.json"
    if not p.exists():
        pytest.skip("run scripts/run_rtl.py first")
    r = json.loads(p.read_text())
    assert r["bit_exact"]["rtl_vs_cpp"]["all"] and r["bit_exact"]["model_vs_cpp"] and r["bit_exact"]["rtl_vs_model"]
    assert r["cycles"]["formula_holds_on_every_vector"] and r["shots"] >= 1000
    assert r["converged_shots"] > 0 and r["not_converged_shots"] > 0
    sv = (ROOT / r["files"][0]).read_text()
    code = get_code(r["code"])
    ch = FastDecoder(code.hz, 0.02, max_iter=r["max_iter"]).channel_llr()
    assert sv == generate_bp_sv(code.hz, ch, r["max_iter"], f"tars_bp_{r['code'].lower()}_hz")   # the file on disk is what the generator emits
