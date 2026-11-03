"""The formal proofs must pass on the generated RTL, and must FAIL (with a counterexample) on RTL with a planted bug."""
import numpy as np
import pytest

from algebra.qldpc import get_code
from decoder.fast import FastDecoder
from hardware.rtl_sim import have_toolchain

pytest.importorskip("z3")
pytestmark = pytest.mark.skipif(not have_toolchain(), reason="yowasp-yosys not installed")

from hardware.formal import prove_all, prove_check_node, prove_stop_test, prove_variable_node  # noqa: E402


def test_generated_blocks_are_proved_for_all_inputs():
    code = get_code("BB72")
    ch = FastDecoder(code.hz, 0.02, max_iter=30).channel_llr()
    r = prove_all(code.hz, ch, 600, log=lambda *_: None)
    assert r["all_proved"]
    assert {x["degree"] for x in r["checks"]} == {6} and all(x["range_lemmas_hold"] for x in r["checks"])


@pytest.mark.parametrize("name,old,new", [
    ("tie-break: <= instead of <", "if (a_0 < m1_0)", "if (a_0 <= m1_0)"),
    ("alpha 0.875 instead of 0.75", "m1_0 - (m1_0 >>> 2)", "m1_0 - (m1_0 >>> 3)"),
    ("second minimum not tracked", "else if (a_0 < m2_0) m2_0 = a_0;", ";"),
    ("sign of the syndrome dropped", "sg_0 = syn_q[0];", "sg_0 = 1'b0;"),
])
def test_planted_check_node_bugs_are_caught(name, old, new):
    r = prove_check_node(3, 300, mutate=lambda t: t.replace(old, new))
    assert old != new and not r["equivalent_for_all_inputs"], name
    assert any("counterexample" in s for s in r["stages"].values())      # the solver produced an input where they differ


def test_planted_variable_node_and_stop_test_bugs_are_caught():
    assert not prove_variable_node(3, 62, 300, mutate=lambda t: t.replace("8000", "8001"))["equivalent_for_all_inputs"]
    assert not prove_variable_node(3, 62, 300, mutate=lambda t: t.replace("sat_add(t_0, r_2)", "(t_0 + r_2)"))["equivalent_for_all_inputs"]
    H = get_code("BB72").hz
    assert not prove_stop_test(H, 300, mutate=lambda t: t.replace("syn_q[0] ^", "", 1))["equivalent_for_all_inputs"]
