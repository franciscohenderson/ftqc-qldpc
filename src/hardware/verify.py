"""One command that re-checks the hardware-facing claims of this repository on the machine that runs it.

    tars-verify                 # everything (needs: pip install -e ".[verify]"; ~3 minutes)
    tars-verify --quick         # skips the RTL compile/execution and the formal proofs (~20 s)
    tars-verify --write-manifest

Each check prints PASS / FAIL / SKIP with its time.  Exit code 0 only if nothing FAILED.  A skipped check is never
reported as passed.  The report is written to results/verify_report.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RTL = ROOT / "exports" / "rtl"
MANIFEST = RTL / "MANIFEST.sha256"
BB72_P, BB72_ITER = 0.02, 30


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write_manifest() -> None:
    names = sorted(f.name for f in RTL.iterdir() if f.suffix in (".sv", ".mem"))
    MANIFEST.write_text("".join(f"{_sha(RTL / n)}  {n}\n" for n in names), encoding="utf-8", newline="\n")


def _setup():
    from algebra.qldpc import get_code
    from decoder.fast import FastDecoder
    code = get_code("BB72")
    dec = FastDecoder(code.hz, BB72_P, max_iter=BB72_ITER)
    return code, dec, dec.channel_llr()


def _syndromes(code, shots: int, seed: int = 2026, p: float = 0.04):
    err = (np.random.default_rng(seed).random((shots, code.n)) < p).astype(np.uint8)
    return ((err @ code.hz.T) % 2).astype(np.uint8)


def check_manifest():
    from .rtl_gen import generate_bp_sv
    code, _, ch = _setup()
    lines = [ln.split() for ln in MANIFEST.read_text().splitlines()] if MANIFEST.exists() else []
    if not lines:
        return "FAIL", "exports/rtl/MANIFEST.sha256 is missing"
    bad = [n for h, n in lines if not (RTL / n).exists() or _sha(RTL / n) != h]
    if bad:
        return "FAIL", f"hash mismatch: {', '.join(bad)}"
    regenerated = generate_bp_sv(code.hz, ch, BB72_ITER, "tars_bp_bb72_hz")
    if regenerated != (RTL / "tars_bp_bb72_hz.sv").read_text():
        return "FAIL", "the committed .sv is not what the generator emits from this checkout"
    return "PASS", f"{len(lines)} files match their SHA-256 and the .sv equals a fresh generator run"


def check_model(shots: int):
    from .rtl_sim import bp_fixed
    code, dec, ch = _setup()
    syn = _syndromes(code, shots)
    post, it, conv, cyc = bp_fixed(code.hz, ch, syn, BB72_ITER)
    bad = 0
    for i in range(shots):
        p, k, st = dec.bp_posterior(syn[i])
        bad += not (np.array_equal(p, post[i]) and k == it[i] and (st == 0) == conv[i])
    ok = bad == 0 and np.array_equal(cyc, 2 * it + 2) and conv.any() and not conv.all()
    return ("PASS" if ok else "FAIL"), f"{shots} syndromes ({int(conv.sum())} converge, {int((~conv).sum())} do not): {bad} mismatches vs the production C++ kernel"


def check_rtl(shots: int):
    from .rtl_sim import RtlSim, have_toolchain
    if not have_toolchain():
        return "SKIP", "yowasp-yosys is not installed (pip install -e '.[verify]')"
    code, dec, _ = _setup()
    syn = _syndromes(code, shots)
    sim = RtlSim(RTL / "tars_bp_bb72_hz.sv", "tars_bp_bb72_hz", code.hz.shape[0], code.n)
    post, it, conv, cyc = sim.run(syn)
    bad = 0
    for i in range(shots):
        p, k, st = dec.bp_posterior(syn[i])
        bad += not (np.array_equal(p, post[i]) and k == it[i] and (st == 0) == conv[i])
    ok = bad == 0 and np.array_equal(cyc, 2 * it + 2)
    return ("PASS" if ok else "FAIL"), f"the generated SystemVerilog, executed clock by clock, on {shots} syndromes: {bad} mismatches vs C++; cycles = 2*iters+2 on every vector"


def check_formal():
    try:
        import z3  # noqa: F401
        from .rtl_sim import have_toolchain
        if not have_toolchain():
            raise ImportError("yowasp-yosys")
    except ImportError as e:
        return "SKIP", f"{e} is not installed (pip install -e '.[verify]')"
    from .formal import prove_all
    code, _, ch = _setup()
    r = prove_all(code.hz, ch, 900, log=lambda *_: None)
    cn = r["checks"][0]
    return ("PASS" if r["all_proved"] else "FAIL"), (f"check node degree {cn['degree']}, variable nodes and stop test equal to the C++ semantics for ALL inputs "
                                                    f"(z3, {cn['equivalence_seconds']:.1f} s on the check node)")


def check_gates(n_shifts: int = 6):
    from algebra.logical_gates import translation_perm, verify_in_memory_circuit, verify_with_stim, induced_logical_matrices, preserves_stabilizers
    from api.services import get_layers
    code, _, _ = _setup()
    layers = get_layers("BB72", "interleaved")
    bad = 0
    shifts = [(1, 0), (0, 1), (2, 3), (3, 1), (5, 4), (1, 1)][:n_shifts]
    for s in shifts:
        perm = translation_perm(code, *s)
        uz, _ = induced_logical_matrices(code, perm)
        v = verify_with_stim(code, perm, uz, trials=2)
        m = verify_in_memory_circuit(code, layers, {1: perm}, rounds=4, shots=16)
        bad += (not preserves_stabilizers(code, perm)) + v["logical_prediction_mismatches"] + m["violations"] + (not v["code_space_preserved"])
    return ("PASS" if bad == 0 else "FAIL"), f"{len(shifts)} translations of BB72 as logical gates, inside the {len(layers)}-layer circuit: {bad} violations"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tars-verify", description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="skip the RTL execution and the formal proofs")
    ap.add_argument("--shots", type=int, default=300)
    ap.add_argument("--write-manifest", action="store_true")
    a = ap.parse_args(argv)
    if a.write_manifest:
        write_manifest()
        print("wrote", MANIFEST)
        return 0
    steps = [("manifest      generated RTL is the committed RTL", check_manifest),
             ("model         fixed-point model == C++ kernel", lambda: check_model(a.shots)),
             ("gates         zero-cost logical gates in Stim", check_gates)]
    if not a.quick:
        steps += [("rtl           SystemVerilog executed == C++ kernel", lambda: check_rtl(a.shots)),
                  ("formal        RTL blocks == C++ for all inputs (z3)", check_formal)]
    report = {"python": platform.python_version(), "platform": platform.platform(), "checks": []}
    failed = 0
    for name, fn in steps:
        t0 = time.perf_counter()
        try:
            status, detail = fn()
        except Exception as e:                                   # a crash is a failure, never a pass
            status, detail = "FAIL", f"{type(e).__name__}: {e}"
        dt = time.perf_counter() - t0
        failed += status == "FAIL"
        print(f"[{status:4}] {name:52s} {dt:6.1f}s  {detail}", flush=True)
        report["checks"].append({"name": name.split()[0], "status": status, "seconds": round(dt, 1), "detail": detail})
    report["failed"] = failed
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "verify_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("\nRESULT:", "ALL CHECKS PASSED" if not failed and not any(c["status"] == "SKIP" for c in report["checks"]) else
          ("FAILED" if failed else "no failures, but some checks were skipped"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
