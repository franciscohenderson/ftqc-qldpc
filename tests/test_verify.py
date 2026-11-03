import json

from hardware import verify


def test_quick_verify_passes_and_writes_a_report(capsys):
    rc = verify.main(["--quick", "--shots", "60"])
    out = capsys.readouterr().out
    assert rc == 0 and "[PASS] manifest" in out and "[PASS] model" in out and "[PASS] gates" in out
    rep = json.loads((verify.ROOT / "results" / "verify_report.json").read_text())
    assert rep["failed"] == 0 and {c["name"] for c in rep["checks"]} == {"manifest", "model", "gates"}


def test_manifest_detects_tampering(tmp_path, monkeypatch):
    sv = verify.RTL / "tars_bp_bb72_hz.sv"
    original = sv.read_text()
    try:
        sv.write_text(original.replace("16'sd62", "16'sd63", 1))
        status, detail = verify.check_manifest()
        assert status == "FAIL" and "hash mismatch" in detail
    finally:
        sv.write_text(original, newline="\n") if False else sv.write_bytes(original.encode())
    assert verify.check_manifest()[0] == "PASS"
