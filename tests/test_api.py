import pytest
from fastapi.testclient import TestClient

from api.server import app
from circuits.qasm import parse_check

client = TestClient(app)


def test_health_and_index():
    assert client.get("/api/v1/health").json()["status"] == "ok"
    r = client.get("/")
    assert r.status_code == 200 and "html" in r.headers["content-type"]


def test_codes_catalog_has_exact_distances_depths_and_tanner():
    r = client.get("/api/v1/codes")
    assert r.status_code == 200
    codes = {c["name"]: c for c in r.json()["codes"]}
    assert set(codes) == {"BB72", "BB90", "BB108", "BB144", "BB288"}
    assert (codes["BB144"]["n"], codes["BB144"]["k"], codes["BB144"]["d_x"], codes["BB144"]["exact"]) == (144, 12, 12, True)
    assert codes["BB72"]["d_z"] == 6 and codes["BB90"]["d_x"] == 10
    assert codes["BB288"]["exact"] is False and codes["BB288"]["d_x_lower"] < codes["BB288"]["d_x"]
    for c in codes.values():
        assert c["depth"]["sequential"] == 12 and c["depth"]["interleaved"] == 7
    t = codes["BB72"]["tanner"]
    assert len(t["data"]) == 72 and len(t["x_checks"]) == 36 and len(t["z_checks"]) == 36
    assert len(t["x_edges"]) == 36 * 6 and len(t["z_edges"]) == 36 * 6
    assert len(t["layers"]["interleaved"]) == 7 and len(t["layers"]["sequential"]) == 12
    ncnot = sum(len(layer) for layer in t["layers"]["interleaved"])
    assert ncnot == codes["BB72"]["cnot_per_round"] == 432
    # no qubit twice in one layer
    for layer in t["layers"]["interleaved"]:
        qs = [q for _, _, q in layer]
        assert len(qs) == len(set(qs))
    slim = client.get("/api/v1/codes", params={"tanner": "false"}).json()["codes"][0]
    assert "tanner" not in slim


def test_compile_returns_parsable_qasm_and_stats():
    r = client.post("/api/v1/compile", json={"code": "BB72", "schedule": "interleaved", "rounds": 2, "p": 1e-3})
    assert r.status_code == 200
    j = r.json()
    parse_check(j["qasm"])
    assert j["stats"]["cnot_layers"] == 7 and j["stats"]["cnot_per_round"] == 432 and j["stats"]["qubits_total"] == 144
    assert j["dem"]["num_detectors"] > 0 and j["dem"]["num_observables"] == 12
    seq = client.post("/api/v1/compile", json={"code": "BB72", "schedule": "sequential", "rounds": 1}).json()
    assert seq["stats"]["cnot_layers"] == 12 and seq["stats"]["depth_per_round"] == 14


def test_compile_t1_t2_noise_and_validation_errors():
    ok = client.post("/api/v1/compile", json={"code": "BB72", "rounds": 1, "p": 1e-3, "t1_us": 100, "t2_us": 80}).json()
    assert ok["noise"]["idle_pauli"] is not None and ok["noise"]["t1_us"] == 100
    assert client.post("/api/v1/compile", json={"code": "NOPE"}).status_code == 404
    assert client.post("/api/v1/compile", json={"code": "BB72", "rounds": 99}).status_code == 422
    assert client.post("/api/v1/compile", json={"code": "BB72", "p": 0.9}).status_code == 422
    assert client.post("/api/v1/compile", json={"code": "BB72", "t1_us": 100}).status_code == 422


def test_benchmarks_serve_real_results():
    j = client.get("/api/v1/benchmarks").json()
    ti, ts = j["runs"]["final_int"]["threshold"]["estimate"], j["runs"]["final_seq"]["threshold"]["estimate"]
    assert 5e-3 < ti < 6.5e-3 and 2.5e-3 < ts < 3.8e-3 and ti > ts
    pts = j["runs"]["final_int"]["points"]
    assert {p["code"] for p in pts} == {"BB72", "BB90", "BB108", "BB144"}
    assert all(p["lo"] <= p["p_logical"] <= p["hi"] for p in pts)
    lat = j["runs"]["final_int"]["latency"]
    assert {l["mode"] for l in lat} == {"batch-amortized", "single-shot"}
    for l in lat:
        assert len(l["hist"]["edges_us"]) == 41 and len(l["hist"]["counts"]) == 40
        pc = l["percentiles_us"]
        assert pc["P50"] <= pc["P95"] <= pc["P99"] <= pc["max"]
    assert j["notes"] and "BB72@0.001" in j["kernel"]
    assert j["kernel"]["BB72@0.001"]["batch_amortized_ns"]["p50"] < 1000   # sub-microsecond amortized BP kernel


def test_ws_code_capacity_stream_and_reconfigure():
    with client.websocket_connect("/api/v1/stream/decode") as ws:
        ws.send_json({"code": "BB72", "mode": "code_capacity", "p": 0.03, "rate_hz": 30, "batch": 8})
        hello = ws.receive_json()
        assert hello["type"] == "hello" and hello["lanes"] == 16
        # first hello may be for the default config; read until we see the requested one
        seen = None
        for _ in range(8):
            m = ws.receive_json()
            if m["type"] == "hello":
                continue
            seen = m
            if m["shots_in_tick"] == 8:
                break
        assert seen["type"] == "tick" and seen["mode"] == "code_capacity"
        d = seen["display"]
        assert set(d) >= {"err_x", "z_syndrome", "corr_x", "tier_x", "logical_fail"}
        lat = seen["latency"]
        assert lat["bp_kernel_ns"]["p50"] > 0 and lat["total_ns"]["p50"] >= 0 and lat["amortized_ns"]["p50"] > 0
        c = seen["counters"]
        assert c["shots"] >= 8 and c["bp_converged"] + c["osd"] + c["gnn"] + c["failed_decode"] >= 16
        assert 0 <= seen["ler"]["wilson_lo"] <= seen["ler"]["rate"] <= seen["ler"]["wilson_hi"] <= 1
        # corrections must reproduce the syndrome of the injected X errors
        from api.services import code_of
        import numpy as np
        code = code_of("BB72")
        ex = np.zeros(72, int); ex[d["err_x"]] = 1
        cx = np.zeros(72, int); cx[d["corr_x"]] = 1
        if d["tier_x"] != 255:
            assert not ((code.hz.astype(int) @ (ex ^ cx)) % 2).any()
        ws.send_json({"p": 5.0})            # invalid: reported, stream survives
        got_err = False
        for _ in range(10):
            m = ws.receive_json()
            if m["type"] == "error":
                got_err = True
                break
        assert got_err


def test_ws_circuit_mode_and_bad_code():
    with client.websocket_connect("/api/v1/stream/decode") as ws:
        ws.send_json({"code": "BB72", "mode": "circuit", "p": 0.004, "rounds": 2, "batch": 8, "rate_hz": 30, "schedule": "interleaved"})
        tick = None
        for _ in range(12):
            m = ws.receive_json()
            if m["type"] == "tick" and m["mode"] == "circuit":
                tick = m
                break
        assert tick is not None
        assert tick["latency"]["total_ns"]["p50"] > 0 and tick["counters"]["shots"] >= 8
        ws.send_json({"code": "BBXX"})
        assert any(ws.receive_json()["type"] == "error" for _ in range(6))


def test_ui_assets_served_and_self_contained():
    import re
    html = client.get("/").text
    assert "Tars" in html and "Telar" not in html and "/ui/live.js" in html and "/ui/tars.js" in html
    for path in ("live.js", "tars.js", "tars.css", "fonts/familjen-grotesk.woff2", "fonts/martian-mono.woff2"):
        r = client.get(f"/ui/{path}")
        assert r.status_code == 200 and len(r.content) > 1000, path
    for name in ("live.js", "tars.js", "tars.css"):
        body = client.get(f"/ui/{name}").text
        assert not re.search(r"https?://(?!localhost|www\.w3\.org/2000/svg)", body), f"{name} references an external URL"   # el espacio de nombres SVG no es una descarga
    assert not re.search(r"<(script|link)[^>]+(https?:)?//(?!localhost)", html)


def test_compile_window_mode_returns_plan_and_feedback_cadence():
    r = client.post("/api/v1/compile", json={"code": "BB72", "rounds": 4, "p": 1e-3, "decoder": "window", "W": 3, "F": 1})
    assert r.status_code == 200
    j = r.json()
    plan = j["decoder"]["plan"]
    assert j["decoder"]["mode"] == "window" and plan["W"] == 3 and plan["F"] == 1
    assert plan["n_windows"] >= 2 and plan["windows"][-1]["rounds"][1] == plan["detector_layers"]
    assert all(w["detectors"] < plan["full"]["detectors"] for w in plan["windows"])
    assert "if (((r + 1) % 1 == 0) || (r == 3))" in j["qasm"]
    f2 = client.post("/api/v1/compile", json={"code": "BB72", "rounds": 4, "decoder": "window", "W": 4, "F": 2}).json()
    assert "if (((r + 1) % 2 == 0) || (r == 3))" in f2["qasm"]
    full = client.post("/api/v1/compile", json={"code": "BB72", "rounds": 2}).json()
    assert full["decoder"]["mode"] == "full" and full["decoder"]["plan"] is None
    assert "if (((r + 1) % 2 == 0) || (r == 1))" in full["qasm"]          # full horizon: one decode, after the last round
    assert client.post("/api/v1/compile", json={"code": "BB72", "rounds": 4, "decoder": "window", "W": 3, "F": 3}).status_code == 422   # no look-ahead
    assert client.post("/api/v1/compile", json={"code": "BB72", "rounds": 2, "decoder": "window", "W": 5, "F": 3}).status_code == 422   # F > rounds
    assert client.post("/api/v1/compile", json={"code": "BB72", "decoder": "window", "W": 2, "F": 3}).status_code == 422


def test_ws_circuit_window_mode_streams_window_telemetry():
    with client.websocket_connect("/api/v1/stream/decode") as ws:
        ws.send_json({"code": "BB72", "mode": "circuit", "decoder": "window", "W": 3, "F": 1, "p": 0.004, "rounds": 4,
                      "batch": 8, "rate_hz": 30})
        tick = None
        for _ in range(15):
            m = ws.receive_json()
            if m["type"] == "tick" and m["mode"] == "circuit" and m["decoder"] == "window":
                tick = m
                break
        assert tick is not None
        assert tick["window"]["W"] == 3 and tick["window"]["F"] == 1 and tick["window"]["n_windows"] >= 2
        lat = tick["latency"]
        assert lat["amortized_ns"]["p50"] > 0 and lat["total_ns"]["p50"] > 0 and lat["bp_kernel_ns"]["p50"] > 0
        assert lat["reaction_batch_ns"]["p50"] > 0
        assert tick["counters"]["shots"] >= 8
        c = tick["counters"]
        assert c["bp_converged"] + c["osd"] + c["failed_decode"] == c["shots"]      # counters are per shot, not per window
        ws.send_json({"W": 2, "F": 3})
        assert any(ws.receive_json()["type"] == "error" for _ in range(8))
        ws.send_json({"W": 3, "F": 3})                         # F == W: no look-ahead round
        assert any(ws.receive_json()["type"] == "error" for _ in range(8))
