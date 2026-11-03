# API contract (FastAPI, http://localhost:8000)

All JSON. The UI is served at `/` from `ui/index.html` (single file, no CDN, no build step).

## GET /api/v1/codes?tanner=true
```json
{ "codes": [ {
  "name": "BB72", "n": 72, "k": 12,
  "d_x": 6, "d_z": 6, "d_x_lower": 6, "d_z_lower": 6, "exact": true, "published_d": 6,
  "m_x": 36, "m_z": 36, "w_col": 3, "w_row": 6, "ilp_seconds": 0.8,
  "depth": {"sequential": 12, "interleaved": 7},
  "cnot_per_round": 432,
  "tanner": {                       // present unless ?tanner=false
    "l": 6, "m": 6,
    "data":     [{"id": 0, "x": 0.0, "y": 0.0, "block": "L"}, ...],     // n entries; x,y in torus units [0,l)x[0,m)
    "x_checks": [{"id": 0, "x": 0.5, "y": 0.0}, ...],                    // m_x entries
    "z_checks": [{"id": 0, "x": 0.0, "y": 0.5}, ...],                    // m_z entries
    "x_edges": [[check, qubit], ...], "z_edges": [[check, qubit], ...],
    "layers": {
      "sequential":  [[["X", check, qubit], ["Z", check, qubit], ...], ...],   // 12 layers
      "interleaved": [[...], ...]                                              // 7 layers
    }
  } } ] }
```
`d_x`/`d_z` are the ILP incumbent (upper bound); `exact` is true iff lower == upper for both.

## POST /api/v1/compile
Request: `{"code":"BB72","schedule":"interleaved"|"sequential","rounds":3,"basis":"Z","p":1e-3,"t1_us":null,"t2_us":null,"layer_time_us":0.05}`
(if `t1_us` and `t2_us` are both given, idle noise is the T1/T2* Pauli twirl and `p` is the gate/prep/measure error).
Response:
```json
{ "code":"BB72","schedule":"interleaved","rounds":3,"qasm":"OPENQASM 3.0;\n...",
  "stats":{"n":72,"ancillas":72,"qubits_total":144,"cnot_layers":7,"depth_per_round":9,"cnot_per_round":432,"cnot_total":1296,
           "qasm_lines":123,"stim_instructions":456},
  "dem":{"num_detectors":144,"num_errors":1152,"num_observables":12,"max_detectors_per_error":6},
  "noise":{"p_gate":0.001,"p_prep":0.001,"p_meas":0.001,"p_idle":0.001,"t1_us":null,"t2_us":null,"idle_pauli":[a,b,c]|null},
  "elapsed_ms": 123.4 }
```
Errors: 404 unknown code, 422 invalid parameters (rounds 1..12, p in (0,0.5)).

## GET /api/v1/benchmarks
```json
{ "machine": "a desktop processor with AVX2 ...",
  "runs": {
    "final_int": { "label":"Interleaved (depth 7)", "schedule":"interleaved", "rounds":6,
      "threshold":{"estimate":0.0056,"crossings":[...],"pseudo_thresholds":{"BB72":0.0014,"BB90":0.0021,...}},
      "points":[{"code":"BB72","n":72,"k":12,"d":6,"p":0.0005,"shots":..,"failures":..,"p_logical":..,"p_logical_per_round":..,
                 "lo":..,"hi":..}, ...],     // lo/hi = Wilson 95 % interval of p_logical
      "latency":[{"mode":"batch-amortized"|"single-shot","code":"BB72","percentiles_us":{"P50":..,"P95":..,"P99":..,"P99.9":..,"mean":..,"max":..},
                  "hist":{"edges_us":[...41 log-spaced...],"counts":[...40...]}}] },
    "final_seq": { ...same shape... } },
  "kernel": { "BB72@0.001": {"matrix":[36,72],"lanes":16,"batch_amortized_ns":{"p50":..,"p95":..,"p99":..},
                             "single_shot_bp_ns":{"p50":..,"p95":..,"p99":..},"bp_converged_fraction":..}, ... },
  "notes": ["...honest caveats..."] }
```

## WS /api/v1/stream/decode
Client sends once (and any time to reconfigure) a JSON text frame:
`{"code":"BB72","mode":"code_capacity"|"circuit","p":0.01,"rate_hz":10,"batch":16,"schedule":"interleaved","rounds":3}`
`code_capacity`: iid X and Z data errors with probability p, ideal syndromes, decoded by the C++ FastDecoder (BP, int16 SIMD) with OSD-CS and the INT8 GNN tier as fallback.
`circuit`: full noisy memory experiment sampled with Stim, DEM decoded by BP+OSD (slower; latency in ms range).

Server frames:
```json
{"type":"hello","config":{...},"gnn_available":true,"lanes":16,"codes":["BB72",...]}
{"type":"tick","seq":17,"mode":"code_capacity","shots_in_tick":16,
 "display":{"shot":3,
    "err_x":[qubit,...],"err_z":[qubit,...],                 // injected physical errors
    "z_syndrome":[zcheck ids firing],"x_syndrome":[xcheck ids firing],   // Z checks detect X errors; X checks detect Z errors
    "corr_x":[qubit,...],"corr_z":[qubit,...],                // Pauli corrections
    "tier_x":0|1|2|255,"tier_z":0|1|2|255,                    // 0 BP, 1 OSD-CS, 2 GNN+OSD, 255 failed
    "logical_fail":false,
    "rounds":[{"round":0,"z_syndrome":[...],"x_syndrome":[...]} ...]},   // circuit mode only: active detectors per round
 "latency":{"bp_kernel_ns":{"p50":..,"p95":..,"max":..},      // single-shot BP kernel, C++ timer, this tick
            "amortized_ns":{"p50":..},                         // 16-shot SIMD pass, per shot
            "total_ns":{"p50":..,"p95":..,"max":..},           // BP + OSD/GNN per shot
            "last_total_ns":123},
 "counters":{"shots":..,"bp_converged":..,"osd":..,"gnn":..,"failed_decode":..},
 "ler":{"shots":..,"failures":..,"rate":..,"wilson_lo":..,"wilson_hi":..},
 "elapsed_s": 12.3}
{"type":"error","message":"..."}
```
