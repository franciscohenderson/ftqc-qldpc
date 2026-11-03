# FACTS (auto-generated from repository data; the only numbers documents may cite)

Machine: desktop processor (6C/12T) with AVX2, no AVX-512, Windows 11, Python 3.14, clang via zig c++ (C++20), 16 x int16 SIMD lanes.

## Code parameters (exact MILP via HiGHS unless stated)

| Code | [[n,k,d]] | d_X | d_Z | exact | w_c / w_r | checks (X+Z) |
|---|---|---|---|---|---|---|
| BB72 | [[72,12,6]] | 6 | 6 | yes | 3 / 6 | 72 |
| BB90 | [[90,8,10]] | 10 | 10 | yes | 3 / 6 | 90 |
| BB108 | [[108,8,10]] | 10 | 10 | yes | 3 / 6 | 108 |
| BB144 | [[144,12,12]] | 12 | 12 | yes | 3 / 6 | 144 |
| BB288 | [[288,12,<=18]] | 4-18 | 4-18 | bounds only | 3 / 6 | 288 |

ILP seconds: BB72 0.8 s, BB90 37.1 s, BB108 57.3 s, BB144 280.3 s, BB288 600.6 s

## Monte Carlo: final_int (interleaved depth-7 schedule)

rounds = [np.int64(6)], decoder = ['bp30-osdcs32-int16'], total shots = 600000

Total logical error rate over the memory experiment (0 = none observed in the shot budget):

| p | BB108 | BB144 | BB72 | BB90 |
|---|---|---|---|---|
| 5.0e-04 | 0.000e+00 | 0.000e+00 | 1.000e-04 | 0.000e+00 |
| 1.0e-03 | 0.000e+00 | 0.000e+00 | 3.000e-04 | 3.333e-05 |
| 1.5e-03 | 1.333e-04 | 0.000e+00 | 1.933e-03 | 2.667e-04 |
| 2.0e-03 | 4.333e-04 | 2.667e-04 | 6.071e-03 | 1.433e-03 |
| 3.0e-03 | 7.250e-03 | 6.000e-03 | 3.791e-02 | 1.436e-02 |
| 4.0e-03 | 4.791e-02 | 4.373e-02 | 1.203e-01 | 6.655e-02 |
| 6.0e-03 | 3.819e-01 | 4.122e-01 | 4.662e-01 | 3.945e-01 |

Failures observed:

| p | BB108 | BB144 | BB72 | BB90 |
|---|---|---|---|---|
| 5.0e-04 | 0 | 0 | 3 | 0 |
| 1.0e-03 | 0 | 0 | 9 | 1 |
| 1.5e-03 | 4 | 0 | 58 | 8 |
| 2.0e-03 | 13 | 8 | 85 | 43 |
| 3.0e-03 | 87 | 84 | 417 | 158 |
| 4.0e-03 | 527 | 481 | 1323 | 732 |
| 6.0e-03 | 4201 | 4534 | 5128 | 4340 |

Shots used:

| p | BB108 | BB144 | BB72 | BB90 |
|---|---|---|---|---|
| 5.0e-04 | 30000 | 30000 | 30000 | 30000 |
| 1.0e-03 | 30000 | 30000 | 30000 | 30000 |
| 1.5e-03 | 30000 | 30000 | 30000 | 30000 |
| 2.0e-03 | 30000 | 30000 | 14000 | 30000 |
| 3.0e-03 | 12000 | 14000 | 11000 | 11000 |
| 4.0e-03 | 11000 | 11000 | 11000 | 11000 |
| 6.0e-03 | 11000 | 11000 | 11000 | 11000 |

Threshold estimate (crossing of consecutive-distance curves): 5.5731e-03; crossings: [0.0055731]; pseudo-thresholds: {'BB108': 0.002586, 'BB144': 0.002704, 'BB72': 0.001398, 'BB90': 0.002147}

Full circuit-level decode latency (BP + OSD-CS), pooled over all p:

| mode | code | P50_us | P95_us | P99_us | P99.9_us | max_us |
|---|---|---|---|---|---|---|
| batch-amortized | BB108 | 289.5 | 2015.9 | 2536.9 | 3036.9 | 5895.6 |
| batch-amortized | BB144 | 539.9 | 3498.7 | 4145.9 | 4605.6 | 5038.0 |
| batch-amortized | BB72 | 80.6 | 870.8 | 1129.5 | 1461.8 | 1715.1 |
| batch-amortized | BB90 | 210.3 | 1365.1 | 1736.5 | 2060.3 | 2466.4 |
| single-shot | BB108 | 216.7 | 4748.6 | 6983.3 | 15535.9 | 40041.4 |
| single-shot | BB144 | 352.5 | 7610.1 | 10474.9 | 15847.6 | 31185.2 |
| single-shot | BB72 | 99.2 | 2434.8 | 3297.7 | 10806.6 | 51071.7 |
| single-shot | BB90 | 158.0 | 3511.7 | 4912.5 | 9209.3 | 21074.3 |

## Monte Carlo: final_seq (sequential depth-12 schedule)

rounds = [np.int64(6)], decoder = ['bp30-osdcs32-int16'], total shots = 460000

Total logical error rate over the memory experiment (0 = none observed in the shot budget):

| p | BB108 | BB144 | BB72 | BB90 |
|---|---|---|---|---|
| 5.0e-04 | 0.000e+00 | 0.000e+00 | 7.000e-04 | 0.000e+00 |
| 1.0e-03 | 8.000e-04 | 3.667e-04 | 6.308e-03 | 8.333e-04 |
| 1.5e-03 | 5.714e-03 | 3.478e-03 | 2.409e-02 | 6.538e-03 |
| 2.0e-03 | 2.145e-02 | 1.745e-02 | 7.473e-02 | 2.600e-02 |
| 3.0e-03 | 1.685e-01 | 1.635e-01 | 2.623e-01 | 1.665e-01 |
| 4.0e-03 | 4.629e-01 | 5.232e-01 | 5.444e-01 | 4.414e-01 |
| 6.0e-03 | 9.402e-01 | 9.715e-01 | 9.265e-01 | 9.105e-01 |

Failures observed:

| p | BB108 | BB144 | BB72 | BB90 |
|---|---|---|---|---|
| 5.0e-04 | 0 | 0 | 21 | 0 |
| 1.0e-03 | 24 | 11 | 82 | 25 |
| 1.5e-03 | 80 | 80 | 265 | 85 |
| 2.0e-03 | 236 | 192 | 822 | 286 |
| 3.0e-03 | 1854 | 1799 | 2885 | 1832 |
| 4.0e-03 | 5092 | 5755 | 5988 | 4855 |
| 6.0e-03 | 10342 | 10687 | 10191 | 10015 |

Shots used:

| p | BB108 | BB144 | BB72 | BB90 |
|---|---|---|---|---|
| 5.0e-04 | 30000 | 30000 | 30000 | 30000 |
| 1.0e-03 | 30000 | 30000 | 13000 | 30000 |
| 1.5e-03 | 14000 | 23000 | 11000 | 13000 |
| 2.0e-03 | 11000 | 11000 | 11000 | 11000 |
| 3.0e-03 | 11000 | 11000 | 11000 | 11000 |
| 4.0e-03 | 11000 | 11000 | 11000 | 11000 |
| 6.0e-03 | 11000 | 11000 | 11000 | 11000 |

Threshold estimate (crossing of consecutive-distance curves): 3.1251e-03; crossings: [0.0031251]; pseudo-thresholds: {'BB108': 0.00106, 'BB144': 0.001247, 'BB72': None, 'BB90': 0.001046}

Full circuit-level decode latency (BP + OSD-CS), pooled over all p:

| mode | code | P50_us | P95_us | P99_us | P99.9_us | max_us |
|---|---|---|---|---|---|---|
| batch-amortized | BB108 | 818.9 | 2940.6 | 3343.8 | 6303.3 | 8979.5 |
| batch-amortized | BB144 | 1469.7 | 5685.2 | 9422.3 | 34945.0 | 51320.9 |
| batch-amortized | BB72 | 381.7 | 1411.4 | 1666.8 | 3202.6 | 8906.3 |
| batch-amortized | BB90 | 423.8 | 1999.9 | 2281.5 | 5196.6 | 12943.7 |
| single-shot | BB108 | 873.9 | 6323.4 | 9817.4 | 18053.6 | 36035.8 |
| single-shot | BB144 | 1650.3 | 10611.9 | 15153.4 | 28296.9 | 103423.7 |
| single-shot | BB72 | 377.9 | 3119.1 | 5861.3 | 14520.0 | 82925.0 |
| single-shot | BB90 | 342.8 | 4135.5 | 7773.8 | 25241.6 | 233140.2 |

## BP kernel latency (code-capacity H_Z: 36x72 for BB72 and 72x144 for BB144, measured by scripts/bench_kernel.py)

```json
{
  "BB72@0.001": {
    "matrix": [
      36,
      72
    ],
    "lanes": 16,
    "shots": 4000,
    "batch_amortized_ns": {
      "p50": 268.0,
      "p95": 312.0,
      "p99": 418.0
    },
    "single_shot_bp_ns": {
      "p50": 600.0,
      "p95": 1200.0,
      "p99": 1300.0
    },
    "bp_converged_fraction": 1.0
  },
  "BB72@0.003": {
    "matrix": [
      36,
      72
    ],
    "lanes": 16,
    "shots": 4000,
    "batch_amortized_ns": {
      "p50": 268.0,
      "p95": 312.0,
      "p99": 312.0
    },
    "single_shot_bp_ns": {
      "p50": 600.0,
      "p95": 1300.0,
      "p99": 1300.0
    },
    "bp_converged_fraction": 1.0
  },
  "BB72@0.01": {
    "matrix": [
      36,
      72
    ],
    "lanes": 16,
    "shots": 4000,
    "batch_amortized_ns": {
      "p50": 356.0,
      "p95": 556.0,
      "p99": 956.0
    },
    "single_shot_bp_ns": {
      "p50": 1200.0,
      "p95": 2300.0,
      "p99": 3500.0
    },
    "bp_converged_fraction": 0.99875
  },
  "BB144@0.001": {
    "matrix": [
      72,
      144
    ],
    "lanes": 16,
    "shots": 4000,
    "batch_amortized_ns": {
      "p50": 556.0,
      "p95": 606.0,
      "p99": 743.0
    },
    "single_shot_bp_ns": {
      "p50": 1200.0,
      "p95": 2700.0,
      "p99": 2800.0
    },
    "bp_converged_fraction": 1.0
  },
  "BB144@0.003": {
    "matrix": [
      72,
      144
    ],
    "lanes": 16,
    "shots": 4000,
    "batch_amortized_ns": {
      "p50": 537.0,
      "p95": 718.0,
      "p99": 737.0
    },
    "single_shot_bp_ns": {
      "p50": 1200.0,
      "p95": 2500.0,
      "p99": 3700.0
    },
    "bp_converged_fraction": 1.0
  },
  "BB144@0.01": {
    "matrix": [
      72,
      144
    ],
    "lanes": 16,
    "shots": 4000,
    "batch_amortized_ns": {
      "p50": 618.0,
      "p95": 850.0,
      "p99": 1237.0
    },
    "single_shot_bp_ns": {
      "p50": 2400.0,
      "p95": 4200.0,
      "p99": 6100.0
    },
    "bp_converged_fraction": 0.99925
  }
}
```

## Sliding-window (W, F) decoding vs full-horizon decoding (scripts/bench_window.py)

Same Stim samples for every decoder; interleaved depth 7; core bp30-osdcs32-int16; desktop processor with AVX2, 16 x int16 lanes. 6 syndrome rounds (7 detector layers). Latencies are measured inside the C++ kernel. "step" = one stationary window step (every window but the last; includes propagating the committed correction); "per round" = step / F; "reaction" = time of the LAST window (what remains to wait after the final syndrome round arrives); for the full-horizon decoder the reaction latency is its whole decode time. Batch = chunk time / 16 shots, so it includes the scalar OSD time of any lane of the chunk (a throughput view); single = one shot at a time (first 400 shots), the per-shot latency. Identical columns created by truncating the future boundary are merged before decoding; windows need F < W.

### BB72 at p = 0.002 (20000 shots, seed 2026; full DEM = 252 detectors x 2232 mechanisms)

| decoder | P_L (6 rounds) | failures/shots | 95 % Wilson | step P50/P95/P99 batch (us) | step P50 single-shot (us) | per-round P50 batch (us) | reaction P50/P95/P99 batch (us) | reaction P50/P95/P99 single-shot (us) | largest window (det x cols; raw cols) |
|---|---|---|---|---|---|---|---|---|---|
| full horizon | 5.950e-03 | 119/20000 | 4.97e-03 - 7.11e-03 | n/a | n/a | n/a | 79.5 / 164.0 / 232.1 | 84.5 / 1181.2 / 1263.5 | 252 x 2232 |
| window (W=2, F=1), 6 windows | 1.145e-02 | 229/20000 | 1.01e-02 - 1.30e-02 | 26.6 / 39.4 / 49.1 | 12.7 | 26.6 | 1.8 / 13.0 / 18.2 | 5.7 / 20.1 / 195.8 | 72 x 648 (raw 720) |
| window (W=3, F=1), 5 windows | 6.550e-03 | 131/20000 | 5.52e-03 - 7.77e-03 | 45.4 / 69.8 / 89.4 | 28.2 | 45.4 | 5.2 / 35.7 / 45.4 | 13.2 / 105.3 / 386.9 | 108 x 1008 (raw 1080) |
| window (W=4, F=2), 3 windows | 6.350e-03 | 127/20000 | 5.34e-03 - 7.55e-03 | 71.4 / 109.7 / 155.9 | 40.0 | 35.7 | 4.8 / 34.4 / 44.1 | 10.2 / 75.7 / 368.3 | 144 x 1368 (raw 1440) |

Shots left with a nonzero residual syndrome / shots where some window failed: W2F1: 0 / 0, W3F1: 0 / 0, W4F2: 0 / 0.

### BB72 at p = 0.003 (10000 shots, seed 2026; full DEM = 252 detectors x 2232 mechanisms)

| decoder | P_L (6 rounds) | failures/shots | 95 % Wilson | step P50/P95/P99 batch (us) | step P50 single-shot (us) | per-round P50 batch (us) | reaction P50/P95/P99 batch (us) | reaction P50/P95/P99 single-shot (us) | largest window (det x cols; raw cols) |
|---|---|---|---|---|---|---|---|---|---|
| full horizon | 3.720e-02 | 372/10000 | 3.37e-02 - 4.11e-02 | n/a | n/a | n/a | 149.5 / 266.3 / 354.4 | 121.2 / 1225.2 / 1440.9 | 252 x 2232 |
| window (W=2, F=1), 6 windows | 5.210e-02 | 521/10000 | 4.79e-02 - 5.66e-02 | 40.5 / 55.9 / 65.7 | 22.6 | 40.5 | 3.2 / 17.9 / 18.4 | 5.7 / 27.2 / 194.0 | 72 x 648 (raw 720) |
| window (W=3, F=1), 5 windows | 3.620e-02 | 362/10000 | 3.27e-02 - 4.00e-02 | 71.9 / 103.8 / 133.0 | 51.7 | 71.9 | 23.9 / 44.6 / 54.9 | 20.0 / 369.2 / 392.2 | 108 x 1008 (raw 1080) |
| window (W=4, F=2), 3 windows | 3.810e-02 | 381/10000 | 3.45e-02 - 4.20e-02 | 117.8 / 161.0 / 198.4 | 82.2 | 58.9 | 23.7 / 44.2 / 54.9 | 17.0 / 367.0 / 378.3 | 144 x 1368 (raw 1440) |

Shots left with a nonzero residual syndrome / shots where some window failed: W2F1: 0 / 0, W3F1: 0 / 0, W4F2: 0 / 0.

### BB144 at p = 0.002 (30000 shots, seed 2026; full DEM = 504 detectors x 4464 mechanisms)

| decoder | P_L (6 rounds) | failures/shots | 95 % Wilson | step P50/P95/P99 batch (us) | step P50 single-shot (us) | per-round P50 batch (us) | reaction P50/P95/P99 batch (us) | reaction P50/P95/P99 single-shot (us) | largest window (det x cols; raw cols) |
|---|---|---|---|---|---|---|---|---|---|
| full horizon | 3.667e-04 | 11/30000 | 2.05e-04 - 6.57e-04 | n/a | n/a | n/a | 344.2 / 703.3 / 928.9 | 236.8 / 3382.3 / 3699.6 | 504 x 4464 |
| window (W=2, F=1), 6 windows | 3.067e-03 | 92/30000 | 2.50e-03 - 3.76e-03 | 75.8 / 110.1 / 139.5 | 45.5 | 75.8 | 4.9 / 39.4 / 51.5 | 11.5 / 47.6 / 422.7 | 144 x 1296 (raw 1440) |
| window (W=3, F=1), 5 windows | 7.000e-04 | 21/30000 | 4.58e-04 - 1.07e-03 | 146.6 / 222.2 / 321.7 | 91.0 | 146.6 | 52.4 / 86.9 / 119.3 | 33.7 / 284.3 / 853.1 | 216 x 2016 (raw 2160) |
| window (W=4, F=2), 3 windows | 7.000e-04 | 21/30000 | 4.58e-04 - 1.07e-03 | 228.6 / 341.1 / 458.2 | 147.9 | 114.3 | 53.9 / 100.1 / 133.3 | 34.6 / 188.2 / 878.5 | 288 x 2736 (raw 2880) |

Shots left with a nonzero residual syndrome / shots where some window failed: W2F1: 0 / 0, W3F1: 0 / 0, W4F2: 0 / 0.

### BB144 at p = 0.003 (10000 shots, seed 2026; full DEM = 504 detectors x 4464 mechanisms)

| decoder | P_L (6 rounds) | failures/shots | 95 % Wilson | step P50/P95/P99 batch (us) | step P50 single-shot (us) | per-round P50 batch (us) | reaction P50/P95/P99 batch (us) | reaction P50/P95/P99 single-shot (us) | largest window (det x cols; raw cols) |
|---|---|---|---|---|---|---|---|---|---|
| full horizon | 6.500e-03 | 65/10000 | 5.10e-03 - 8.28e-03 | n/a | n/a | n/a | 677.5 / 1132.5 / 1409.7 | 443.1 / 3352.2 / 3827.1 | 504 x 4464 |
| window (W=2, F=1), 6 windows | 3.350e-02 | 335/10000 | 3.01e-02 - 3.72e-02 | 126.0 / 162.1 / 215.7 | 442.9 | 126.0 | 27.6 / 53.7 / 67.6 | 18.5 / 413.5 / 426.5 | 144 x 1296 (raw 1440) |
| window (W=3, F=1), 5 windows | 1.210e-02 | 121/10000 | 1.01e-02 - 1.44e-02 | 251.0 / 327.2 / 492.0 | 796.2 | 251.0 | 77.7 / 131.6 / 176.5 | 60.1 / 825.9 / 858.6 | 216 x 2016 (raw 2160) |
| window (W=4, F=2), 3 windows | 1.080e-02 | 108/10000 | 8.95e-03 - 1.30e-02 | 396.0 / 501.9 / 691.1 | 1159.5 | 198.0 | 78.1 / 131.9 / 178.5 | 61.1 / 828.2 / 855.9 | 288 x 2736 (raw 2880) |

Shots left with a nonzero residual syndrome / shots where some window failed: W2F1: 0 / 0, W3F1: 0 / 0, W4F2: 0 / 0.


## Circuit-level distance d_circ (scripts/run_circuit_distance.py; MILP on the detector error model, HiGHS)

d_circ = fewest DEM mechanisms with zero detector syndrome that flip a logical observable (Z-basis memory, uniform noise 1e-3, identical symptoms merged). `certified` means HiGHS proved lower = upper within the time limit; otherwise the row is a bound, not a value. `max weight asked` marks a decision problem (is there an undetected logical fault of weight <= W?).

| code | layers | rounds | max weight asked | d_circ lower | d_circ upper | certified | seconds | time limit s |
|---|---|---|---|---|---|---|---|---|
| BB108 | 12 | 6 | - | 2 | 14 | False | 300 | 300.0 |
| BB108 | 7 | 6 | - | 2 | 132 | False | 300 | 300.0 |
| BB144 | 12 | 6 | - | 2 | 32 | False | 300 | 300.0 |
| BB144 | 7 | 6 | - | 2 | 14 | False | 301 | 300.0 |
| BB72 | 12 | 6 | - | 4 | 4 | True | 233 | 900.0 |
| BB72 | 7 | 6 | - | 3 | 5 | False | 900 | 900.0 |
| BB72 | 7 | 6 | 4 | 5 | - | False | 3903 | 300 |
| BB90 | 12 | 6 | - | 2 | 9 | False | 300 | 300.0 |
| BB90 | 7 | 6 | - | 2 | 12 | False | 300 | 300.0 |

## Zero-cost logical gates from translation automorphisms (scripts/run_logical_gates.py)

| code | [[n,k]] | translations | automorphisms | non-trivial logical | distinct logical matrices | Stim predictions | Stim mismatches | gates inserted in the depth-7 circuit | violations | gauge detectors |
|---|---|---|---|---|---|---|---|---|---|---|
| BB72 | [[72,12]] | 35 | 35 | 35 | 35 | 7560 | 0 | 35 | 0 | 0 |
| BB90 | [[90,8]] | 44 | 44 | 40 | 8 | 896 | 0 | 44 | 0 | 0 |
| BB108 | [[108,8]] | 53 | 53 | 48 | 8 | 896 | 0 | 53 | 0 | 0 |
| BB144 | [[144,12]] | 71 | 71 | 70 | 35 | 7560 | 0 | 71 | 0 | 0 |

## Fused C++ window loop vs Python loop (scripts/run_window_fused.py)

Same Stim samples (seed 2026, 6 rounds, p = 2e-3), best of 3 wall-clock runs per loop. `prop` = sub-syndrome gather + commit + propagate. Single-shot step = median per-window time of one-shot calls (BP-bound, not accelerated by fusion).

| code | (W,F) | shots | identical decisions | logical failures | Python us/shot | fused us/shot | reduction | BP us | OSD us | prop us | step single p50 Python us | step single p50 fused us |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BB72 | (3,1) | 1024 | True | 4 | 250 | 180 | 28 % | 15.8 | 20.2 | 0.6 | 27.7 | 28.9 |
| BB72 | (4,2) | 1024 | True | 5 | 219 | 153 | 30 % | 22.9 | 35.5 | 1.1 | 39.0 | 46.6 |
| BB144 | (3,1) | 1024 | True | 1 | 838 | 606 | 28 % | 34.4 | 94.1 | 1.4 | 88.0 | 90.5 |
| BB144 | (4,2) | 1024 | True | 0 | 665 | 495 | 26 % | 49.7 | 152.2 | 2.1 | 138.7 | 149.8 |

## BP kernel as SystemVerilog, executed and compared with qdecoder.cpp (scripts/run_rtl.py)

* BB72 H_Z 36x72 (216 edges), int16 saturating normalized min-sum alpha=0.75, max 30 iterations, channel LLR 62, 3390 lines of SystemVerilog-2012 (`exports/rtl/`).
* 1000 noisy syndromes (iid X errors p=0.04, seed 2026): 953 converge, 47 reach the iteration limit.
* RTL (Yosys -> CXXRTL -> zig c++, executed clock edge by clock edge) vs C++: posterior LLRs True, hard decisions True, iterations True, converged flag True; fixed-point model vs C++ True.
* Cycles = 2*iterations + 2 (holds on every vector: True): min 2, p50 6, p95 42, max 62, mean 9.60; iterations mean 3.80, max 30.
* Nominal-clock conversion of those RTL cycles (the clocks are assumptions, not achieved frequencies; no FPGA flow was run): 250MHz: p50 0.024 us, p95 0.168 us, max 0.248 us; 400MHz: p50 0.015 us, p95 0.105 us, max 0.155 us.
* For context only (a different quantity): CPU BP single-shot on the same syndromes, p50 1.3 us, p95 8.9 us.
* Size: 4665 register bits (exact); 166895 generic 2-input gates from per-block Yosys synthesis (no ABC, no LUT mapping: an upper bound, not an FPGA report); deepest block 163 gates.
* The generated testbench was executed: False (no Verilog simulator is available on this machine).

## Formal proof of the generated RTL blocks (scripts/run_formal.py; z3 on SMT-LIB exported by Yosys)

* Scope: check node (per degree), variable node (per degree and channel LLR) and stop test, as combinational functions, for every 16-bit input; range lemmas for the check node; NOT the register/FSM sequencing or the wiring between blocks (covered by executing the whole RTL).
* All blocks proved: True. Check node degree 6: equivalent to the C++ semantics for every 16-bit input (193 input bits), 6.7 s of solver time over 3 stages; range lemmas hold (with |tot| <= 8000 and |r| <= 24576: q = tot - r never saturates, q is never -32768 (so |q| never wraps), and every output has |r_new| <= 24576).
* Variable nodes: degree 3, channel LLR 62: equivalent True, posterior always within +-8000: True.
* Stop test (36 checks, 72 variables): equivalent True.
* Non-vacuity: tests/test_formal.py plants 7 bugs in the RTL (tie-break, alpha, second minimum, syndrome sign, clamp, unsaturated add, dropped stop-test term); the proof finds a counterexample for every one.


## Other measured facts

* Interleaved schedule depth 7 vs sequential depth 12 CNOT layers per round (0 ordering violations, Stim detectors deterministic with noise off).
* CNOTs per round = |H_X| + |H_Z| (BB72: 432, BB144: 864).
* BB288 also admits the depth-7 interleaved schedule (found in 23 s, 0 ordering violations, Stim detectors deterministic with noise off over 2 rounds); BB288 was not included in the Monte Carlo sweeps.
* Audit by agy found OSD-CS degrading to OSD-0 once the OSD basis was full (the column scan stopped at rank = m). Effect on the interleaved 6-round total logical error rate, sweep before the fix (results/int6) vs after (results/final_int): BB72 @ p=2e-03: 0.83 % (75/9000) -> 0.61 % (85/14000); BB90 @ p=3e-03: 1.56 % (140/9000) -> 1.44 % (158/11000); BB108 @ p=3e-03: 0.80 % (72/9000) -> 0.73 % (87/12000); BB144 @ p=3e-03: 0.70 % (63/9000) -> 0.60 % (84/14000). The improvement is modest (roughly 8-27 % fewer failures) and of the order of the sampling error at these shot counts; a separate 3000-shot spot check (10 failures) is not used because it is not consistent with the sweep. All final sweeps were re-run after the fix.
* GNN (INT8) tier: BB72 H_Z code-capacity p=0.06, BP10, 2000 shots: INT8/fp32 sign agreement 99.84 %; logical failure 23.65 % (BP+OSD) -> 22.65 % (with GNN-ordered OSD); 24 shots resolved by tier 2.
* ILP formulation: min sum z s.t. H z - 2u = 0, L z - 2v = y, sum y >= 1 (z, y binary; u, v integer).
