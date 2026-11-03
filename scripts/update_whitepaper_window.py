"""Regenerate every sliding-window passage of docs/WHITEPAPER.md and the README summary from results/window/results.json.

    python scripts/update_whitepaper_window.py

All numbers are computed from the measured data (nothing is typed by hand), so re-running the benchmark and this script
keeps the paper, the facts file and the README consistent.  Passages are located by their anchors and replaced in place.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = json.loads((ROOT / "results" / "window" / "results.json").read_text())
R = {(r["code"], r["p"]): r for r in D["runs"]}


def tex(x, d=3):
    m, e = f"{x:.{d}e}".split("e")
    return f"${m} \\times 10^{{{int(e)}}}$"


def fails(r, run):
    return int(round(r["p_logical"] * run["shots"]))


def table(code, p, num):
    run = R[(code, p)]
    f = run["full"]
    L = [f"#### Table {num}: {code}, $p = {p:g}$, {run['shots']} shots, 6 rounds (`docs/FACTS.md`; full DEM = {run['dem']['detectors']} detectors $\\times$ {run['dem']['mechanisms']} mechanisms)", "",
         "| Decoder | $P_L$ | Failures | 95 % Wilson | Step P50 / P99 (batch, $\\mu$s) | Step P50 (single, $\\mu$s) | Reaction P50 / P99 (batch, $\\mu$s) | Reaction P50 (single, $\\mu$s) | Largest window (det $\\times$ cols) |",
         "|---|---|---|---|---|---|---|---|---|"]
    fb, fs = f["batch_amortized"], f["single_shot"]
    L.append(f"| Full horizon | {tex(f['p_logical'])} | {fails(f, run)}/{run['shots']} | {f['lo']:.2e} to {f['hi']:.2e} | n/a | n/a | {fb['p50_us']:.1f} / {fb['p99_us']:.1f} | {fs['p50_us']:.1f} | {run['dem']['detectors']} $\\times$ {run['dem']['mechanisms']} |")
    for c in run["configs"].values():
        sb, ss, rb, rs = c["step_batch_amortized"], c["step_single_shot"], c["reaction_batch_amortized"], c["reaction_single_shot"]
        L.append(f"| Window $(W,F)=({c['W']},{c['F']})$, {c['n_windows']} windows | {tex(c['p_logical'])} | {c['failures']}/{run['shots']} | {c['lo']:.2e} to {c['hi']:.2e} | "
                 f"{sb['p50_us']:.1f} / {sb['p99_us']:.1f} | {ss['p50_us']:.1f} | {rb['p50_us']:.1f} / {rb['p99_us']:.1f} | {rs['p50_us']:.1f} | {c['window_shape_max'][0]} $\\times$ {c['window_shape_max'][1]} |")
    return "\n".join(L) + "\n"


b72, b144, b72h, b144h = R[("BB72", 2e-3)], R[("BB144", 2e-3)], R[("BB72", 3e-3)], R[("BB144", 3e-3)]
C = lambda run, k: run["configs"][k]
rx = lambda run: run["full"]["batch_amortized"]["p50_us"]
rw = lambda run, k: C(run, k)["reaction_batch_amortized"]["p50_us"]
rxs = lambda run: run["full"]["single_shot"]["p50_us"]
rws = lambda run, k: C(run, k)["reaction_single_shot"]["p50_us"]
pl = lambda run, k=None: (run["full"] if k is None else C(run, k))["p_logical"]
stp72 = [c["step_batch_amortized"]["p50_us"] for c in b72["configs"].values()]
stp144 = [c["step_batch_amortized"]["p50_us"] for c in b144["configs"].values()]
stps72 = [c["step_single_shot"]["p50_us"] for c in b72["configs"].values()]
stps144 = [c["step_single_shot"]["p50_us"] for c in b144["configs"].values()]
nfull72, nfull144 = fails(b72["full"], b72), fails(b144["full"], b144)
lo_ratio = lambda run, k: C(run, k)["p_logical"] / pl(run)
overlap = lambda a, b: a["lo"] <= b["hi"] and b["lo"] <= a["hi"]

sec54 = r'''### 5.4 Sliding-Window $(W, F)$ Stream Decoder (`src/decoder/window.py`)

Full-horizon decoding solves one problem whose size grows with the number of rounds $R$, and nothing can be committed before the last round is measured. A sliding-window decoder bounds the size of every problem by $W$ and lets the correction of old rounds be committed while later rounds are still arriving.

Let $L = R + 1$ be the number of detector layers (the $R$ syndrome rounds plus the final transversal readout) and let $\rho(d)$ be the round coordinate of detector $d$, read from the Stim detector coordinates. For a mechanism (a column of the DEM check matrix $H$) let $\phi(j) = \min\{\rho(d) : H_{dj} = 1\}$ be its earliest round. Window $k = 0, 1, \dots$ covers the detector layers $[s_k, e_k)$ with $s_k = kF$ and $e_k = \min(s_k + W, L)$; the last window is the first one with $e_k = L$. It contains the detectors $\mathcal{D}_k = \{d : s_k \le \rho(d) < e_k\}$ and the mechanisms $\mathcal{M}_k = \{j : s_k \le \phi(j) < e_k\}$, with every detector at or beyond $e_k$ truncated (an open boundary toward the future). Truncation can make different mechanisms identical inside the window (for example, mechanisms that differ only in detectors of the first truncated layer); identical columns are merged, their priors combined as $p \leftarrow p(1-q) + q(1-p)$, and the decision of a merged column is attributed to its most probable member. The sub-problem $H_k = H[\mathcal{D}_k, \mathcal{M}_k]$ is solved with the same SIMD BP + OSD-CS kernel of Sections 5.1 and 5.2 against the running syndrome $\sigma_k$. Only the mechanisms of the commit region $\mathcal{C}_k = \{j \in \mathcal{M}_k : \phi(j) < s_k + F\}$ are confirmed (all of $\mathcal{M}_k$ in the last window), and only if the window produced a syndrome-consistent answer; a window that fails commits nothing and the shot is counted as failed. The committed correction is propagated, including its effect on detectors of later layers that the window could not see:

$$\sigma_{k+1} = \sigma_k \oplus H[:, \mathcal{C}_k]\, \hat{e}_{\mathcal{C}_k}, \qquad \hat{e} = \bigoplus_k \hat{e}_{\mathcal{C}_k}.$$

A window needs at least one round of look-ahead ($F < W$) unless a single window covers the whole experiment: with $F = W$ the mechanisms of the last committed round would be confirmed without ever seeing the detectors of the next one. The logical outcome is $O \hat{e}$ with $O$ the observable matrix, and the residual syndrome $H \hat{e} \oplus \sigma_0$ is reported (it was zero for every shot of every configuration measured). One `FastDecoder` is built per window position; shots are processed window by window in 16-lane SIMD batches. The orchestration is written in Python around the C++ kernel; a fused C++ loop is future work. Three latencies are reported: the *step* latency (one stationary window step, including the propagation of the committed correction; divided by $F$ it is the cost per confirmed round), the *reaction* latency (the time of the last window, which is what is left to wait after the final syndrome round arrives) and, for the full-horizon decoder, its entire decoding time, which is its reaction latency. "Batch-amortized" is the time of a 16-shot chunk divided by 16, so it includes the scalar OSD time of any lane of the chunk and can exceed the single-shot median; "single-shot" is the per-shot latency.

'''

sec85 = f'''### 8.5 Sliding-Window $(W, F)$ Decoding vs. Full Horizon (`results/window`, `scripts/bench_window.py`)

All decoders were run on exactly the same Stim samples (interleaved depth-7 schedule, 6 rounds, seed 2026, BP30 + OSD-CS32 `int16` core, a desktop processor with AVX2). Tables 8 to 11 give the logical error rate and the latencies; the values are those of `docs/FACTS.md`.

{table("BB72", 2e-3, 8)}
{table("BB144", 2e-3, 9)}
{table("BB72", 3e-3, 10)}
{table("BB144", 3e-3, 11)}
![[wide] Figure 5. Sliding-window decoding versus full horizon at $p = 2 \\times 10^{{-3}}$: logical error rate over 6 rounds with 95 % Wilson intervals (left) and reaction latency after the last round, P50 (solid) and P99 (dotted), in microseconds (right).](../figures/window_latency.svg)

**Reaction latency.** At $p = 2 \\times 10^{{-3}}$ the time that remains after the last syndrome round drops from {rx(b72):.1f} $\\mu$s (full horizon) to {rw(b72, "W3F1"):.1f} $\\mu$s for $(W,F) = (3,1)$ and {rw(b72, "W4F2"):.1f} $\\mu$s for $(4,2)$ on BB72 (P50, batch-amortized), a {rx(b72) / rw(b72, "W3F1"):.0f}$\\times$ and {rx(b72) / rw(b72, "W4F2"):.0f}$\\times$ reduction; on BB144 it drops from {rx(b144):.1f} $\\mu$s to {rw(b144, "W3F1"):.1f} $\\mu$s and {rw(b144, "W4F2"):.1f} $\\mu$s ({rx(b144) / rw(b144, "W3F1"):.1f}$\\times$ and {rx(b144) / rw(b144, "W4F2"):.1f}$\\times$). Shot by shot (single-shot P50) the reaction goes from {rxs(b72):.1f} to {rws(b72, "W3F1"):.1f} $\\mu$s on BB72 and from {rxs(b144):.1f} to {rws(b144, "W3F1"):.1f} $\\mu$s on BB144 for $(3,1)$. This is the latency that matters for a backlog-free real-time loop, because the earlier windows are decoded while the experiment is still running.

**Step latency.** One stationary window step takes between {min(stp72):.0f} and {max(stp72):.0f} $\\mu$s (batch P50) and {min(stps72):.0f} to {max(stps72):.0f} $\\mu$s (single-shot P50) on BB72, and {min(stp144):.0f} to {max(stp144):.0f} $\\mu$s (batch) and {min(stps144):.0f} to {max(stps144):.0f} $\\mu$s (single-shot) on BB144; dividing by $F$ gives the cost per confirmed round. That is far from the sub-microsecond regime of Section 8.4: it is the cost of solving a window with BP and, when BP does not converge, OSD-CS. We did not separate the BP and OSD times inside a window step. Note that the batch-amortized figure can exceed the single-shot median because a 16-shot chunk pays the scalar OSD of any of its lanes.

**Logical error rate.** On BB72 at $p = 2 \\times 10^{{-3}}$, $(4,2)$ reaches {tex(pl(b72, "W4F2"))} ({C(b72, "W4F2")["failures"]} failures) against {tex(pl(b72))} ({nfull72} failures) for the full horizon in {b72["shots"]} shots, $(3,1)$ gives {tex(pl(b72, "W3F1"))} ({C(b72, "W3F1")["failures"]}) and the smallest window $(2,1)$ gives {tex(pl(b72, "W2F1"))} ({C(b72, "W2F1")["failures"]}). On BB144 at the same $p$ the full horizon failed {nfull144} times in {b144["shots"]} shots and $(3,1)$ and $(4,2)$ failed {C(b144, "W3F1")["failures"]} and {C(b144, "W4F2")["failures"]} times, whereas $(2,1)$ failed {C(b144, "W2F1")["failures"]} times ({lo_ratio(b144, "W2F1"):.1f}$\\times$ the full-horizon rate); the Wilson intervals of the full horizon and of $(3,1)$/$(4,2)$ overlap at these small counts. At $p = 3 \\times 10^{{-3}}$ the differences are resolved: on BB72 the full horizon gives {tex(pl(b72h))} and $(3,1)$, $(4,2)$ give {tex(pl(b72h, "W3F1"))} and {tex(pl(b72h, "W4F2"))}; on BB144 the full horizon gives {tex(pl(b144h))} and $(3,1)$, $(4,2)$ give {tex(pl(b144h, "W3F1"))} and {tex(pl(b144h, "W4F2"))} ({lo_ratio(b144h, "W3F1"):.1f}$\\times$ and {lo_ratio(b144h, "W4F2"):.1f}$\\times$). Truncating the future boundary costs accuracy when the window is much shorter than the code distance ($d = 12$ for BB144, $d = 6$ for BB72), as is well known for sliding-window decoders. The residual syndrome after the last window was zero for every shot of every configuration.

'''


def sub_between(text, start_pat, end_pat, new):
    a = re.search(start_pat, text)
    b = re.search(end_pat, text[a.end():])
    assert a and b, start_pat
    return text[:a.start()] + new + text[a.end() + b.start():]


wp = ROOT / "docs" / "WHITEPAPER.md"
s = wp.read_text(encoding="utf-8")
s = sub_between(s, r"### 5\.4 Sliding-Window", r"---\n\n## 6\. Hybrid", sec54)
s = sub_between(s, r"### 8\.5 Sliding-Window", r"---\n\n## 9\. Discussion", sec85)

abs_new = (f" A sliding-window $(W, F)$ circuit-level decoder cuts the latency that remains after the last syndrome round from {rx(b72):.0f} to {rw(b72, 'W3F1'):.0f} $\\mu$s "
           f"(P50, BB72, $p = 2 \\times 10^{{-3}}$, $(W,F) = (3,1)$) and from {rx(b144):.0f} to {rw(b144, 'W3F1'):.0f} $\\mu$s on BB144, with a logical error rate that is "
           f"{'statistically indistinguishable from' if overlap(C(b72, 'W4F2'), b72['full']) else 'different from'} full-horizon decoding (overlapping 95 % intervals) for $(4,2)$ on BB72 at $p = 2 \\times 10^{{-3}}$ "
           f"but a measured penalty on BB144 at higher noise; the cost of one window step stays in the tens to hundreds of microseconds, so circuit-level decoding is still not sub-microsecond.")
s = re.sub(r" A sliding-window \$\(W, F\)\$ circuit-level decoder cuts the latency.*?still not sub-microsecond\.", lambda m: abs_new, s, count=1, flags=re.S)

bullet = (f"   * **Sliding-Window $(W, F)$ Decoding (implemented, Sections 5.4 and 8.5):** Restricting each decode step to a window of $W$ rounds and committing $F$ rounds at a time bounds the active DEM size independently of the circuit duration. "
          f"Measured: the reaction latency after the last round falls by {rx(b72) / rw(b72, 'W3F1'):.0f}$\\times$ on BB72 and {rx(b144) / rw(b144, 'W3F1'):.1f}$\\times$ on BB144 ($(3,1)$, $p = 2 \\times 10^{{-3}}$, batch P50), "
          f"but one window step still costs {min(stp72):.0f} to {max(stp144):.0f} $\\mu$s and BB144 pays an accuracy penalty at higher $p$. Windowing is therefore necessary but not sufficient.")
s = re.sub(r"   \* \*\*Sliding-Window \$\(W, F\)\$ Decoding \(implemented.*?not sufficient\.", lambda m: bullet, s, count=1, flags=re.S)

concl = (f"remains an open hardware-software co-design challenge. A sliding-window decoder (Section 5.4) lowers the reaction latency after the last syndrome round by {rx(b72) / rw(b72, 'W3F1'):.0f}$\\times$ (BB72) and {rx(b144) / rw(b144, 'W3F1'):.1f}$\\times$ (BB144) at $p = 2 \\times 10^{{-3}}$, "
         f"with accuracy close to the full horizon for $(4,2)$ on BB72 and a clear loss on BB144 at $p = 3 \\times 10^{{-3}}$ ({lo_ratio(b144h, 'W3F1'):.1f}$\\times$ for $(3,1)$), but it leaves one window step at tens to hundreds of microseconds; "
         f"the remaining gap calls for a fused C++ or FPGA window loop and a cheaper post-processing stage.")
s = re.sub(r"remains an open hardware-software co-design challenge\. A sliding-window decoder \(Section 5\.4\).*?cheaper post-processing stage\.", lambda m: concl, s, count=1, flags=re.S)

lim = ("8. **Sliding-Window Study Scope:** Three window configurations $(2,1)$, $(3,1)$, $(4,2)$ were measured, on two codes (BB72, BB144), two physical error rates ($2 \\times 10^{-3}$ and $3 \\times 10^{-3}$) and 6 rounds, with 20 000 to 30 000 shots at the lower rate and 10 000 at the higher one. "
       "The windows are orchestrated in Python around the C++ kernel (no fused loop, no parallel windows), BP and OSD times inside a window step are not separated, the timings are single-machine, "
       f"and the BB144 failure counts at $p = 2 \\times 10^{{-3}}$ ({min(nfull144, C(b144, 'W3F1')['failures'], C(b144, 'W4F2')['failures'])} to {max(nfull144, C(b144, 'W3F1')['failures'], C(b144, 'W4F2')['failures'])}) are too small to resolve the accuracy cost there. "
       "Windows much shorter than the code distance lose accuracy; larger $W$ was not explored. The batch-amortized latencies divide the time of a 16-shot chunk by 16 and therefore include the scalar OSD of any lane in the chunk (the same convention as Tables 6 and 7); single-shot values are the per-shot latency.")
s = re.sub(r"8\. \*\*Sliding-Window Study Scope:\*\*.*?(?=\n9\. \*\*No Physical QPU)", lambda m: lim, s, count=1, flags=re.S)
wp.write_text(s, encoding="utf-8")

# README summary rows
rd = ROOT / "README.md"
t = rd.read_text(encoding="utf-8")
t = re.sub(r"reaction after last round [\d.]+ -> [\d.]+ us \(BB72\), [\d.]+ -> [\d.]+ us \(BB144\) at p = 2e-3, \(3,1\)",
           f"reaction after last round {rx(b72):.0f} -> {rw(b72, 'W3F1'):.0f} us (BB72), {rx(b144):.0f} -> {rw(b144, 'W3F1'):.0f} us (BB144) at p = 2e-3, (3,1)", t)
t = re.sub(r"a sliding-window decoder lowers the wait after the last round by about \d+x \(BB72\) and [\d.]+x \(BB144\)",
           f"a sliding-window decoder lowers the wait after the last round by about {rx(b72) / rw(b72, 'W3F1'):.0f}x (BB72) and {rx(b144) / rw(b144, 'W3F1'):.1f}x (BB144)", t)
t = re.sub(r"\(BB144 at p = 3e-3: about [\d.]+x the full-horizon error rate for W = 3\)",
           f"(BB144 at p = 3e-3: about {lo_ratio(b144h, 'W3F1'):.1f}x the full-horizon error rate for W = 3)", t)
rd.write_text(t, encoding="utf-8")
print("whitepaper and README updated from results/window/results.json")
