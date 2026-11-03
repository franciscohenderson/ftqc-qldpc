"""Stim circuit synthesis: circuit-level memory experiments for CSS qLDPC codes.

Qubit layout: data 0..n-1, X-ancillas n..n+m_x-1, Z-ancillas n+m_x..n+m_x+m_z-1.
Every round: reset ancillas (RX for X-checks, R for Z-checks) -> CNOT layers from ``schedule`` ->
measure ancillas (MX / M), each phase followed by the configured noise.  Detectors compare
consecutive rounds of every check that is deterministic in the memory basis; observables are the
logical operators of that basis measured through the final transversal data readout.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from algebra.qldpc import CSSCode

from .noise import NoiseModel
from .schedule import Layer, commutation_violations, sequential_schedule, validate_schedule


@dataclass
class MemoryExperiment:
    code: CSSCode
    circuit: stim.Circuit
    layers: list[Layer]
    rounds: int
    basis: str
    noise: NoiseModel
    n_det_per_round: int

    @property
    def depth_per_round(self) -> int:
        return len(self.layers) + 2  # + reset layer + measurement layer

    @property
    def cnot_per_round(self) -> int:
        return int(self.code.hx.sum() + self.code.hz.sum())

    def dem(self, **kw) -> stim.DetectorErrorModel:
        return self.circuit.detector_error_model(**{"decompose_errors": False, "allow_gauge_detectors": False,
                                                    "approximate_disjoint_errors": True, **kw})


class _Rec:
    def __init__(self):
        self.count = 0

    def add(self, k: int) -> list[int]:
        idx = list(range(self.count, self.count + k))
        self.count += k
        return idx

    def tgt(self, idx: int) -> stim.GateTarget:
        return stim.target_rec(idx - self.count)


def build_memory_experiment(code: CSSCode, rounds: int, noise: NoiseModel | None = None,
                            basis: str = "Z", layers: list[Layer] | None = None,
                            detectors: str = "matched") -> MemoryExperiment:
    """Build the full noisy memory circuit.

    basis: "Z" (protect Z-basis logical states, detect X errors with Z checks) or "X".
    detectors: "matched" (only checks deterministic in ``basis``) or "all" (both check types;
    the other type gets gauge-free detectors from round 2 on).
    """
    if rounds < 1:
        raise ValueError("rounds >= 1")
    noise = noise or NoiseModel()
    n, mx, mz = code.n, code.hx.shape[0], code.hz.shape[0]
    layers = layers if layers is not None else sequential_schedule(code.hx, code.hz)
    validate_schedule(layers, code.hx, code.hz)
    if commutation_violations(layers, code.hx, code.hz):
        raise ValueError("schedule violates X/Z ordering parity: it would not measure the stabilizers")
    xa = [n + i for i in range(mx)]
    za = [n + mx + i for i in range(mz)]
    data = list(range(n))
    allq = data + xa + za
    c = stim.Circuit()
    rec = _Rec()
    idle = noise.idle_pauli

    def idle_noise(qubits, dur=1.0):
        ch = idle(dur)
        if ch and qubits:
            c.append("PAULI_CHANNEL_1", qubits, list(ch))

    # ---- data preparation
    c.append("R" if basis == "Z" else "RX", data)
    if noise.p_prep:
        c.append("X_ERROR" if basis == "Z" else "Z_ERROR", data, noise.p_prep)
    c.append("TICK")

    det_basis_checks = (code.hz, za) if basis == "Z" else (code.hx, xa)
    other_checks = (code.hx, xa) if basis == "Z" else (code.hz, za)
    prev_meas: dict[str, list[int]] = {}
    ndet_round = 0
    coords_t = 0

    def measure_round(r: int):
        nonlocal ndet_round
        # ---- ancilla reset
        c.append("RX", xa)
        c.append("R", za)
        if noise.p_prep:
            c.append("Z_ERROR", xa, noise.p_prep)
            c.append("X_ERROR", za, noise.p_prep)
        idle_noise(data)
        c.append("TICK")
        # ---- entangling layers
        for layer in layers:
            used = set()
            pairs = []
            for kind, ci, q in layer:
                a = xa[ci] if kind == "X" else za[ci]
                pairs += [a, q] if kind == "X" else [q, a]
                used.update((a, q))
            c.append("CX", pairs)
            if noise.p_gate:
                c.append("DEPOLARIZE2", pairs, noise.p_gate)
            idle_noise([q for q in allq if q not in used])
            c.append("TICK")
        # ---- measurement (X ancillas then Z ancillas)
        c.append("MX", xa, noise.p_meas) if noise.p_meas else c.append("MX", xa)
        c.append("M", za, noise.p_meas) if noise.p_meas else c.append("M", za)
        mx_idx, mz_idx = rec.add(mx), rec.add(mz)
        idle_noise(data)
        c.append("TICK")
        cur = {"X": mx_idx, "Z": mz_idx}
        want = [("Z" if basis == "Z" else "X")] + ([("X" if basis == "Z" else "Z")] if detectors == "all" else [])
        for kind in want:
            deterministic = kind == ("Z" if basis == "Z" else "X")
            for i, idx in enumerate(cur[kind]):
                if kind in prev_meas:
                    c.append("DETECTOR", [rec.tgt(idx), rec.tgt(prev_meas[kind][i])], [i, r, 0 if deterministic else 1])
                elif deterministic:
                    c.append("DETECTOR", [rec.tgt(idx)], [i, r, 0])
        prev_meas.update(cur)

    for r in range(rounds):
        measure_round(r)

    # ---- transversal readout + final detectors + observables
    c.append("M" if basis == "Z" else "MX", data, noise.p_meas) if noise.p_meas else c.append("M" if basis == "Z" else "MX", data)
    dat_idx = rec.add(n)
    kind = "Z" if basis == "Z" else "X"
    h = det_basis_checks[0]
    for i in range(h.shape[0]):
        supp = np.flatnonzero(h[i])
        c.append("DETECTOR", [rec.tgt(dat_idx[q]) for q in supp] + [rec.tgt(prev_meas[kind][i])], [i, rounds, 0])
    logicals = code.logical_z() if basis == "Z" else code.logical_x()
    for j, row in enumerate(logicals):
        c.append("OBSERVABLE_INCLUDE", [rec.tgt(dat_idx[q]) for q in np.flatnonzero(row)], j)

    return MemoryExperiment(code, c, layers, rounds, basis, noise, det_basis_checks[0].shape[0])
