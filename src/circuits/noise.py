"""Circuit-level noise models.

``NoiseModel`` is a standard circuit-level depolarizing model (two-qubit depolarizing after every
CNOT, preparation flips, measurement flips) with an optional phenomenological idle channel derived
from T1/T2*:   p_x = p_y = (1 - exp(-t/T1))/4,   p_z = (1 - exp(-t/T2))/2 - p_x   (Pauli twirl of
amplitude + phase damping).  If T1/T2 are not given, idle qubits suffer depolarizing noise p_idle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class NoiseModel:
    p_gate: float = 0.0          # two-qubit depolarizing after each CNOT
    p_prep: float = 0.0          # reset error (wrong state)
    p_meas: float = 0.0          # measurement flip
    p_idle: float = 0.0          # single-qubit depolarizing on idle qubits (if no T1/T2)
    t1_us: float | None = None
    t2_us: float | None = None
    layer_time_us: float = 0.05  # duration of one CNOT layer

    @classmethod
    def uniform(cls, p: float) -> "NoiseModel":
        """Uniform circuit-level depolarizing noise at physical error rate p."""
        return cls(p_gate=p, p_prep=p, p_meas=p, p_idle=p)

    @classmethod
    def from_coherence(cls, p_gate: float, t1_us: float, t2_us: float, layer_time_us: float = 0.05,
                       p_prep: float | None = None, p_meas: float | None = None) -> "NoiseModel":
        return cls(p_gate=p_gate, p_prep=p_gate if p_prep is None else p_prep,
                   p_meas=p_gate if p_meas is None else p_meas, t1_us=t1_us, t2_us=t2_us,
                   layer_time_us=layer_time_us)

    def idle_pauli(self, duration_layers: float = 1.0) -> tuple[float, float, float] | None:
        """(p_x, p_y, p_z) of the idle channel for a given number of layers, or None if noiseless."""
        if self.t1_us is not None and self.t2_us is not None:
            t = self.layer_time_us * duration_layers
            px = py = (1 - math.exp(-t / self.t1_us)) / 4
            pz = max(0.0, (1 - math.exp(-t / self.t2_us)) / 2 - px)
            return px, py, pz
        if self.p_idle > 0:
            q = self.p_idle * duration_layers / 3
            return q, q, q
        return None
