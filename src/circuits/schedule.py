"""Entangling schedules for syndrome extraction.

A schedule is a list of layers; each layer is a list of ``(kind, check, qubit)`` triples, with
``kind`` in {"X", "Z"}.  Within one layer no data qubit and no ancilla appears twice, so the layer
is a single parallel CNOT time-step.  X-check CNOTs go ancilla -> data (ancilla prepared in |+>);
Z-check CNOTs go data -> ancilla (ancilla in |0>).

``sequential_schedule`` colours each check graph with exactly Delta colours (Konig's theorem,
alternating-path algorithm) so depth = Delta_X + Delta_Z, optimal for a non-interleaved schedule.
``validate_schedule`` proves every layer is conflict-free and covers every Tanner edge exactly once;
``commutation_violations`` additionally checks the X/Z ordering parity condition required for an
interleaved schedule to measure the correct stabilizers.
"""
from __future__ import annotations

import numpy as np

Layer = list[tuple[str, int, int]]


def edge_colour_bipartite(h: np.ndarray) -> list[list[tuple[int, int]]]:
    """Proper edge colouring of the Tanner graph of ``h`` with max-degree colours.

    Returns ``classes`` where ``classes[c]`` is a matching of (check, qubit) edges.
    """
    m, n = h.shape
    delta = int(max(h.sum(0).max(), h.sum(1).max()))
    # colour_at[side][vertex][colour] -> neighbour or -1
    ck = np.full((m, delta), -1, dtype=np.int64)
    qb = np.full((n, delta), -1, dtype=np.int64)
    for i, j in zip(*np.nonzero(h)):
        a = int(np.flatnonzero(ck[i] < 0)[0])      # free colour at check i
        b = int(np.flatnonzero(qb[j] < 0)[0])      # free colour at qubit j
        if a != b and qb[j, a] >= 0:
            # Flip the a/b alternating path starting at qubit j so that colour a becomes free at j.
            path = []
            side, v, col = 1, j, a
            while True:
                nxt = qb[v, col] if side == 1 else ck[v, col]
                if nxt < 0:
                    break
                path.append((side, v, int(nxt), col))
                side, v = 1 - side, int(nxt)
                col = b if col == a else a
            for s, v, w, col in path:           # clear
                if s == 1:
                    qb[v, col] = -1; ck[w, col] = -1
                else:
                    ck[v, col] = -1; qb[w, col] = -1
            for s, v, w, col in path:           # recolour with the swapped colour
                nc = b if col == a else a
                if s == 1:
                    qb[v, nc] = w; ck[w, nc] = v
                else:
                    ck[v, nc] = w; qb[w, nc] = v
        ck[i, a] = j
        qb[j, a] = i
    return [[(int(i), int(ck[i, c])) for i in range(m) if ck[i, c] >= 0] for c in range(delta)]


def sequential_schedule(hx: np.ndarray, hz: np.ndarray) -> list[Layer]:
    layers: list[Layer] = []
    for kind, h in (("X", hx), ("Z", hz)):
        for cls in edge_colour_bipartite(h):
            layers.append([(kind, i, j) for i, j in cls])
    return layers


def validate_schedule(layers: list[Layer], hx: np.ndarray, hz: np.ndarray) -> None:
    seen = {"X": set(), "Z": set()}
    for t, layer in enumerate(layers):
        data, anc = set(), set()
        for kind, c, q in layer:
            if q in data or (kind, c) in anc:
                raise ValueError(f"layer {t}: qubit/ancilla used twice")
            data.add(q); anc.add((kind, c))
            if (c, q) in seen[kind]:
                raise ValueError(f"layer {t}: duplicate edge")
            seen[kind].add((c, q))
    for kind, h in (("X", hx), ("Z", hz)):
        want = set(zip(*[a.tolist() for a in np.nonzero(h)]))
        if seen[kind] != want:
            raise ValueError(f"schedule does not cover the {kind} Tanner graph")


def commutation_violations(layers: list[Layer], hx: np.ndarray, hz: np.ndarray) -> int:
    """Count (X-check, Z-check) pairs whose shared-qubit ordering parity is odd (must be 0)."""
    tx = np.full(hx.shape, -1)
    tz = np.full(hz.shape, -1)
    for t, layer in enumerate(layers):
        for kind, c, q in layer:
            (tx if kind == "X" else tz)[c, q] = t
    bad = 0
    for s in range(hx.shape[0]):
        for r in range(hz.shape[0]):
            shared = np.flatnonzero(hx[s] & hz[r])
            if shared.size and np.sum(tx[s, shared] < tz[r, shared]) % 2:
                bad += 1
    return bad
