"""Export parity-check matrices: .alist (MacKay), sparse CSR (.npz) and JSON Tanner-graph metadata."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .qldpc import CSSCode


def write_alist(h: np.ndarray, path) -> None:
    """MacKay alist: columns = variables (n), rows = checks (m)."""
    m, n = h.shape
    cols = [np.flatnonzero(h[:, j]) + 1 for j in range(n)]
    rows = [np.flatnonzero(h[i]) + 1 for i in range(m)]
    maxc, maxr = max(len(c) for c in cols), max(len(r) for r in rows)
    lines = [f"{n} {m}", f"{maxc} {maxr}",
             " ".join(str(len(c)) for c in cols), " ".join(str(len(r)) for r in rows)]
    lines += [" ".join(map(str, c)) + " 0" * (maxc - len(c)) for c in cols]
    lines += [" ".join(map(str, r)) + " 0" * (maxr - len(r)) for r in rows]
    Path(path).write_text("\n".join(lines) + "\n")


def read_alist(path) -> np.ndarray:
    tok = Path(path).read_text().split("\n")
    n, m = map(int, tok[0].split())
    maxc, _ = map(int, tok[1].split())
    h = np.zeros((m, n), dtype=np.uint8)
    for j in range(n):
        for i in map(int, tok[4 + j].split()):
            if i:
                h[i - 1, j] = 1
    return h


def tanner_json(code: CSSCode, extra: dict | None = None) -> dict:
    def edges(h):
        r, c = np.nonzero(h)
        return [[int(i), int(j)] for i, j in zip(r, c)]

    meta = {k: v for k, v in code.meta.items()}
    return {
        "name": code.name, "n": code.n, "k": code.k,
        "rank_hx": code.rank_x, "rank_hz": code.rank_z,
        "m_x": int(code.hx.shape[0]), "m_z": int(code.hz.shape[0]),
        "weights": code.weights, "meta": meta,
        "tanner_x_edges_check_qubit": edges(code.hx),
        "tanner_z_edges_check_qubit": edges(code.hz),
        **(extra or {}),
    }


def export_code(code: CSSCode, out_dir, extra: dict | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for tag, h in (("hx", code.hx), ("hz", code.hz)):
        paths[f"{tag}.alist"] = out / f"{code.name}_{tag}.alist"
        write_alist(h, paths[f"{tag}.alist"])
        paths[f"{tag}.npz"] = out / f"{code.name}_{tag}.npz"
        sp.save_npz(paths[f"{tag}.npz"], sp.csr_matrix(h))
    paths["json"] = out / f"{code.name}.json"
    paths["json"].write_text(json.dumps(tanner_json(code, extra), default=str))
    return paths
