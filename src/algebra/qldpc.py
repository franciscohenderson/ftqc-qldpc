"""Generalized asymmetric qLDPC code construction.

Two families, both CSS with H_X H_Z^T = 0 (mod 2) enforced at construction:

* ``lifted_product``: A, B are protograph matrices (lists of lists) whose entries are tuples of
  exponents of a polynomial over F2[x]/(x^l - 1).  With H1=A (m1 x n1), H2=B (m2 x n2):
      H_X = [A (x) I_{n2} | I_{m1} (x) B*],   H_Z = [I_{n1} (x) B | A* (x) I_{m2}]
  (Panteleev-Kalachev lifted product; * = conjugate transpose, i.e. binary transpose of the lift).
* ``bivariate_bicycle``: the BB family of Bravyi et al. (Nature 627, 2024) -- the lifted product of
  1x1 protographs over F2[x,y]/(x^l-1, y^m-1).  H_X=[A|B], H_Z=[B^T|A^T].
* ``hypergraph_product``: Tillich-Zemor product of two classical parity-check matrices.

"Asymmetric": A and B are independent so d_X != d_Z is allowed; weights w_c, w_r are read from the
Tanner graph and checked against the low-density bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from .gf2 import as_gf2, nullspace, rank, rowspace_complement


@dataclass
class CSSCode:
    name: str
    hx: np.ndarray
    hz: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.hx, self.hz = as_gf2(self.hx), as_gf2(self.hz)
        if self.hx.shape[1] != self.hz.shape[1]:
            raise ValueError("H_X and H_Z need equal column count")
        if np.any((self.hx.astype(np.int64) @ self.hz.T.astype(np.int64)) & 1):
            raise ValueError("symplectic commutativity violated: H_X H_Z^T != 0 mod 2")

    @property
    def n(self) -> int:
        return self.hx.shape[1]

    @property
    def rank_x(self) -> int:
        return rank(self.hx)

    @property
    def rank_z(self) -> int:
        return rank(self.hz)

    @property
    def k(self) -> int:
        return self.n - self.rank_x - self.rank_z

    @property
    def weights(self) -> dict:
        return {
            "col_x": int(self.hx.sum(0).max()), "row_x": int(self.hx.sum(1).max()),
            "col_z": int(self.hz.sum(0).max()), "row_z": int(self.hz.sum(1).max()),
        }

    def is_low_density(self, wc: int = 6, wr: int = 6) -> bool:
        w = self.weights
        return max(w["col_x"], w["col_z"]) <= wc and max(w["row_x"], w["row_z"]) <= wr

    def logical_z(self) -> np.ndarray:
        """Z-type logicals: ker(H_X) modulo rowspace(H_Z). Shape (k, n)."""
        return rowspace_complement(self.hz, nullspace(self.hx))

    def logical_x(self) -> np.ndarray:
        """X-type logicals: ker(H_Z) modulo rowspace(H_X). Shape (k, n)."""
        return rowspace_complement(self.hx, nullspace(self.hz))


def _circulant(l: int, exps) -> np.ndarray:
    m = np.zeros((l, l), dtype=np.uint8)
    for e in exps:
        m[(np.arange(l) + e) % l, np.arange(l)] ^= 1
    return m


def _blocks(proto, l: int) -> list[list[np.ndarray]]:
    return [[_circulant(l, e) for e in row] for row in proto]


def _transpose_blocks(b):
    return [[b[i][j].T for i in range(len(b))] for j in range(len(b[0]))]


def _kron_id_right(b, k: int) -> np.ndarray:
    """M (x) I_k at block level (each l x l block replicated along a k-diagonal)."""
    m, n, l = len(b), len(b[0]), b[0][0].shape[0]
    out = np.zeros((m * k * l, n * k * l), dtype=np.uint8)
    for i in range(m):
        for j in range(n):
            for t in range(k):
                out[(i * k + t) * l:(i * k + t + 1) * l, (j * k + t) * l:(j * k + t + 1) * l] = b[i][j]
    return out


def _kron_id_left(b, k: int) -> np.ndarray:
    """I_k (x) M at block level."""
    m, n, l = len(b), len(b[0]), b[0][0].shape[0]
    out = np.zeros((k * m * l, k * n * l), dtype=np.uint8)
    for t in range(k):
        for i in range(m):
            for j in range(n):
                out[(t * m + i) * l:(t * m + i + 1) * l, (t * n + j) * l:(t * n + j + 1) * l] = b[i][j]
    return out


def lifted_product(a_proto, b_proto, l: int, name: str = "LP") -> CSSCode:
    A, B = _blocks(a_proto, l), _blocks(b_proto, l)
    m1, n1, m2, n2 = len(A), len(A[0]), len(B), len(B[0])
    hx = np.hstack([_kron_id_right(A, n2), _kron_id_left(_transpose_blocks(B), m1)])
    hz = np.hstack([_kron_id_left(B, n1), _kron_id_right(_transpose_blocks(A), m2)])
    return CSSCode(name, hx, hz, {"family": "lifted_product", "l": l})


def bivariate_bicycle(l: int, m: int, a_terms, b_terms, name: str | None = None) -> CSSCode:
    """BB code on Z_l x Z_m.  Terms are ('x'|'y', exponent)."""
    S = {"x": np.kron(_circulant(l, [1]), np.eye(m, dtype=np.uint8)),
         "y": np.kron(np.eye(l, dtype=np.uint8), _circulant(m, [1]))}

    def poly(terms):
        out = np.zeros((l * m, l * m), dtype=np.uint8)
        for var, e in terms:
            out ^= np.linalg.matrix_power(S[var].astype(np.int64), e).astype(np.uint8) & 1
        return out

    A, B = poly(a_terms), poly(b_terms)
    return CSSCode(name or f"BB{2 * l * m}", np.hstack([A, B]), np.hstack([B.T, A.T]),
                   {"family": "bivariate_bicycle", "l": l, "m": m, "A": a_terms, "B": b_terms})


def hypergraph_product(h1: np.ndarray, h2: np.ndarray, name: str = "HGP") -> CSSCode:
    h1, h2 = as_gf2(h1), as_gf2(h2)
    m1, n1 = h1.shape
    m2, n2 = h2.shape
    hx = np.hstack([np.kron(h1, np.eye(n2, dtype=np.uint8)), np.kron(np.eye(m1, dtype=np.uint8), h2.T)])
    hz = np.hstack([np.kron(np.eye(n1, dtype=np.uint8), h2), np.kron(h1.T, np.eye(m2, dtype=np.uint8))])
    return CSSCode(name, hx, hz, {"family": "hypergraph_product"})


def repetition_check(n: int, ring: bool = False) -> np.ndarray:
    h = np.zeros((n if ring else n - 1, n), dtype=np.uint8)
    for i in range(h.shape[0]):
        h[i, i] = h[i, (i + 1) % n] = 1
    return h


# Published BB codes (Bravyi et al. 2024): name -> (l, m, A, B, k, d)
BB_ZOO = {
    "BB72":  (6, 6,  [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)], 12, 6),
    "BB90":  (15, 3, [("x", 9), ("y", 1), ("y", 2)], [("x", 0), ("x", 2), ("x", 7)], 8, 10),
    "BB108": (9, 6,  [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)], 8, 10),
    "BB144": (12, 6, [("x", 3), ("y", 1), ("y", 2)], [("y", 3), ("x", 1), ("x", 2)], 12, 12),
    "BB288": (12, 12, [("x", 3), ("y", 2), ("y", 7)], [("y", 3), ("x", 1), ("x", 2)], 12, 18),
}


def get_code(name: str) -> CSSCode:
    l, m, a, b, k, d = BB_ZOO[name]
    c = bivariate_bicycle(l, m, a, b, name)
    c.meta.update(published_k=k, published_d=d)
    return c


def to_csr(h: np.ndarray) -> sp.csr_matrix:
    return sp.csr_matrix(as_gf2(h))
