import numpy as np
import pytest

from algebra.distance import distance_x, distance_z, verify_logical
from algebra.export import export_code, read_alist
from algebra.gf2 import nullspace, rank, rref, solve
from algebra.qldpc import (BB_ZOO, CSSCode, get_code, hypergraph_product, lifted_product,
                           repetition_check)


def test_gf2_basics():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, (20, 30)).astype(np.uint8)
    ns = nullspace(a)
    assert ns.shape[0] == 30 - rank(a)
    assert not ((a.astype(int) @ ns.T.astype(int)) % 2).any()
    x = rng.integers(0, 2, 30)
    b = (a.astype(int) @ x) % 2
    sol = solve(a, b)
    assert sol is not None and not (((a.astype(int) @ sol) % 2) ^ b).any()
    assert rref(a)[0].shape[0] == rank(a)


@pytest.mark.parametrize("name", list(BB_ZOO))
def test_bb_commutation_weights_and_k(name):
    c = get_code(name)
    assert not ((c.hx.astype(int) @ c.hz.T.astype(int)) % 2).any()   # H_X H_Z^T = 0 mod 2
    assert c.is_low_density(6, 6)
    assert c.weights["row_x"] == 6 and c.weights["col_x"] == 3
    assert c.k == c.meta["published_k"]
    assert c.logical_z().shape[0] == c.k and c.logical_x().shape[0] == c.k


def test_non_commuting_matrices_rejected():
    with pytest.raises(ValueError):
        CSSCode("bad", np.array([[1, 1, 0]]), np.array([[1, 0, 0]]))


def test_hypergraph_and_lifted_product_commute():
    hgp = hypergraph_product(repetition_check(4), repetition_check(5))
    assert hgp.k == 1 and hgp.is_low_density(6, 6)
    rng = np.random.default_rng(3)
    for _ in range(5):
        a = [[tuple(sorted(rng.choice(7, 2, replace=False)))]]
        b = [[tuple(sorted(rng.choice(7, 3, replace=False)))]]
        lifted_product(a, b, 7)  # constructor raises if H_X H_Z^T != 0
    lp = lifted_product([[(0,), (1,)], [(2,), (0,)]], [[(0, 3), (1,)]], 5)
    assert lp.hx.shape[1] == lp.hz.shape[1]


@pytest.mark.parametrize("name,d", [("BB72", 6), ("BB90", 10)])
def test_exact_distance_via_ilp(name, d):
    c = get_code(name)
    for res, kind in ((distance_z(c, 120), "Z"), (distance_x(c, 120), "X")):
        assert res.exact and res.upper == d
        assert verify_logical(c, res.witness, kind) and res.witness.sum() == d


def test_export_roundtrip(tmp_path):
    c = get_code("BB72")
    paths = export_code(c, tmp_path, {"d": 6})
    assert (read_alist(paths["hx.alist"]) == c.hx).all() and (read_alist(paths["hz.alist"]) == c.hz).all()
    import json
    import scipy.sparse as sp
    meta = json.loads(paths["json"].read_text())
    assert meta["n"] == 72 and meta["k"] == 12 and meta["d"] == 6
    assert (sp.load_npz(paths["hx.npz"]).toarray() == c.hx).all()
