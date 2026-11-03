"""Low-depth interleaved X/Z CNOT schedules for qLDPC syndrome extraction.

An interleaved schedule mixes X-check CNOTs (ancilla -> data) and Z-check CNOTs (data -> ancilla)
within the same time-step while satisfying two conditions:
1. Conflict freedom (``validate_schedule``): every Tanner edge appears once and no data qubit or
   ancilla is used twice in the same layer.
2. Symplectic commutation parity (``commutation_violations == 0``): for every (X-check, Z-check)
   pair sharing data qubits, an even number of shared qubits execute their X-CNOT before their
   Z-CNOT, cancelling hook-like ancilla-ancilla coupling (Bravyi et al., Nature 627, 2024).
"""
from __future__ import annotations

import time

import numpy as np

from algebra.qldpc import CSSCode, _circulant

from .schedule import (
    Layer,
    commutation_violations,
    edge_colour_bipartite,
    sequential_schedule,
    validate_schedule,
)


def schedule_depth(layers: list[Layer]) -> int:
    """Return the number of parallel CNOT layers in ``layers``."""
    return len(layers)


def _term_matrix(l: int, m: int, var: str, exp: int) -> np.ndarray:
    """Permutation matrix on Z_l x Z_m for a single monomial ('x'|'y', exp)."""
    if var == "x":
        return np.kron(_circulant(l, [exp % l]), np.eye(m, dtype=np.uint8))
    if var == "y":
        return np.kron(np.eye(l, dtype=np.uint8), _circulant(m, [exp % m]))
    raise ValueError(f"unknown variable {var!r}; expected 'x' or 'y'")


def _decompose_bb_blocks(
    hx: np.ndarray,
    hz: np.ndarray,
    meta: dict | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    """Extract commuting permutation terms (A_mats, B_mats) when H_X=[A|B], H_Z=[B^T|A^T]."""
    mx, n = hx.shape
    mz, nz = hz.shape
    if mx != mz or n != 2 * mx or nz != n or mx == 0:
        return None
    a_blk = hx[:, :mx]
    b_blk = hx[:, mx:]
    if not (np.array_equal(hz[:, :mx], b_blk.T) and np.array_equal(hz[:, mx:], a_blk.T)):
        return None

    # 1. Use explicit BB metadata if available and consistent with (hx, hz).
    if meta and "l" in meta and "m" in meta and "A" in meta and "B" in meta:
        l, m = int(meta["l"]), int(meta["m"])
        if l * m == mx:
            a_mats = [_term_matrix(l, m, v, int(e)) for v, e in meta["A"]]
            b_mats = [_term_matrix(l, m, v, int(e)) for v, e in meta["B"]]
            a_sum = np.zeros_like(a_blk)
            for mat in a_mats:
                a_sum ^= mat
            b_sum = np.zeros_like(b_blk)
            for mat in b_mats:
                b_sum ^= mat
            if np.array_equal(a_sum, a_blk) and np.array_equal(b_sum, b_blk):
                return a_mats, b_mats

    # 2. Auto-detect Z_l x Z_m translation decomposition directly from (a_blk, b_blk).
    a_cols0 = np.flatnonzero(a_blk[0])
    b_cols0 = np.flatnonzero(b_blk[0])
    for l in range(1, mx + 1):
        if mx % l != 0:
            continue
        m = mx // l
        u_c = np.arange(mx) // m
        v_c = np.arange(mx) % m
        a_mats, b_mats = [], []
        a_sum = np.zeros_like(a_blk)
        b_sum = np.zeros_like(b_blk)
        for q0 in a_cols0:
            u0, v0 = divmod(int(q0), m)
            cols = ((u_c + u0) % l) * m + ((v_c + v0) % m)
            mat = np.zeros((mx, mx), dtype=np.uint8)
            mat[np.arange(mx), cols] = 1
            a_mats.append(mat)
            a_sum ^= mat
        if not np.array_equal(a_sum, a_blk):
            continue
        for q0 in b_cols0:
            u0, v0 = divmod(int(q0), m)
            cols = ((u_c + u0) % l) * m + ((v_c + v0) % m)
            mat = np.zeros((mx, mx), dtype=np.uint8)
            mat[np.arange(mx), cols] = 1
            b_mats.append(mat)
            b_sum ^= mat
        if np.array_equal(b_sum, b_blk):
            return a_mats, b_mats
    return None


def _layers_from_bb_times(
    a_mats: list[np.ndarray],
    b_mats: list[np.ndarray],
    xl_t: list[int] | np.ndarray,
    xr_t: list[int] | np.ndarray,
    zl_t: list[int] | np.ndarray,
    zr_t: list[int] | np.ndarray,
) -> list[Layer]:
    """Expand translation-invariant edge-type times into concrete CNOT layers."""
    mx = a_mats[0].shape[0]
    max_t = int(max(max(xl_t, default=0), max(xr_t, default=0), max(zl_t, default=0), max(zr_t, default=0)))
    buckets: list[Layer] = [[] for _ in range(max_t + 1)]
    rows = np.arange(mx, dtype=np.int64)
    for i, mat in enumerate(a_mats):
        cols = np.argmax(mat, axis=1)
        t = int(xl_t[i])
        buckets[t].extend(("X", int(c), int(q)) for c, q in zip(rows, cols))
    for j, mat in enumerate(b_mats):
        cols = np.argmax(mat, axis=1)
        t = int(xr_t[j])
        buckets[t].extend(("X", int(c), int(mx + q)) for c, q in zip(rows, cols))
    for j, mat in enumerate(b_mats):
        cols = np.argmax(mat.T, axis=1)
        t = int(zl_t[j])
        buckets[t].extend(("Z", int(r), int(q)) for r, q in zip(rows, cols))
    for i, mat in enumerate(a_mats):
        cols = np.argmax(mat.T, axis=1)
        t = int(zr_t[i])
        buckets[t].extend(("Z", int(r), int(mx + q)) for r, q in zip(rows, cols))
    return [layer for layer in buckets if layer]


def _bb_commutation_ok(
    ab_overlap: list[list[tuple[int, int]]],
    xl_t: np.ndarray,
    xr_t: np.ndarray,
    zl_t: np.ndarray,
    zr_t: np.ndarray,
) -> bool:
    """Fast vectorized check of translation-invariant commutation parity from check 0."""
    # Pairwise indicator matrices: left_lt[i, j] = (xl_t[i] < zl_t[j]), right_lt[i, j] = (xr_t[j] < zr_t[i])
    left_lt = xl_t[:, None] < zl_t[None, :]
    right_lt = xr_t[None, :] < zr_t[:, None]
    if np.array_equal(left_lt, right_lt):
        return True
    diff = left_lt ^ right_lt
    for pairs in ab_overlap:
        parity = 0
        for i, j in pairs:
            parity ^= int(diff[i, j])
        if parity:
            return False
    return True


def bb_translation_schedule(
    code: CSSCode,
    depth: int = 8,
    seed: int = 0,
    max_tries: int = 20000,
) -> list[Layer] | None:
    """Find a translation-invariant interleaved schedule of CNOT depth <= ``depth`` for a BB code.

    Edges of the same (block, polynomial-term) type share a time-step in ``0..depth-1``, reducing
    the schedule to ``2*(|A| + |B|)`` integer variables.
    """
    decomp = _decompose_bb_blocks(code.hx, code.hz, getattr(code, "meta", None))
    if decomp is None:
        return None
    a_mats, b_mats = decomp
    wa, wb = len(a_mats), len(b_mats)
    min_req = max(wa + wb, 1)
    if depth < min_req:
        return None

    # Precompute which (i, j) pairs connect X-check 0 to each Z-check r via A_i B_j = B_j A_i.
    mx = code.hx.shape[0]
    by_r: dict[int, list[tuple[int, int]]] = {}
    for i, a_mat in enumerate(a_mats):
        q_l = int(np.argmax(a_mat[0]))
        for j, b_mat in enumerate(b_mats):
            r = int(np.argmax(b_mat[q_l]))
            by_r.setdefault(r, []).append((i, j))
    ab_overlap = [pairs for pairs in by_r.values() if pairs]

    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    k_max = max(wa, wb)
    # Exact staircase solution: uses 2*k_max + 1 layers (depth 7 for weight-6 BB codes)
    # where (xl[i] < zl[j]) <=> (i <= j) <=> (xr[j] < zr[i]) holds for all (i, j).
    if depth >= 2 * k_max + 1:
        xl0 = np.array([2 * i for i in range(wa)], dtype=np.int64)
        zl0 = np.array([2 * j + 1 for j in range(wb)], dtype=np.int64)
        xr0 = np.array([2 * (k_max - 1 - j) + 1 for j in range(wb)], dtype=np.int64)
        zr0 = np.array([2 * (k_max - i) for i in range(wa)], dtype=np.int64)
        candidates.append((xl0, xr0, zl0, zr0))
    if depth >= 2 * k_max + 2:
        xl1 = np.array([2 * i for i in range(wa)], dtype=np.int64)
        zl1 = np.array([2 * j + 1 for j in range(wb - 1)] + [2 * k_max], dtype=np.int64)
        xr1 = np.array([2 * (k_max - 1 - j) + 1 for j in range(wb)], dtype=np.int64)
        zr1 = np.array([2 * k_max + 1] + [2 * (k_max - i) for i in range(1, wa)], dtype=np.int64)
        candidates.append((xl1, xr1, zl1, zr1))

    best_layers: list[Layer] | None = None
    for xl_t, xr_t, zl_t, zr_t in candidates:
        if (
            len(set(xl_t.tolist()) | set(xr_t.tolist())) == wa + wb
            and len(set(zl_t.tolist()) | set(zr_t.tolist())) == wa + wb
            and len(set(xl_t.tolist()) | set(zl_t.tolist())) == wa + wb
            and len(set(xr_t.tolist()) | set(zr_t.tolist())) == wa + wb
            and _bb_commutation_ok(ab_overlap, xl_t, xr_t, zl_t, zr_t)
        ):
            layers = _layers_from_bb_times(a_mats, b_mats, xl_t, xr_t, zl_t, zr_t)
            if len(layers) <= depth:
                validate_schedule(layers, code.hx, code.hz)
                if commutation_violations(layers, code.hx, code.hz) == 0:
                    if best_layers is None or len(layers) < len(best_layers):
                        best_layers = layers
                    if len(best_layers) <= 2 * k_max + 1:
                        return best_layers

    # Vectorized stochastic search over conflict-free 12-variable assignments in 0..d-1.
    rng = np.random.default_rng(seed)
    all_colors = np.arange(depth, dtype=np.int64)
    for _ in range(max_tries):
        perm_x = rng.permutation(all_colors)
        xl_t = perm_x[:wa]
        xr_t = perm_x[wa : wa + wb]
        # zl_t must avoid xl_t
        avail_zl = perm_x[wa:]
        if avail_zl.size < wb:
            continue
        zl_t = rng.choice(avail_zl, size=wb, replace=False)
        # zr_t must avoid both xr_t and zl_t
        forbidden_zr = set(xr_t.tolist()) | set(zl_t.tolist())
        avail_zr = np.array([c for c in all_colors if int(c) not in forbidden_zr], dtype=np.int64)
        if avail_zr.size < wa:
            continue
        zr_t = rng.choice(avail_zr, size=wa, replace=False)
        if _bb_commutation_ok(ab_overlap, xl_t, xr_t, zl_t, zr_t):
            layers = _layers_from_bb_times(a_mats, b_mats, xl_t, xr_t, zl_t, zr_t)
            if len(layers) <= depth:
                validate_schedule(layers, code.hx, code.hz)
                if commutation_violations(layers, code.hx, code.hz) == 0:
                    if best_layers is None or len(layers) < len(best_layers):
                        best_layers = layers
                    if len(best_layers) <= 2 * k_max + 1:
                        break
    return best_layers


def _bipartite_kempe_sa(
    hx: np.ndarray,
    hz: np.ndarray,
    depth: int,
    rng: np.random.Generator,
    max_steps: int,
    deadline: float,
) -> list[Layer] | None:
    """Conflict-free colouring + Kempe/recolour simulated annealing on commutation violations."""
    mx, n = hx.shape
    mz = hz.shape[0]
    h_comb = np.vstack([hx, hz])
    delta_comb = int(max(h_comb.sum(0).max(initial=0), h_comb.sum(1).max(initial=0)))
    if depth < delta_comb:
        return None

    # Build randomized proper edge-colouring of the combined bipartite Tanner graph in <= depth colours.
    ck = np.full((mx + mz, depth), -1, dtype=np.int64)
    qb = np.full((n, depth), -1, dtype=np.int64)
    edges_u, edges_q = np.nonzero(h_comb)
    order = rng.permutation(edges_u.size)
    for idx in order:
        i = int(edges_u[idx])
        j = int(edges_q[idx])
        free_i = np.flatnonzero(ck[i] < 0)
        free_j = np.flatnonzero(qb[j] < 0)
        common = np.intersect1d(free_i, free_j, assume_unique=True)
        if common.size:
            a = int(rng.choice(common))
        else:
            a = int(rng.choice(free_i))
            b = int(rng.choice(free_j))
            path = []
            side, v, col = 1, j, a
            while True:
                nxt = qb[v, col] if side == 1 else ck[v, col]
                if nxt < 0:
                    break
                path.append((side, v, int(nxt), col))
                side, v = 1 - side, int(nxt)
                col = b if col == a else a
            for s, v, w, col in path:
                if s == 1:
                    qb[v, col] = -1
                    ck[w, col] = -1
                else:
                    ck[v, col] = -1
                    qb[w, col] = -1
            for s, v, w, col in path:
                nc = b if col == a else a
                if s == 1:
                    qb[v, nc] = w
                    ck[w, nc] = v
                else:
                    ck[v, nc] = w
                    qb[w, nc] = v
        ck[i, a] = j
        qb[j, a] = i

    tx = np.full((mx, n), -1, dtype=np.int64)
    tz = np.full((mz, n), -1, dtype=np.int64)
    edge_col = np.zeros(edges_u.size, dtype=np.int64)
    for idx in range(edges_u.size):
        u = int(edges_u[idx])
        q = int(edges_q[idx])
        col = int(np.flatnonzero(ck[u] == q)[0])
        edge_col[idx] = col
        if u < mx:
            tx[u, q] = col
        else:
            tz[u - mx, q] = col

    x_at_q = [np.flatnonzero(hx[:, q]).tolist() for q in range(n)]
    z_at_q = [np.flatnonzero(hz[:, q]).tolist() for q in range(n)]
    parity = np.zeros((mx, mz), dtype=np.uint8)
    has_shared = np.zeros((mx, mz), dtype=bool)
    for s in range(mx):
        for r in range(mz):
            sh = np.flatnonzero(hx[s] & hz[r])
            if sh.size:
                has_shared[s, r] = True
                parity[s, r] = int(np.sum(tx[s, sh] < tz[r, sh]) & 1)
    num_bad = int(parity.sum())

    def set_edge_time(u: int, q: int, new_t: int) -> None:
        nonlocal num_bad
        if u < mx:
            old_t = int(tx[u, q])
            if old_t == new_t:
                return
            for r in z_at_q[q]:
                tz_rq = int(tz[r, q])
                if (old_t < tz_rq) != (new_t < tz_rq):
                    if parity[u, r]:
                        parity[u, r] = 0
                        num_bad -= 1
                    else:
                        parity[u, r] = 1
                        num_bad += 1
            tx[u, q] = new_t
        else:
            r = u - mx
            old_t = int(tz[r, q])
            if old_t == new_t:
                return
            for s in x_at_q[q]:
                tx_sq = int(tx[s, q])
                if (tx_sq < old_t) != (tx_sq < new_t):
                    if parity[s, r]:
                        parity[s, r] = 0
                        num_bad -= 1
                    else:
                        parity[s, r] = 1
                        num_bad += 1
            tz[r, q] = new_t

    for step in range(max_steps):
        if num_bad == 0:
            break
        if (step & 63) == 0 and time.perf_counter() >= deadline:
            break
        temp = max(0.05, 1.5 * (1.0 - step / max(max_steps, 1)))
        # Bias edge selection toward edges participating in violated check pairs.
        bad_s, bad_r = np.nonzero(parity)
        if bad_s.size and rng.random() < 0.85:
            pick = int(rng.integers(0, bad_s.size))
            s0, r0 = int(bad_s[pick]), int(bad_r[pick])
            sh = np.flatnonzero(hx[s0] & hz[r0])
            q = int(rng.choice(sh))
            u = s0 if rng.random() < 0.5 else (mx + r0)
        else:
            idx = int(rng.integers(0, edges_u.size))
            u, q = int(edges_u[idx]), int(edges_q[idx])

        a = int(tx[u, q] if u < mx else tz[u - mx, q])
        b = int(rng.integers(0, depth - 1))
        if b >= a:
            b += 1

        old_bad = num_bad
        # Case 1: colour b is free at both endpoints -> O(1) conflict-free recolour.
        if ck[u, b] < 0 and qb[q, b] < 0:
            ck[u, a] = -1
            qb[q, a] = -1
            ck[u, b] = q
            qb[q, b] = u
            set_edge_time(u, q, b)
            delta = num_bad - old_bad
            if delta > 0 and rng.random() >= np.exp(-delta / temp):
                ck[u, b] = -1
                qb[q, b] = -1
                ck[u, a] = q
                qb[q, a] = u
                set_edge_time(u, q, a)
            continue

        # Case 2: flip maximal (a, b)-alternating component containing (u, q).
        comp_edges: list[tuple[int, int, int]] = [(u, q, a)]
        seen_e = {(u, q)}
        for start_side, start_v, start_col in ((0, u, b), (1, q, b)):
            side, v, col = start_side, start_v, start_col
            while True:
                nxt = ck[v, col] if side == 0 else qb[v, col]
                if nxt < 0:
                    break
                uu, qq = (v, int(nxt)) if side == 0 else (int(nxt), v)
                if (uu, qq) in seen_e:
                    break
                seen_e.add((uu, qq))
                comp_edges.append((uu, qq, col))
                side, v = 1 - side, int(nxt)
                col = a if col == b else b

        for uu, qq, col in comp_edges:
            ck[uu, col] = -1
            qb[qq, col] = -1
        for uu, qq, col in comp_edges:
            nc = b if col == a else a
            ck[uu, nc] = qq
            qb[qq, nc] = uu
            set_edge_time(uu, qq, nc)

        delta = num_bad - old_bad
        if delta > 0 and rng.random() >= np.exp(-delta / temp):
            for uu, qq, col in comp_edges:
                nc = b if col == a else a
                ck[uu, nc] = -1
                qb[qq, nc] = -1
            for uu, qq, col in comp_edges:
                ck[uu, col] = qq
                qb[qq, col] = uu
                set_edge_time(uu, qq, col)

    if num_bad != 0:
        return None

    buckets: list[Layer] = [[] for _ in range(depth)]
    for u in range(mx):
        for q in np.flatnonzero(hx[u]):
            buckets[int(tx[u, q])].append(("X", u, int(q)))
    for r in range(mz):
        for q in np.flatnonzero(hz[r]):
            buckets[int(tz[r, q])].append(("Z", r, int(q)))
    layers = [layer for layer in buckets if layer]
    validate_schedule(layers, hx, hz)
    if commutation_violations(layers, hx, hz) != 0:
        return None
    return layers


def find_interleaved_schedule(
    hx: np.ndarray | CSSCode,
    hz: np.ndarray | None = None,
    target_depth: int | None = None,
    max_tries: int = 20000,
    seed: int = 0,
    time_limit_s: float = 60.0,
) -> list[Layer] | None:
    """Find a low-depth valid interleaved CNOT schedule for ``(hx, hz)``.

    Uses translation invariance for bivariate bicycle codes and conflict-free bipartite edge
    colouring with Kempe-chain simulated annealing for general CSS codes. Falls back to
    ``sequential_schedule(hx, hz)`` if no shorter valid interleaved schedule is found.
    """
    code_obj: CSSCode | None = None
    if isinstance(hx, CSSCode):
        code_obj = hx
        hx_arr, hz_arr = code_obj.hx, code_obj.hz
    else:
        if hz is None:
            raise ValueError("hz is required when hx is an ndarray")
        hx_arr = np.asarray(hx, dtype=np.uint8)
        hz_arr = np.asarray(hz, dtype=np.uint8)

    start = time.perf_counter()
    deadline = start + float(time_limit_s)
    seq = sequential_schedule(hx_arr, hz_arr)
    best_layers: list[Layer] = seq

    if code_obj is None:
        code_obj = CSSCode("code", hx_arr, hz_arr)

    # 1. Try translation-invariant BB schedule when applicable.
    bb_max_d = target_depth if target_depth is not None else max(len(seq) - 1, 8)
    bb_sched = bb_translation_schedule(code_obj, depth=bb_max_d, seed=seed, max_tries=max_tries)
    if bb_sched is not None and len(bb_sched) < len(best_layers):
        best_layers = bb_sched
        if target_depth is not None and len(best_layers) <= target_depth:
            return best_layers

    # 2. General bipartite colouring + Kempe-chain SA for non-BB codes or tighter target depths.
    h_comb = np.vstack([hx_arr, hz_arr])
    delta_comb = int(max(h_comb.sum(0).max(initial=0), h_comb.sum(1).max(initial=0)))
    if delta_comb == 0:
        return []

    upper_d = min(len(best_layers) - 1, target_depth if target_depth is not None else len(best_layers) - 1)
    lower_d = delta_comb if target_depth is None else max(delta_comb, min(target_depth, upper_d))
    rng = np.random.default_rng(seed)

    for d in range(upper_d, lower_d - 1, -1):
        if time.perf_counter() >= deadline:
            break
        restarts = max(4, min(32, max_tries // 500))
        steps_per_restart = max(200, max_tries // restarts)
        found_at_d = False
        for _ in range(restarts):
            if time.perf_counter() >= deadline:
                break
            cand = _bipartite_kempe_sa(hx_arr, hz_arr, d, rng, steps_per_restart, deadline)
            if cand is not None and len(cand) < len(best_layers):
                best_layers = cand
                found_at_d = True
                break
        if not found_at_d and best_layers is not seq:
            break

    return best_layers
