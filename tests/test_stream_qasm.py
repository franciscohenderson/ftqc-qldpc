import asyncio

import numpy as np

from algebra.qldpc import get_code
from circuits.qasm import compile_openqasm3, parse_check
from circuits.schedule import sequential_schedule
from decoder.stream import AsyncDecoderService, PauliCorrector


def test_qasm_parses_and_has_realtime_feedback():
    c = get_code("BB72")
    q = compile_openqasm3(c, sequential_schedule(c.hx, c.hz), 2)
    ast = parse_check(q)
    assert ast.statements
    assert "extern decode_x" in q and "measure anc_x" in q and "if (corr_x[i] == 1)" in q
    assert q.count("cx anc_x[") == 2 * 0 + int(c.hx.sum()) * 1  # emitted once inside the round loop


def test_pauli_corrector_fixes_single_qubit_errors_both_types():
    c = get_code("BB72")
    pc = PauliCorrector(c, 0.01)
    rng = np.random.default_rng(0)
    B = 20
    ex = np.zeros((B, c.n), np.uint8); ez = np.zeros((B, c.n), np.uint8)
    ex[np.arange(B), rng.integers(0, c.n, B)] = 1
    ez[np.arange(B), rng.integers(0, c.n, B)] = 1
    sz = (c.hz.astype(int) @ ex.T % 2).T.astype(np.uint8)
    sx = (c.hx.astype(int) @ ez.T % 2).T.astype(np.uint8)
    for corr, a, b in zip(pc.decode(sx, sz), ex, ez):
        assert corr.ok
        assert not ((c.hz.astype(int) @ (corr.x ^ a)) % 2).any()
        assert not ((c.hx.astype(int) @ (corr.z ^ b)) % 2).any()


def test_async_service_no_backlog_and_correct_order():
    c = get_code("BB72")
    pc = PauliCorrector(c, 0.01)
    rng = np.random.default_rng(1)
    N = 200
    ex = (rng.random((N, c.n)) < 0.005).astype(np.uint8)
    sz = (c.hz.astype(int) @ ex.T % 2).T.astype(np.uint8)
    sx = np.zeros((N, c.hx.shape[0]), np.uint8)

    async def src():
        for i in range(N):
            yield sx[i], sz[i]
            await asyncio.sleep(0)

    async def main():
        svc = AsyncDecoderService(pc, workers=2)
        await svc.start()
        outs = [o async for o in svc.stream(src())]
        await svc.stop()
        return outs, svc.stats

    outs, st = asyncio.run(main())
    assert len(outs) == N and st.completed == N
    for o, s in zip(outs, sz):
        assert not (((c.hz.astype(int) @ o.x) % 2) ^ s).any()
