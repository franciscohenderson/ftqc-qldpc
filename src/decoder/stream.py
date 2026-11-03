"""Asynchronous streaming interface: detector stream in, Pauli corrections out.

``PauliCorrector`` splits the problem the way CSS hardware does: Z-check syndromes decode X
errors (H_Z e_X = s_Z) and X-check syndromes decode Z errors (H_X e_Z = s_X), returning the
correction pair (X, Z).  ``AsyncDecoderService`` micro-batches syndromes arriving on an asyncio
queue (up to one SIMD pass = ``lanes`` shots, or ``max_wait_us``), decodes them off the event loop
in worker threads (cffi releases the GIL) and tracks queue backlog and end-to-end latency, so a
decoding backlog is observable rather than silent.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from algebra.qldpc import CSSCode

from .fast import FastDecoder


@dataclass
class Correction:
    x: np.ndarray            # X-type Pauli correction on data qubits
    z: np.ndarray            # Z-type Pauli correction on data qubits
    ok: bool                 # both decoders produced a syndrome-consistent estimate
    latency_ns: int = 0      # submit -> result (end to end, includes queueing)
    kernel_ns: int = 0       # decoder-kernel time only (C++ measured)


class PauliCorrector:
    def __init__(self, code: CSSCode, p: float = 0.01, max_iter: int = 20):
        self.code = code
        self.dx = FastDecoder(code.hz, p, max_iter=max_iter)   # decodes X errors from Z-check syndrome
        self.dz = FastDecoder(code.hx, p, max_iter=max_iter)   # decodes Z errors from X-check syndrome

    def decode(self, sx: np.ndarray, sz: np.ndarray) -> list[Correction]:
        """sx: (B, m_x) X-check syndromes; sz: (B, m_z) Z-check syndromes."""
        rx = self.dx.decode_batch(sz)
        rz = self.dz.decode_batch(sx)
        out = []
        for i in range(rx.errors.shape[0]):
            out.append(Correction(rx.errors[i], rz.errors[i], bool(rx.status[i] != 3 and rz.status[i] != 3),
                                  0, int(rx.latency_ns[i] + rz.latency_ns[i])))
        return out


@dataclass
class ServiceStats:
    submitted: int = 0
    completed: int = 0
    max_backlog: int = 0
    latencies_ns: list = field(default_factory=list)
    kernel_ns: list = field(default_factory=list)


class AsyncDecoderService:
    def __init__(self, corrector: PauliCorrector, workers: int = 2, max_wait_us: float = 20.0):
        self.c = corrector
        self.max_batch = corrector.dx.lanes
        self.max_wait = max_wait_us * 1e-6
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.q: asyncio.Queue = asyncio.Queue()
        self.stats = ServiceStats()
        self._task: asyncio.Task | None = None

    async def start(self):
        self._task = asyncio.create_task(self._run())

    async def stop(self):
        await self.q.put(None)
        if self._task:
            await self._task
        self.pool.shutdown()

    async def submit(self, sx: np.ndarray, sz: np.ndarray) -> Correction:
        fut = asyncio.get_running_loop().create_future()
        self.stats.submitted += 1
        await self.q.put((time.perf_counter_ns(), sx, sz, fut))
        self.stats.max_backlog = max(self.stats.max_backlog, self.q.qsize())
        return await fut

    async def _run(self):
        loop = asyncio.get_running_loop()
        stop = False
        while not stop:
            item = await self.q.get()
            if item is None:
                break
            batch = [item]
            deadline = time.perf_counter() + self.max_wait
            while len(batch) < self.max_batch:
                timeout = deadline - time.perf_counter()
                try:
                    nxt = self.q.get_nowait() if timeout <= 0 else await asyncio.wait_for(self.q.get(), timeout)
                except (asyncio.QueueEmpty, asyncio.TimeoutError):
                    break
                if nxt is None:
                    stop = True
                    break
                batch.append(nxt)
            sx = np.stack([b[1] for b in batch]).astype(np.uint8)
            sz = np.stack([b[2] for b in batch]).astype(np.uint8)
            corr = await loop.run_in_executor(self.pool, self.c.decode, sx, sz)
            now = time.perf_counter_ns()
            for (t0, _, _, fut), c in zip(batch, corr):
                c.latency_ns = now - t0
                self.stats.latencies_ns.append(c.latency_ns)
                self.stats.kernel_ns.append(c.kernel_ns)
                self.stats.completed += 1
                fut.set_result(c)

    async def stream(self, source):
        """Consume an async iterator of (sx, sz) and yield Corrections in order."""
        pending: deque = deque()
        async for sx, sz in source:
            pending.append(asyncio.create_task(self.submit(sx, sz)))
            while pending and pending[0].done():
                yield pending.popleft().result()
        while pending:
            yield await pending.popleft()
