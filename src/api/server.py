"""FastAPI + WebSocket server.   python -m uvicorn api.server:app --app-dir src --port 8000"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import services as svc
from .stream import Config, DecodeSession

UI_DIR = svc.ROOT / "ui"

app = FastAPI(title="FTQC qLDPC API", version="1.0.0",
              description="BB-code catalog, OpenQASM 3.0 compiler, measured benchmarks, and live SIMD decode telemetry.")


class CompileRequest(BaseModel):
    code: str
    schedule: Literal["interleaved", "sequential"] = "interleaved"
    rounds: int = Field(3, ge=1, le=12)
    basis: Literal["Z", "X"] = "Z"
    p: float = Field(1e-3, gt=0, lt=0.5)
    t1_us: float | None = Field(None, gt=0)
    t2_us: float | None = Field(None, gt=0)
    layer_time_us: float = Field(0.05, gt=0)
    decoder: Literal["full", "window"] = "full"
    W: int = Field(3, ge=1, le=12)
    F: int = Field(1, ge=1, le=12)


@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "codes": svc.CODE_NAMES}


@app.get("/api/v1/codes")
async def codes(tanner: bool = Query(True)):
    return await run_in_threadpool(svc.catalog, tanner)


@app.post("/api/v1/compile")
async def compile_(req: CompileRequest):
    if req.code not in svc.CODE_NAMES:
        raise HTTPException(404, f"unknown code {req.code!r}; choose from {svc.CODE_NAMES}")
    if (req.t1_us is None) != (req.t2_us is None):
        raise HTTPException(422, "provide both t1_us and t2_us, or neither")
    try:
        return await run_in_threadpool(svc.compile_request, req.code, req.schedule, req.rounds, req.basis,
                                       req.p, req.t1_us, req.t2_us, req.layer_time_us, req.decoder, req.W, req.F)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))


@app.get("/api/v1/benchmarks")
async def benchmarks():
    try:
        return await run_in_threadpool(svc.benchmarks)
    except FileNotFoundError as e:
        raise HTTPException(503, f"benchmark data missing: {e}")


@app.get("/api/v1/hardware")
async def hardware():
    return await run_in_threadpool(svc.hardware)


@app.get("/api/v1/rtl/vector/{index}")
async def rtl_vector(index: int):
    try:
        return await run_in_threadpool(svc.rtl_vector, index)
    except IndexError:
        raise HTTPException(404, "vector index out of range")
    except FileNotFoundError as e:
        raise HTTPException(503, f"RTL results missing: {e}")


@app.get("/api/v1/rtl/file/{name}")
async def rtl_file(name: str):
    if name not in svc.RTL_FILES:                      # whitelist: nothing else under exports/ is ever served
        raise HTTPException(404, "unknown file")
    path = svc.ROOT / "exports" / "rtl" / name
    if not path.exists():
        raise HTTPException(404, "file not generated yet")
    return FileResponse(path, media_type="text/plain; charset=utf-8", filename=name)


@app.websocket("/api/v1/stream/decode")
async def stream_decode(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    cfg = Config()
    session: DecodeSession | None = None
    pending: asyncio.Queue = asyncio.Queue()

    async def reader():
        try:
            while True:
                await pending.put(json.loads(await ws.receive_text()))
        except (WebSocketDisconnect, RuntimeError):
            await pending.put(None)
        except ValueError:
            await pending.put({"__bad_json__": True})

    task = asyncio.create_task(reader())
    try:
        msg = {}
        while True:
            # apply the latest queued config (if any) and rebuild the session
            changed = session is None
            while not pending.empty():
                msg = pending.get_nowait()
                if msg is None:
                    return
                if msg.get("__bad_json__"):
                    await ws.send_json({"type": "error", "message": "invalid JSON"})
                    continue
                try:
                    cfg = Config.from_msg({**(cfg_to_dict(cfg)), **msg})
                    changed = True
                except ValueError as e:
                    await ws.send_json({"type": "error", "message": str(e)})
            if changed:
                session = await run_in_threadpool(DecodeSession, cfg)
                await ws.send_json(session.hello())
            frame = await loop.run_in_executor(None, session.tick)
            await ws.send_json(frame)
            await asyncio.sleep(1.0 / cfg.rate_hz)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        task.cancel()


def cfg_to_dict(cfg: Config) -> dict:
    from .stream import cfg_dict
    return cfg_dict(cfg)


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=UI_DIR), name="ui")



@app.get("/", response_class=HTMLResponse)
async def index():
    f = UI_DIR / "index.html"
    if not f.exists():
        return HTMLResponse("<h1>FTQC qLDPC API</h1><p>UI not built yet. See <a href='/docs'>/docs</a>.</p>")
    return FileResponse(f, media_type="text/html")
