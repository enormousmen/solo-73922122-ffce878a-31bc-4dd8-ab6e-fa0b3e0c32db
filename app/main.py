"""FastAPI service: ingest 64x64 grayscale frames from 3 fixed cameras,
classify them on CPU with a local ONNX model, and serve per-lane results.

Endpoints:
  POST /cameras/{lane}/frames?frame_no=N   body = raw 4096-byte gray image
  GET  /cameras/{lane}/result              last successful classification
  GET  /cameras                            queue / lane overview
  GET  /healthz
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .hub import FrameRejected, Hub, UnknownLane
from .model import IMAGE_BYTES, OnnxClassifier

log = logging.getLogger("vision")

LANES = tuple(
    name.strip()
    for name in os.environ.get("LANES", "cam0,cam1,cam2").split(",")
    if name.strip()
)
BATCH_LANES = int(os.environ.get("BATCH_LANES", "2"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "classifier.onnx"),
)
INFER_DELAY_MS = int(os.environ.get("INFER_DELAY_MS", "0"))


async def inference_loop(hub: Hub, model: OnnxClassifier) -> None:
    """Single inference channel: front-most lanes, one frame each."""
    while True:
        batch = await hub.take_batch()
        try:
            outputs = await model.infer([p.image for _, p in batch])
        except Exception:
            # keep previous results, drop failed frames, keep draining the queue
            log.exception("inference batch failed for lanes %s",
                          [name for name, _ in batch])
            await hub.fail_batch(batch)
            continue
        await hub.finish_batch(batch, outputs)
        log.info("batch done: %s",
                 [(name, p.frame_no) for name, p in batch])


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = Hub(LANES, batch_lanes=BATCH_LANES)
    model = OnnxClassifier(MODEL_PATH, infer_delay_ms=INFER_DELAY_MS)
    worker = asyncio.create_task(inference_loop(hub, model))
    app.state.hub = hub
    app.state.model = model
    log.info("service up: lanes=%s batch_lanes=%d model=%s",
             LANES, BATCH_LANES, MODEL_PATH)
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


app = FastAPI(title="vision-edge", version="1.0.0", lifespan=lifespan)


@app.exception_handler(UnknownLane)
async def unknown_lane_handler(_: Request, exc: UnknownLane):
    return JSONResponse(
        status_code=404,
        content={"detail": f"unknown lane {exc.args[0]!r}", "lanes": list(LANES)},
    )


@app.exception_handler(FrameRejected)
async def frame_rejected_handler(_: Request, exc: FrameRejected):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/cameras/{lane}/frames", status_code=202)
async def submit_frame(
    request: Request,
    lane: str,
    frame_no: int = Query(..., ge=0, description="monotonic per-lane frame number"),
    body: bytes = Body(..., media_type="application/octet-stream"),
):
    if len(body) != IMAGE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"expected {IMAGE_BYTES} raw gray bytes, got {len(body)}",
        )
    meta = await request.app.state.hub.submit(lane, frame_no, body)
    return {"lane": lane, "frame_no": frame_no, "accepted": True, **meta}


@app.get("/cameras/{lane}/result")
async def get_result(request: Request, lane: str):
    res = await request.app.state.hub.result(lane)
    if res is None:
        return {"lane": lane, "frame_no": None, "class_id": None,
                "scores": None, "computed_at": None}
    return {
        "lane": lane,
        "frame_no": res.frame_no,
        "class_id": res.class_id,
        "scores": res.scores,
        "computed_at": res.computed_at,
    }


@app.get("/cameras")
async def get_cameras(request: Request):
    return await request.app.state.hub.snapshot()
