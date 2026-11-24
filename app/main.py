"""FastAPI service: ingest 64x64 grayscale frames from 3 fixed cameras,
classify them on CPU with a local ONNX model, and serve per-lane results.

Endpoints:
  POST /cameras/{lane}/frames?frame_no=N   body = raw 4096-byte gray image
  GET  /cameras/{lane}/result              last successful classification
  GET  /cameras                            queue / lane overview
  GET  /healthz

Two inference channels (configurable via INFER_CHANNELS) run as separate
asyncio workers but share one loaded ONNX CPU session. At most that many
batches are computed concurrently. A free channel takes a batch from the
front of the shared queue; a lane whose earlier frame is still being
computed can be queued again and picked up by the other channel.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .hub import FrameRejected, Hub, UnknownLane
from .model import IMAGE_BYTES, OnnxClassifier

log = logging.getLogger("vision")
if not log.handlers:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

LANES = tuple(
    name.strip()
    for name in os.environ.get("LANES", "cam0,cam1,cam2").split(",")
    if name.strip()
)
BATCH_LANES = int(os.environ.get("BATCH_LANES", "2"))
INFER_CHANNELS = int(os.environ.get("INFER_CHANNELS", "2"))
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "model", "classifier.onnx"),
)
INFER_DELAY_MS = int(os.environ.get("INFER_DELAY_MS", "0"))
# Per-frame delay header is a demo knob; clamp it to a sane maximum.
MAX_HINT_DELAY_MS = int(os.environ.get("MAX_HINT_DELAY_MS", "10000"))


async def inference_loop(
    channel: int, hub: Hub, model: OnnxClassifier, default_delay_ms: int
) -> None:
    """One inference channel: waits for a free slot, takes a front batch."""
    while True:
        batch = await hub.take_batch()
        # Per-frame demo overrides win over the service-wide default; the
        # slowest frame in the batch sets the batch delay.
        hints = [p.infer_delay_ms for _, p in batch if p.infer_delay_ms is not None]
        delay_ms = max(hints) if hints else default_delay_ms
        try:
            outputs = await model.infer(
                [p.image for _, p in batch], delay_ms=delay_ms
            )
        except Exception:
            # keep previous results, drop failed frames, keep draining the queue
            log.exception(
                "channel %d inference batch failed for lanes %s",
                channel,
                [name for name, _ in batch],
            )
            await hub.fail_batch(batch)
            continue
        await hub.finish_batch(batch, outputs)
        log.info(
            "channel %d batch done (delay_ms=%s): %s",
            channel,
            delay_ms,
            [(name, p.frame_no) for name, p in batch],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub = Hub(LANES, batch_lanes=BATCH_LANES, channels=INFER_CHANNELS)
    # One model session is loaded once and shared by every channel.
    model = OnnxClassifier(MODEL_PATH, infer_delay_ms=INFER_DELAY_MS)
    workers = [
        asyncio.create_task(
            inference_loop(i, hub, model, INFER_DELAY_MS),
            name=f"inference-channel-{i}",
        )
        for i in range(INFER_CHANNELS)
    ]
    app.state.hub = hub
    app.state.model = model
    log.info(
        "service up: lanes=%s batch_lanes=%d channels=%d model=%s",
        LANES,
        BATCH_LANES,
        INFER_CHANNELS,
        MODEL_PATH,
    )
    try:
        yield
    finally:
        for worker in workers:
            worker.cancel()
        for worker in workers:
            try:
                await worker
            except asyncio.CancelledError:
                pass


app = FastAPI(title="vision-edge", version="1.1.0", lifespan=lifespan)


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
    x_infer_delay_ms: int | None = Header(
        default=None,
        ge=0,
        alias="X-Infer-Delay-Ms",
        description="demo knob: force this frame's batch to sleep N ms on CPU",
    ),
):
    if len(body) != IMAGE_BYTES:
        raise HTTPException(
            status_code=422,
            detail=f"expected {IMAGE_BYTES} raw gray bytes, got {len(body)}",
        )
    if x_infer_delay_ms is not None and x_infer_delay_ms > MAX_HINT_DELAY_MS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"X-Infer-Delay-Ms {x_infer_delay_ms} exceeds limit "
                f"{MAX_HINT_DELAY_MS}"
            ),
        )
    meta = await request.app.state.hub.submit(
        lane, frame_no, body, infer_delay_ms=x_infer_delay_ms
    )
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
