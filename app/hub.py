"""Per-lane pending-frame hub with one global FIFO queue.

Semantics (one instance shared by API handlers and the inference worker):

- Fixed set of lanes; each lane holds at most ONE pending frame.
- A submitted frame is accepted only if its frame number is greater than
  the lane's maximum received frame number; otherwise it is rejected.
- A newer frame replaces the lane's pending frame *in place*: the lane
  keeps its original position in the global queue.
- The single inference worker pops up to ``batch_lanes`` lanes from the
  front of the queue and computes one frame per lane. A lane that was
  popped re-enters the queue (at the tail) only when a newer frame is
  submitted for it.
- Results always carry the frame number that was actually computed.
  A failed batch keeps each lane's previous successful result.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class PendingFrame:
    frame_no: int
    image: bytes
    received_at: float


@dataclass(slots=True)
class LaneResult:
    frame_no: int
    class_id: int
    scores: list[float]
    computed_at: float


@dataclass(slots=True)
class Lane:
    name: str
    max_received: int | None = None
    pending: PendingFrame | None = None
    queued: bool = False
    in_batch: bool = False
    last_result: LaneResult | None = None
    replaced_frames: int = 0  # pending frames overwritten before ever computed
    failed_frames: int = 0    # frames lost to failed batches


class FrameRejected(Exception):
    """Frame number did not exceed the lane's max received frame number."""


class UnknownLane(Exception):
    pass


class Hub:
    def __init__(self, lane_names: list[str] | tuple[str, ...], batch_lanes: int = 2):
        if not lane_names:
            raise ValueError("at least one lane is required")
        if batch_lanes < 1:
            raise ValueError("batch_lanes must be >= 1")
        self.lanes: dict[str, Lane] = {name: Lane(name) for name in lane_names}
        self.batch_lanes = batch_lanes
        self._queue: deque[str] = deque()  # lane names, front == left
        self._cond = asyncio.Condition()

    def _lane(self, lane: str) -> Lane:
        try:
            return self.lanes[lane]
        except KeyError:
            raise UnknownLane(lane) from None

    # ------------------------------------------------------------------
    # producer side (API handlers)
    # ------------------------------------------------------------------
    async def submit(self, lane: str, frame_no: int, image: bytes) -> dict:
        """Accept or reject a frame. Returns queue metadata for the lane."""
        async with self._cond:
            L = self._lane(lane)
            if L.max_received is not None and frame_no <= L.max_received:
                raise FrameRejected(
                    f"frame_no {frame_no} <= max received {L.max_received} on {lane}"
                )
            replaced = L.pending is not None
            if replaced:
                L.replaced_frames += 1
            L.max_received = frame_no
            L.pending = PendingFrame(frame_no=frame_no, image=image,
                                     received_at=time.time())
            if not L.queued:
                L.queued = True
                self._queue.append(lane)
            position = list(self._queue).index(lane)
            self._cond.notify_all()
            return {
                "replaced": replaced,
                "in_batch": L.in_batch,
                "queue_position": position,
                "queue": list(self._queue),
            }

    # ------------------------------------------------------------------
    # consumer side (single inference worker)
    # ------------------------------------------------------------------
    async def take_batch(self) -> list[tuple[str, PendingFrame]]:
        """Wait for work, then pop up to ``batch_lanes`` lanes from the front."""
        async with self._cond:
            while not self._queue:
                await self._cond.wait()
            batch: list[tuple[str, PendingFrame]] = []
            while self._queue and len(batch) < self.batch_lanes:
                name = self._queue.popleft()
                L = self.lanes[name]
                L.queued = False
                if L.pending is None:  # defensive: queued implies pending
                    continue
                L.in_batch = True
                batch.append((name, L.pending))
                L.pending = None
            return batch

    async def finish_batch(
        self,
        batch: list[tuple[str, PendingFrame]],
        outputs: list[tuple[int, list[float]]],
    ) -> None:
        """Store per-lane results tagged with the actually computed frame."""
        async with self._cond:
            for (name, pending), (class_id, scores) in zip(batch, outputs):
                L = self.lanes[name]
                L.in_batch = False
                L.last_result = LaneResult(
                    frame_no=pending.frame_no,
                    class_id=class_id,
                    scores=scores,
                    computed_at=time.time(),
                )

    async def fail_batch(self, batch: list[tuple[str, PendingFrame]]) -> None:
        """Batch failed: keep previous results, drop the failed frames."""
        async with self._cond:
            for name, _ in batch:
                L = self.lanes[name]
                L.in_batch = False
                L.failed_frames += 1

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    async def result(self, lane: str) -> LaneResult | None:
        async with self._cond:
            return self._lane(lane).last_result

    async def snapshot(self) -> dict:
        async with self._cond:
            queue = list(self._queue)
            lanes = {}
            for name, L in self.lanes.items():
                lanes[name] = {
                    "max_received": L.max_received,
                    "has_pending": L.pending is not None,
                    "pending_frame_no": L.pending.frame_no if L.pending else None,
                    "queued": L.queued,
                    "queue_position": queue.index(name) if L.queued else None,
                    "in_batch": L.in_batch,
                    "last_result_frame_no": (
                        L.last_result.frame_no if L.last_result else None
                    ),
                    "replaced_frames": L.replaced_frames,
                    "failed_frames": L.failed_frames,
                }
            return {"queue": queue, "batch_lanes": self.batch_lanes, "lanes": lanes}
