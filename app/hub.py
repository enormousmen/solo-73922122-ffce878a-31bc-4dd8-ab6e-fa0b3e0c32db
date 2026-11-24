"""Per-lane pending-frame hub with one global FIFO queue and N channels.

Semantics (one instance shared by API handlers and the inference workers):

- Fixed set of lanes; each lane holds at most ONE pending frame.
- A submitted frame is accepted only if its frame number is greater than
  the lane's maximum received frame number; otherwise it is rejected.
- A newer frame replaces the lane's pending frame *in place*: the lane
  keeps its original position in the global queue.
- Up to ``channels`` inference workers (two by default) share one loaded
  CPU model. A free worker pops up to ``batch_lanes`` lanes from the front
  of the queue and computes one frame per lane, so at most ``channels``
  batches are ever in flight at the same time.
- A popped lane re-enters the queue (at the tail) as soon as a newer frame
  arrives, even while an earlier frame of the same lane is still being
  computed in another batch. A lane can therefore have up to ``channels``
  frames in flight concurrently, but still at most ONE pending frame.
- Results always carry the frame number that was actually computed. A
  computed frame is published only when it is newer than the lane's last
  published result, so a slow earlier batch can never overwrite a newer
  frame's result ("success out of order, publish in order of frame_no").
  Frames that finish after a newer result was already published are
  counted as ``stale_frames``.
- A failed batch keeps each lane's previous successful result; frames
  submitted earlier and already in another batch are unaffected and are
  still published under the same rule when they finish.
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
    infer_delay_ms: int | None = None  # per-frame demo override


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
    in_flight: int = 0  # frames of this lane currently being computed
    last_result: LaneResult | None = None
    replaced_frames: int = 0   # pending frames overwritten before ever computed
    failed_frames: int = 0     # frames lost to failed batches
    stale_frames: int = 0      # computed frames superseded by a newer published result

    @property
    def in_batch(self) -> bool:
        """True while the lane has at least one frame in a running batch."""
        return self.in_flight > 0


class FrameRejected(Exception):
    """Frame number did not exceed the lane's max received frame number."""


class UnknownLane(Exception):
    pass


class Hub:
    def __init__(
        self,
        lane_names: list[str] | tuple[str, ...],
        batch_lanes: int = 2,
        channels: int = 2,
    ):
        if not lane_names:
            raise ValueError("at least one lane is required")
        if batch_lanes < 1:
            raise ValueError("batch_lanes must be >= 1")
        if channels < 1:
            raise ValueError("channels must be >= 1")
        self.lanes: dict[str, Lane] = {name: Lane(name) for name in lane_names}
        self.batch_lanes = batch_lanes
        self.channels = channels
        self._queue: deque[str] = deque()  # lane names, front == left
        self._active_batches = 0  # batches currently computed by the channels
        self._cond = asyncio.Condition()

    def _lane(self, lane: str) -> Lane:
        try:
            return self.lanes[lane]
        except KeyError:
            raise UnknownLane(lane) from None

    # ------------------------------------------------------------------
    # producer side (API handlers)
    # ------------------------------------------------------------------
    async def submit(
        self,
        lane: str,
        frame_no: int,
        image: bytes,
        infer_delay_ms: int | None = None,
    ) -> dict:
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
            L.pending = PendingFrame(
                frame_no=frame_no,
                image=image,
                received_at=time.time(),
                infer_delay_ms=infer_delay_ms,
            )
            if not L.queued:
                # A lane whose earlier frame is still in flight re-enters the
                # queue at the tail just like any other lane.
                L.queued = True
                self._queue.append(lane)
            position = list(self._queue).index(lane)
            self._cond.notify_all()
            return {
                "replaced": replaced,
                "in_batch": L.in_batch,
                "in_flight": L.in_flight,
                "queue_position": position,
                "queue": list(self._queue),
            }

    # ------------------------------------------------------------------
    # consumer side (one inference worker per channel; may run concurrently)
    # ------------------------------------------------------------------
    async def take_batch(self) -> list[tuple[str, PendingFrame]]:
        """Wait for a free channel and work, then pop up to ``batch_lanes``
        lanes from the front.

        The condition lock makes the whole pop atomic: two free channels can
        never grab the same queued lane, and a third channel blocks here
        until one of the two running batches finishes.
        """
        async with self._cond:
            while self._active_batches >= self.channels or not self._queue:
                await self._cond.wait()
            batch: list[tuple[str, PendingFrame]] = []
            while self._queue and len(batch) < self.batch_lanes:
                name = self._queue.popleft()
                L = self.lanes[name]
                L.queued = False
                if L.pending is None:  # defensive: queued implies pending
                    continue
                L.in_flight += 1
                batch.append((name, L.pending))
                L.pending = None
            self._active_batches += 1
            return batch

    async def finish_batch(
        self,
        batch: list[tuple[str, PendingFrame]],
        outputs: list[tuple[int, list[float]]],
    ) -> None:
        """Publish per-lane results tagged with the actually computed frame.

        A frame is published only if it is newer than the lane's current
        result; older frames finishing late are counted as stale and never
        overwrite a newer published result.
        """
        async with self._cond:
            for (name, pending), (class_id, scores) in zip(batch, outputs):
                L = self.lanes[name]
                L.in_flight = max(0, L.in_flight - 1)
                if L.last_result is None or pending.frame_no > L.last_result.frame_no:
                    L.last_result = LaneResult(
                        frame_no=pending.frame_no,
                        class_id=class_id,
                        scores=scores,
                        computed_at=time.time(),
                    )
                else:
                    # A newer frame was already published while this one was
                    # still being computed (slow earlier batch).
                    L.stale_frames += 1
            self._active_batches = max(0, self._active_batches - 1)
            self._cond.notify_all()

    async def fail_batch(self, batch: list[tuple[str, PendingFrame]]) -> None:
        """Batch failed: keep previous results, drop the failed frames.

        Earlier frames of the same lane running in another batch keep
        computing and are still published normally when they finish.
        """
        async with self._cond:
            for name, _ in batch:
                L = self.lanes[name]
                L.in_flight = max(0, L.in_flight - 1)
                L.failed_frames += 1
            self._active_batches = max(0, self._active_batches - 1)
            self._cond.notify_all()

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
                    "in_flight": L.in_flight,
                    "last_result_frame_no": (
                        L.last_result.frame_no if L.last_result else None
                    ),
                    "replaced_frames": L.replaced_frames,
                    "failed_frames": L.failed_frames,
                    "stale_frames": L.stale_frames,
                }
            return {
                "queue": queue,
                "batch_lanes": self.batch_lanes,
                "channels": self.channels,
                "active_batches": self._active_batches,
                "lanes": lanes,
            }
