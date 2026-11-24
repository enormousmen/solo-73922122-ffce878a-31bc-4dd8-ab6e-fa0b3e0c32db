"""Unit tests for the Hub state machine (no plugins, plain asyncio)."""

import asyncio
import unittest
from collections import deque

from app.hub import FrameRejected, Hub, UnknownLane

IMG = b"\x00" * 4096


def img(seed: int) -> bytes:
    return bytes([seed % 256]) * 4096


class HubTest(unittest.IsolatedAsyncioTestCase):
    async def test_reject_stale_frame_numbers(self):
        hub = Hub(["a", "b", "c"])
        await hub.submit("a", 5, img(5))
        with self.assertRaises(FrameRejected):
            await hub.submit("a", 5, img(5))  # equal -> reject
        with self.assertRaises(FrameRejected):
            await hub.submit("a", 4, img(4))  # lower -> reject
        meta = await hub.submit("a", 6, img(6))
        self.assertTrue(meta["replaced"])  # frame 5 was still pending
        self.assertEqual(hub.lanes["a"].max_received, 6)

    async def test_unknown_lane(self):
        hub = Hub(["a"])
        with self.assertRaises(UnknownLane):
            await hub.submit("zzz", 1, IMG)
        with self.assertRaises(UnknownLane):
            await hub.result("zzz")

    async def test_replace_keeps_queue_position(self):
        hub = Hub(["a", "b", "c"])
        await hub.submit("a", 1, img(1))
        await hub.submit("b", 1, img(1))
        await hub.submit("c", 1, img(1))
        meta = await hub.submit("a", 2, img(2))  # replace a's pending frame
        self.assertTrue(meta["replaced"])
        self.assertEqual(meta["queue_position"], 0)  # original slot kept
        self.assertEqual(meta["queue"], ["a", "b", "c"])

        batch = await hub.take_batch()
        self.assertEqual([n for n, _ in batch], ["a", "b"])
        self.assertEqual(batch[0][1].frame_no, 2)  # the replacement is computed
        self.assertEqual(batch[1][1].frame_no, 1)

    async def test_batch_takes_front_two_lanes(self):
        hub = Hub(["a", "b", "c"], batch_lanes=2)
        for lane in ("a", "b", "c"):
            await hub.submit(lane, 1, img(1))
        batch = await hub.take_batch()
        self.assertEqual([n for n, _ in batch], ["a", "b"])
        batch2 = await hub.take_batch()
        self.assertEqual([n for n, _ in batch2], ["c"])

    async def test_single_waiting_lane_computes_one(self):
        hub = Hub(["a", "b", "c"])
        await hub.submit("b", 7, img(7))
        batch = await hub.take_batch()
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0][0], "b")
        self.assertEqual(batch[0][1].frame_no, 7)

    async def test_lane_requeues_after_being_taken(self):
        hub = Hub(["a", "b"])
        await hub.submit("a", 1, img(1))
        batch = await hub.take_batch()
        self.assertEqual([n for n, _ in batch], ["a"])
        self.assertTrue(hub.lanes["a"].in_batch)
        # new frame for the taken lane goes back into the queue
        meta = await hub.submit("a", 2, img(2))
        self.assertFalse(meta["replaced"])  # nothing pending while in batch
        self.assertEqual(meta["queue_position"], 0)
        batch2 = await hub.take_batch()
        self.assertEqual(batch2[0][1].frame_no, 2)

    async def test_finish_batch_stores_actual_frame(self):
        hub = Hub(["a"])
        await hub.submit("a", 3, img(3))
        batch = await hub.take_batch()
        await hub.finish_batch(batch, [(1, [0.1, 0.8, 0.1])])
        res = await hub.result("a")
        self.assertEqual(res.frame_no, 3)
        self.assertEqual(res.class_id, 1)
        self.assertFalse(hub.lanes["a"].in_batch)

    async def test_failed_batch_keeps_previous_result(self):
        hub = Hub(["a"])
        await hub.submit("a", 1, img(1))
        batch = await hub.take_batch()
        await hub.finish_batch(batch, [(0, [0.9, 0.05, 0.05])])
        good = await hub.result("a")

        await hub.submit("a", 2, img(2))
        batch2 = await hub.take_batch()
        await hub.fail_batch(batch2)
        res = await hub.result("a")
        self.assertEqual(res.frame_no, good.frame_no)  # untouched
        self.assertEqual(res.class_id, good.class_id)
        self.assertEqual(hub.lanes["a"].failed_frames, 1)
        # hub keeps accepting work afterwards
        await hub.submit("a", 3, img(3))
        batch3 = await hub.take_batch()
        self.assertEqual(batch3[0][1].frame_no, 3)

    async def test_take_batch_waits_for_work(self):
        hub = Hub(["a"])
        taken = []

        async def consumer():
            taken.append(await hub.take_batch())

        task = asyncio.create_task(consumer())
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())  # blocked, queue empty
        await hub.submit("a", 1, img(1))
        await asyncio.wait_for(task, timeout=1)
        self.assertEqual(taken[0][0][0], "a")

    async def test_same_lane_can_be_taken_twice_concurrently(self):
        # two channels: the lane's earlier frame stays in flight while a
        # newer frame is queued again and taken by the free channel.
        hub = Hub(["a", "b"], channels=2)
        await hub.submit("a", 1, img(1))
        batch1 = await hub.take_batch()
        self.assertEqual(batch1[0][1].frame_no, 1)
        self.assertTrue(hub.lanes["a"].in_batch)
        self.assertEqual(hub.lanes["a"].in_flight, 1)

        meta = await hub.submit("a", 2, img(2))
        self.assertFalse(meta["replaced"])  # no pending slot while frame 1 runs
        self.assertTrue(meta["in_batch"])   # still in batch via frame 1
        self.assertEqual(meta["in_flight"], 1)
        batch2 = await hub.take_batch()
        self.assertEqual([n for n, _ in batch2], ["a"])
        self.assertEqual(batch2[0][1].frame_no, 2)
        self.assertEqual(hub.lanes["a"].in_flight, 2)

        # last frame finishes first: in_batch stays true until frame 1 ends
        await hub.finish_batch(batch2, [(1, [0.1, 0.8, 0.1])])
        self.assertTrue(hub.lanes["a"].in_batch)
        self.assertEqual(hub.lanes["a"].in_flight, 1)
        await hub.finish_batch(batch1, [(0, [0.9, 0.05, 0.05])])
        self.assertFalse(hub.lanes["a"].in_batch)
        self.assertEqual(hub.lanes["a"].in_flight, 0)

    async def test_newer_frame_publishes_first_older_goes_stale(self):
        # slow earlier batch finishes after the newer one: the newer result
        # is queryable immediately and the late old frame never overwrites it.
        hub = Hub(["a"], channels=2)
        await hub.submit("a", 1, img(1))
        batch1 = await hub.take_batch()
        await hub.submit("a", 2, img(2))
        batch2 = await hub.take_batch()

        await hub.finish_batch(batch2, [(2, [0.1, 0.1, 0.8])])
        res = await hub.result("a")
        self.assertEqual(res.frame_no, 2)  # newer frame published at once
        self.assertEqual(hub.lanes["a"].stale_frames, 0)

        await hub.finish_batch(batch1, [(0, [0.9, 0.05, 0.05])])
        res = await hub.result("a")
        self.assertEqual(res.frame_no, 2)  # old frame must not overwrite
        self.assertEqual(hub.lanes["a"].stale_frames, 1)

    async def test_failure_keeps_result_and_other_inflight_frame(self):
        # the newer frame fails: previous result survives and the earlier
        # in-flight frame is still published normally when it finishes.
        hub = Hub(["a"], channels=2)
        await hub.submit("a", 1, img(1))
        batch1 = await hub.take_batch()
        await hub.finish_batch(batch1, [(0, [0.9, 0.05, 0.05])])

        await hub.submit("a", 2, img(2))
        batch2 = await hub.take_batch()
        await hub.submit("a", 3, img(3))
        batch3 = await hub.take_batch()

        await hub.fail_batch(batch3)  # frame 3 fails
        res = await hub.result("a")
        self.assertEqual(res.frame_no, 1)  # previous result untouched
        self.assertEqual(hub.lanes["a"].failed_frames, 1)
        self.assertTrue(hub.lanes["a"].in_batch)  # frame 2 still computing

        await hub.finish_batch(batch2, [(1, [0.1, 0.8, 0.1])])
        res = await hub.result("a")
        self.assertEqual(res.frame_no, 2)
        self.assertFalse(hub.lanes["a"].in_batch)

    async def test_two_channels_cap_concurrent_batches(self):
        # with two channels a third take_batch blocks until a batch finishes.
        hub = Hub(["a", "b", "c", "d"], channels=2)
        for lane in ("a", "b", "c", "d"):
            await hub.submit(lane, 1, img(1))
        b1 = await hub.take_batch()  # channels take a,b ...
        b2 = await hub.take_batch()  # ... then c,d: both channels now busy
        self.assertEqual(len(b1), 2)
        self.assertEqual(len(b2), 2)

        await hub.submit("a", 2, img(2))  # re-queued behind nobody
        self.assertEqual(hub._queue, deque(["a"]))

        async def grab():
            return await hub.take_batch()

        task = asyncio.create_task(grab())
        await asyncio.sleep(0.05)
        self.assertFalse(task.done())  # both channels busy: third batch waits
        await hub.finish_batch(b1, [(0, [1.0, 0.0, 0.0])] * len(b1))
        b3 = await asyncio.wait_for(asyncio.shield(task), timeout=1)
        self.assertEqual([n for n, _ in b3], ["a"])
        self.assertEqual(b3[0][1].frame_no, 2)
        await hub.finish_batch(b2, [(0, [1.0, 0.0, 0.0])] * len(b2))
        await hub.finish_batch(b3, [(0, [1.0, 0.0, 0.0])])
        self.assertEqual(hub.lanes["a"].in_flight, 0)

    async def test_per_frame_delay_hint_is_carried(self):
        hub = Hub(["a", "b"])
        await hub.submit("a", 1, img(1), infer_delay_ms=500)
        await hub.submit("b", 1, img(1))
        batch = await hub.take_batch()
        hints = {name: p.infer_delay_ms for name, p in batch}
        self.assertEqual(hints["a"], 500)
        self.assertIsNone(hints["b"])


if __name__ == "__main__":
    unittest.main()
