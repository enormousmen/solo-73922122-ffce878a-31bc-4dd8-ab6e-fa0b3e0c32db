"""End-to-end API tests: ingestion rules, queueing, results, failure path."""

import time
import unittest

from fastapi.testclient import TestClient

from app.main import app

LANES = ("cam0", "cam1", "cam2")


def frame(seed: int) -> bytes:
    return bytes((seed + i) % 256 for i in range(4096))


def post(client, lane, frame_no, body):
    return client.post(
        f"/cameras/{lane}/frames",
        params={"frame_no": frame_no},
        content=body,
        headers={"Content-Type": "application/octet-stream"},
    )


def wait_result(client, lane, want_frame=None, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/cameras/{lane}/result").json()
        got = r["frame_no"]
        if got is not None and (want_frame is None or got >= want_frame):
            return r
        time.sleep(0.02)
    raise AssertionError(f"no result for {lane} within {timeout}s")


def wait_lane_state(client, lane, key, want, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = client.get("/cameras").json()
        if snap["lanes"][lane][key] == want:
            return snap
        time.sleep(0.02)
    raise AssertionError(f"{lane}.{key} never became {want!r}")


class ApiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._ctx = TestClient(app)
        cls.client = cls._ctx.__enter__()  # runs lifespan: hub + model + worker

    @classmethod
    def tearDownClass(cls):
        cls._ctx.__exit__(None, None, None)

    def test_01_health(self):
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_02_reject_wrong_size(self):
        r = post(self.client, "cam0", 1, b"\x00" * 100)
        self.assertEqual(r.status_code, 422)

    def test_03_reject_unknown_lane(self):
        r = post(self.client, "cam9", 1, frame(1))
        self.assertEqual(r.status_code, 404)
        r = self.client.get("/cameras/cam9/result")
        self.assertEqual(r.status_code, 404)

    def test_04_accept_then_reject_stale(self):
        r = post(self.client, "cam0", 10, frame(10))
        self.assertEqual(r.status_code, 202)
        self.assertTrue(r.json()["accepted"])
        self.assertEqual(post(self.client, "cam0", 10, frame(10)).status_code, 409)
        self.assertEqual(post(self.client, "cam0", 3, frame(3)).status_code, 409)
        self.assertEqual(post(self.client, "cam0", 11, frame(11)).status_code, 202)

    def test_05_result_carries_computed_frame(self):
        # cam1 gets one frame; it must be computed and queryable
        r = post(self.client, "cam1", 1, frame(1))
        self.assertEqual(r.status_code, 202)
        res = wait_result(self.client, "cam1")
        self.assertEqual(res["frame_no"], 1)
        self.assertIn(res["class_id"], (0, 1, 2))
        self.assertEqual(len(res["scores"]), 3)
        self.assertAlmostEqual(sum(res["scores"]), 1.0, places=4)

    def test_06_churning_lane_does_not_starve_others(self):
        # cam0 churns frames rapidly; cam2 submits a single frame.
        post(self.client, "cam2", 100, frame(100))
        for n in range(200, 240):
            r = post(self.client, "cam0", n, frame(n))
            self.assertEqual(r.status_code, 202)
        res2 = wait_result(self.client, "cam2")
        self.assertEqual(res2["frame_no"], 100)  # cam2's only frame computed
        res0 = wait_result(self.client, "cam0")
        # cam0's result is some frame that was pending when its batch was taken
        self.assertGreaterEqual(res0["frame_no"], 200)
        self.assertLessEqual(res0["frame_no"], 239)

    def test_07_overview_snapshot(self):
        r = self.client.get("/cameras")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(sorted(data["lanes"].keys()), sorted(LANES))
        self.assertEqual(data["batch_lanes"], 2)
        for lane in LANES:
            self.assertIn("max_received", data["lanes"][lane])

    def test_08_batch_failure_keeps_result_and_continues(self):
        post(self.client, "cam2", 500, frame(500))
        good = wait_result(self.client, "cam2", want_frame=500)

        model = app.state.model  # same instance the worker loop uses
        orig_infer = model.infer
        calls = {"n": 0}

        async def flaky(images):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("boom")
            return await orig_infer(images)

        model.infer = flaky
        try:
            post(self.client, "cam2", 501, frame(501))
            wait_lane_state(self.client, "cam2", "failed_frames", 1)
            res = self.client.get("/cameras/cam2/result").json()
            self.assertEqual(res["frame_no"], good["frame_no"])  # original kept
            # worker keeps draining the queue after the failure
            post(self.client, "cam2", 502, frame(502))
            res2 = wait_result(self.client, "cam2", want_frame=502)
            self.assertEqual(res2["frame_no"], 502)
        finally:
            model.infer = orig_infer


if __name__ == "__main__":
    unittest.main()
