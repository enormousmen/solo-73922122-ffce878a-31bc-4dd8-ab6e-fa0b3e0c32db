#!/usr/bin/env python3
"""Demo: one lane churns frames while another lane still gets computed
according to its queue position. Uses only the standard library.

Run the service first (docker compose up, or uvicorn app.main:app).
For visible queueing/replacement, start it with INFER_DELAY_MS=200
(already set in docker-compose.yml). Then:

    python3 demo/demo.py

The script is re-runnable: it numbers new frames from each lane's
current max_received + 1.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
CHURN_FRAMES = 60  # frames fired at cam0 as fast as possible


def req(method, path, body=None):
    r = urllib.request.Request(
        BASE + path,
        data=body,
        method=method,
        headers={"Content-Type": "application/octet-stream"} if body else {},
    )
    try:
        with urllib.request.urlopen(r, timeout=10) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def frame(seed: int) -> bytes:
    return bytes((seed + i) % 256 for i in range(64 * 64))


def wait_result(lane, want_frame, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, r = req("GET", f"/cameras/{lane}/result")
        got = r.get("frame_no")
        if got is not None and got >= want_frame:
            return r
        time.sleep(0.05)
    raise SystemExit(f"timeout waiting for {lane} frame {want_frame}")


def main():
    print(f"== demo against {BASE} ==\n")

    for _ in range(50):
        s, _ = req("GET", "/healthz")
        if s == 200:
            break
        time.sleep(0.2)
    else:
        raise SystemExit("service not healthy")

    # re-runnable: continue numbering where each lane left off
    _, snap = req("GET", "/cameras")
    base = {lane: (st["max_received"] or 0)
            for lane, st in snap["lanes"].items()}
    n1, n2 = base.get("cam1", 0), base.get("cam2", 0)

    print(f"[1] frame numbers must increase per lane "
          f"(cam1 gets frame {n1 + 1} twice):")
    print("   ", req("POST", f"/cameras/cam1/frames?frame_no={n1 + 1}",
                      frame(n1 + 1)))
    print("   ", req("POST", f"/cameras/cam1/frames?frame_no={n1 + 1}",
                      frame(n1 + 1)))
    print()

    print(f"[2] cam1 and cam2 submit one frame each, then cam0 churns "
          f"{CHURN_FRAMES} frames:")
    print("    cam1 <-", req("POST", f"/cameras/cam1/frames?frame_no={n1 + 2}",
                             frame(n1 + 2))[0:2])
    print("    cam2 <-", req("POST", f"/cameras/cam2/frames?frame_no={n2 + 1}",
                             frame(n2 + 1))[0:2])
    t0 = time.monotonic()
    for n in range(1, CHURN_FRAMES + 1):
        req("POST", f"/cameras/cam0/frames?frame_no={base.get('cam0', 0) + n}",
            frame(n))
    dt = time.monotonic() - t0
    print(f"    cam0 <- {CHURN_FRAMES} frames accepted in {dt:.2f}s "
          f"(ingestion is not blocked by ongoing inference)\n")

    print("[3] snapshot taken right after the burst (inference still busy):")
    _, snap = req("GET", "/cameras")
    for lane, st in snap["lanes"].items():
        print(f"    {lane}: pending={st['pending_frame_no']} "
              f"queued={st['queued']} in_batch={st['in_batch']} "
              f"last_result={st['last_result_frame_no']}")
    print(f"    queue: {snap['queue']}\n")

    print("[4] final results:")
    r1 = wait_result("cam1", want_frame=n1 + 2)
    print(f"    cam1: frame_no={r1['frame_no']} class_id={r1['class_id']} "
          f"scores={r1['scores']}")
    assert r1["frame_no"] == n1 + 2, "cam1's queued frame must be computed"
    r2 = wait_result("cam2", want_frame=n2 + 1)
    print(f"    cam2: frame_no={r2['frame_no']} class_id={r2['class_id']}")
    r0 = wait_result("cam0", want_frame=base.get("cam0", 0) + 1)
    print(f"    cam0: frame_no={r0['frame_no']} class_id={r0['class_id']} "
          f"<- the frame actually pending when its batch was taken;")
    print(f"        earlier frames were replaced while waiting\n")

    _, snap = req("GET", "/cameras")
    st0 = snap["lanes"]["cam0"]
    print(f"[5] cam0 counters: max_received={st0['max_received']} "
          f"replaced_frames={st0['replaced_frames']} "
          f"failed_frames={st0['failed_frames']}")
    if st0["replaced_frames"] == 0:
        print("    note: inference outpaced ingestion, so no frame was "
              "replaced; restart the service with INFER_DELAY_MS=200 "
              "(as in docker-compose.yml) to observe replacement.")

    print("\nOK: cam1/cam2 were computed from their queue slots while cam0 "
          "churned; replaced frames kept their lane's queue position.")


if __name__ == "__main__":
    main()
