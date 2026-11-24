#!/usr/bin/env python3
"""Demo: two inference channels share one CPU model.

Scenario A (main): on ONE lane, a slow frame is still computing on
channel 1 when a newer fast frame arrives; the free channel 2 computes
the newer frame first and it becomes queryable immediately. When the slow
older frame finishes later it must NOT overwrite the newer result.

Scenario B: while channel 1 is busy with a slow batch, another lane's
frame is computed promptly on channel 2 (the duty console sees fast
batches' classifications while a slow batch is still running).

Run the service first:

    docker compose up --build        # INFER_DELAY_MS=200 by default
    python3 demo/demo_channels.py    # only needs the standard library

The per-frame X-Infer-Delay-Ms header forces the batch containing that
frame to sleep the given number of milliseconds on CPU, so the ordering
below is deterministic regardless of host speed.
"""

import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"


def req(method, path, body=None, delay_ms=None):
    headers = {}
    if body is not None:
        headers["Content-Type"] = "application/octet-stream"
    if delay_ms is not None:
        headers["X-Infer-Delay-Ms"] = str(delay_ms)
    r = urllib.request.Request(BASE + path, data=body, method=method,
                               headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=15) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def frame(seed: int) -> bytes:
    return bytes((seed + i) % 256 for i in range(64 * 64))


def snapshot():
    _, snap = req("GET", "/cameras")
    return snap


def wait_state(lane, key, want, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = snapshot()["lanes"][lane]
        if st[key] == want:
            return st
        time.sleep(0.02)
    raise SystemExit(f"timeout: {lane}.{key} never became {want!r}")


def wait_result(lane, want_frame, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        _, r = req("GET", f"/cameras/{lane}/result")
        if r.get("frame_no") is not None and r["frame_no"] >= want_frame:
            return r
        time.sleep(0.02)
    raise SystemExit(f"timeout waiting for {lane} frame {want_frame}")


def show_snap(title):
    snap = snapshot()
    print(f"    {title}: active_batches={snap['active_batches']} "
          f"queue={snap['queue']}")
    for lane, st in snap["lanes"].items():
        print(f"      {lane}: pending={st['pending_frame_no']} "
              f"in_flight={st['in_flight']} in_batch={st['in_batch']} "
              f"last_result={st['last_result_frame_no']} "
              f"stale={st['stale_frames']}")


def main():
    print(f"== dual-channel demo against {BASE} ==\n")
    for _ in range(50):
        s, _ = req("GET", "/healthz")
        if s == 200:
            break
        time.sleep(0.2)
    else:
        raise SystemExit("service not healthy")

    snap = snapshot()
    print(f"channels={snap['channels']} batch_lanes={snap['batch_lanes']}\n")
    # re-runnable: keep numbering above anything previously received
    base = {lane: (st["max_received"] or 0)
            for lane, st in snap["lanes"].items()}
    slow_no = base["cam1"] + 1
    fast_no = slow_no + 1

    print(f"[A1] cam1 frame {slow_no} with a 1500ms controlled delay "
          f"(slow batch on channel 1):")
    s, meta = req("POST", f"/cameras/cam1/frames?frame_no={slow_no}",
                  frame(slow_no), delay_ms=1500)
    print(f"     -> {s} {meta}")
    wait_state("cam1", "in_flight", 1)
    t_slow = time.monotonic()
    show_snap("slow frame now in flight")

    print(f"\n[A2] while it is still computing, cam1 frame {fast_no} "
          f"arrives with a 100ms delay:")
    s, meta = req("POST", f"/cameras/cam1/frames?frame_no={fast_no}",
                  frame(fast_no), delay_ms=100)
    print(f"     -> {s} {meta}")
    assert meta["in_batch"] is True, "older frame is still being computed"
    wait_state("cam1", "in_flight", 2)
    show_snap("same lane now has TWO frames in flight (two channels)")

    print(f"\n[A3] poll the result endpoint: the newer frame must appear "
          f"first, without waiting for the slow batch:")
    r = wait_result("cam1", fast_no, timeout=5)
    dt_new = time.monotonic() - t_slow
    print(f"     frame {r['frame_no']} published after {dt_new:.2f}s "
          f"(class_id={r['class_id']}, scores={r['scores']})")
    assert r["frame_no"] == fast_no
    assert dt_new < 1.0, "newer frame was blocked behind the slow 1.5s batch"

    print(f"\n[A4] wait for the slow older frame to finish; the published "
          f"result must stay at frame {fast_no}:")
    wait_state("cam1", "in_flight", 0, timeout=5)
    _, r = req("GET", "/cameras/cam1/result")
    st = snapshot()["lanes"]["cam1"]
    print(f"     result frame_no={r['frame_no']} (still the newer one)")
    print(f"     cam1 stale_frames={st['stale_frames']} "
          f"(slow old frame finished late and was not published)")
    assert r["frame_no"] == fast_no
    assert st["stale_frames"] >= 1
    assert st["in_batch"] is False

    print("\n[B1] another slow frame occupies channel 1, then cam2's fast "
          "frame is served by the free channel:")
    slow2 = base["cam0"] + 1
    req("POST", f"/cameras/cam0/frames?frame_no={slow2}",
        frame(slow2), delay_ms=1200)
    wait_state("cam0", "in_flight", 1)
    t0 = time.monotonic()
    cam2_no = base["cam2"] + 1
    req("POST", f"/cameras/cam2/frames?frame_no={cam2_no}",
        frame(cam2_no), delay_ms=50)
    r2 = wait_result("cam2", cam2_no, timeout=5)
    dt = time.monotonic() - t0
    print(f"     cam2 frame {r2['frame_no']} done in {dt:.2f}s while "
          f"cam0's slow batch is still running")
    assert dt < 1.0
    r0 = wait_result("cam0", slow2, timeout=5)
    print(f"     cam0 slow frame {r0['frame_no']} finished afterwards")

    print("\nOK: two channels share one model; a same-lane newer frame "
          "publishes first and late older frames never overwrite it.")


if __name__ == "__main__":
    main()
