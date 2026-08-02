#!/usr/bin/env python3
"""Capture every MJPEG frame delivered during a supervised base motion test."""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


CAMERAS = ("head", "base")


def fetch_json(url: str, method: str = "GET") -> dict:
    request = urllib.request.Request(
        url,
        data=b"{}" if method == "POST" else None,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=4.0) as response:
        return json.loads(response.read())


def capture_stream(
    base_url: str,
    camera: str,
    output_dir: Path,
    stop_event: threading.Event,
    manifest: list[dict],
    errors: list[str],
) -> None:
    camera_dir = output_dir / "raw" / camera
    camera_dir.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        f"{base_url}/stream/{camera}",
        headers={"User-Agent": "thu-vr-final-motion/1.0"},
    )
    buffer = b""
    sequence = 0
    seen: set[bytes] = set()
    try:
        with urllib.request.urlopen(request, timeout=60.0) as response:
            while not stop_event.is_set():
                # HTTPResponse.read(n) waits to fill n bytes and can time out
                # between slow camera frames. read1() returns bytes already
                # available from the current multipart chunk immediately.
                chunk = response.read1(65536)
                if not chunk:
                    break
                buffer += chunk
                while True:
                    start = buffer.find(b"\xff\xd8")
                    if start < 0:
                        buffer = buffer[-2:]
                        break
                    end = buffer.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        buffer = buffer[start:]
                        break
                    jpeg = buffer[start:end + 2]
                    buffer = buffer[end + 2:]
                    digest = hashlib.sha256(jpeg).digest()
                    # The upstream endpoint blocks on the next ROS frame, but
                    # retain this guard so retries can never duplicate evidence.
                    if digest in seen:
                        continue
                    seen.add(digest)
                    captured_at = datetime.now(timezone.utc).isoformat()
                    filename = f"{camera}_move_{sequence:06d}.jpg"
                    (camera_dir / filename).write_bytes(jpeg)
                    manifest.append({
                        "camera": camera,
                        "frame": filename,
                        "captured_at": captured_at,
                        "sha256": digest.hex(),
                        "bytes": len(jpeg),
                    })
                    sequence += 1
    except Exception as exc:
        errors.append(f"{camera}: {type(exc).__name__}: {exc}")


def odom_distance(status: dict, start: dict) -> float:
    dx = float(status["base_odom_x"]) - float(start["base_odom_x"])
    dy = float(status["base_odom_y"]) - float(start["base_odom_y"])
    return (dx * dx + dy * dy) ** 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("turn", "shuttle"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--camera-url", default="http://127.0.0.1:5000")
    parser.add_argument("--control-url", default="http://127.0.0.1:5080")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifests = {camera: [] for camera in CAMERAS}
    capture_errors: list[str] = []
    stop_capture = threading.Event()
    threads = [
        threading.Thread(
            target=capture_stream,
            args=(args.camera_url, camera, args.output_dir, stop_capture,
                  manifests[camera], capture_errors),
            daemon=True,
        )
        for camera in CAMERAS
    ]
    for thread in threads:
        thread.start()

    control_log: list[dict] = []
    initial = fetch_json(f"{args.control_url}/api/status")
    if initial.get("base_is_moving"):
        raise RuntimeError("base is already moving")
    prime_deadline = time.monotonic() + 8.0
    while (
        time.monotonic() < prime_deadline
        and not capture_errors
        and any(len(manifests[camera]) < 2 for camera in CAMERAS)
    ):
        time.sleep(0.1)
    if capture_errors or any(len(manifests[camera]) < 2 for camera in CAMERAS):
        stop_capture.set()
        raise RuntimeError(
            f"camera capture did not prime: counts="
            f"{ {camera: len(manifests[camera]) for camera in CAMERAS} }, "
            f"errors={capture_errors}"
        )
    endpoint = "base_turn_test" if args.mode == "turn" else "base_shuttle"
    trigger = fetch_json(f"{args.control_url}/api/{endpoint}", "POST")
    if not trigger.get("success"):
        raise RuntimeError(trigger.get("msg") or "motion trigger failed")

    saw_motion = False
    saw_far_endpoint = False
    deadline = time.monotonic() + (45.0 if args.mode == "turn" else 40.0)
    try:
        while time.monotonic() < deadline:
            status = fetch_json(f"{args.control_url}/api/status")
            status["sampled_at"] = datetime.now(timezone.utc).isoformat()
            status["distance_from_start_m"] = round(odom_distance(status, initial), 5)
            control_log.append(status)
            saw_motion = saw_motion or bool(status.get("base_is_moving"))
            if args.mode == "shuttle":
                saw_far_endpoint = saw_far_endpoint or status["distance_from_start_m"] >= 0.45
                if (
                    saw_far_endpoint
                    and status.get("base_shuttle_leg") == "forward"
                    and status["distance_from_start_m"] <= 0.025
                ):
                    fetch_json(f"{args.control_url}/api/base_stop", "POST")
                if saw_motion and saw_far_endpoint and not status.get("base_is_moving"):
                    break
            elif saw_motion and not status.get("base_is_moving"):
                break
            time.sleep(0.05)
        else:
            raise TimeoutError(f"{args.mode} motion did not finish before timeout")
    finally:
        fetch_json(f"{args.control_url}/api/base_stop", "POST")
        time.sleep(1.0)
        stop_capture.set()
        for thread in threads:
            thread.join(timeout=4.0)

    final = fetch_json(f"{args.control_url}/api/status")
    metadata = {
        "schema_version": 1,
        "mode": args.mode,
        "algorithm_target": "2026-08-02-motion-temporal-v4.1",
        "started_at": control_log[0]["sampled_at"] if control_log else None,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "initial_status": initial,
        "final_status": final,
        "trigger_response": trigger,
        "frame_count_by_camera": {
            camera: len(manifests[camera]) for camera in CAMERAS
        },
        "capture_errors": capture_errors,
        "frames": {
            camera: sorted(manifests[camera], key=lambda item: item["frame"])
            for camera in CAMERAS
        },
        "control_samples": control_log,
    }
    (args.output_dir / "capture_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "ok": True,
        "mode": args.mode,
        "frame_count_by_camera": metadata["frame_count_by_camera"],
        "final_status": final,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
