#!/usr/bin/env python3
"""Capture every received ROS image during the final robot motion tests."""

from __future__ import annotations

import argparse
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from final_motion_capture import fetch_json, odom_distance


CAMERAS = {
    "head": "/head_rgbd/color/image_raw",
    "base": "/base_rgbd/color/image_raw",
}


class FrameCapture(Node):
    def __init__(self, output_dir: Path) -> None:
        super().__init__("thu_vr_final_motion_capture")
        self.output_dir = output_dir
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.manifests = {camera: [] for camera in CAMERAS}
        self.errors: list[str] = []
        self.callback_group = ReentrantCallbackGroup()
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=30,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.image_subscriptions = [
            self.create_subscription(
                Image,
                topic,
                lambda message, name=camera: self._on_image(name, message),
                qos,
                callback_group=self.callback_group,
            )
            for camera, topic in CAMERAS.items()
        ]
        for camera in CAMERAS:
            (output_dir / "raw" / camera).mkdir(parents=True, exist_ok=True)

    def _on_image(self, camera: str, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            ok, encoded = cv2.imencode(
                ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92]
            )
            if not ok:
                raise RuntimeError("JPEG encoding failed")
            with self.lock:
                sequence = len(self.manifests[camera])
                filename = f"{camera}_move_{sequence:06d}.jpg"
                payload = encoded.tobytes()
                (self.output_dir / "raw" / camera / filename).write_bytes(payload)
                self.manifests[camera].append({
                    "camera": camera,
                    "frame": filename,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "ros_stamp": {
                        "sec": int(message.header.stamp.sec),
                        "nanosec": int(message.header.stamp.nanosec),
                    },
                    "bytes": len(payload),
                    "resolution": [int(message.width), int(message.height)],
                    "encoding": message.encoding,
                })
        except Exception as exc:
            with self.lock:
                if len(self.errors) < 20:
                    self.errors.append(
                        f"{camera}: {type(exc).__name__}: {exc}"
                    )

    def counts(self) -> dict[str, int]:
        with self.lock:
            return {
                camera: len(self.manifests[camera]) for camera in CAMERAS
            }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("probe", "turn", "shuttle"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--control-url", default="http://127.0.0.1:5080")
    parser.add_argument("--probe-seconds", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=False)
    rclpy.init()
    node = FrameCapture(args.output_dir)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    control_log: list[dict] = []
    trigger = None
    initial = fetch_json(f"{args.control_url}/api/status")
    if initial.get("base_is_moving"):
        raise RuntimeError("base is already moving")

    try:
        prime_deadline = time.monotonic() + 8.0
        while time.monotonic() < prime_deadline:
            if all(count >= 2 for count in node.counts().values()):
                break
            time.sleep(0.05)
        else:
            raise RuntimeError(f"ROS cameras did not prime: {node.counts()}")

        if args.mode == "probe":
            time.sleep(max(0.1, args.probe_seconds))
        else:
            endpoint = "base_turn_test" if args.mode == "turn" else "base_shuttle"
            trigger = fetch_json(f"{args.control_url}/api/{endpoint}", "POST")
            if not trigger.get("success"):
                raise RuntimeError(trigger.get("msg") or "motion trigger failed")
            saw_motion = False
            saw_far_endpoint = False
            deadline = time.monotonic() + (45.0 if args.mode == "turn" else 40.0)
            while time.monotonic() < deadline:
                status = fetch_json(f"{args.control_url}/api/status")
                status["sampled_at"] = datetime.now(timezone.utc).isoformat()
                status["distance_from_start_m"] = round(odom_distance(status, initial), 5)
                control_log.append(status)
                saw_motion = saw_motion or bool(status.get("base_is_moving"))
                if args.mode == "shuttle":
                    saw_far_endpoint = (
                        saw_far_endpoint
                        or status["distance_from_start_m"] >= 0.45
                    )
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
            fetch_json(f"{args.control_url}/api/base_stop", "POST")
            time.sleep(1.0)
    finally:
        if args.mode != "probe":
            try:
                fetch_json(f"{args.control_url}/api/base_stop", "POST")
            except Exception:
                pass
        executor.shutdown(timeout_sec=3.0)
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=3.0)

    final = fetch_json(f"{args.control_url}/api/status")
    counts = node.counts()
    metadata = {
        "schema_version": 1,
        "mode": args.mode,
        "source": "direct ROS Image subscriptions",
        "topics": CAMERAS,
        "algorithm_target": "2026-08-02-motion-temporal-v4.1",
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "initial_status": initial,
        "final_status": final,
        "trigger_response": trigger,
        "frame_count_by_camera": counts,
        "capture_errors": node.errors,
        "frames": node.manifests,
        "control_samples": control_log,
    }
    (args.output_dir / "capture_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({
        "ok": not node.errors,
        "mode": args.mode,
        "frame_count_by_camera": counts,
        "capture_errors": node.errors,
        "final_status": final,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
