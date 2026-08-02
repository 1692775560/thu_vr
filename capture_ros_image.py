#!/usr/bin/env python3
"""Capture one color frame from a ROS 2 Image topic.

This helper only subscribes to a camera topic. It does not publish robot
commands or move any actuator.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


class SingleFrameCapture(Node):
    def __init__(self, topic: str, output: Path) -> None:
        super().__init__("thu_vr_single_frame_capture")
        self.output = output
        self.bridge = CvBridge()
        self.saved = False
        self.error: Exception | None = None
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.subscription = self.create_subscription(Image, topic, self._on_image, qos)

    def _on_image(self, message: Image) -> None:
        if self.saved or self.error is not None:
            return
        try:
            frame = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
            self.output.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self.output), frame):
                raise RuntimeError(f"OpenCV could not write {self.output}")
            self.saved = True
        except Exception as exc:  # surfaced by main after ROS cleanup
            self.error = exc


def main() -> int:
    parser = argparse.ArgumentParser(description="从 ROS 2 彩色图像话题抓取一帧")
    parser.add_argument("--topic", default="/head_rgbd/color/image_raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()

    rclpy.init(args=None)
    node = SingleFrameCapture(args.topic, args.output)
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not node.saved and node.error is None:
            if time.monotonic() >= deadline:
                print(f"在 {args.timeout:.1f}s 内未从 {args.topic} 收到图像", file=sys.stderr)
                return 2
            rclpy.spin_once(node, timeout_sec=0.2)
        if node.error is not None:
            raise node.error
        print(args.output)
        return 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
