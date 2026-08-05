#!/usr/bin/env python3
"""Verify that a sticker moved to a different shelf row keeps its own prefix.

Two rows in one frame carry different bay numbers (A05 above A04).  Swapping
one sticker between them reproduces a label filed under the wrong bay.  The
pipeline has to report the bay printed on the sticker, not the bay of the row
it now sits in, and has to flag it -- the prefix analogue of
test_misplaced_labels.py.

Run with a frame directory that shows two populated rows:
    python test_misplaced_prefix.py /path/to/frames --camera head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from label_ocr_engine import RecognitionEngine
from label_pipeline import LabelTracker, read_frame
from label_reader import group_rows, locate_stickers
from test_misplaced_labels import swap_stickers


def read_labels(engine, frames: list[np.ndarray]) -> list[dict]:
    tracker = LabelTracker()
    for phase, frame in enumerate(frames):
        tracker.update(read_frame(engine, frame, phase=phase))
    tracker.refine(engine)
    return tracker.results()


def nearest(entries: list[dict], box: tuple[int, int, int, int]) -> dict | None:
    """The published entry sitting on the given sticker box."""
    x, y, width, height = box
    centre = (x + width / 2.0, y + height / 2.0)
    best, best_distance = None, float("inf")
    for entry in entries:
        if not entry["resolved"]:
            continue
        distance = (
            abs(entry["center"][0] - centre[0]) + abs(entry["center"][1] - centre[1])
        )
        if distance < best_distance:
            best, best_distance = entry, distance
    if best is None or best_distance > width:
        return None
    return best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--upper-slot", type=int, default=5)
    parser.add_argument("--lower-slot", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.frame_dir.glob(f"{args.camera}_move_*.jpg"))[:args.frames]
    if not paths:
        raise SystemExit(f"no {args.camera}_move_*.jpg frames in {args.frame_dir}")
    originals = [cv2.imread(str(path)) for path in paths]

    engine = RecognitionEngine()
    rows = sorted(
        (row for row in group_rows(locate_stickers(originals[0])) if len(row) >= 8),
        key=lambda row: sum(entry["box"][1] for entry in row) / len(row),
    )
    if len(rows) < 2:
        raise SystemExit("need two populated rows; pick a frame set showing both")
    upper, lower = rows[0], rows[1]
    upper_box = upper[args.upper_slot]["box"]
    lower_box = lower[args.lower_slot]["box"]

    baseline = read_labels(engine, originals)
    before_upper = nearest(baseline, upper_box)
    before_lower = nearest(baseline, lower_box)
    if not before_upper or not before_lower:
        raise SystemExit("baseline did not publish both chosen slots")
    print(f"原始: 上行槽位 {before_upper['label']}, 下行槽位 {before_lower['label']}")
    if before_upper["prefix"] == before_lower["prefix"]:
        raise SystemExit("both rows read the same bay; pick a frame set with two bays")

    swapped = [
        swap_stickers(frame, upper[args.upper_slot], lower[args.lower_slot])
        for frame in originals
    ]
    result = read_labels(engine, swapped)
    after_upper = nearest(result, upper_box)
    after_lower = nearest(result, lower_box)
    print(
        f"错位后: 上行槽位 {after_upper['label'] if after_upper else None}, "
        f"下行槽位 {after_lower['label'] if after_lower else None}"
    )
    print(
        f"应当读出: 上行槽位 {before_lower['label']}, "
        f"下行槽位 {before_upper['label']}"
    )

    failures = []
    for position, entry, want in (
        ("上行", after_upper, before_lower["label"]),
        ("下行", after_lower, before_upper["label"]),
    ):
        if entry is None:
            failures.append(f"{position}槽位未发布标签")
            continue
        if entry["label"] != want:
            failures.append(
                f"{position}槽位读成 {entry['label']}，应为 {want}"
                "（前缀被所在行改写）"
            )
        if entry.get("prefix_status") != "out_of_bay":
            failures.append(
                f"{position}槽位前缀状态为 {entry.get('prefix_status')}，"
                "应标记 out_of_bay"
            )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS 跨行错位标签按像素读出前缀，并被标记为 out_of_bay")
    return 0


if __name__ == "__main__":
    sys.exit(main())
