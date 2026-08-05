#!/usr/bin/env python3
"""Verify that a physically misplaced sticker is read, not "corrected".

A sticker put in the wrong slot must still report the serial printed on it.
The test swaps two stickers' pixels inside a real frame, so the row sequence
becomes ...0012 0018 0014... and checks that the pipeline reports 0018 at that
position and flags it as out of sequence rather than rewriting it to 0013.

Run with a frame directory:
    python test_misplaced_labels.py /path/to/frames --camera head
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


def longest_row(image: np.ndarray) -> list[dict]:
    rows = group_rows(locate_stickers(image))
    if not rows:
        raise SystemExit("no sticker row found in the frame")
    return max(rows, key=len)


def swap_stickers(
    image: np.ndarray, left: dict, right: dict
) -> np.ndarray:
    """Exchange the pixels of two stickers, keeping each slot's geometry."""
    swapped = image.copy()
    lx, ly, lw, lh = left["box"]
    rx, ry, rw, rh = right["box"]
    pad = 2
    left_patch = image[
        max(0, ly - pad):ly + lh + pad, max(0, lx - pad):lx + lw + pad
    ].copy()
    right_patch = image[
        max(0, ry - pad):ry + rh + pad, max(0, rx - pad):rx + rw + pad
    ].copy()
    target_left = swapped[
        max(0, ly - pad):ly + lh + pad, max(0, lx - pad):lx + lw + pad
    ]
    target_right = swapped[
        max(0, ry - pad):ry + rh + pad, max(0, rx - pad):rx + rw + pad
    ]
    target_left[:] = cv2.resize(
        right_patch, (target_left.shape[1], target_left.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    target_right[:] = cv2.resize(
        left_patch, (target_right.shape[1], target_right.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    return swapped


def read_labels(engine, frames: list[np.ndarray], camera: str) -> list[dict]:
    tracker = LabelTracker()
    for phase, frame in enumerate(frames):
        tracker.update(read_frame(engine, frame, phase=phase))
    return tracker.results()


def row_serials(entries: list[dict], row: int) -> list[str | None]:
    picked = [entry for entry in entries if entry["row"] == row]
    picked.sort(key=lambda entry: entry["center"][0])
    return [entry["serial"] if entry["resolved"] else None for entry in picked]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--left", type=int, default=3)
    parser.add_argument("--right", type=int, default=8)
    args = parser.parse_args()

    paths = sorted(args.frame_dir.glob(f"{args.camera}_move_*.jpg"))[:args.frames]
    if not paths:
        raise SystemExit(f"no {args.camera}_move_*.jpg frames in {args.frame_dir}")
    originals = [cv2.imread(str(path)) for path in paths]

    engine = RecognitionEngine()
    baseline = read_labels(engine, originals, args.camera)
    target_row = max(
        {entry["row"] for entry in baseline},
        key=lambda row: sum(
            1 for entry in baseline
            if entry["row"] == row and entry["resolved"]
        ),
    )
    before = row_serials(baseline, target_row)
    print(f"原始行序号: {before}")
    if before.count(None) > 0 or len(before) < 9:
        raise SystemExit("baseline row is not fully read; pick a clearer trial")

    reference_row = longest_row(originals[0])
    left, right = reference_row[args.left], reference_row[args.right]
    swapped_frames = [
        swap_stickers(frame, left, right) for frame in originals
    ]
    result = read_labels(engine, swapped_frames, args.camera)
    swapped_row = max(
        {entry["row"] for entry in result},
        key=lambda row: sum(
            1 for entry in result if entry["row"] == row and entry["resolved"]
        ),
    )
    after = row_serials(result, swapped_row)
    print(f"错位后序号: {after}")

    expected = list(before)
    expected[args.left], expected[args.right] = (
        expected[args.right], expected[args.left]
    )
    print(f"应当读出  : {expected}")

    entries = [
        entry for entry in result if entry["row"] == swapped_row
    ]
    entries.sort(key=lambda entry: entry["center"][0])
    flagged = [
        entry["serial"] for entry in entries
        if entry.get("sequence_status") == "out_of_sequence"
    ]
    print(f"被标记乱序: {flagged}")

    failures = []
    if after != expected:
        failures.append(
            "序号未按像素读出（错位标签被序列先验改写或漏读）"
        )
    moved = {expected[args.left], expected[args.right]}
    # Exactly the moved stickers, no more: flagging their neighbours too would
    # send an operator to the wrong shelf slots.
    if set(flagged) != moved:
        failures.append(
            f"乱序标记不精确: 标记了 {sorted(flagged)}，应当只标 {sorted(moved)}"
        )

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("PASS 错位标签按像素正确读出，并被标记为乱序")
    return 0


if __name__ == "__main__":
    sys.exit(main())
