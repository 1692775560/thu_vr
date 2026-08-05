#!/usr/bin/env python3
"""Verify prefixes whose zone letter or bay decade differs from the row's.

Every sticker in the recordings is in zone A bay 04/05, so a B12 sticker can
only be tested by repainting one: the sticker's own plate is reused and its two
text lines are redrawn at the same size and contrast.

Three cases, all of which used to fail:

  one sticker   a B12 sticker among A04 ones must read B12 and be flagged
  partial row   five of them must read B12 without dragging the A04 stickers
                off their own bay
  whole row     a row entirely in B12 belongs to B12; nothing is misplaced

Run with a frame directory showing two populated rows:
    python test_zone_letter.py /path/to/frames --camera head
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


def repaint(image: np.ndarray, box, prefix: str, serial: str) -> None:
    """Redraw a sticker's two text lines in place, keeping its plate."""
    x, y, width, height = box
    plate = image[y:y + height, x:x + width]
    if plate.size == 0:
        return
    grey = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    paper = int(np.percentile(grey, 88))
    ink = int(np.percentile(grey, 6))
    inset = max(1, height // 12)
    plate[inset:height - inset, inset:width - inset] = paper
    for line, text in enumerate((f"{prefix}-", serial)):
        band = (height - 2 * inset) / 2.0
        scale = band * 0.62 / 22.0
        thickness = max(1, int(round(scale * 2.2)))
        (text_width, text_height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness
        )
        cv2.putText(
            image, text,
            (int(x + (width - text_width) / 2),
             int(y + inset + band * line + (band + text_height) / 2)),
            cv2.FONT_HERSHEY_DUPLEX, scale, (ink, ink, ink), thickness,
            cv2.LINE_AA,
        )


def read_row(engine, frames: list[np.ndarray], reference: list[dict]) -> list[dict]:
    """Published entries of the row holding the reference stickers, left to right."""
    tracker = LabelTracker()
    for phase, frame in enumerate(frames):
        tracker.update(read_frame(engine, frame, phase=phase))
    tracker.refine(engine)
    mid = sum(
        entry["box"][1] + entry["box"][3] / 2.0 for entry in reference
    ) / len(reference)
    height = float(np.median([entry["box"][3] for entry in reference]))
    picked = [
        entry for entry in tracker.results()
        if entry["resolved"] and abs(entry["center"][1] - mid) < height
    ]
    picked.sort(key=lambda entry: entry["center"][0])
    return picked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--zone", default="B12")
    parser.add_argument("--start", type=int, default=4)
    parser.add_argument("--count", type=int, default=5)
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
    target_row = rows[1]

    baseline = read_row(engine, originals, target_row)
    if len(baseline) != len(target_row):
        raise SystemExit(
            f"baseline published {len(baseline)} of {len(target_row)} stickers; "
            "pick a clearer trial"
        )
    home = baseline[0]["prefix"]
    serials = [entry["serial"] for entry in baseline]
    print(f"原始行: {home} 共 {len(baseline)} 张")

    failures: list[str] = []
    for name, span in (
        ("单张混入", range(args.start, args.start + 1)),
        ("部分混入", range(args.start, args.start + args.count)),
        ("整行换区", range(len(target_row))),
    ):
        indexes = [index for index in span if index < len(target_row)]
        frames = []
        for frame in originals:
            copy = frame.copy()
            for index in indexes:
                repaint(copy, target_row[index]["box"], args.zone, serials[index])
            frames.append(copy)
        published = read_row(engine, frames, target_row)
        labels = [entry["label"] for entry in published]
        flagged = {
            entry["label"] for entry in published
            if entry.get("prefix_status") == "out_of_bay"
        }
        want = [
            f"{args.zone if index in indexes else home}-{serials[index]}"
            for index in range(len(target_row))
        ]
        # A row that is entirely in the other zone belongs to that zone, so
        # nothing in it is misplaced.  A minority is.
        want_flagged = (
            set() if len(indexes) == len(target_row)
            else {f"{args.zone}-{serials[index]}" for index in indexes}
        )
        print(f"\n{name}（{len(indexes)} 张 -> {args.zone}）")
        print(f"  读出: {' '.join(str(label) for label in labels)}")
        if labels != want:
            failures.append(f"{name}: 读出 {labels}，应为 {want}")
        if flagged != want_flagged:
            failures.append(
                f"{name}: 标记 {sorted(flagged)}，应标 {sorted(want_flagged)}"
            )
        if labels == want and flagged == want_flagged:
            print(f"  标记: {sorted(flagged) or '无'}  OK")

    if failures:
        print()
        for failure in failures:
            print(f"FAIL {failure}")
        return 1
    print("\nPASS 混入其他字母区的贴纸按像素读出完整前缀，且不影响同行其他贴纸")
    return 0


if __name__ == "__main__":
    sys.exit(main())
