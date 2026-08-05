#!/usr/bin/env python3
"""Verify that stickers merged into one bright blob are cut apart again.

Where a shelf sits far enough away and the backdrop is bright enough, the gaps
between neighbouring plates stop separating them: a whole row can come out of
thresholding as two or three blobs, each holding several stickers.  A blob
holding two stickers can only ever publish one serial, so the row silently
loses half its labels while still looking like it was found -- boxes are drawn,
they just straddle two plates each.

The merged blob is taller as well as wider, because it takes in the book spine
under the plates.  That is what makes this hard to catch: its aspect ratio
still resembles a single sticker's, so counting stickers by aspect alone
under-splits it by exactly the factor the extra height introduced.

Segmentation cannot be judged frame by frame -- background clutter clusters
into convincing "rows", and a real row gives up only some of its stickers in
any one frame -- so the test runs the pass and counts the slots the tracker
ends up with, which is what a merged blob actually costs.

Run with a frame directory:
    python test_touching_split.py /path/to/frames --camera head --expect 11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from label_ocr_engine import RecognitionEngine
from label_pipeline import LabelTracker, read_frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("--camera", default="head")
    parser.add_argument("--frames", type=int, default=0,
                        help="0 uses the whole pass")
    parser.add_argument("--expect", type=int, default=11,
                        help="stickers on each shelf row in this scene")
    parser.add_argument("--rows", type=int, default=2,
                        help="shelf rows the camera sees")
    args = parser.parse_args()

    paths = sorted(args.frame_dir.glob(f"{args.camera}_move_*.jpg"))
    if args.frames:
        paths = paths[: args.frames]
    if not paths:
        raise SystemExit(f"no {args.camera}_move_*.jpg frames in {args.frame_dir}")

    engine = RecognitionEngine()
    tracker = LabelTracker()
    for phase, path in enumerate(paths):
        tracker.update(read_frame(engine, cv2.imread(str(path)), phase=phase))
    entries = tracker.results()

    counts: dict[int, int] = {}
    published: dict[int, int] = {}
    for entry in entries:
        counts[entry["row"]] = counts.get(entry["row"], 0) + 1
        published[entry["row"]] = published.get(entry["row"], 0) + bool(
            entry["resolved"]
        )
    shelves = sorted(counts, key=lambda row: published.get(row, 0), reverse=True)
    shelves = shelves[: args.rows]

    print(f"{args.frame_dir.name}/{args.camera}: {len(paths)} 帧")
    # A blob that swallowed its neighbour leaves the row one slot short and one
    # label short.  Spare slots that publish nothing are a different matter --
    # a stray bright patch tracked and then rejected costs no label -- so only
    # a shortfall is a failure here.
    ok = True
    for row in shelves:
        short = counts[row] < args.expect or published[row] != args.expect
        print(f"  第{row}排 槽位 {counts[row]}（不应少于 {args.expect}） "
              f"发布 {published[row]}（应为 {args.expect}）  "
              f"{'不符' if short else 'OK'}")
        ok &= not short
    if len(shelves) < args.rows:
        print(f"FAIL 只找到 {len(shelves)} 排，应有 {args.rows} 排")
        return 1
    if not ok:
        print("FAIL 有排的槽位或标签少于贴纸数，说明有块横跨多张贴纸")
        return 1
    print(f"PASS 每排都读满 {args.expect} 张，没有块横跨多张贴纸")
    return 0


if __name__ == "__main__":
    sys.exit(main())
