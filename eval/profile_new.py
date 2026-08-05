#!/usr/bin/env python3
"""Break the end-to-end cost into stages, and check how few frames suffice.

Reported per trial: model load, per-frame accumulation (split into sticker
segmentation and recognition), the temporal refine pass, and the publishing
decision.  Then the same trial is rerun on evenly-spaced subsets of the frames
so the accuracy-versus-time trade-off is measurable rather than guessed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# The data root holds eval/<trial>/<camera>_move_*.jpg and receives results/.
# It defaults to this script's own directory, which is the layout used while
# developing; point LABEL_DATA at that directory to run from anywhere else.
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("LABEL_DATA", HERE))
# The pipeline modules sit either beside this script's directory (in the repo)
# or in a sibling checkout (when run from a scratch directory).
sys.path.insert(0, str(
    HERE.parent if (HERE.parent / "label_pipeline.py").exists()
    else HERE.parent / "thu_vr"
))
sys.path.insert(0, str(HERE))

import cv2  # noqa: E402

import label_pipeline as pipeline  # noqa: E402
from ground_truth import expected_labels  # noqa: E402
from label_ocr_engine import RecognitionEngine  # noqa: E402
from label_reader import group_rows, locate_stickers  # noqa: E402

TRIALS = [
    ("A2__dark", "head"),
    ("A5_A4-5__light", "head"),
    ("A4__light", "base"),
]


def stage_timings(engine, images: list) -> dict:
    """Time each stage of a full run over the given frames."""
    segment = 0.0
    tracker = pipeline.LabelTracker()
    started = time.monotonic()
    for phase, image in enumerate(images):
        mark = time.monotonic()
        group_rows(locate_stickers(image))
        segment += time.monotonic() - mark
        tracker.update(pipeline.read_frame(engine, image, phase=phase))
    frames_total = time.monotonic() - started

    mark = time.monotonic()
    tracker.refine(engine)
    refine = time.monotonic() - mark

    mark = time.monotonic()
    entries = tracker.results()
    publish = time.monotonic() - mark
    return {
        "segment": segment,
        # The segmentation above is a separate measuring call, so subtract it
        # to leave the recognition and bookkeeping cost of the frame loop.
        "recognize": frames_total - segment * 2,
        "refine": refine,
        "publish": publish,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", default="1,2,4,8,0")
    parser.add_argument("--out", default=str(WORK / "results" / "profile.json"))
    args = parser.parse_args()

    started = time.monotonic()
    engine = RecognitionEngine()
    load = time.monotonic() - started
    print(f"识别后端 {engine.backend}，模型加载 {load:.2f} s\n", flush=True)

    report = {"model_load_seconds": round(load, 2), "trials": []}
    for trial, camera in TRIALS:
        paths = sorted((WORK / "eval" / trial).glob(f"{camera}_move_*.jpg"))
        images = [cv2.imread(str(path)) for path in paths]
        images = [image for image in images if image is not None]
        want = expected_labels(trial.split("__")[0], camera)

        timing = stage_timings(engine, images)
        total = sum(
            timing[key] for key in ("segment", "recognize", "refine", "publish")
        )
        predicted = {
            entry["label"] for entry in timing["entries"]
            if entry["resolved"] and entry["label"]
        }
        print(
            f"{trial} {camera}  {len(images)} 帧  合计 {total:.1f} s\n"
            f"   分割 {timing['segment']:.1f}s  识别 {timing['recognize']:.1f}s  "
            f"时序精修 {timing['refine']:.1f}s  发布判定 {timing['publish']:.3f}s  "
            f"召回 {len(predicted & want) / len(want):.3f}",
            flush=True,
        )

        subsets = []
        for step in [int(v) for v in args.subsets.split(",")]:
            if step == 0:
                count = len(images)
                picked = images
            else:
                picked = images[::max(1, len(images) // step)][:step]
                count = len(picked)
            mark = time.monotonic()
            tracker = pipeline.LabelTracker()
            for phase, image in enumerate(picked):
                tracker.update(pipeline.read_frame(engine, image, phase=phase))
            tracker.refine(engine)
            got = {
                entry["label"] for entry in tracker.results()
                if entry["resolved"] and entry["label"]
            }
            elapsed = time.monotonic() - mark
            hit = got & want
            row = {
                "frames": count,
                "seconds": round(elapsed, 1),
                "recall": round(len(hit) / len(want), 4),
                "precision": round(len(hit) / len(got), 4) if got else 0.0,
            }
            subsets.append(row)
            print(
                f"      {count:2d} 帧 -> {elapsed:5.1f} s  "
                f"召回 {row['recall']:.3f}  精确 {row['precision']:.3f}",
                flush=True,
            )

        report["trials"].append({
            "trial": trial, "camera": camera, "frames": len(images),
            "seconds_total": round(total, 1),
            "seconds_segment": round(timing["segment"], 1),
            "seconds_recognize": round(timing["recognize"], 1),
            "seconds_refine": round(timing["refine"], 1),
            "seconds_publish": round(timing["publish"], 3),
            "subsets": subsets,
        })
        print(flush=True)

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
