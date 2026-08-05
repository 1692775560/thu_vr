#!/usr/bin/env python3
"""Evaluate the pixel-first pipeline on every trial against verified truth."""

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

from ground_truth import GROUND_TRUTH, expected_labels  # noqa: E402
from label_ocr_engine import RecognitionEngine  # noqa: E402
from label_pipeline import LabelTracker, read_frame  # noqa: E402


def evaluate(engine, trial_dir: Path, camera: str, expected: set[str]) -> dict:
    tracker = LabelTracker()
    paths = sorted(trial_dir.glob(f"{camera}_move_*.jpg"))
    started = time.monotonic()
    per_frame = []
    for phase, path in enumerate(paths):
        image = cv2.imread(str(path))
        if image is None:
            continue
        rows = read_frame(engine, image, phase=phase)
        tracker.update(rows)
        if phase % 4 == 3:
            published = {
                entry["label"] for entry in tracker.results()
                if entry["resolved"] and entry["label"]
            }
            per_frame.append([phase, len(published & expected), len(published)])
    tracker.refine(engine)
    entries = tracker.results()
    predicted = {
        entry["label"] for entry in entries
        if entry["resolved"] and entry["label"]
    }
    elapsed = time.monotonic() - started
    hit = predicted & expected
    return {
        "frames": len(paths),
        "seconds_per_frame": round(elapsed / max(1, len(paths)), 2),
        "predicted": sorted(predicted),
        "recall": round(len(hit) / len(expected), 4) if expected else 0.0,
        "precision": round(len(hit) / len(predicted), 4) if predicted else 0.0,
        "exact": predicted == expected,
        "missed": sorted(expected - predicted),
        "wrong": sorted(predicted - expected),
        "progression": per_frame,
        "entries": entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="new")
    parser.add_argument("--cameras", default="head,base")
    parser.add_argument("--trials", default="")
    args = parser.parse_args()

    engine = RecognitionEngine()
    print(f"识别后端: {engine.backend}\n")
    out_dir = WORK / "results" / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    cameras = [c for c in args.cameras.split(",") if c]
    wanted = [t for t in args.trials.split(",") if t]

    rows = []
    for trial in sorted((WORK / "eval").iterdir()):
        if not trial.is_dir() or (wanted and trial.name not in wanted):
            continue
        pose, lighting = trial.name.split("__")
        for camera in cameras:
            if camera not in GROUND_TRUTH[pose]:
                continue
            expected = expected_labels(pose, camera)
            result = evaluate(engine, trial, camera, expected)
            (out_dir / f"{trial.name}_{camera}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            rows.append({
                "pose": pose, "lighting": lighting, "camera": camera,
                "expected": len(expected), **{
                    key: result[key] for key in
                    ("recall", "precision", "exact", "frames",
                     "seconds_per_frame", "missed", "wrong")
                },
            })
            mark = "OK " if result["exact"] else "FAIL"
            print(
                f"{mark} {pose:8s} {lighting:5s} {camera:4s} "
                f"期望{len(expected):3d} 召回{result['recall']:.3f} "
                f"精确{result['precision']:.3f} {result['seconds_per_frame']:.2f}s/帧"
            )
            if not result["exact"]:
                if result["missed"]:
                    print(f"       漏检 {result['missed']}")
                if result["wrong"]:
                    print(f"       误检 {result['wrong']}")

    (out_dir / "table.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n"
    )
    if rows:
        exact = sum(1 for row in rows if row["exact"])
        mean = lambda key: sum(row[key] for row in rows) / len(rows)
        print(
            f"\n== {args.tag} ==  全对 {exact}/{len(rows)} 组  "
            f"平均召回 {mean('recall'):.4f}  平均精确 {mean('precision'):.4f}"
        )


if __name__ == "__main__":
    main()
