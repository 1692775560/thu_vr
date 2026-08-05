#!/usr/bin/env python3
"""Score every trial on evenly-spaced frame subsets to find the frame budget.

The full takes run 15-45 frames, but most of that is the robot still moving
past labels it has already read.  This measures what a fixed small budget
actually costs in accuracy.
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

from ground_truth import GROUND_TRUTH, expected_labels  # noqa: E402
from label_ocr_engine import RecognitionEngine  # noqa: E402
from label_pipeline import LabelTracker, read_frame  # noqa: E402


def pick(paths: list[Path], budget: int) -> list[Path]:
    if budget <= 0 or budget >= len(paths):
        return paths
    step = len(paths) / budget
    return [paths[min(len(paths) - 1, int(i * step))] for i in range(budget)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budgets", default="2,4,6,8")
    parser.add_argument("--out", default=str(WORK / "results" / "frame_budget.json"))
    args = parser.parse_args()

    engine = RecognitionEngine()
    print(f"识别后端: {engine.backend}\n", flush=True)
    report = []

    for budget in [int(v) for v in args.budgets.split(",")]:
        recalls, precisions, exact, seconds = [], [], 0, 0.0
        for trial in sorted((WORK / "eval").iterdir()):
            if not trial.is_dir():
                continue
            pose = trial.name.split("__")[0]
            for camera in ("head", "base"):
                if camera not in GROUND_TRUTH[pose]:
                    continue
                paths = pick(
                    sorted(trial.glob(f"{camera}_move_*.jpg")), budget
                )
                started = time.monotonic()
                tracker = LabelTracker()
                for phase, path in enumerate(paths):
                    image = cv2.imread(str(path))
                    if image is None:
                        continue
                    tracker.update(read_frame(engine, image, phase=phase))
                tracker.refine(engine)
                seconds += time.monotonic() - started
                got = {
                    entry["label"] for entry in tracker.results()
                    if entry["resolved"] and entry["label"]
                }
                want = expected_labels(pose, camera)
                hit = got & want
                recalls.append(len(hit) / len(want))
                precisions.append(len(hit) / len(got) if got else 0.0)
                exact += int(got == want)
        row = {
            "budget": budget,
            "recall": round(sum(recalls) / len(recalls), 4),
            "precision": round(sum(precisions) / len(precisions), 4),
            "exact": exact,
            "trials": len(recalls),
            "seconds_per_trial": round(seconds / len(recalls), 1),
        }
        report.append(row)
        print(
            f"{budget:2d} 帧预算  召回 {row['recall']:.4f}  "
            f"精确 {row['precision']:.4f}  全对 {row['exact']}/{row['trials']}  "
            f"每组 {row['seconds_per_trial']:.1f} s",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
