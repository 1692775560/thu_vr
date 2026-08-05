#!/usr/bin/env python3
"""Ablate one mechanism at a time and rescore, without touching the pipeline.

Each mechanism is disabled by rebinding the name `label_pipeline` calls it by,
so the production code stays free of feature flags.  Scores are reported on a
fixed subset of trials that covers the easy and the hard end of the set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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

SUBSET = [
    ("A2__dark", "head"),
    ("A2__light", "base"),
    ("A3__light", "head"),
    ("A5_A4-5__dark", "head"),
]

ORIGINALS = {
    name: getattr(pipeline, name)
    for name in (
        "propose_missing", "propose_row_ends", "build_row_prefix_composites",
        "settle_uncertain_serials", "_unique_row_units",
        "withdraw_repeated_serials", "withdraw_weak_out_of_sequence",
        "withdraw_isolated_row", "withdraw_low_evidence_labels",
    )
}


def restore() -> None:
    for name, function in ORIGINALS.items():
        setattr(pipeline, name, function)


def apply(config: str) -> bool:
    """Disable one mechanism.  Returns whether the refine pass should run."""
    restore()
    if config == "full":
        return True
    if config == "single_frame":
        return True
    if config == "no_refine":
        return False
    if config == "no_proposals":
        pipeline.propose_missing = lambda *a, **k: []
        pipeline.propose_row_ends = lambda *a, **k: []
        return True
    if config == "no_row_prefix":
        pipeline.build_row_prefix_composites = lambda *a, **k: ([], [])
        pipeline._unique_row_units = lambda rankings, pinned=None: {
            row: (ranking[0] if ranking else (None, 0.0, 0.0))
            for row, ranking in rankings.items()
        }
        return True
    if config == "no_sequence_settle":
        pipeline.settle_uncertain_serials = lambda *a, **k: None
        return True
    if config == "no_withdrawals":
        for name in (
            "withdraw_repeated_serials", "withdraw_weak_out_of_sequence",
            "withdraw_isolated_row", "withdraw_low_evidence_labels",
        ):
            setattr(pipeline, name, lambda *a, **k: None)
        return True
    raise SystemExit(f"unknown config {config}")


def score(engine, config: str, subset: list[tuple[str, str]]) -> dict:
    refine = apply(config)
    recalls, precisions, exact = [], [], 0
    for trial, camera in subset:
        paths = sorted((WORK / "eval" / trial).glob(f"{camera}_move_*.jpg"))
        if config == "single_frame":
            paths = [paths[len(paths) // 2]]
        tracker = pipeline.LabelTracker()
        for phase, path in enumerate(paths):
            image = cv2.imread(str(path))
            if image is None:
                continue
            tracker.update(pipeline.read_frame(engine, image, phase=phase))
        if refine:
            tracker.refine(engine)
        predicted = {
            entry["label"] for entry in tracker.results()
            if entry["resolved"] and entry["label"]
        }
        want = expected_labels(trial.split("__")[0], camera)
        hit = predicted & want
        recalls.append(len(hit) / len(want))
        precisions.append(len(hit) / len(predicted) if predicted else 0.0)
        exact += int(predicted == want)
    restore()
    return {
        "config": config,
        "recall": round(sum(recalls) / len(recalls), 4),
        "precision": round(sum(precisions) / len(precisions), 4),
        "exact": exact,
        "trials": len(subset),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        default="full,single_frame,no_refine,no_proposals,no_row_prefix,"
                "no_sequence_settle,no_withdrawals",
    )
    parser.add_argument("--subset", default="")
    parser.add_argument("--out", default=str(WORK / "results" / "ablation.json"))
    args = parser.parse_args()

    subset = SUBSET
    if args.subset:
        subset = [
            tuple(item.split(":")) for item in args.subset.split(",") if item
        ]

    engine = RecognitionEngine()
    print(f"识别后端: {engine.backend}\n")
    rows = []
    for config in args.configs.split(","):
        row = score(engine, config, subset)
        rows.append(row)
        print(
            f"{row['config']:20s} 召回 {row['recall']:.4f} "
            f"精确 {row['precision']:.4f} 全对 {row['exact']}/{row['trials']}",
            flush=True,
        )
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
