#!/usr/bin/env python3
"""Evaluate the live OCR algorithm on a time-ordered motion-frame capture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR

from live_ocr_worker import (
    ALGORITHM_VERSION,
    annotate,
    draw_candidate_labels,
    recognize_label_candidates,
)
from ocr_temporal import (
    apply_fused_to_current,
    confirm_sequence_anomalies,
    fuse_candidate_history,
    scene_compatible,
    stabilize_complete_row_grids,
)


SUFFIXES = tuple(f"{number:04d}" for number in range(10, 21))


def expected_labels(rows: list[str]) -> set[str]:
    return {f"{row}-{suffix}" for row in rows for suffix in SUFFIXES}


def metrics(predictions: set[str], expected: set[str]) -> dict:
    true_positive = predictions & expected
    false_positive = predictions - expected
    missed = expected - predictions
    return {
        "predicted_count": len(predictions),
        "true_positive": len(true_positive),
        "precision": round(len(true_positive) / len(predictions), 4) if predictions else 0.0,
        "recall": round(len(true_positive) / len(expected), 4) if expected else 0.0,
        "exact": predictions == expected,
        "wrong": sorted(false_positive),
        "missed": sorted(missed),
    }


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def unique_frames(input_dir: Path, camera: str) -> list[Path]:
    result = []
    seen = set()
    for path in sorted(input_dir.rglob(f"{camera}_move_*.jpg")):
        digest = hashlib.sha256(path.read_bytes()).digest()
        if digest not in seen:
            seen.add(digest)
            result.append(path)
    return result


def evaluate_camera(
    engine: RapidOCR,
    input_dir: Path,
    camera: str,
    rows: list[str],
    temporal_window: int,
    annotated_dir: Path | None = None,
) -> dict:
    expected = expected_labels(rows)
    history: deque[list[dict]] = deque(maxlen=temporal_window)
    frames = []
    previous_predictions: set[str] | None = None
    similarities = []

    for path in unique_frames(input_dir, camera):
        started = time.monotonic()
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"cannot read {path}")
        ocr_result, _ = engine(image)
        _, detections, _ = annotate(image, ocr_result)
        labels, candidate_seconds = recognize_label_candidates(
            engine, image, detections, camera
        )
        raw_labels = copy.deepcopy(labels)
        raw_predictions = {
            item["label"] for item in raw_labels if item.get("resolved") and item.get("label")
        }
        history_reset = bool(history and not scene_compatible(history[-1], raw_labels))
        if history_reset:
            history.clear()
        history.append(raw_labels)
        stable = fuse_candidate_history(list(history))
        fused_labels = apply_fused_to_current(labels, stable)
        fused_labels = stabilize_complete_row_grids(fused_labels)
        fused_labels = confirm_sequence_anomalies(fused_labels, list(history))
        if annotated_dir is not None:
            camera_dir = annotated_dir / camera
            camera_dir.mkdir(parents=True, exist_ok=True)
            visualization = image.copy()
            draw_candidate_labels(visualization, fused_labels)
            cv2.imwrite(str(camera_dir / path.name), visualization)
        predictions = {
            item["label"] for item in fused_labels if item.get("resolved") and item.get("label")
        }
        if previous_predictions is not None:
            similarities.append(jaccard(previous_predictions, predictions))
        previous_predictions = predictions
        frames.append({
            "frame": path.name,
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "candidate_seconds": round(candidate_seconds, 2),
            "detection_count": len(detections),
            "candidate_count": len(labels),
            "history_size": len(history),
            "history_reset": history_reset,
            "raw": metrics(raw_predictions, expected),
            "temporal": metrics(predictions, expected),
            "predictions": sorted(predictions),
            "wrong_labels": sorted({
                item.get("observed_label") for item in fused_labels
                if item.get("anomaly_confirmed") and item.get("observed_label")
            }),
            "suspected_wrong_labels": sorted({
                item.get("observed_label") for item in fused_labels
                if item.get("sequence_status") == "suspected_wrong_label"
                and item.get("observed_label")
            }),
            "raw_candidates": [
                {
                    "row": item.get("row"),
                    "center": item.get("center"),
                    "label": item.get("label"),
                    "resolved": item.get("resolved"),
                    "ocr_prefix": item.get("ocr_prefix"),
                    "ocr_prefix_score": item.get("ocr_prefix_score"),
                    "row_visual_prefix": item.get("row_visual_prefix"),
                    "prefix_source": item.get("prefix_source"),
                    "ocr_number": item.get("ocr_number"),
                    "number": item.get("number"),
                    "number_source": item.get("number_source"),
                }
                for item in raw_labels
            ],
        })

    temporal = [frame["temporal"] for frame in frames]
    raw = [frame["raw"] for frame in frames]
    return {
        "camera": camera,
        "expected_rows": rows,
        "expected_count": len(expected),
        "unique_frame_count": len(frames),
        "frames": frames,
        "summary": {
            "raw_mean_recall": round(sum(item["recall"] for item in raw) / len(raw), 4) if raw else 0.0,
            "temporal_mean_recall": round(sum(item["recall"] for item in temporal) / len(temporal), 4) if temporal else 0.0,
            "temporal_mean_precision": round(sum(item["precision"] for item in temporal) / len(temporal), 4) if temporal else 0.0,
            "temporal_exact_frames": sum(item["exact"] for item in temporal),
            "temporal_exact_rate": round(sum(item["exact"] for item in temporal) / len(temporal), 4) if temporal else 0.0,
            "consecutive_jaccard": round(sum(similarities) / len(similarities), 4) if similarities else 1.0,
            "history_resets": sum(frame["history_reset"] for frame in frames),
            "confirmed_wrong_labels": sorted({
                label for frame in frames for label in frame["wrong_labels"]
            }),
            "suspected_wrong_labels": sorted({
                label for frame in frames
                for label in frame["suspected_wrong_labels"]
            }),
        },
    }


def parse_rows(value: str) -> list[str]:
    return [item.strip().upper() for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--head-rows", default="A05,A04")
    parser.add_argument("--base-rows", default="A02,A01")
    parser.add_argument("--temporal-window", type=int, default=5)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--cameras", default="head,base",
        help="comma-separated subset of head,base",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--annotated-dir", type=Path, default=None)
    args = parser.parse_args()

    # Match the live service's site-confirmed serial format.  Prefixes remain
    # unknown and must still come from pixels/temporal evidence.
    os.environ.setdefault("THU_VR_LABEL_ROW_PREFIX_STEP", "-1")
    os.environ.setdefault("THU_VR_LABEL_ROW_PREFIX_ALLOW_SINGLE_ANCHOR", "0")

    engine = RapidOCR(
        intra_op_num_threads=max(1, args.threads),
        inter_op_num_threads=1,
    )
    result = {
        "algorithm_version": ALGORITHM_VERSION,
        "input_dir": str(args.input_dir),
        "temporal_window": args.temporal_window,
        "cameras": {},
    }
    selected_cameras = {
        value.strip() for value in args.cameras.split(",") if value.strip()
    }
    if not selected_cameras or not selected_cameras <= {"head", "base"}:
        parser.error("--cameras must contain head and/or base")
    for camera, rows in (
        ("head", parse_rows(args.head_rows)),
        ("base", parse_rows(args.base_rows)),
    ):
        if camera not in selected_cameras:
            continue
        result["cameras"][camera] = evaluate_camera(
            engine, args.input_dir, camera, rows, args.temporal_window,
            args.annotated_dir,
        )
        summary = result["cameras"][camera]["summary"]
        print(
            f"{camera}: frames={result['cameras'][camera]['unique_frame_count']} "
            f"raw_recall={summary['raw_mean_recall']:.3f} "
            f"temporal_recall={summary['temporal_mean_recall']:.3f} "
            f"exact={summary['temporal_exact_rate']:.3f} "
            f"jaccard={summary['consecutive_jaccard']:.3f}",
            flush=True,
        )
    payload = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
