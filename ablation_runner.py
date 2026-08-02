#!/usr/bin/env python3
"""Run reproducible OCR ablations on captured robot-camera trials."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter
from pathlib import Path

import cv2
from rapidocr_onnxruntime import RapidOCR

from live_ocr_worker import (
    annotate,
    enhance_candidate_crop,
    extract_codes,
    recognize_label_candidates,
)
from ocr_temporal import fuse_candidate_history


ENHANCEMENT_MODES = ("color", "gray", "clahe", "clahe_unsharp", "adaptive")
STEPS_PER_FRAME = 1 + len(ENHANCEMENT_MODES) + 4


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.stem + ".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def write_progress(
    status_path: Path | None,
    *,
    state: str,
    completed_steps: int,
    total_steps: int,
    current_frame: str | None = None,
    current_variant: str | None = None,
    error: str | None = None,
) -> None:
    if status_path is None:
        return
    atomic_json(status_path, {
        "state": state,
        "updated_at": time.time(),
        "completed_steps": completed_steps,
        "total_steps": total_steps,
        "progress": round(completed_steps / total_steps, 4) if total_steps else 0.0,
        "current_frame": current_frame,
        "current_variant": current_variant,
        "error": error,
    })


def score_predictions(expected: set[str], predicted: set[str]) -> dict:
    true_positive = expected & predicted
    false_positive = predicted - expected
    false_negative = expected - predicted
    precision = len(true_positive) / len(predicted) if predicted else 0.0
    recall = len(true_positive) / len(expected) if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "expected_count": len(expected),
        "predicted_count": len(predicted),
        "true_positive_count": len(true_positive),
        "false_positive_count": len(false_positive),
        "false_negative_count": len(false_negative),
        "precision": round(precision, 6),
        "success_rate": round(recall, 6),
        "f1": round(f1, 6),
        "exact_set_match": expected == predicted,
        "correct_labels": sorted(true_positive),
        "false_positive_labels": sorted(false_positive),
        "missed_labels": sorted(false_negative),
    }


def direct_candidate_predictions(labels: list[dict]) -> set[str]:
    return {
        f"{item['ocr_prefix']}-{item['ocr_number']}"
        for item in labels
        if item.get("ocr_prefix") and item.get("ocr_number")
    }


def row_prior_predictions(labels: list[dict], target_rows: list[str]) -> set[str]:
    predictions: set[str] = set()
    for item in labels:
        row_index = int(item.get("row", 0)) - 1
        number = item.get("ocr_number")
        if number and 0 <= row_index < len(target_rows):
            predictions.add(f"{target_rows[row_index]}-{number}")
    return predictions


def dynamic_row_predictions(labels: list[dict]) -> set[str]:
    """Fill a row prefix only from repeated OCR evidence in that image."""
    prefixes = dynamic_row_prefixes(labels)
    predictions: set[str] = set()
    for item in labels:
        row_number = int(item.get("row", 0))
        if row_number in prefixes and item.get("ocr_number"):
            predictions.add(f"{prefixes[row_number]}-{item['ocr_number']}")
    return predictions


def row_visual_predictions(labels: list[dict]) -> set[str]:
    """Use only a prefix read from aligned, repeated top-line pixels."""
    return {
        f"{item['row_visual_prefix']}-{item['ocr_number']}"
        for item in labels
        if item.get("row_visual_prefix") and item.get("ocr_number")
    }


def dynamic_row_prefixes(labels: list[dict]) -> dict[int, str]:
    prefixes = {}
    row_numbers = sorted({int(item.get("row", 0)) for item in labels})
    for row_number in row_numbers:
        row = [item for item in labels if int(item.get("row", 0)) == row_number]
        prefix_counts: Counter[str] = Counter(
            item["ocr_prefix"] for item in row if item.get("ocr_prefix")
        )
        ranked = prefix_counts.most_common(2)
        if not ranked or ranked[0][1] < 2:
            continue
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            continue
        prefixes[row_number] = ranked[0][0]
    return prefixes


def add_derived_dynamic_variant(frame: dict, expected: set[str]) -> None:
    variants = frame.get("variants", {})
    color = variants.get("candidate_color")
    if not color:
        return
    labels = color.get("candidates", [])
    if "candidate_color_dynamic_row_consensus" in variants:
        variants["candidate_color_dynamic_row_consensus"].setdefault("candidates", labels)
        return
    predictions = dynamic_row_predictions(labels)
    confidence_values = []
    for item in labels:
        if not item.get("ocr_number"):
            continue
        score = item.get("score")
        if score is None:
            score = max(
                (float(reading.get("score", 0.0)) for reading in item.get("raw_texts", [])),
                default=0.0,
            )
        confidence_values.append(score)
    variants["candidate_color_dynamic_row_consensus"] = {
        "runtime_seconds": 0.0,
        "predictions": sorted(predictions),
        "metrics": score_predictions(expected, predictions),
        "mean_ocr_confidence": mean_confidence(confidence_values),
        "candidates": labels,
        "note": "从已有彩色候选 OCR 结果派生的动态行前缀共识",
    }


def compact_candidates(labels: list[dict]) -> list[dict]:
    return [
        {
            "center": item.get("center"),
            "row": item.get("row"),
            "ocr_prefix": item.get("ocr_prefix"),
            "ocr_prefix_score": item.get("ocr_prefix_score"),
            "ocr_number": item.get("ocr_number"),
            "ocr_number_score": item.get("ocr_number_score"),
            "row_visual_prefix": item.get("row_visual_prefix"),
            "row_visual_prefix_score": item.get("row_visual_prefix_score"),
            "row_visual_prefix_candidates": item.get("row_visual_prefix_candidates", []),
            "fused_label": item.get("label"),
            "fused_inferred": item.get("inferred"),
            "score": item.get("score"),
            "box": item.get("box"),
            "raw_texts": item.get("raw_texts", []),
        }
        for item in labels
    ]


def mean_confidence(values) -> float:
    scores = [float(value) for value in values if value is not None]
    return round(sum(scores) / len(scores), 6) if scores else 0.0


def run_frame(
    engine: RapidOCR,
    image_path: Path,
    expected: set[str],
    target_rows: list[str],
    camera: str,
    on_variant_complete=None,
) -> dict:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"无法读取图像: {image_path}")

    full_started = time.monotonic()
    full_result, _ = engine(image)
    _, detections, _ = annotate(image, full_result)
    full_codes = extract_codes(detections)
    full_predictions = {item["label"] for item in full_codes}
    variants = {
        "full_frame_raw": {
            "runtime_seconds": round(time.monotonic() - full_started, 4),
            "predictions": sorted(full_predictions),
            "metrics": score_predictions(expected, full_predictions),
            "text_detection_count": len(detections),
            "mean_ocr_confidence": mean_confidence(item.get("score") for item in full_codes),
        }
    }
    if on_variant_complete:
        on_variant_complete("full_frame_raw")

    color_labels: list[dict] | None = None
    for mode in ENHANCEMENT_MODES:
        started = time.monotonic()
        # Every enhancement owns its full-frame OCR pass.  Reusing the raw
        # detections here makes a crop-only comparison and can hide failures
        # introduced by an enhancement before candidate recognition.
        if mode == "color":
            mode_detections = detections
        else:
            enhanced_frame = enhance_candidate_crop(image, mode)
            enhanced_result, _ = engine(enhanced_frame)
            _, mode_detections, _ = annotate(enhanced_frame, enhanced_result)
        labels, engine_seconds = recognize_label_candidates(
            engine,
            image,
            mode_detections,
            camera,
            enhance_mode=mode,
        )
        strict_predictions = direct_candidate_predictions(labels)
        variants[f"candidate_{mode}"] = {
            "runtime_seconds": round(time.monotonic() - started, 4),
            "engine_seconds": round(engine_seconds, 4),
            "candidate_count": len(labels),
            "full_frame_detection_count": len(mode_detections),
            "predictions": sorted(strict_predictions),
            "metrics": score_predictions(expected, strict_predictions),
            "candidates": compact_candidates(labels),
            "mean_ocr_confidence": mean_confidence(
                item.get("score") for item in labels
                if item.get("ocr_prefix") and item.get("ocr_number")
            ),
        }
        if on_variant_complete:
            on_variant_complete(f"candidate_{mode}")
        if mode == "color":
            color_labels = labels

    if color_labels is not None:
        dynamic_predictions = dynamic_row_predictions(color_labels)
        variants["candidate_color_dynamic_row_consensus"] = {
            "runtime_seconds": 0.0,
            "predictions": sorted(dynamic_predictions),
            "metrics": score_predictions(expected, dynamic_predictions),
            "mean_ocr_confidence": mean_confidence(
                item.get("score") for item in color_labels if item.get("ocr_number")
            ),
            "candidates": compact_candidates(color_labels),
            "note": "行前缀必须在当前图像中至少被 OCR 重复读到两次；不使用人工前缀或固定编号范围",
        }
        if on_variant_complete:
            on_variant_complete("candidate_color_dynamic_row_consensus")
        visual_predictions = row_visual_predictions(color_labels)
        variants["candidate_color_row_visual_stack"] = {
            "runtime_seconds": 0.0,
            "predictions": sorted(visual_predictions),
            "metrics": score_predictions(expected, visual_predictions),
            "mean_ocr_confidence": mean_confidence(
                item.get("row_visual_prefix_score") for item in color_labels
                if item.get("row_visual_prefix") and item.get("ocr_number")
            ),
            "candidates": compact_candidates(color_labels),
            "note": "同一行标签上半部对齐叠加后动态识别前缀；不使用已知前缀或编号范围",
        }
        if on_variant_complete:
            on_variant_complete("candidate_color_row_visual_stack")
        known_row_predictions = row_prior_predictions(color_labels, target_rows)
        fused_predictions = {
            item["label"] for item in color_labels
            if item.get("resolved") and item.get("label") != "未读全"
        }
        variants["candidate_color_known_row"] = {
            "runtime_seconds": 0.0,
            "predictions": sorted(known_row_predictions),
            "metrics": score_predictions(expected, known_row_predictions),
            "mean_ocr_confidence": mean_confidence(
                item.get("score") for item in color_labels if item.get("ocr_number")
            ),
            "note": "使用人工给定的可见行前缀；编号仍必须由 OCR 直接读出",
        }
        if on_variant_complete:
            on_variant_complete("candidate_color_known_row")
        variants["candidate_color_spatial_sequence_fusion"] = {
            "runtime_seconds": 0.0,
            "predictions": sorted(fused_predictions),
            "metrics": score_predictions(expected, fused_predictions),
            "mean_ocr_confidence": mean_confidence(
                item.get("score") for item in color_labels if item.get("resolved")
            ),
            "note": "使用已知行前缀与 0010–0020 顺序先验；单独报告，不与纯 OCR 混算",
        }
        if on_variant_complete:
            on_variant_complete("candidate_color_spatial_sequence_fusion")

    return {
        "frame": image_path.name,
        "camera": camera,
        "resolution": [image.shape[1], image.shape[0]],
        "variants": variants,
    }


def labels_for_rows(rows: list[str]) -> set[str]:
    return {f"{row}-{number:04d}" for row in rows for number in range(10, 21)}


def ordered_labels_for_rows(rows: list[str]) -> list[str]:
    return [f"{row}-{number:04d}" for row in rows for number in range(10, 21)]


def candidate_prediction(item: dict, variant: str, dynamic_prefixes: dict[int, str]) -> str | None:
    number = item.get("ocr_number")
    if not number:
        return None
    if "row_visual" in variant:
        prefix = item.get("row_visual_prefix")
    elif "dynamic_row" in variant:
        prefix = dynamic_prefixes.get(int(item.get("row", 0)))
    else:
        prefix = item.get("ocr_prefix")
    return f"{prefix}-{number}" if prefix else None


def build_label_audit(entries: list[dict], target_rows: list[str], variant: str) -> dict:
    ordered_expected = ordered_labels_for_rows(target_rows)
    expected = set(ordered_expected)
    counters = {
        label: {
            "correct_frames": 0,
            "wrong_frames": 0,
            "missed_frames": 0,
            "wrong_predictions": Counter(),
        }
        for label in ordered_expected
    }
    wrong_regions = []
    false_prediction_counts: Counter[str] = Counter()
    false_prediction_regions: dict[str, list[dict]] = {}

    for frame_index, entry in enumerate(entries):
        predictions = set(entry.get("predictions", []))
        candidates = entry.get("candidates", [])
        dynamic_prefixes = dynamic_row_prefixes(candidates) if "dynamic_row" in variant else {}
        regions_by_truth: dict[str, list[dict]] = {}
        regions_by_prediction: dict[str, list[dict]] = {}
        for candidate in candidates:
            truth = candidate.get("fused_label")
            if truth not in expected:
                truth = None
            predicted = candidate_prediction(candidate, variant, dynamic_prefixes)
            region = {
                "frame_index": frame_index,
                "center": candidate.get("center"),
                "box": candidate.get("box"),
                "ground_truth_label": truth,
                "predicted_label": predicted,
                "raw_texts": candidate.get("raw_texts", []),
            }
            if truth:
                regions_by_truth.setdefault(truth, []).append(region)
            if predicted:
                regions_by_prediction.setdefault(predicted, []).append(region)

        for label in ordered_expected:
            if label in predictions:
                counters[label]["correct_frames"] += 1
                continue
            wrong = [
                region for region in regions_by_truth.get(label, [])
                if region.get("predicted_label") and region["predicted_label"] != label
            ]
            if wrong:
                counters[label]["wrong_frames"] += 1
                for region in wrong:
                    counters[label]["wrong_predictions"][region["predicted_label"]] += 1
                    if len(wrong_regions) < 80:
                        wrong_regions.append(region)
            else:
                counters[label]["missed_frames"] += 1

        for predicted in predictions - expected:
            false_prediction_counts[predicted] += 1
            stored_regions = false_prediction_regions.setdefault(predicted, [])
            for region in regions_by_prediction.get(predicted, []):
                if len(stored_regions) < 5:
                    stored_regions.append(region)

    label_rows = []
    total_frames = len(entries)
    for label in ordered_expected:
        values = counters[label]
        label_rows.append({
            "label": label,
            "total_frames": total_frames,
            "correct_frames": values["correct_frames"],
            "wrong_frames": values["wrong_frames"],
            "missed_frames": values["missed_frames"],
            "recognition_rate": round(values["correct_frames"] / total_frames, 6) if total_frames else 0.0,
            "wrong_predictions": [
                {"label": predicted, "frames": count}
                for predicted, count in values["wrong_predictions"].most_common()
            ],
        })
    return {
        "truth_source": "人工选择的行 + 实验板从左到右编号顺序",
        "labels": label_rows,
        "wrong_regions": wrong_regions,
        "false_predictions": [
            {
                "label": predicted,
                "frames": count,
                "regions": false_prediction_regions.get(predicted, []),
            }
            for predicted, count in false_prediction_counts.most_common(30)
        ],
    }


def aggregate_metrics(metrics: list[dict]) -> dict:
    expected_total = sum(item["expected_count"] for item in metrics)
    predicted_total = sum(item["predicted_count"] for item in metrics)
    true_positive_total = sum(item["true_positive_count"] for item in metrics)
    precision = true_positive_total / predicted_total if predicted_total else 0.0
    success_rate = true_positive_total / expected_total if expected_total else 0.0
    f1 = 2 * precision * success_rate / (precision + success_rate) if precision + success_rate else 0.0
    return {
        "expected_total": expected_total,
        "predicted_total": predicted_total,
        "true_positive_total": true_positive_total,
        "precision": round(precision, 6),
        "success_rate": round(success_rate, 6),
        "f1": round(f1, 6),
        "exact_frame_rate": round(
            sum(bool(item["exact_set_match"]) for item in metrics) / len(metrics), 6
        ) if metrics else 0.0,
    }


def row_metrics(entries: list[dict], target_rows: list[str]) -> dict:
    rows = {}
    for row in target_rows:
        expected = labels_for_rows([row])
        metrics = []
        for entry in entries:
            predictions = {
                label for label in entry.get("predictions", [])
                if label.startswith(f"{row}-")
            }
            metrics.append(score_predictions(expected, predictions))
        rows[row] = aggregate_metrics(metrics)
    return rows


def temporal_vote_summary(
    frame_results: list[dict],
    target_rows: list[str],
    source_variant: str,
    output_variant: str,
) -> dict | None:
    source_entries = [
        frame["variants"][source_variant]
        for frame in frame_results
        if source_variant in frame.get("variants", {})
    ]
    if not source_entries:
        return None
    votes: Counter[str] = Counter()
    for entry in source_entries:
        votes.update(set(entry.get("predictions", [])))
    threshold = len(source_entries) // 2 + 1
    predictions = {label for label, count in votes.items() if count >= threshold}
    expected = labels_for_rows(target_rows)
    metrics = score_predictions(expected, predictions)
    rows = {}
    for row in target_rows:
        row_expected = labels_for_rows([row])
        row_predictions = {label for label in predictions if label.startswith(f"{row}-")}
        rows[row] = aggregate_metrics([score_predictions(row_expected, row_predictions)])
    return {
        "variant": output_variant,
        "frame_count": len(source_entries),
        **aggregate_metrics([metrics]),
        "rows": rows,
        "mean_ocr_confidence": mean_confidence(
            entry.get("mean_ocr_confidence") for entry in source_entries
        ),
        "mean_runtime_seconds": 0.0,
        "evaluation_basis": f"{len(source_entries)} 帧多数投票（至少 {threshold} 帧）",
        "predictions": sorted(predictions),
        "label_audit": build_label_audit(source_entries, target_rows, source_variant),
    }


def spatial_temporal_summary(
    frame_results: list[dict],
    target_rows: list[str],
) -> dict | None:
    source_entries = [
        frame["variants"]["candidate_color_row_visual_stack"]
        for frame in frame_results
        if "candidate_color_row_visual_stack" in frame.get("variants", {})
    ]
    if not source_entries:
        return None
    fused = fuse_candidate_history([
        entry.get("candidates", []) for entry in source_entries
    ])
    predictions = {item["label"] for item in fused}
    expected = labels_for_rows(target_rows)
    metrics = score_predictions(expected, predictions)
    rows = {}
    for row in target_rows:
        row_expected = labels_for_rows([row])
        row_predictions = {
            label for label in predictions if label.startswith(f"{row}-")
        }
        rows[row] = aggregate_metrics([
            score_predictions(row_expected, row_predictions)
        ])
    audit_entry = {
        "predictions": sorted(predictions),
        "candidates": [],
    }
    return {
        "variant": "candidate_color_spatial_temporal_fusion",
        "frame_count": len(source_entries),
        **aggregate_metrics([metrics]),
        "rows": rows,
        "mean_ocr_confidence": mean_confidence(
            item.get("confidence") for item in fused
        ),
        "mean_runtime_seconds": 0.0,
        "evaluation_basis": f"{len(source_entries)} 帧按标签位置跟踪，前缀与编号分别融合",
        "predictions": sorted(predictions),
        "temporal_tracks": fused,
        "label_audit": build_label_audit(
            [audit_entry], target_rows, "candidate_color_spatial_temporal_fusion"
        ),
    }


def summarize(frame_results: list[dict], target_rows: list[str]) -> list[dict]:
    variant_names = sorted({name for frame in frame_results for name in frame["variants"]})
    summary = []
    for name in variant_names:
        entries = [frame["variants"][name] for frame in frame_results if name in frame["variants"]]
        metrics = [entry["metrics"] for entry in entries]
        summary.append({
            "variant": name,
            "frame_count": len(entries),
            **aggregate_metrics(metrics),
            "rows": row_metrics(entries, target_rows),
            "label_audit": build_label_audit(entries, target_rows, name),
            "mean_ocr_confidence": mean_confidence(
                entry.get("mean_ocr_confidence") for entry in entries
            ),
            "mean_runtime_seconds": round(sum(entry["runtime_seconds"] for entry in entries) / len(entries), 4),
        })
    for source_variant, output_variant in (
        ("candidate_color", "candidate_color_temporal_vote"),
        ("candidate_color_dynamic_row_consensus", "candidate_color_dynamic_row_temporal_vote"),
    ):
        temporal = temporal_vote_summary(
            frame_results,
            target_rows,
            source_variant,
            output_variant,
        )
        if temporal:
            summary.append(temporal)
    spatial_temporal = spatial_temporal_summary(frame_results, target_rows)
    if spatial_temporal:
        summary.append(spatial_temporal)
    return sorted(summary, key=lambda item: (-item["success_rate"], -item["precision"], item["variant"]))


def ground_truth_check(frame_results: list[dict], target_rows: list[str]) -> dict:
    observed_counts: Counter[str] = Counter()
    for frame in frame_results:
        predictions = frame.get("variants", {}).get("full_frame_raw", {}).get("predictions", [])
        observed_counts.update(
            label.split("-", 1)[0]
            for label in predictions
            if "-" in label
        )
    observed_rows = sorted(observed_counts, key=lambda row: (-observed_counts[row], row))
    selected = set(target_rows)
    observed = set(observed_rows)
    missing = sorted(selected - observed)
    unexpected = sorted(observed - selected)
    warning = ""
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"勾选但原图 OCR 未观察到：{', '.join(missing)}")
        if unexpected:
            parts.append(f"原图 OCR 观察到但未勾选：{', '.join(unexpected)}")
        warning = "；".join(parts) + "。请核对本条记录的真值行，统计不会自动篡改人工真值。"
    return {
        "state": "warning" if warning else "consistent",
        "selected_rows": target_rows,
        "observed_rows": observed_rows,
        "observed_frame_counts": dict(observed_counts),
        "warning": warning,
    }


def build_result(metadata: dict, target_rows_by_camera: dict[str, list[str]], frame_results: list[dict], *, partial: bool = False) -> dict:
    camera_results = {}
    for camera in ("head", "base"):
        camera_frames = [frame for frame in frame_results if frame.get("camera", "head") == camera]
        if not camera_frames:
            continue
        target_rows = target_rows_by_camera.get(camera, [])
        camera_results[camera] = {
            "frame_count": len(camera_frames),
            "target_rows": target_rows,
            "summary": summarize(camera_frames, target_rows),
            "ground_truth_check": ground_truth_check(camera_frames, target_rows),
        }
    head_result = camera_results.get("head", {})
    result = {
        "schema_version": 3,
        "trial_id": metadata.get("trial_id"),
        "distance_m": metadata.get("distance_m"),
        "target_rows_by_camera": target_rows_by_camera,
        "robot": metadata.get("robot"),
        "frames": frame_results,
        "cameras": camera_results,
        # Compatibility for records/API clients created before schema 3.
        "summary": head_result.get("summary", []),
        "ground_truth_check": head_result.get("ground_truth_check"),
    }
    if partial:
        result["partial"] = True
    return result


def write_summary_csv(result: dict, path: Path) -> None:
    csv_rows = []
    for camera, camera_result in result["cameras"].items():
        for item in camera_result["summary"]:
            row = {
                key: value for key, value in item.items()
                if key not in ("rows", "label_audit", "predictions")
            }
            row["camera"] = camera
            for label_row, metrics in item.get("rows", {}).items():
                row[f"{label_row}_success_rate"] = metrics["success_rate"]
                row[f"{label_row}_hits"] = (
                    f"{metrics['true_positive_total']}/{metrics['expected_total']}"
                )
            csv_rows.append(row)
    with path.open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in csv_rows for key in row)) if csv_rows else ["camera", "variant"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)


def run_trial(
    engine: RapidOCR,
    trial_dir: Path,
    max_frames: int | None,
    status_path: Path | None = None,
) -> dict:
    metadata_path = trial_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    target_rows_by_camera = metadata.get("target_rows_by_camera") or {
        "head": metadata.get("target_rows") or [],
        "base": [],
    }
    frame_specs = []
    camera_counts: Counter[str] = Counter()
    for item in metadata.get("frames", []):
        camera = item.get("camera", "head")
        if camera not in ("head", "base"):
            continue
        if max_frames is not None and camera_counts[camera] >= max_frames:
            continue
        camera_counts[camera] += 1
        frame_specs.append((camera, trial_dir / item["file"]))
    if not frame_specs:
        raise RuntimeError(f"{trial_dir.name} 没有可分析的相机帧")
    for camera in camera_counts:
        rows = target_rows_by_camera.get(camera) or []
        if not rows:
            raise RuntimeError(f"{trial_dir.name} 缺少 {camera} 的真值行，不能计算成功率")
        # Ground-truth row prefixes are for scoring only.  Feeding them back
        # into candidate recognition would turn field-unknown prefixes into
        # an oracle and inflate every deployable variant.
        os.environ.pop(f"THU_VR_{camera.upper()}_LABEL_PREFIXES", None)
        os.environ.pop(f"THU_VR_{camera.upper()}_LABEL_PREFIX", None)
    os.environ.pop("THU_VR_LABEL_PREFIXES", None)
    os.environ.pop("THU_VR_LABEL_PREFIX", None)
    os.environ["THU_VR_LABEL_SEQUENCE_START"] = "10"
    os.environ["THU_VR_LABEL_SEQUENCE_END"] = "20"
    total_steps = len(frame_specs) * STEPS_PER_FRAME
    completed_steps = 0
    write_progress(
        status_path,
        state="running",
        completed_steps=0,
        total_steps=total_steps,
    )
    frame_results = []
    for camera, frame_path in frame_specs:
        def on_variant_complete(variant: str) -> None:
            nonlocal completed_steps
            completed_steps += 1
            write_progress(
                status_path,
                state="running",
                completed_steps=completed_steps,
                total_steps=total_steps,
                current_frame=frame_path.name,
                current_variant=variant,
            )

        frame_results.append(run_frame(
            engine,
            frame_path,
            labels_for_rows(target_rows_by_camera[camera]),
            target_rows_by_camera[camera],
            camera,
            on_variant_complete=on_variant_complete,
        ))
        partial_result = build_result(metadata, target_rows_by_camera, frame_results, partial=True)
        atomic_json(trial_dir / "ablation_partial.json", partial_result)
    result = build_result(metadata, target_rows_by_camera, frame_results)
    atomic_json(trial_dir / "ablation_results.json", result)
    write_summary_csv(result, trial_dir / "ablation_summary.csv")
    (trial_dir / "ablation_partial.json").unlink(missing_ok=True)
    write_progress(
        status_path,
        state="complete",
        completed_steps=total_steps,
        total_steps=total_steps,
    )
    return result


def find_trials(path: Path) -> list[Path]:
    if (path / "metadata.json").exists():
        return [path]
    return sorted(metadata.parent for metadata in path.glob("*/metadata.json"))


def main() -> None:
    parser = argparse.ArgumentParser(description="头部相机标签 OCR 消融实验")
    parser.add_argument("path", type=Path, help="单次 trial 目录或 ablation_data 根目录")
    parser.add_argument("--max-frames", type=int, default=None, help="每个条件最多分析多少帧")
    parser.add_argument("--status-file", type=Path, default=None, help="实时写入分析进度 JSON")
    parser.add_argument("--rebuild-summary", action="store_true", help="不重跑 OCR，只从逐帧结果重建分相机/分行汇总")
    args = parser.parse_args()

    trials = find_trials(args.path)
    if args.rebuild_summary:
        for trial_dir in trials:
            metadata = json.loads((trial_dir / "metadata.json").read_text())
            old_result = json.loads((trial_dir / "ablation_results.json").read_text())
            target_rows_by_camera = metadata.get("target_rows_by_camera") or {
                "head": metadata.get("target_rows") or [],
                "base": [],
            }
            frames = old_result.get("frames", [])
            for frame in frames:
                camera = frame.get("camera", "head")
                add_derived_dynamic_variant(
                    frame,
                    labels_for_rows(target_rows_by_camera.get(camera, [])),
                )
            rebuilt = build_result(metadata, target_rows_by_camera, frames)
            atomic_json(trial_dir / "ablation_results.json", rebuilt)
            write_summary_csv(rebuilt, trial_dir / "ablation_summary.csv")
            print(f"{trial_dir.name}: summary rebuilt", flush=True)
        return

    threads = max(1, int(os.environ.get("THU_VR_OCR_THREADS", "2")))
    engine = RapidOCR(intra_op_num_threads=threads, inter_op_num_threads=1)
    for trial_dir in trials:
        started = time.monotonic()
        status_path = args.status_file
        try:
            result = run_trial(engine, trial_dir, args.max_frames, status_path=status_path)
        except Exception as exc:
            write_progress(
                status_path,
                state="error",
                completed_steps=0,
                total_steps=0,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        best = result["summary"][0] if result["summary"] else None
        if best:
            print(
                f"{trial_dir.name}: best={best['variant']} "
                f"success={best['success_rate']:.3f} precision={best['precision']:.3f} "
                f"elapsed={time.monotonic() - started:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
