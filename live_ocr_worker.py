#!/usr/bin/env python3
"""Latest-frame OCR worker for the dual-camera dashboard."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR

from ocr_temporal import (
    apply_fused_to_current,
    apply_cross_camera_row_prefix_model,
    confirm_sequence_anomalies,
    fuse_candidate_history,
    scene_compatible,
    stabilize_complete_row_grids,
)


CAMERAS = ("head", "base")
ALGORITHM_VERSION = "2026-08-02-motion-temporal-v4.22-stable"
FULL_CODE_RE = re.compile(r"^([A-Z])([0-9OQDILSZBAC]{2})[-_ ]?([0-9OQDILSZBAC]{4})$")
PREFIX_RE = re.compile(r"^([A-Z])([0-9OQDILSZBAC]{2})$")
NUMBER_RE = re.compile(r"^[0-9OQDILSZBAC]{4}$")
STICKER_ASPECT_RATIO = 1.78
DIGIT_MAP = str.maketrans({
    "O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
    "S": "5", "Z": "2", "B": "8", "A": "4", "C": "0",
})


def clean_token(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def digits(text: str) -> str:
    return text.translate(DIGIT_MAP)


def box_center(points: np.ndarray) -> tuple[float, float]:
    return float(points[:, 0].mean()), float(points[:, 1].mean())


def _elapsed_seconds(elapsed: object) -> float:
    """Normalize RapidOCR timing output, which may legitimately be None."""
    if elapsed is None:
        return 0.0
    if isinstance(elapsed, (list, tuple, np.ndarray)):
        return sum(_elapsed_seconds(value) for value in elapsed)
    try:
        return float(elapsed)
    except (TypeError, ValueError):
        return 0.0


def extract_codes(detections: list[dict]) -> list[dict]:
    codes: list[dict] = []
    prefixes: list[dict] = []
    numbers: list[dict] = []
    for item in detections:
        token = clean_token(item["text"])
        match = FULL_CODE_RE.fullmatch(token)
        if match:
            code = f"{match.group(1)}{digits(match.group(2))}-{digits(match.group(3))}"
            codes.append({"label": code, "score": item["score"], "center": item["center"]})
            continue
        match = PREFIX_RE.match(token)
        if match:
            prefixes.append({
                "letter": match.group(1),
                "number": digits(match.group(2)),
                "score": item["score"],
                "center": item["center"],
            })
            continue
        if NUMBER_RE.match(token):
            numbers.append({
                "number": digits(token),
                "score": item["score"],
                "center": item["center"],
            })

    used_numbers: set[int] = set()
    for prefix in prefixes:
        choices = []
        for index, number in enumerate(numbers):
            if index in used_numbers:
                continue
            dx = prefix["center"][0] - number["center"][0]
            dy = prefix["center"][1] - number["center"][1]
            distance = float((dx * dx + dy * dy) ** 0.5)
            if abs(dx) <= 65 and distance <= 95:
                choices.append((distance, index, number))
        if not choices:
            continue
        _, index, number = min(choices, key=lambda choice: choice[0])
        used_numbers.add(index)
        codes.append({
            "label": f"{prefix['letter']}{prefix['number']}-{number['number']}",
            "score": round((prefix["score"] + number["score"]) / 2.0, 4),
            "center": [
                (prefix["center"][0] + number["center"][0]) / 2.0,
                (prefix["center"][1] + number["center"][1]) / 2.0,
            ],
        })
    return codes


def detect_label_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Locate the bright rectangular inventory stickers before OCR.

    Full-frame text detection downsizes a 1080p frame and misses the small
    characters.  The sticker itself remains easy to segment, so locating it
    first lets the recognition model see each text line at useful scale.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # A high global percentile finds the well-lit upper row but can erase a
    # darker row in the same frame.  Sticker paper remains above 165 in the
    # validated dark and bright scenes, while OCR evidence later rejects
    # fabric highlights, so cap the threshold instead of letting it rise with
    # the brightest row.
    threshold = int(np.clip(np.percentile(gray, 99.0), 155, 165))
    mask = (gray >= threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = max(12, int(width * 0.006))
    max_width = int(width * 0.14)
    min_height = max(8, int(height * 0.006))
    max_height = int(height * 0.07)
    blobs: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (min_width <= box_width <= max_width and min_height <= box_height <= max_height):
            continue
        fill_ratio = cv2.contourArea(contour) / max(1, box_width * box_height)
        if fill_ratio < 0.45 or box_width / box_height < 0.8:
            continue
        blobs.append((x, y, box_width, box_height))

    box_candidates: list[tuple[tuple[int, int, int, int], bool]] = []
    for x, y, box_width, box_height in blobs:
        # Adjacent stickers can touch after thresholding.  Estimate their
        # count both from aspect ratio and from the median width of isolated
        # stickers in this frame.  The latter remains reliable when a slanted
        # row makes a multi-sticker blob artificially tall.
        center_y = y + box_height / 2.0
        same_row_shapes = [
            (other_width, other_height)
            for _, other_y, other_width, other_height in blobs
            if abs(center_y - (other_y + other_height / 2.0))
            <= max(box_height, other_height) * 1.20
            and other_width / max(1, other_height)
            <= STICKER_ASPECT_RATIO * 1.45
        ]
        reference_width = (
            float(np.median([shape[0] for shape in same_row_shapes]))
            if same_row_shapes else 0.0
        )
        reference_height = (
            float(np.median([shape[1] for shape in same_row_shapes]))
            if same_row_shapes else 0.0
        )
        aspect_estimate = box_width / max(box_height * STICKER_ASPECT_RATIO, 1)
        width_estimate = box_width / reference_width if reference_width else 1.0
        # When a touching blob has the normal row height, its aspect ratio is
        # the better count estimate.  Using only the narrowest isolated-label
        # width can over-split four compressed stickers into five.  Retain the
        # width estimate for genuinely tall/slanted blobs, where aspect ratio
        # is known to under-count.
        if (
            reference_height
            and box_height <= reference_height * 1.30
            and aspect_estimate >= 1.45
            and width_estimate - aspect_estimate >= 0.80
        ):
            estimated_parts = aspect_estimate
        else:
            estimated_parts = max(aspect_estimate, width_estimate)
        # np/Python round uses bankers' rounding (2.5 -> 2), which can merge
        # exactly three touching stickers into two crops.  Sticker counts need
        # conventional half-up rounding instead.
        part_count = int(np.clip(np.floor(estimated_parts + 0.5), 1, 8))
        part_width = box_width / part_count
        for part_index in range(part_count):
            part_x1 = int(round(x + part_index * part_width))
            part_x2 = int(round(x + (part_index + 1) * part_width))
            candidate_width = part_x2 - part_x1
            if candidate_width / box_height < 0.65:
                continue
            # Parts intentionally cut from one wide contour must stay
            # separate.  Only unsplit contours are eligible for the later
            # repair that rejoins threshold-induced fragments.
            box_candidates.append((
                (part_x1, y, candidate_width, box_height),
                part_count == 1,
            ))

    box_candidates.sort(
        key=lambda candidate: (
            candidate[0][1] + candidate[0][3] / 2,
            candidate[0][0],
        )
    )

    # The percentile mask can occasionally split one physical sticker into
    # two touching fragments.  Join only fragments whose union still has the
    # geometry of one sticker; two real adjacent stickers are much wider.
    coalesced: list[tuple[tuple[int, int, int, int], bool]] = []
    for box, merge_eligible in box_candidates:
        if not coalesced:
            coalesced.append((box, merge_eligible))
            continue
        left, left_merge_eligible = coalesced[-1]
        lx, ly, lw, lh = left
        x, y, box_width, box_height = box
        gap = x - (lx + lw)
        vertical_overlap = max(0, min(ly + lh, y + box_height) - max(ly, y))
        overlap_ratio = vertical_overlap / max(1, min(lh, box_height))
        union_x1 = min(lx, x)
        union_y1 = min(ly, y)
        union_x2 = max(lx + lw, x + box_width)
        union_y2 = max(ly + lh, y + box_height)
        union_width = union_x2 - union_x1
        union_height = union_y2 - union_y1
        union_aspect = union_width / max(1, union_height)
        if (
            left_merge_eligible
            and merge_eligible
            and gap <= max(3, int(min(lh, box_height) * 0.10))
            and gap >= -max(3, int(min(lh, box_height) * 0.12))
            and overlap_ratio >= 0.70
            and union_aspect <= STICKER_ASPECT_RATIO * 1.38
        ):
            coalesced[-1] = (
                (union_x1, union_y1, union_width, union_height),
                True,
            )
        else:
            coalesced.append((box, merge_eligible))
    return [box for box, _ in coalesced]


def _parse_prefix(text: str) -> str | None:
    token = clean_token(text)
    match = re.search(r"([A-Z])([0-9OQDILSZB]{2})", token)
    if match:
        return f"{match.group(1)}{digits(match.group(2))}"
    # At long range the zero in prefixes such as A02 often disappears while
    # the outer characters remain legible ("A2").  The site format is one
    # letter plus two digits, so an exact two-character token has a unique,
    # format-preserving recovery.
    short_match = re.fullmatch(r"([A-Z])([0-9OQDILSZB])", token)
    if short_match:
        return f"{short_match.group(1)}0{digits(short_match.group(2))}"
    return None


def _parse_number(text: str) -> str | None:
    token = clean_token(text)
    full_match = FULL_CODE_RE.fullmatch(token)
    if full_match:
        return digits(full_match.group(3))
    # Try the four-character serial before prefix stripping. OCR commonly
    # emits O/C/Q for a leading zero (for example C020 for 0020); those must
    # not be mistaken for a three-character row prefix.
    if NUMBER_RE.fullmatch(token):
        return digits(token)
    if _parse_prefix(token):
        token = token[3:]
    if not NUMBER_RE.fullmatch(token):
        return None
    return digits(token)


def _box_overlap(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> tuple[float, float]:
    """Return IoU and coverage of the smaller box."""
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    intersection_width = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    intersection_height = max(0, min(ay + ah, by + bh) - max(ay, by))
    intersection = intersection_width * intersection_height
    if not intersection:
        return 0.0, 0.0
    first_area = max(1, aw * ah)
    second_area = max(1, bw * bh)
    union = first_area + second_area - intersection
    return intersection / union, intersection / min(first_area, second_area)


def _box_geometry_error(box: tuple[int, int, int, int]) -> float:
    aspect = box[2] / max(1, box[3])
    return abs(float(np.log(max(0.05, aspect) / STICKER_ASPECT_RATIO)))


def _ocr_candidate_boxes(
    detections: list[dict],
    image_shape: tuple[int, ...],
) -> list[tuple[int, int, int, int]]:
    """Turn full-frame number/prefix detections into sticker candidates."""
    image_height, image_width = image_shape[:2]
    prefixes = [item for item in detections if _parse_prefix(item["text"])]
    numbers = [item for item in detections if _parse_number(item["text"])]
    boxes: list[tuple[int, int, int, int]] = []
    for number in numbers:
        number_points = np.asarray(number["box"], dtype=np.float32)
        nx1, ny1 = number_points.min(axis=0)
        nx2, ny2 = number_points.max(axis=0)
        number_width = max(1.0, nx2 - nx1)
        number_height = max(1.0, ny2 - ny1)
        number_x, number_y = number["center"]
        prefix_choices = []
        for prefix in prefixes:
            prefix_x, prefix_y = prefix["center"]
            vertical_gap = number_y - prefix_y
            if abs(number_x - prefix_x) <= max(65.0, number_width * 1.8) and -5.0 <= vertical_gap <= 55.0:
                prefix_choices.append((abs(number_x - prefix_x) + vertical_gap * 0.35, prefix))

        if prefix_choices:
            _, prefix = min(prefix_choices, key=lambda value: value[0])
            prefix_points = np.asarray(prefix["box"], dtype=np.float32)
            all_points = np.vstack((number_points, prefix_points))
            x1, y1 = all_points.min(axis=0)
            x2, y2 = all_points.max(axis=0)
            pad_x = max(3.0, (x2 - x1) * 0.10)
            pad_y = max(2.0, (y2 - y1) * 0.10)
            x1, y1, x2, y2 = x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y
        else:
            # A number line normally occupies the lower half of a sticker.
            x1 = nx1 - max(4.0, number_width * 0.14)
            x2 = nx2 + max(4.0, number_width * 0.14)
            y1 = ny1 - number_height * 1.35
            y2 = ny2 + number_height * 0.25

        x1 = int(np.clip(round(x1), 0, image_width - 1))
        y1 = int(np.clip(round(y1), 0, image_height - 1))
        x2 = int(np.clip(round(x2), x1 + 1, image_width))
        y2 = int(np.clip(round(y2), y1 + 1, image_height))
        boxes.append((x1, y1, x2 - x1, y2 - y1))
    return boxes


def _merge_candidate_boxes(
    image: np.ndarray,
    full_frame_detections: list[dict],
) -> list[tuple[int, int, int, int]]:
    # White sticker contours are substantially more stable than OCR boxes.
    # Keep them as canonical crops and use OCR-derived boxes only to recover a
    # sticker whose bright region was genuinely missed.
    boxes = detect_label_boxes(image)
    reference_width = float(np.median([box[2] for box in boxes])) if boxes else 0.0
    reference_height = float(np.median([box[3] for box in boxes])) if boxes else 0.0
    for candidate in _ocr_candidate_boxes(full_frame_detections, image.shape):
        x, y, width, height = candidate
        if reference_width and not (reference_width * 0.50 <= width <= reference_width * 1.80):
            continue
        if reference_height and not (reference_height * 0.50 <= height <= reference_height * 2.00):
            continue
        center_x, center_y = x + width / 2.0, y + height / 2.0
        duplicate_index = None
        for index, (other_x, other_y, other_width, other_height) in enumerate(boxes):
            other_center_x = other_x + other_width / 2.0
            other_center_y = other_y + other_height / 2.0
            iou, smaller_coverage = _box_overlap(
                candidate,
                (other_x, other_y, other_width, other_height),
            )
            if (
                iou >= 0.16
                or smaller_coverage >= 0.48
                or (
                    abs(center_x - other_center_x) <= max(10.0, min(width, other_width) * 0.48)
                    and abs(center_y - other_center_y) <= max(8.0, min(height, other_height) * 0.60)
                )
            ):
                duplicate_index = index
                break
        if duplicate_index is None:
            boxes.append(candidate)
        elif _box_geometry_error(candidate) + 0.08 < _box_geometry_error(boxes[duplicate_index]):
            # OCR-derived geometry can repair a contour that contains only a
            # narrow bright fragment.  Otherwise retain the more stable white
            # contour as the canonical crop.
            boxes[duplicate_index] = candidate
    boxes.sort(key=lambda box: (box[1] + box[3] / 2.0, box[0] + box[2] / 2.0))
    return boxes


def _reading_line(center_y: float, crop_top: float, crop_bottom: float) -> str:
    """Classify a text fragment as the prefix or serial-number line."""
    relative_y = (center_y - crop_top) / max(1.0, crop_bottom - crop_top)
    if relative_y < 0.48:
        return "prefix"
    if relative_y > 0.52:
        return "number"
    return "full"


def _aligned_prefix_composite(
    image: np.ndarray,
    row: list[dict],
    height_fraction: float,
) -> np.ndarray | None:
    """Align and median-stack the prefix repeated on every sticker in a row."""
    patches: list[np.ndarray] = []
    image_height, image_width = image.shape[:2]
    for item in row:
        points = np.asarray(item.get("box", []), dtype=np.float32)
        if points.shape != (4, 2):
            continue
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        crop_x1 = int(np.clip(round(x1 - max(2.0, width * 0.06)), 0, image_width - 1))
        crop_x2 = int(np.clip(round(x2 + max(2.0, width * 0.06)), crop_x1 + 1, image_width))
        crop_y1 = int(np.clip(round(y1 - max(1.0, height * 0.04)), 0, image_height - 1))
        crop_y2 = int(np.clip(
            round(y1 + height * height_fraction + max(1.0, height * 0.04)),
            crop_y1 + 1,
            image_height,
        ))
        crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        patches.append(cv2.resize(gray, (240, 80), interpolation=cv2.INTER_CUBIC))
    if len(patches) < 3:
        return None

    sharpness = [float(cv2.Laplacian(patch, cv2.CV_32F).var()) for patch in patches]
    reference = patches[int(np.argmax(sharpness))].astype(np.float32) / 255.0
    aligned: list[np.ndarray] = []
    for patch in patches:
        transform = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(
                reference,
                patch.astype(np.float32) / 255.0,
                transform,
                cv2.MOTION_TRANSLATION,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-4),
                None,
                3,
            )
            aligned_patch = cv2.warpAffine(
                patch,
                transform,
                (patch.shape[1], patch.shape[0]),
                flags=cv2.INTER_CUBIC | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        except cv2.error:
            aligned_patch = patch
        aligned.append(aligned_patch)

    composite = np.median(np.stack(aligned), axis=0).astype(np.uint8)
    composite = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(composite)
    return cv2.cvtColor(composite, cv2.COLOR_GRAY2BGR)


def _recognize_candidate_lines(
    engine: RapidOCR,
    image: np.ndarray,
    labels: list[dict],
) -> float:
    """Recognize known prefix/number line crops without text detection.

    At long range a sticker can be only about 45x22 pixels.  The full OCR
    detector often misses such text even though its recognition model can
    still read an enlarged, already-localized line.
    """
    if not labels or not hasattr(engine, "text_rec"):
        return 0.0
    image_height, image_width = image.shape[:2]
    views: list[np.ndarray] = []
    metadata: list[tuple[int, str]] = []
    fast_mode = os.environ.get("THU_VR_OCR_FAST_MODE", "1") != "0"
    for index, item in enumerate(labels):
        points = np.asarray(item.get("box", []), dtype=np.float32)
        if points.shape != (4, 2):
            continue
        x1, y1 = points.min(axis=0)
        x2, y2 = points.max(axis=0)
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        crop_x1 = int(np.clip(round(x1 - 3), 0, image_width - 1))
        crop_x2 = int(np.clip(round(x2 + 3), crop_x1 + 1, image_width))

        # The <=5 s path learns the repeated prefix from two row composites
        # below. Per-sticker prefix crops are retained only for slow ablation
        # and audit runs; doing both on every live frame doubled recognition
        # latency without improving the row identity.
        if not fast_mode:
            crop_y1 = int(np.clip(round(y1 - 3), 0, image_height - 1))
            crop_y2 = int(np.clip(
                round(y1 + height * 0.62 + 3), crop_y1 + 1, image_height
            ))
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size:
                enlarged_prefix = cv2.resize(
                    crop, None, fx=8.0, fy=8.0, interpolation=cv2.INTER_CUBIC
                )
                prefix_gray = cv2.cvtColor(enlarged_prefix, cv2.COLOR_BGR2GRAY)
                prefix_enhanced = cv2.createCLAHE(
                    clipLimit=2.5, tileGridSize=(4, 4)
                ).apply(prefix_gray)
                views.append(cv2.cvtColor(prefix_enhanced, cv2.COLOR_GRAY2BGR))
                metadata.append((index, "prefix"))

        if not item.get("ocr_number") or float(item.get("ocr_number_score") or 0.0) < 0.90:
            crop_y1 = int(np.clip(
                round(y1 + height * 0.40 - 2), 0, image_height - 1
            ))
            crop_y2 = int(np.clip(round(y2 + 2), crop_y1 + 1, image_height))
            crop = image[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size:
                enlarged = cv2.resize(
                    crop, None, fx=6.0, fy=6.0, interpolation=cv2.INTER_CUBIC
                )
                gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.createCLAHE(
                    clipLimit=2.5, tileGridSize=(4, 4)
                ).apply(gray)
                views.append(cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR))
                metadata.append((index, "number"))

    if not views:
        return 0.0
    results, elapsed = engine.text_rec(views)
    for (index, line), (text, score) in zip(metadata, results):
        text = str(text)
        score = float(score)
        item = labels[index]
        item["raw_texts"].append({
            "text": text,
            "score": round(score, 4),
            "source": "candidate_line",
            "line": line,
        })
        if line == "prefix":
            prefix = _parse_prefix(text)
            if prefix and score >= 0.32 and (
                not item.get("ocr_prefix")
                or score > float(item.get("ocr_prefix_score") or 0.0)
            ):
                item["ocr_prefix"] = prefix
                item["ocr_prefix_score"] = round(score, 4)
        else:
            number = _parse_number(text)
            if number and score >= 0.50 and (
                not item.get("ocr_number")
                or score > float(item.get("ocr_number_score") or 0.0)
            ):
                item["ocr_number"] = number
                item["ocr_number_score"] = round(score, 4)
        item["score"] = round(max(float(item.get("score") or 0.0), score), 4)
    return _elapsed_seconds(elapsed)


def _recognize_row_prefixes(
    engine: RapidOCR,
    image: np.ndarray,
    labels: list[dict],
) -> float:
    """Infer a row prefix from pixels, without a configured prefix value."""
    fast_mode = os.environ.get("THU_VR_OCR_FAST_MODE", "1") != "0"
    fractions = (0.52, 0.63) if fast_mode else (0.50, 0.58, 0.65)
    rows = _candidate_rows(labels)
    views: list[np.ndarray] = []
    view_rows: list[int] = []
    for row_index, row in enumerate(rows):
        for fraction in fractions:
            composite = _aligned_prefix_composite(image, row, fraction)
            if composite is not None:
                views.append(composite)
                view_rows.append(row_index)
    if not views or not hasattr(engine, "text_rec"):
        return 0.0

    results, elapsed = engine.text_rec(views)
    evidence_by_row: dict[int, list[dict]] = {}
    digit_evidence_by_row: dict[int, list[dict]] = {}
    for row_index, (text, score) in zip(view_rows, results):
        prefix = _parse_prefix(str(text))
        if prefix:
            evidence_by_row.setdefault(row_index, []).append({
                "prefix": prefix,
                "text": str(text),
                "score": round(float(score), 4),
            })
            continue
        token = clean_token(str(text))
        if re.fullmatch(r"[0-9OQDILSZBA]{2}", token):
            digit_evidence_by_row.setdefault(row_index, []).append({
                "digits": digits(token),
                "text": str(text),
                "score": round(float(score), 4),
            })

    for row_index, row in enumerate(rows):
        evidence = evidence_by_row.get(row_index, [])
        ranked = Counter(item["prefix"] for item in evidence).most_common(2)
        selected = None
        selected_score = 0.0
        if ranked:
            candidate, votes = ranked[0]
            selected_score = max(
                item["score"] for item in evidence if item["prefix"] == candidate
            )
            tied = len(ranked) > 1 and ranked[1][1] == votes
            # One high-confidence crop can still confuse A01/A03 at this
            # distance.  A row-wide prefix requires agreement from at least
            # two independently cropped composite views.
            if not tied and votes >= 2:
                selected = candidate
        if not selected:
            digit_evidence = digit_evidence_by_row.get(row_index, [])
            digit_ranked = Counter(
                item["digits"] for item in digit_evidence
            ).most_common(2)
            letter_ranked = Counter(
                item["ocr_prefix"][0]
                for item in row
                if item.get("ocr_prefix")
            ).most_common(2)
            if digit_ranked and letter_ranked:
                digit_value, digit_votes = digit_ranked[0]
                digit_tied = (
                    len(digit_ranked) > 1
                    and digit_ranked[1][1] == digit_votes
                )
                letter, letter_votes = letter_ranked[0]
                letter_tied = (
                    len(letter_ranked) > 1
                    and letter_ranked[1][1] == letter_votes
                )
                if digit_votes >= 2 and not digit_tied and not letter_tied:
                    selected = f"{letter}{digit_value}"
                    digit_score = max(
                        item["score"] for item in digit_evidence
                        if item["digits"] == digit_value
                    )
                    letter_score = max(
                        float(item.get("ocr_prefix_score") or 0.0)
                        for item in row
                        if (item.get("ocr_prefix") or "").startswith(letter)
                    )
                    selected_score = min(digit_score, letter_score)
                    evidence.append({
                        "prefix": selected,
                        "text": f"{letter}+{digit_value}",
                        "score": round(selected_score, 4),
                    })

        # A median-stacked row view can occasionally erase the last digit
        # (for example A01 -> A00).  When at least three independently
        # cropped stickers agree on another full prefix, that repeated pixel
        # evidence is stronger than the correlated composite views.
        direct_ranked = Counter(
            item["ocr_prefix"] for item in row if item.get("ocr_prefix")
        ).most_common(2)
        if direct_ranked:
            direct_prefix, direct_votes = direct_ranked[0]
            direct_tied = (
                len(direct_ranked) > 1
                and direct_ranked[1][1] == direct_votes
            )
            if direct_votes >= 3 and not direct_tied and selected != direct_prefix:
                selected = direct_prefix
                selected_score = max(
                    float(item.get("ocr_prefix_score") or 0.0)
                    for item in row if item.get("ocr_prefix") == direct_prefix
                )
                # Do not retain the conflicting composite as temporal
                # evidence; the individual crops are already recorded below.
                evidence = [
                    item for item in evidence
                    if item.get("prefix") == direct_prefix
                ]
        for item_index, item in enumerate(row):
            item["row_visual_prefix"] = selected
            item["row_visual_prefix_score"] = round(selected_score, 4)
            item["row_visual_prefix_candidates"] = evidence if item_index == 0 else []
    return _elapsed_seconds(elapsed)


def enhance_candidate_crop(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "color":
        return image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if mode == "gray":
        output = gray
    elif mode in ("clahe", "clahe_unsharp"):
        output = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        if mode == "clahe_unsharp":
            blurred = cv2.GaussianBlur(output, (0, 0), 1.4)
            output = cv2.addWeighted(output, 1.65, blurred, -0.65, 0)
    elif mode == "adaptive":
        output = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
    else:
        raise ValueError(f"未知标签增强模式: {mode}")
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def _candidate_rows(labels: list[dict]) -> list[list[dict]]:
    if not labels:
        return []
    median_height = float(np.median([item["height"] for item in labels]))
    tolerance = max(46.0, median_height * 2.8)
    rows: list[list[dict]] = []
    for item in sorted(labels, key=lambda value: (value["center"][1], value["center"][0])):
        if not rows:
            rows.append([item])
            continue
        row_y = float(np.median([value["center"][1] for value in rows[-1]]))
        if abs(item["center"][1] - row_y) <= tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])
    for row in rows:
        row.sort(key=lambda value: value["center"][0])
        if len(row) >= 4:
            gaps = [
                row[index + 1]["center"][0] - row[index]["center"][0]
                for index in range(len(row) - 1)
            ]
            typical_gap = float(np.median(gaps)) if gaps else 0.0
            isolation_limit = max(220.0, typical_gap * 4.0)
            filtered = []
            for index, item in enumerate(row):
                neighbor_gaps = []
                if index:
                    neighbor_gaps.append(item["center"][0] - row[index - 1]["center"][0])
                if index + 1 < len(row):
                    neighbor_gaps.append(row[index + 1]["center"][0] - item["center"][0])
                if neighbor_gaps and min(neighbor_gaps) <= isolation_limit:
                    filtered.append(item)
            if len(filtered) >= 2:
                row[:] = filtered
    return rows


def _camera_setting(camera: str, name: str) -> str:
    return os.environ.get(f"THU_VR_{camera.upper()}_{name}", os.environ.get(f"THU_VR_{name}", ""))


def _reading_sources(item: dict, kind: str, value: str | None) -> list[str]:
    """Return independent OCR views that read the same prefix or serial."""
    if not value:
        return []
    parser = _parse_prefix if kind == "prefix" else _parse_number
    return sorted({
        str(reading.get("source") or "unknown")
        for reading in item.get("raw_texts", [])
        if parser(str(reading.get("text") or "")) == value
    })


def _configured_sequence(camera: str) -> list[int]:
    start_text = _camera_setting(camera, "LABEL_SEQUENCE_START")
    end_text = _camera_setting(camera, "LABEL_SEQUENCE_END")
    if not start_text or not end_text:
        return []
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        return []
    if 0 <= start <= end <= 9999 and end - start <= 99:
        return list(range(start, end + 1))
    return []


def _retain_configured_row_clusters(labels: list[dict], camera: str) -> list[dict]:
    """Drop isolated lookalike clusters beside a complete configured row."""
    sequence = _configured_sequence(camera)
    expected_count = len(sequence)
    if not expected_count:
        return labels
    retained: list[dict] = []
    rows = _candidate_rows(labels)
    minimum_substantial_count = max(3, int(np.ceil(expected_count * 0.65)))
    has_substantial_row = any(
        len(row) >= minimum_substantial_count for row in rows
    )
    for row in rows:
        # When one or more real grid rows are present, a separate tiny row is
        # overwhelmingly a pair of fabric highlights or scene clutter.  Keep
        # short rows when they are the only visible labels (close/partial
        # views), but do not let them pollute a full-wall evaluation.
        if has_substantial_row and len(row) < minimum_substantial_count:
            continue
        if len(row) <= expected_count:
            retained.extend(row)
            continue
        ordered = sorted(row, key=lambda item: float(item["center"][0]))
        gaps = [
            float(ordered[index + 1]["center"][0])
            - float(ordered[index]["center"][0])
            for index in range(len(ordered) - 1)
        ]
        typical_gap = float(np.median(gaps)) if gaps else 0.0
        split_limit = max(220.0, typical_gap * 5.0)
        clusters: list[list[dict]] = [[ordered[0]]]
        for index, gap in enumerate(gaps):
            if gap > split_limit:
                clusters.append([])
            clusters[-1].append(ordered[index + 1])
        eligible = [cluster for cluster in clusters if len(cluster) <= expected_count]
        if not eligible:
            retained.extend(ordered)
            continue
        selected = max(
            eligible,
            key=lambda cluster: (
                len(cluster) == expected_count,
                -abs(len(cluster) - expected_count),
                sum(bool(item.get("raw_texts")) for item in cluster),
                len(cluster),
            ),
        )
        # Only prune when one cluster explains a substantial row; otherwise a
        # split may simply be the intentional spacing between label groups.
        if len(selected) >= minimum_substantial_count:
            retained.extend(selected)
        else:
            retained.extend(ordered)
    return sorted(
        retained,
        key=lambda item: (float(item["center"][1]), float(item["center"][0])),
    )


def _repair_dynamic_row_sequence(row: list[dict]) -> None:
    """Learn an arbitrary 0000..9999 +1 sequence and expose sparse errors.

    The previous implementation repaired an outlier to the expected value.
    That is useful for reading a clean wall, but it hides a genuinely wrong
    physical sticker.  Keep the pixel-derived value immutable, learn the row
    start from the majority of ``value - physical_index`` votes, and record
    both observed and expected values when they disagree.
    """
    observations: list[int | None] = []
    weighted_offsets: Counter[int] = Counter()
    offset_counts: Counter[int] = Counter()
    for index, item in enumerate(row):
        number = item.get("ocr_number")
        value = int(number) if number and str(number).isdigit() else None
        observations.append(value)
        item["observed_number"] = f"{value:04d}" if value is not None else None
        item["direct_ocr_number"] = item["observed_number"]
        if value is not None:
            offset = value - index
            offset_counts[offset] += 1
            weighted_offsets[offset] += max(
                0.25, float(item.get("ocr_number_score") or 0.0)
            )

    model_offset: int | None = None
    if len(observations) >= 3 and offset_counts:
        ranked = sorted(
            offset_counts,
            key=lambda value: (offset_counts[value], weighted_offsets[value]),
            reverse=True,
        )
        candidate = ranked[0]
        runner_count = offset_counts[ranked[1]] if len(ranked) > 1 else 0
        valid_count = sum(offset_counts.values())
        minimum_support = max(3, int(np.ceil(valid_count * 0.50)))
        if (
            offset_counts[candidate] >= minimum_support
            and offset_counts[candidate] > runner_count
        ):
            model_offset = candidate

    for index, (item, observed) in enumerate(zip(row, observations)):
        item["number_inferred"] = False
        item["number_anomaly_candidate"] = False
        if model_offset is None:
            item["expected_number"] = None
            item["number"] = item["observed_number"]
            item["number_source"] = "ocr_unverified" if observed is not None else "unread"
            item["sequence_status"] = "unverified" if observed is not None else "unread"
            continue

        expected = model_offset + index
        if not 0 <= expected <= 9999:
            item["expected_number"] = None
            item["number"] = item["observed_number"]
            item["number_source"] = "ocr_unverified" if observed is not None else "unread"
            item["sequence_status"] = "unverified" if observed is not None else "unread"
            continue
        item["expected_number"] = f"{expected:04d}"
        if observed is None:
            item["number"] = item["expected_number"]
            item["number_inferred"] = True
            item["number_source"] = "dynamic_sequence_inferred"
            item["sequence_status"] = "unread_inferred"
        elif observed == expected:
            item["number"] = item["observed_number"]
            item["number_source"] = "ocr_sequence_anchor"
            item["sequence_status"] = "correct"
        else:
            # Preserve the conflicting physical reading.  Temporal fusion
            # decides later whether it is a repeatable wrong sticker or a
            # one-frame OCR error.
            item["number"] = item["observed_number"]
            item["number_source"] = "ocr_sequence_conflict"
            item["number_anomaly_candidate"] = True
            item["number_anomaly_sources"] = _reading_sources(
                item, "number", item["observed_number"]
            )
            item["sequence_status"] = "suspected_wrong_label"


def _propagate_equal_row_serial_grid(rows: list[list[dict]]) -> None:
    """Share a learned +1 serial grid with an equally sized adjacent row.

    A row clipped by the image edge can expose all sticker rectangles while
    making every serial unreadable.  When another complete row independently
    establishes a consecutive sequence, equal sticker count and ordering are
    strong geometric evidence that both rows use the same serial columns.
    Partial rows are intentionally excluded.
    """
    references: list[list[str]] = []
    for row in rows:
        values = [item.get("expected_number") for item in row]
        if len(row) < 3 or any(value is None for value in values):
            continue
        integers = [int(str(value)) for value in values]
        if all(right == left + 1 for left, right in zip(integers, integers[1:])):
            references.append([f"{value:04d}" for value in integers])

    if not references:
        return
    reference_counts = Counter(tuple(reference) for reference in references)
    for row in rows:
        if len(row) < 3 or sum(bool(item.get("expected_number")) for item in row) >= 3:
            continue
        compatible = [
            reference for reference in reference_counts
            if len(reference) == len(row)
        ]
        if not compatible:
            continue
        compatible.sort(
            key=lambda reference: (reference_counts[reference], reference),
            reverse=True,
        )
        reference = compatible[0]
        for item, expected in zip(row, reference):
            observed = item.get("observed_number") or item.get("ocr_number")
            item["expected_number"] = expected
            item["number"] = expected
            item["number_inferred"] = observed != expected
            item["number_source"] = "cross_row_sequence_inferred"
            if observed and observed != expected:
                item["number_anomaly_candidate"] = True
                item["number_anomaly_sources"] = _reading_sources(
                    item, "number", str(observed)
                )
                item["sequence_status"] = "suspected_wrong_label"
            elif observed == expected:
                item["sequence_status"] = "correct"
            else:
                item["sequence_status"] = "unread_inferred"


def _apply_configured_sequence(labels: list[dict], camera: str) -> None:
    row_prefixes = [
        clean_token(value)
        for value in _camera_setting(camera, "LABEL_PREFIXES").split(",")
        if re.fullmatch(r"[A-Z][0-9]{2}", clean_token(value))
    ]
    single_prefix = clean_token(_camera_setting(camera, "LABEL_PREFIX"))
    if not re.fullmatch(r"[A-Z][0-9]{2}", single_prefix):
        single_prefix = ""
    sequence = _configured_sequence(camera)
    rows = _candidate_rows(labels)
    prefix_infos: list[dict] = []
    for row_index, row in enumerate(rows):
        visual_prefixes = [
            item["row_visual_prefix"] for item in row
            if item.get("row_visual_prefix")
        ]
        visual_prefix = Counter(visual_prefixes).most_common(1)[0][0] if visual_prefixes else ""
        recognized_prefixes = [
            item["ocr_prefix"] for item in row
            if item.get("ocr_prefix") and re.fullmatch(r"[A-Z][0-9]{2}", item["ocr_prefix"])
        ]
        prefix_counts = Counter(recognized_prefixes)
        ranked_prefixes = prefix_counts.most_common(2)
        prefix_from_row_ocr = (
            ranked_prefixes[0][0]
            if ranked_prefixes and ranked_prefixes[0][1] >= 2 else ""
        )
        fallback_prefix = row_prefixes[row_index] if row_index < len(row_prefixes) else single_prefix
        if (
            visual_prefix
            and prefix_from_row_ocr
            and visual_prefix != prefix_from_row_ocr
            and prefix_counts[prefix_from_row_ocr] >= 3
        ):
            prefix = prefix_from_row_ocr
            prefix_source = "row_ocr"
        else:
            prefix = visual_prefix or prefix_from_row_ocr or fallback_prefix
            prefix_source = (
                "row_visual_stack" if visual_prefix else
                ("row_ocr" if prefix_from_row_ocr else
                 ("configured" if prefix else "unread"))
            )
        prefix_infos.append({"prefix": prefix, "source": prefix_source})

    # Infer unread row prefixes from the configured *relationship* between
    # adjacent rows, never from fixed prefix values.  For example, A04/A03 in
    # one camera establishes a -1 step and dynamically predicts A02; A01 in
    # the other camera then predicts the unread row above as A02.  Only rows
    # containing the complete configured serial grid participate.
    step_text = _camera_setting(camera, "LABEL_ROW_PREFIX_STEP")
    try:
        prefix_step = int(step_text) if step_text else 0
    except ValueError:
        prefix_step = 0

    # Resolve a misleading per-row majority with a relationship supported by
    # independent sticker crops in multiple rows.  Example: a distant A04 row
    # may contain four A00 misreads and three correct A04 reads; an adjacent
    # A05 row makes the generic -1 model A05/A04 the only model with evidence
    # in two physical rows.  This learns the letter and numbers from pixels;
    # no A-specific value is encoded here.
    if prefix_step and -9 <= prefix_step <= 9 and sequence:
        direct_model_scores: Counter[tuple[str, int]] = Counter()
        direct_model_rows: dict[tuple[str, int], set[int]] = {}
        for row_index, row in enumerate(rows):
            if len(row) != len(sequence):
                continue
            counts = Counter(
                item.get("ocr_prefix") for item in row
                if re.fullmatch(r"[A-Z][0-9]{2}", item.get("ocr_prefix") or "")
            )
            for candidate, count in counts.items():
                if count < 2:
                    continue
                match = re.fullmatch(r"([A-Z])([0-9]{2})", candidate)
                assert match is not None
                model = (
                    match.group(1),
                    int(match.group(2)) - prefix_step * row_index,
                )
                direct_model_scores[model] += float(count)
                direct_model_rows.setdefault(model, set()).add(row_index)
        ranked_direct_models = sorted(
            (
                (len(direct_model_rows[model]), score, model)
                for model, score in direct_model_scores.items()
            ),
            reverse=True,
        )
        if ranked_direct_models and ranked_direct_models[0][0] >= 2:
            top_row_support, top_score, (letter, constant) = ranked_direct_models[0]
            runner_key = (
                ranked_direct_models[1][0], ranked_direct_models[1][1]
            ) if len(ranked_direct_models) > 1 else (0, 0.0)
            if (top_row_support, top_score) > runner_key:
                for row_index, (row, info) in enumerate(zip(rows, prefix_infos)):
                    if len(row) != len(sequence):
                        continue
                    number = constant + prefix_step * row_index
                    if 0 <= number <= 99:
                        info["prefix"] = f"{letter}{number:02d}"
                        info["source"] = "row_sequence"

    allow_single_anchor = _camera_setting(
        camera, "LABEL_ROW_PREFIX_ALLOW_SINGLE_ANCHOR"
    ).lower() in ("1", "true", "yes", "on")
    prefix_model: tuple[str, int] | None = None
    prefix_model_support = 0
    if prefix_step and -9 <= prefix_step <= 9 and sequence:
        model_counts: Counter[tuple[str, int]] = Counter()
        for row_index, (row, info) in enumerate(zip(rows, prefix_infos)):
            prefix = info["prefix"]
            match = re.fullmatch(r"([A-Z])([0-9]{2})", prefix or "")
            if match and len(row) == len(sequence):
                constant = int(match.group(2)) - prefix_step * row_index
                model_counts[(match.group(1), constant)] += 1
        ranked_models = model_counts.most_common(2)
        if ranked_models:
            candidate_model, support = ranked_models[0]
            second_support = ranked_models[1][1] if len(ranked_models) > 1 else 0
            if (
                (support >= 2 and support > second_support)
                or (support == 1 and not second_support and allow_single_anchor)
            ):
                prefix_model = candidate_model
                prefix_model_support = support
    if prefix_model:
        letter, constant = prefix_model
        for row_index, (row, info) in enumerate(zip(rows, prefix_infos)):
            if len(row) != len(sequence):
                continue
            number = constant + prefix_step * row_index
            if not 0 <= number <= 99:
                continue
            predicted = f"{letter}{number:02d}"
            if not info["prefix"] or (
                prefix_model_support >= 2 and info["prefix"] != predicted
            ):
                info["prefix"] = predicted
                info["source"] = "row_sequence"

    for row_index, (row, prefix_info) in enumerate(zip(rows, prefix_infos)):
        prefix = prefix_info["prefix"]
        prefix_source = prefix_info["source"]
        if sequence and len(row) <= len(sequence):
            # Align the monotonically ordered stickers to the configured code
            # sequence. OCR readings are anchors; missing readings are filled
            # without pretending that they came from the pixels.
            observations: list[int | None] = []
            for item in row:
                parsed = item.get("ocr_number")
                item["direct_ocr_number"] = parsed
                value = int(parsed) if parsed and int(parsed) in sequence else None
                observations.append(value)

            count = len(row)
            sequence_count = len(sequence)
            if count == sequence_count:
                assignments: list[int | None] = list(sequence)
            else:
                costs = np.full((count, sequence_count), np.inf, dtype=np.float64)
                parents = np.full((count, sequence_count), -1, dtype=np.int32)
                for item_index in range(count):
                    for seq_index, number in enumerate(sequence):
                        if seq_index < item_index or sequence_count - seq_index < count - item_index:
                            continue
                        observed = observations[item_index]
                        read_cost = 0.0 if observed is None else (0.0 if observed == number else 8.0 + abs(observed - number))
                        if count == 1 or sequence_count == 1:
                            position_cost = 0.0
                        else:
                            position_cost = abs(item_index / (count - 1) - seq_index / (sequence_count - 1)) * 0.35
                        local_cost = read_cost + position_cost
                        if item_index == 0:
                            costs[item_index, seq_index] = local_cost
                            continue
                        best_parent = -1
                        best_cost = np.inf
                        for previous in range(seq_index):
                            if costs[item_index - 1, previous] < best_cost:
                                best_cost = costs[item_index - 1, previous]
                                best_parent = previous
                        if best_parent >= 0:
                            costs[item_index, seq_index] = best_cost + local_cost
                            parents[item_index, seq_index] = best_parent

                final_index = int(np.argmin(costs[-1])) if count else -1
                assignments = [None] * count
                if final_index >= 0 and np.isfinite(costs[-1, final_index]):
                    cursor = final_index
                    for item_index in range(count - 1, -1, -1):
                        assignments[item_index] = sequence[cursor]
                        cursor = int(parents[item_index, cursor]) if item_index else -1
            for item, observed, assigned in zip(row, observations, assignments):
                if assigned is not None:
                    item["number"] = f"{assigned:04d}"
                    item["number_inferred"] = observed != assigned
                    item["number_source"] = "configured_sequence"
        elif not sequence:
            _repair_dynamic_row_sequence(row)

        for item in row:
            # Prefer a prefix repeatedly read from pixels in the same row.
            # A configured prefix is only an optional experimental oracle and
            # is intentionally absent from the production service.
            direct_prefix = item.get("ocr_prefix")
            direct_prefix_score = float(item.get("ocr_prefix_score") or 0.0)
            # A single medium-confidence prefix can turn one otherwise valid
            # four-digit serial into a confident false label (A71-0017 in the
            # motion test).  Without row/temporal consensus, require a strong
            # per-sticker prefix before exposing a complete code.
            # A standalone high-score crop is not enough to establish a row
            # prefix: at long range A01 can repeatedly look like A00.  Prefix
            # output requires row-wide or temporal consensus; the direct
            # reading is still retained as anomaly evidence below.
            selected_prefix = prefix or None
            selected_number = (
                item["number"] if "number" in item else item.get("ocr_number")
            )
            if item.get("number_source") in ("ocr_unverified", "unread"):
                # A four-digit crop without row-sequence support is evidence,
                # not yet a formal label.  Temporal fusion may promote it
                # after the exact reading recurs; one-frame values such as
                # 0015 -> 0515 must stay out of the production output.
                selected_number = None
            item["observed_prefix"] = direct_prefix
            item["expected_prefix"] = prefix or None
            prefix_conflict = bool(
                prefix
                and direct_prefix
                and direct_prefix != prefix
                and direct_prefix_score >= 0.76
            )
            item["prefix_anomaly_candidate"] = prefix_conflict
            item["prefix_anomaly_sources"] = (
                _reading_sources(item, "prefix", direct_prefix)
                if prefix_conflict else []
            )
            # A high-confidence conflicting prefix may be the sparse bad
            # physical label requested by the field test.  Preserve it as the
            # observed value; the row majority remains the expected value.
            actual_prefix = direct_prefix if prefix_conflict else selected_prefix
            item["prefix"] = actual_prefix
            item["number"] = selected_number
            item["prefix_source"] = prefix_source if prefix else ("ocr" if item.get("ocr_prefix") else "unread")
            item["resolved"] = bool(actual_prefix and selected_number)
            prefix_inferred = bool(
                actual_prefix
                and item.get("ocr_prefix") != actual_prefix
                and item["prefix_source"] in (
                    "row_visual_stack", "row_ocr", "row_sequence", "configured"
                )
            )
            item["inferred"] = bool(item.get("number_inferred", False) or prefix_inferred)
            expected_number = item.get("expected_number") or selected_number
            expected_prefix = item.get("expected_prefix") or selected_prefix
            item["expected_label"] = (
                f"{expected_prefix}-{expected_number}"
                if expected_prefix and expected_number else None
            )
            observed_number = item.get("observed_number")
            observed_prefix = item.get("observed_prefix")
            item["observed_label"] = (
                f"{observed_prefix or expected_prefix}-{observed_number or expected_number}"
                if (observed_prefix or expected_prefix) and (observed_number or expected_number)
                else None
            )
            item["anomaly_candidate"] = bool(
                item.get("number_anomaly_candidate") or prefix_conflict
            )
            item["anomaly_evidence_sources"] = sorted(set(
                item.get("number_anomaly_sources", [])
                + item.get("prefix_anomaly_sources", [])
            ))
            if item["anomaly_candidate"]:
                item["sequence_status"] = "suspected_wrong_label"
            if item["resolved"]:
                item["label"] = f"{actual_prefix}-{selected_number}"
            else:
                item["label"] = "未读全"
            item["row"] = row_index + 1

    if not sequence:
        _propagate_equal_row_serial_grid(rows)


def recognize_label_candidates(
    engine: RapidOCR,
    image: np.ndarray,
    full_frame_detections: list[dict],
    camera: str,
    enhance_mode: str = "color",
) -> tuple[list[dict], float]:
    boxes = _merge_candidate_boxes(image, full_frame_detections)
    if not boxes:
        return [], 0.0

    columns = max(2, int(os.environ.get("THU_VR_OCR_SHEET_COLUMNS", "6")))
    cell_width = max(120, int(os.environ.get("THU_VR_OCR_CELL_WIDTH", "220")))
    cell_height = max(80, int(os.environ.get("THU_VR_OCR_CELL_HEIGHT", "140")))
    rows = (len(boxes) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 70, dtype=np.uint8)
    image_height, image_width = image.shape[:2]
    labels: list[dict] = []
    sheet_regions: list[tuple[float, float]] = []
    for index, (x, y, width, height) in enumerate(boxes):
        pad_x = max(3, int(width * 0.12))
        pad_y = max(2, int(height * 0.12))
        x1, x2 = max(0, x - pad_x), min(image_width, x + width + pad_x)
        y1, y2 = max(0, y - pad_y), min(image_height, y + height + pad_y)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        available_width, available_height = cell_width - 22, cell_height - 22
        scale = min(available_width / crop.shape[1], available_height / crop.shape[0])
        enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        enlarged = enhance_candidate_crop(enlarged, enhance_mode)
        cell_x = (index % columns) * cell_width
        cell_y = (index // columns) * cell_height
        offset_x = cell_x + (cell_width - enlarged.shape[1]) // 2
        offset_y = cell_y + (cell_height - enlarged.shape[0]) // 2
        sheet[offset_y:offset_y + enlarged.shape[0], offset_x:offset_x + enlarged.shape[1]] = enlarged
        sheet_regions.append((float(offset_y), float(offset_y + enlarged.shape[0])))
        labels.append({
            "index": index,
            "label": "未读全",
            "resolved": False,
            "inferred": False,
            "raw_texts": [],
            "score": 0.0,
            "center": [round(x + width / 2.0, 1), round(y + height / 2.0, 1)],
            "box": [[x, y], [x + width, y], [x + width, y + height], [x, y + height]],
            "height": height,
        })

    # Fast production mode already has stable sticker geometry and performs a
    # batched recognition-only pass for every prefix/number line below.  A
    # second detector over this large contact sheet dominated total latency
    # (roughly 8-12 seconds) while re-reading the same pixels.  Keep it as an
    # opt-in ablation path, but skip it in the live <=5 s pipeline.
    fast_mode = os.environ.get("THU_VR_OCR_FAST_MODE", "1") != "0"
    if fast_mode:
        sheet_result, elapsed = [], 0.0
    else:
        sheet_result, elapsed = engine(sheet)
    for points, text, score in sheet_result or []:
        center_x, center_y = box_center(np.asarray(points, dtype=np.float32))
        column = int(center_x // cell_width)
        row = int(center_y // cell_height)
        index = row * columns + column
        if 0 <= column < columns and 0 <= index < len(labels):
            crop_top, crop_bottom = sheet_regions[index]
            labels[index]["raw_texts"].append({
                "text": str(text),
                "score": round(float(score), 4),
                "source": "candidate_sheet",
                "line": _reading_line(center_y, crop_top, crop_bottom),
            })

    # Full-frame OCR can read number lines that contact-sheet OCR misses. Map
    # those results back to the corresponding sticker and retain both sources.
    for detection in full_frame_detections:
        center_x, center_y = detection["center"]
        choices = []
        for index, item in enumerate(labels):
            points = np.asarray(item["box"])
            x1, y1 = points[:, 0].min() - 12, points[:, 1].min() - 10
            x2, y2 = points[:, 0].max() + 12, points[:, 1].max() + 10
            if x1 <= center_x <= x2 and y1 <= center_y <= y2:
                dx = center_x - item["center"][0]
                dy = center_y - item["center"][1]
                choices.append((dx * dx + dy * dy, index))
        if choices:
            _, index = min(choices)
            points = np.asarray(labels[index]["box"], dtype=np.float32)
            labels[index]["raw_texts"].append({
                "text": detection["text"],
                "score": detection["score"],
                "source": "full_frame",
                "line": _reading_line(
                    center_y,
                    float(points[:, 1].min()),
                    float(points[:, 1].max()),
                ),
            })

    for item in labels:
        unique_texts: list[dict] = []
        seen: set[str] = set()
        for reading in sorted(item["raw_texts"], key=lambda value: value["score"], reverse=True):
            key = clean_token(reading["text"])
            if not key or key in seen:
                continue
            seen.add(key)
            unique_texts.append(reading)
        item["raw_texts"] = unique_texts
        prefix_values = [
            (_parse_prefix(reading["text"]), float(reading["score"]))
            for reading in unique_texts
            if reading.get("line", "full") != "number"
        ]
        number_values = [
            (_parse_number(reading["text"]), float(reading["score"]))
            for reading in unique_texts
            if reading.get("line", "full") != "prefix"
        ]
        prefixes = [(value, score) for value, score in prefix_values if value]
        numbers = [(value, score) for value, score in number_values if value]
        item["ocr_prefix"] = prefixes[0][0] if prefixes else None
        item["ocr_prefix_score"] = round(prefixes[0][1], 4) if prefixes else 0.0
        item["ocr_number"] = numbers[0][0] if numbers else None
        item["ocr_number_score"] = round(numbers[0][1], 4) if numbers else 0.0
        item["score"] = round(max((reading["score"] for reading in unique_texts), default=0.0), 4)

    line_recognition_elapsed = _recognize_candidate_lines(engine, image, labels)

    # A bright fabric edge can resemble a row of stickers. Retain a row only
    # when OCR found text in at least two candidates (or the only candidate in
    # a one-label row); unread stickers inside a proven row are still kept.
    proven_labels: list[dict] = []
    for row in _candidate_rows(labels):
        evidence = sum(bool(item["raw_texts"]) for item in row)
        if evidence >= min(2, len(row)):
            proven_labels.extend(row)
    labels = _retain_configured_row_clusters(proven_labels, camera)

    row_prefix_elapsed = 0.0
    if enhance_mode == "color":
        row_prefix_elapsed = _recognize_row_prefixes(engine, image, labels)
    _apply_configured_sequence(labels, camera)
    labels.sort(key=lambda item: (item["row"], item["center"][0]))
    elapsed_seconds = _elapsed_seconds(elapsed)
    return labels, elapsed_seconds + line_recognition_elapsed + row_prefix_elapsed


def merge_full_frame_evidence(labels: list[dict], detections: list[dict]) -> None:
    """Merge concurrently produced full-frame OCR into structured labels.

    The fast geometry/line path has already learned the row grid.  Full-frame
    OCR is therefore used only as direct pixel evidence at the matching box;
    it cannot replace inferred values or create a row by itself.
    """
    for detection in detections:
        center_x, center_y = detection.get("center") or (0.0, 0.0)
        choices: list[tuple[float, dict]] = []
        for item in labels:
            points = np.asarray(item.get("box", []), dtype=np.float32)
            if points.shape != (4, 2):
                continue
            x1, y1 = points.min(axis=0)
            x2, y2 = points.max(axis=0)
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            if (
                x1 - width * 0.20 <= center_x <= x2 + width * 0.20
                and y1 - height * 0.25 <= center_y <= y2 + height * 0.25
            ):
                dx = center_x - float((x1 + x2) * 0.5)
                dy = center_y - float((y1 + y2) * 0.5)
                choices.append((dx * dx + dy * dy, item))
        if not choices:
            continue
        _, item = min(choices, key=lambda value: value[0])
        text = str(detection.get("text") or "")
        score = float(detection.get("score") or 0.0)
        number = _parse_number(text)
        prefix = _parse_prefix(text)
        if not number and not prefix:
            continue
        line = "number" if number and not prefix else ("prefix" if prefix and not number else "full")
        item.setdefault("raw_texts", []).append({
            "text": text,
            "score": round(score, 4),
            "source": "full_frame",
            "line": line,
        })
        if number and score >= 0.50:
            if score >= float(item.get("ocr_number_score") or 0.0):
                item["ocr_number"] = number
                item["ocr_number_score"] = round(score, 4)
            item["direct_ocr_number"] = number
            item["observed_number"] = number
            expected_number = item.get("expected_number")
            expected_prefix = item.get("expected_prefix") or item.get("prefix")
            if expected_prefix:
                item["observed_label"] = f"{expected_prefix}-{number}"
            if expected_number and number != str(expected_number):
                item["number_anomaly_candidate"] = True
                item["number_anomaly_sources"] = sorted(set(
                    item.get("number_anomaly_sources", []) + ["full_frame"]
                ))
                item["anomaly_candidate"] = True
                item["anomaly_evidence_sources"] = sorted(set(
                    item.get("anomaly_evidence_sources", []) + ["full_frame"]
                ))
                item["sequence_status"] = "suspected_wrong_label"
            elif expected_number == number:
                item["sequence_status"] = "correct"
        if prefix and score >= 0.76:
            if score >= float(item.get("ocr_prefix_score") or 0.0):
                item["ocr_prefix"] = prefix
                item["ocr_prefix_score"] = round(score, 4)
            item["observed_prefix"] = prefix
            expected_prefix = item.get("expected_prefix") or item.get("prefix")
            if expected_prefix and prefix != expected_prefix:
                item["prefix_anomaly_candidate"] = True
                item["prefix_anomaly_sources"] = sorted(set(
                    item.get("prefix_anomaly_sources", []) + ["full_frame"]
                ))
                item["anomaly_candidate"] = True
                item["anomaly_evidence_sources"] = sorted(set(
                    item.get("anomaly_evidence_sources", []) + ["full_frame"]
                ))
                item["sequence_status"] = "suspected_wrong_label"


def audit_crop(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop audit OCR to the wall region that actually contains labels."""
    boxes = detect_label_boxes(image)
    if len(boxes) < 2:
        return image, (0, 0)
    image_height, image_width = image.shape[:2]
    x1 = min(box[0] for box in boxes)
    y1 = min(box[1] for box in boxes)
    x2 = max(box[0] + box[2] for box in boxes)
    y2 = max(box[1] + box[3] for box in boxes)
    median_width = float(np.median([box[2] for box in boxes]))
    median_height = float(np.median([box[3] for box in boxes]))
    pad_x = max(20, int(round(median_width * 1.2)))
    pad_y = max(20, int(round(median_height * 1.5)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(image_width, x2 + pad_x)
    y2 = min(image_height, y2 + pad_y)
    if (x2 - x1) * (y2 - y1) >= image_width * image_height * 0.85:
        return image, (0, 0)
    return image[y1:y2, x1:x2], (x1, y1)


def offset_ocr_result(result, offset: tuple[int, int]):
    """Map audit OCR boxes from crop coordinates to full-frame pixels."""
    offset_x, offset_y = offset
    if not offset_x and not offset_y:
        return result
    delta = np.asarray([offset_x, offset_y], dtype=np.float32)
    return [
        (np.asarray(points, dtype=np.float32) + delta, text, score)
        for points, text, score in (result or [])
    ]


def draw_candidate_labels(output: np.ndarray, labels: list[dict]) -> None:
    for item in labels:
        points = np.asarray(item["box"], dtype=np.int32)
        visible_suspect = bool(
            item.get("sequence_status") == "suspected_wrong_label"
            and (
                item.get("anomaly_structured_displacement")
                or item.get("anomaly_reciprocal")
                or (
                    int(item.get("anomaly_observations") or 0) >= 3
                    and float(item.get("anomaly_peak_score") or 0.0) >= 0.97
                )
            )
        )
        if item.get("anomaly_confirmed"):
            color = (30, 30, 245)
        elif visible_suspect:
            color = (0, 155, 255)
        elif not item["resolved"]:
            # Visible, but deliberately neutral: this is a detected sticker,
            # not yet a recognized result or a warning.
            color = (145, 145, 145)
        elif item["inferred"]:
            color = (255, 190, 45)
        else:
            color = (45, 225, 90)
        thickness = 1 if not item.get("resolved") else 2
        cv2.polylines(output, [points], True, color, thickness, cv2.LINE_AA)
        x = max(0, int(points[:, 0].min()))
        y = max(22, int(points[:, 1].min()) - 5)
        if item.get("anomaly_confirmed"):
            observed = item.get("observed_label") or item.get("label") or "?"
            expected = item.get("expected_label") or "?"
            observed_part = observed.split("-", 1)[-1]
            expected_part = expected.split("-", 1)[-1]
            if observed_part == expected_part:
                observed_part = observed.split("-", 1)[0]
                expected_part = expected.split("-", 1)[0]
            caption = f"ERR:{observed_part}->{expected_part}"
        elif visible_suspect:
            observed = item.get("observed_label") or "?"
            expected = item.get("expected_label") or "?"
            observed_part = observed.split("-", 1)[-1]
            expected_part = expected.split("-", 1)[-1]
            if observed_part == expected_part:
                observed_part = observed.split("-", 1)[0]
                expected_part = expected.split("-", 1)[0]
            caption = f"?{observed_part}->{expected_part}"
        elif not item.get("resolved"):
            # The outline already communicates "detected but unread". Avoid
            # rows of question marks that obscure the source image.
            caption = ""
        else:
            # Prefixes are shared by a row; drawing the complete label above
            # every tightly packed sticker creates unreadable overlap.  The
            # full observed/expected labels remain available in JSON.
            caption = item.get("number") or "?"
        if caption:
            cv2.putText(
                output,
                caption,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                color,
                1,
                cv2.LINE_AA,
            )


def annotate(image: np.ndarray, result) -> tuple[np.ndarray, list[dict], list[dict]]:
    output = image.copy()
    detections: list[dict] = []
    for points, text, score in result or []:
        pts = np.asarray(points, dtype=np.int32)
        center = box_center(pts)
        detections.append({
            "text": str(text),
            "score": round(float(score), 4),
            "center": [round(center[0], 1), round(center[1], 1)],
            "box": pts.tolist(),
        })
        cv2.polylines(output, [pts], True, (0, 220, 255), 2, cv2.LINE_AA)
        x = max(0, int(pts[:, 0].min()))
        y = max(24, int(pts[:, 1].min()) - 7)
        caption = f"{text} {float(score):.2f}"
        cv2.putText(output, caption, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (0, 220, 255), 2, cv2.LINE_AA)

    codes = extract_codes(detections)
    for code in codes:
        x, y = (int(code["center"][0]), int(code["center"][1]))
        cv2.circle(output, (x, y), 7, (40, 230, 90), -1, cv2.LINE_AA)
        cv2.putText(output, code["label"], (x + 10, y - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (40, 230, 90), 2, cv2.LINE_AA)
    return output, detections, codes


def atomic_jpeg(path: Path, image: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.jpg")
    if not cv2.imwrite(str(temporary), image, [int(cv2.IMWRITE_JPEG_QUALITY), 90]):
        raise RuntimeError(f"无法写入 {temporary}")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="双相机最新帧 OCR 工作进程")
    parser.add_argument("--shared-dir", type=Path, required=True)
    parser.add_argument("--camera", choices=CAMERAS)
    args = parser.parse_args()
    args.shared_dir.mkdir(parents=True, exist_ok=True)
    active_cameras = (args.camera,) if args.camera else CAMERAS

    ocr_threads = max(1, int(os.environ.get("THU_VR_OCR_THREADS", "2")))
    print(f"[OCR] 加载 RapidOCR 模型（{ocr_threads} 计算线程）...", flush=True)
    engine = RapidOCR(
        intra_op_num_threads=ocr_threads,
        inter_op_num_threads=1,
    )
    fast_mode = os.environ.get("THU_VR_OCR_FAST_MODE", "1") != "0"
    full_engine = (
        RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
        if fast_mode else None
    )
    full_executor = ThreadPoolExecutor(max_workers=1) if full_engine else None
    print(f"[OCR] 模型就绪，开始轮询 {','.join(active_cameras)} 最新帧", flush=True)
    processed_mtime = {camera: 0 for camera in active_cameras}
    last_started = {camera: 0.0 for camera in active_cameras}
    temporal_window = max(2, int(os.environ.get("THU_VR_OCR_TEMPORAL_WINDOW", "5")))
    candidate_history = {
        camera: deque(maxlen=temporal_window) for camera in active_cameras
    }
    latest_camera_labels: dict[str, list[dict]] = {}
    minimum_interval = float(os.environ.get("THU_VR_OCR_MIN_INTERVAL", "10.0"))

    while True:
        did_work = False
        for camera in active_cameras:
            input_path = args.shared_dir / f"{camera}_input.jpg"
            try:
                source_mtime = input_path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            if source_mtime <= processed_mtime[camera]:
                continue
            if time.monotonic() - last_started[camera] < minimum_interval:
                continue
            did_work = True
            last_started[camera] = time.monotonic()
            started = time.monotonic()
            status_path = args.shared_dir / f"{camera}_status.json"
            try:
                image = cv2.imread(str(input_path))
                if image is None:
                    raise RuntimeError("输入图像读取失败")
                if fast_mode:
                    assert full_executor is not None and full_engine is not None
                    full_image, full_offset = audit_crop(image)
                    full_future = full_executor.submit(full_engine, full_image)
                    result, engine_elapsed = [], 0.0
                else:
                    full_future = None
                    full_offset = (0, 0)
                    result, engine_elapsed = engine(image)
                _, detections, _ = annotate(image, result)
                annotated = image.copy()
                labels, candidate_elapsed = recognize_label_candidates(engine, image, detections, camera)
                if full_future is not None:
                    result, engine_elapsed = full_future.result()
                    result = offset_ocr_result(result, full_offset)
                    _, detections, _ = annotate(image, result)
                    merge_full_frame_evidence(labels, detections)
                    # Rebuild the row model now that both independent OCR
                    # paths have contributed direct pixel observations.
                    _apply_configured_sequence(labels, camera)
                    labels.sort(key=lambda item: (item["row"], item["center"][0]))
                for item in labels:
                    item.pop("height", None)
                raw_labels = copy.deepcopy(labels)
                history = candidate_history[camera]
                if history and not scene_compatible(history[-1], raw_labels):
                    history.clear()
                history.append(raw_labels)
                stable_labels = fuse_candidate_history(list(history))
                labels = apply_fused_to_current(labels, stable_labels)
                labels = stabilize_complete_row_grids(labels)
                # Parallel camera workers exchange their latest stabilized
                # rows through the existing status files.  This preserves the
                # dynamic cross-camera prefix model without serializing OCR.
                for other_camera in CAMERAS:
                    if other_camera == camera:
                        continue
                    other_status_path = args.shared_dir / f"{other_camera}_status.json"
                    try:
                        other_payload = json.loads(other_status_path.read_text())
                        other_labels = other_payload.get("labels")
                        if isinstance(other_labels, list):
                            existing = latest_camera_labels.get(other_camera)
                            new_quality = sum(
                                bool(item.get("resolved")) for item in other_labels
                            )
                            existing_quality = sum(
                                bool(item.get("resolved"))
                                for item in (existing or [])
                            )
                            if (
                                not existing
                                or not scene_compatible(existing, other_labels)
                                or new_quality >= existing_quality
                            ):
                                latest_camera_labels[other_camera] = other_labels
                    except (FileNotFoundError, OSError, ValueError, TypeError):
                        pass
                labels = apply_cross_camera_row_prefix_model(
                    labels, camera, latest_camera_labels
                )
                latest_camera_labels[camera] = copy.deepcopy(labels)
                labels = confirm_sequence_anomalies(labels, list(history))
                draw_candidate_labels(annotated, labels)
                codes = [
                    {
                        "label": item["label"],
                        "score": item["score"],
                        "center": item["center"],
                        "inferred": item["inferred"],
                        "sequence_status": item.get("sequence_status"),
                        "observed_label": item.get("observed_label"),
                        "expected_label": item.get("expected_label"),
                        "anomaly_confirmed": bool(item.get("anomaly_confirmed")),
                    }
                    for item in labels if item["resolved"]
                ]
                elapsed = time.monotonic() - started
                payload = {
                    "camera": camera,
                    "algorithm_version": ALGORITHM_VERSION,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "processing_seconds": round(elapsed, 2),
                    "engine_elapsed": engine_elapsed,
                    "detection_count": len(detections),
                    "candidate_processing_seconds": round(candidate_elapsed, 2),
                    "candidate_count": len(labels),
                    "resolved_count": sum(1 for item in labels if item["resolved"]),
                    "inferred_count": sum(1 for item in labels if item["resolved"] and item["inferred"]),
                    "unresolved_count": sum(1 for item in labels if not item["resolved"]),
                    "temporal_window": len(history),
                    "temporal_label_count": len(stable_labels),
                    "codes": codes,
                    "labels": labels,
                    "detections": detections,
                    "source_mtime_ns": source_mtime,
                }
                atomic_json(status_path, payload)
                atomic_jpeg(args.shared_dir / f"{camera}_annotated.jpg", annotated)
                print(
                    f"[OCR] {camera}: {elapsed:.2f}s, {len(detections)} texts, "
                    f"{len(labels)} stickers, {len(codes)} labels",
                    flush=True,
                )
            except Exception as exc:
                atomic_json(status_path, {
                    "camera": camera,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_mtime_ns": source_mtime,
                })
                print(f"[OCR] {camera} 失败: {type(exc).__name__}: {exc}", flush=True)
            processed_mtime[camera] = source_mtime
        if not did_work:
            time.sleep(0.1)


if __name__ == "__main__":
    main()
