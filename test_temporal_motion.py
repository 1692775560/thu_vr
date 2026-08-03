#!/usr/bin/env python3
"""Deterministic checks for motion-tolerant, arbitrary-prefix temporal OCR."""

from __future__ import annotations

import copy
import os

from ocr_temporal import (
    _build_tracks,
    apply_cross_camera_row_prefix_model,
    apply_fused_to_current,
    fuse_candidate_history,
    scene_compatible,
    stabilize_complete_row_grids,
)
from live_ocr_worker import _apply_configured_sequence


def frame(scale: float, shift_x: float, *, include_prefix: bool) -> list[dict]:
    labels = []
    for row, prefix in ((1, "B05"), (2, "B04")):
        for offset, number in enumerate(range(10, 21)):
            x = shift_x + scale * (350 + offset * 70)
            y = scale * (240 + row * 330)
            ocr_prefix = prefix if include_prefix and offset < 4 else None
            labels.append({
                "row": row,
                "center": [x, y],
                "box": [[x - 22, y - 14], [x + 22, y - 14], [x + 22, y + 14], [x - 22, y + 14]],
                "ocr_prefix": ocr_prefix,
                "ocr_prefix_score": 0.92 if ocr_prefix else 0.0,
                "row_visual_prefix": None,
                "row_visual_prefix_candidates": [],
                "prefix": None,
                "prefix_source": "unread",
                "ocr_number": f"{number:04d}",
                "ocr_number_score": 0.95,
                "number": f"{number:04d}",
                "number_source": "configured_sequence",
                "resolved": False,
                "inferred": True,
                "label": "未读全",
            })
    return labels


def main() -> None:
    os.environ["THU_VR_LABEL_SEQUENCE_START"] = "10"
    os.environ["THU_VR_LABEL_SEQUENCE_END"] = "20"
    os.environ["THU_VR_LABEL_ROW_PREFIX_STEP"] = "-1"
    near = frame(1.0, 0.0, include_prefix=True)
    far = frame(0.68, 180.0, include_prefix=True)
    blurred = frame(0.82, -130.0, include_prefix=False)
    assert scene_compatible(near, far)
    assert scene_compatible(far, blurred)
    changed_scene = copy.deepcopy(far)
    for item in changed_scene:
        if item.get("ocr_prefix"):
            item["ocr_prefix"] = "N05" if item["row"] == 1 else "N04"
    # Prefix changes alone age through the bounded temporal window; OCR
    # prefix hallucinations must not erase motion/anomaly history.
    assert scene_compatible(far, changed_scene)

    # A one-frame serial hallucination must not jump horizontally to the
    # real occurrence of that serial and swap two physical tracks.
    clean_row = [copy.deepcopy(item) for item in near if item["row"] == 1]
    noisy_row = copy.deepcopy(clean_row)
    noisy_row[0]["ocr_number"] = noisy_row[8]["ocr_number"]
    noisy_row[0]["number"] = noisy_row[8]["number"]
    tracks = _build_tracks([clean_row, noisy_row])
    assert len(tracks) == 11
    assert all(len(track["observations"]) == 2 for track in tracks)
    fused = fuse_candidate_history([near, far, blurred])
    assert len(fused) == 22
    output = apply_fused_to_current(copy.deepcopy(blurred), fused)
    predictions = {item["label"] for item in output if item["resolved"]}
    expected = {
        f"{prefix}-{number:04d}"
        for prefix in ("B05", "B04")
        for number in range(10, 21)
    }
    assert predictions == expected

    # If only the lower row is readable, its strong repeated B04 evidence may
    # infer the unread adjacent B05 row using the configured -1 relationship.
    anchor_only = [copy.deepcopy(near), copy.deepcopy(far), copy.deepcopy(blurred)]
    for candidates in anchor_only[:2]:
        for item in candidates:
            if item["row"] == 1:
                item["ocr_prefix"] = None
                item["ocr_prefix_score"] = 0.0
    inferred = fuse_candidate_history(anchor_only)
    assert {item["label"] for item in inferred} == expected

    # The lower physical row can be ordinal row 1 when the upper row is out
    # of view, then become ordinal row 2 after the upper row enters.  Match it
    # by vertical trajectory instead of attaching B04 to the new upper row.
    lower_only = [copy.deepcopy(item) for item in near if item["row"] == 2]
    for item in lower_only:
        item["row"] = 1
    full_after_entry = frame(1.0, 160.0, include_prefix=False)
    ordinal_shift_fused = fuse_candidate_history([
        lower_only,
        copy.deepcopy(lower_only),
        full_after_entry,
    ])
    assert {item["label"] for item in ordinal_shift_fused} == expected

    # A clipped upper row may repeatedly hallucinate another letter.  The
    # complete lower row's stronger pixel evidence selects B dynamically and
    # the conflicting D prefix must never be exposed.
    letter_conflict = frame(1.0, 0.0, include_prefix=False)
    for item in letter_conflict:
        offset = int(item["number"]) - 10
        if item["row"] == 2:
            item["ocr_prefix"] = "B04"
            item["ocr_prefix_score"] = 0.92
        elif offset < 3:
            item["ocr_prefix"] = "D01"
            item["ocr_prefix_score"] = 0.92
    conflict_fused = fuse_candidate_history([
        copy.deepcopy(letter_conflict),
        copy.deepcopy(letter_conflict),
    ])
    conflict_predictions = {item["label"] for item in conflict_fused}
    assert conflict_predictions == expected
    assert not any(label.startswith("D") for label in conflict_predictions)

    # A robust relationship inferred from the current frame must not be
    # overwritten by a stale temporal prefix from an earlier moving frame.
    current_structural = [copy.deepcopy(near[0])]
    current_structural[0].update({
        "prefix": "B05",
        "prefix_source": "row_sequence",
        "number": "0010",
        "number_source": "configured_sequence",
        "resolved": True,
        "inferred": True,
        "label": "B05-0010",
    })
    stale = [{
        "row": 1,
        "center": current_structural[0]["center"],
        "box": current_structural[0]["box"],
        "prefix": "B06",
        "number": "0010",
        "label": "B06-0010",
        "confidence": 0.99,
        "track_observations": 3,
    }]
    protected = apply_fused_to_current(current_structural, stale)
    assert protected[0]["label"] == "B05-0010"

    # Without a configured serial range, two dynamically read adjacent rows
    # establish the generic B05/B04/B03 relationship and override a sparse
    # clipped-row hallucination B00.  The same rule works for any letter.
    os.environ.pop("THU_VR_LABEL_SEQUENCE_START", None)
    os.environ.pop("THU_VR_LABEL_SEQUENCE_END", None)
    dynamic_three = frame(1.0, 0.0, include_prefix=True)
    for offset, number in enumerate(range(3470, 3481)):
        x = 350 + offset * 70
        y = 240 + 3 * 330
        dynamic_three.append({
            "row": 3,
            "center": [x, y],
            "box": [[x - 22, y - 14], [x + 22, y - 14],
                    [x + 22, y + 14], [x - 22, y + 14]],
            "ocr_prefix": "B00" if offset == 0 else None,
            "ocr_prefix_score": 0.90 if offset == 0 else 0.0,
            "row_visual_prefix": None,
            "row_visual_prefix_candidates": [],
            "prefix": None,
            "prefix_source": "unread",
            "ocr_number": f"{number:04d}",
            "ocr_number_score": 0.95,
            "number": f"{number:04d}",
            "number_source": "dynamic_sequence_inferred",
            "resolved": False,
            "inferred": True,
            "label": "未读全",
        })
    dynamic_fused = fuse_candidate_history([
        copy.deepcopy(dynamic_three), copy.deepcopy(dynamic_three)
    ])
    assert {item["prefix"] for item in dynamic_fused} == {"B05", "B04", "B03"}

    # An equally sized clipped row may borrow only the serial columns from a
    # complete +1 row. Prefix letters/numbers still come from pixel/temporal
    # evidence and are not configured here.
    clipped_grid = frame(1.0, 0.0, include_prefix=True)
    for item in clipped_grid:
        item["height"] = 28
        item["raw_texts"] = []
        item["score"] = 0.95
        if item["row"] == 1:
            item["ocr_prefix"] = None
            item["ocr_prefix_score"] = 0.0
            item["ocr_number"] = None
            item["number"] = None
            item["number_source"] = "unread"
        item.pop("expected_number", None)
    _apply_configured_sequence(clipped_grid, "head")
    clipped_top = [item for item in clipped_grid if item["row"] == 1]
    assert [item["number"] for item in clipped_top] == [
        f"{number:04d}" for number in range(10, 21)
    ]
    assert {
        item["number_source"] for item in clipped_top
    } == {"cross_row_sequence_inferred"}

    # Repair a minority of temporally repeated serial hallucinations in a
    # complete row, then use only that validated grid for its unread peer.
    corrupt = frame(1.0, 0.0, include_prefix=False)
    for item in corrupt:
        item["resolved"] = item["row"] == 2
        item["prefix"] = "B04" if item["row"] == 2 else None
        item["prefix_source"] = (
            "spatial_temporal" if item["row"] == 2 else "unread"
        )
        item["label"] = (
            f"B04-{item['number']}" if item["row"] == 2 else "未读全"
        )
        if item["row"] == 1:
            item["number"] = None
            item["ocr_number"] = None
            item["number_source"] = "unread"
    lower = [item for item in corrupt if item["row"] == 2]
    lower[1]["number"] = "0211"
    lower[3]["number"] = "0011"
    stabilized = stabilize_complete_row_grids(corrupt)
    assert {
        item["label"] for item in stabilized if item["row"] == 2
    } == {f"B04-{number:04d}" for number in range(10, 21)}
    assert {
        item["label"] for item in stabilized if item["row"] == 1
    } == {f"B05-{number:04d}" for number in range(10, 21)}

    # Head/base are one contiguous wall. Three agreeing head rows correct a
    # systematic one-row prefix shift in the base view without hardcoding B.
    head_global = []
    for row, prefix in ((1, "B05"), (2, "B04"), (3, "B03")):
        for number in range(10, 21):
            head_global.append({
                "row": row,
                "center": [float(number * 50), float(row * 200)],
                "prefix": prefix,
                "number": f"{number:04d}",
                "label": f"{prefix}-{number:04d}",
                "resolved": True,
                "inferred": False,
            })
    base_shifted = []
    for row, wrong_prefix in ((1, "B01"), (2, "B00")):
        for number in range(10, 21):
            base_shifted.append({
                "row": row,
                "center": [float(number * 50), float(row * 200)],
                "prefix": wrong_prefix,
                "number": f"{number:04d}",
                "label": f"{wrong_prefix}-{number:04d}",
                "resolved": True,
                "inferred": True,
            })
    corrected_base = apply_cross_camera_row_prefix_model(
        base_shifted, "base", {"head": head_global}
    )
    assert {
        item["label"] for item in corrected_base if item["row"] == 1
    } == {f"B02-{number:04d}" for number in range(10, 21)}
    assert {
        item["label"] for item in corrected_base if item["row"] == 2
    } == {f"B01-{number:04d}" for number in range(10, 21)}

    # The same arbitrary serial grid is shared across cameras when two head
    # rows independently establish it. Base labels can therefore resolve in
    # the fast path even if their tiny serial glyphs are unread in one frame.
    base_unread = copy.deepcopy(base_shifted)
    for item in base_unread:
        item["number"] = None
        item["ocr_number"] = None
        item["resolved"] = False
        item["label"] = "未读全"
    resolved_base = apply_cross_camera_row_prefix_model(
        base_unread, "base", {"head": head_global}
    )
    assert {item["label"] for item in resolved_base} == {
        f"{prefix}-{number:04d}"
        for prefix in ("B02", "B01")
        for number in range(10, 21)
    }

    clipped_base = [
        copy.deepcopy(item) for index, item in enumerate(base_unread)
        if index not in (2, 15)
    ]
    resolved_clipped_base = apply_cross_camera_row_prefix_model(
        clipped_base, "base", {"head": head_global}
    )
    assert len(resolved_clipped_base) == 20
    assert all(item["resolved"] for item in resolved_clipped_base)
    assert all(item["expected_label"] for item in resolved_clipped_base)

    # A competing B01/B00 head hallucination would imply B-1 on the visible
    # third head row, so it is invalid across the full observed row span.
    head_shifted = copy.deepcopy(head_global)
    for item in head_shifted:
        if item["row"] <= 2:
            wrong_prefix = "B01" if item["row"] == 1 else "B00"
            item["prefix"] = wrong_prefix
            item["label"] = f"{wrong_prefix}-{item['number']}"
        else:
            item["resolved"] = False
            item["prefix"] = None
            item["label"] = "未读全"
    corrected_head = apply_cross_camera_row_prefix_model(
        head_shifted, "head", {"base": corrected_base}
    )
    assert {
        item["prefix"] for item in corrected_head if item["row"] == 1
    } == {"B05"}
    assert {
        item["prefix"] for item in corrected_head if item["row"] == 2
    } == {"B04"}
    print("temporal-motion-dynamic-prefix-ok")


if __name__ == "__main__":
    main()
