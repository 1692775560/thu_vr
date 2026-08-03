#!/usr/bin/env python3
"""Deterministic checks for arbitrary ranges and sparse physical wrong labels."""

from __future__ import annotations

import copy
import os

from live_ocr_worker import _apply_configured_sequence, _parse_number
from ocr_temporal import confirm_sequence_anomalies


def make_row(letter_prefix: str, start: int) -> list[dict]:
    labels = []
    for index in range(11):
        number = start + index
        prefix = letter_prefix
        if index == 4:
            number = 9999
        if index == 7:
            prefix = "Z42"
        x = 120.0 + index * 72.0
        y = 280.0
        labels.append({
            "center": [x, y],
            "box": [[x - 30, y - 16], [x + 30, y - 16],
                    [x + 30, y + 16], [x - 30, y + 16]],
            "height": 32,
            "label": "未读全",
            "resolved": False,
            "inferred": False,
            "raw_texts": [
                {"text": f"{prefix}-{number:04d}", "score": 0.96,
                 "source": "candidate_sheet"},
                {"text": f"{prefix}-{number:04d}", "score": 0.94,
                 "source": "candidate_line"},
                {"text": f"{prefix}-{number:04d}", "score": 0.92,
                 "source": "full_frame"},
            ],
            "score": 0.96,
            "ocr_prefix": prefix,
            "ocr_prefix_score": 0.96,
            "row_visual_prefix": letter_prefix,
            "row_visual_prefix_candidates": [],
            "row_visual_prefix_score": 0.94,
            "ocr_number": f"{number:04d}",
            "ocr_number_score": 0.97,
        })
    return labels


def main() -> None:
    # Production must not know the site's old 0010..0020 range.
    os.environ.pop("THU_VR_LABEL_SEQUENCE_START", None)
    os.environ.pop("THU_VR_LABEL_SEQUENCE_END", None)
    assert _parse_number("C020") == "0020"

    first = make_row("N42", 3470)
    second = copy.deepcopy(first)
    third = copy.deepcopy(first)
    _apply_configured_sequence(first, "head")
    _apply_configured_sequence(second, "head")
    _apply_configured_sequence(third, "head")

    assert first[0]["label"] == "N42-3470"
    assert first[-1]["label"] == "N42-3480"
    assert first[4]["observed_label"] == "N42-9999"
    assert first[4]["expected_label"] == "N42-3474"
    assert first[7]["observed_label"] == "Z42-3477"
    assert first[7]["expected_label"] == "N42-3477"

    confirmed = confirm_sequence_anomalies(third, [first, second, third])
    assert confirmed[4]["anomaly_confirmed"]
    assert confirmed[4]["sequence_status"] == "wrong_label"
    assert confirmed[4]["label"] == "N42-9999"
    assert confirmed[7]["anomaly_confirmed"]
    assert confirmed[7]["label"] == "Z42-3477"
    assert sum(bool(item.get("anomaly_confirmed")) for item in confirmed) == 2

    # Repeated contact-sheet/line results are still one physical crop. They
    # must not confirm a wrong sticker without full-frame agreement.
    crop_history = [make_row("N42", 3470) for _ in range(4)]
    for frame in crop_history:
        for item in frame:
            item["raw_texts"] = [
                reading for reading in item["raw_texts"]
                if reading["source"] != "full_frame"
            ]
            item["ocr_number_score"] = 0.75
            item["ocr_prefix_score"] = 0.75
        _apply_configured_sequence(frame, "head")
    unconfirmed = confirm_sequence_anomalies(
        crop_history[-1], crop_history
    )
    assert not any(item.get("anomaly_confirmed") for item in unconfirmed)

    strong_crop_history = [make_row("N42", 3470) for _ in range(3)]
    for frame in strong_crop_history:
        for item in frame:
            item["raw_texts"] = [
                reading for reading in item["raw_texts"]
                if reading["source"] != "full_frame"
            ]
        _apply_configured_sequence(frame, "head")
    strong_crop_confirmed = confirm_sequence_anomalies(
        strong_crop_history[-1], strong_crop_history
    )
    assert strong_crop_confirmed[4]["anomaly_confirmed"]
    assert strong_crop_confirmed[7]["anomaly_confirmed"]

    # A blurred current frame that reads the expected value must retain a
    # three-frame confirmed conflict at the same expected row/column.
    clean_current = copy.deepcopy(third)
    for index in (4, 7):
        item = clean_current[index]
        item["anomaly_candidate"] = False
        item["observed_label"] = item["expected_label"]
        item["observed_number"] = item["expected_number"]
        item["observed_prefix"] = item["expected_prefix"]
        item["sequence_status"] = "correct"
    persistent = confirm_sequence_anomalies(
        clean_current, [first, second, third, clean_current]
    )
    assert persistent[4]["anomaly_confirmed"]
    assert persistent[4]["observed_label"] == "N42-9999"
    assert persistent[7]["anomaly_confirmed"]
    assert persistent[7]["observed_label"] == "Z42-3477"

    reciprocal = copy.deepcopy(third)
    for index, observed_number in ((4, "3480"), (10, "3474")):
        item = reciprocal[index]
        item["observed_number"] = observed_number
        item["observed_prefix"] = "N42"
        item["observed_label"] = f"N42-{observed_number}"
        item["ocr_number"] = observed_number
        item["ocr_number_score"] = 0.98
        item["anomaly_candidate"] = True
        item["number_anomaly_candidate"] = True
        item["anomaly_evidence_sources"] = ["candidate_sheet"]
    reciprocal_result = confirm_sequence_anomalies(
        reciprocal, [reciprocal]
    )
    assert reciprocal_result[4]["anomaly_confirmed"]
    assert reciprocal_result[10]["anomaly_confirmed"]

    displaced_history = [copy.deepcopy(third) for _ in range(2)]
    for frame in displaced_history:
        item = frame[4]
        item["observed_number"] = "3480"
        item["observed_prefix"] = "N42"
        item["observed_label"] = "N42-3480"
        item["ocr_number"] = "3480"
        item["ocr_number_score"] = 0.75
        item["ocr_prefix_score"] = 0.75
        item["anomaly_candidate"] = True
        item["number_anomaly_candidate"] = True
        item["anomaly_evidence_sources"] = ["candidate_sheet"]
    displaced = confirm_sequence_anomalies(
        displaced_history[-1], displaced_history
    )
    assert displaced[4]["anomaly_confirmed"]
    assert displaced[4]["anomaly_structured_displacement"]

    # Even a structurally plausible displacement is not confirmed from one
    # frame: 0016 can momentarily look like 0015. Two frames remain fast
    # enough in the balanced worker and suppress that live false alarm.
    single_fast = copy.deepcopy(displaced_history[-1])
    single_fast[4]["ocr_number_score"] = 0.85
    single_fast[4]["direct_ocr_number"] = "3480"
    single_fast_result = confirm_sequence_anomalies(single_fast, [single_fast])
    assert not single_fast_result[4]["anomaly_confirmed"]
    assert single_fast_result[4]["anomaly_observations"] == 1

    one_character_history = [copy.deepcopy(third) for _ in range(2)]
    for frame in one_character_history:
        item = frame[4]
        item["observed_number"] = "3478"
        item["observed_prefix"] = "N42"
        item["observed_label"] = "N42-3478"
        item["ocr_number"] = "3478"
        item["ocr_number_score"] = 0.75
        item["ocr_prefix_score"] = 0.75
        item["anomaly_candidate"] = True
        item["number_anomaly_candidate"] = True
        item["anomaly_evidence_sources"] = ["candidate_sheet"]
    one_character = confirm_sequence_anomalies(
        one_character_history[-1], one_character_history
    )
    assert not one_character[4]["anomaly_confirmed"]
    assert not one_character[4]["anomaly_structured_displacement"]

    # Even crop/full-frame agreement can repeat the same one-glyph mistake.
    # Two 0.93 reads must remain unconfirmed; three 0.98 reads may promote it.
    one_glyph_cross_view = [copy.deepcopy(third) for _ in range(2)]
    for frame in one_glyph_cross_view:
        item = frame[4]
        item["observed_number"] = "3478"
        item["observed_prefix"] = "N42"
        item["observed_label"] = "N42-3478"
        item["ocr_number"] = "3478"
        item["ocr_number_score"] = 0.93
        item["anomaly_candidate"] = True
        item["number_anomaly_candidate"] = True
        item["anomaly_evidence_sources"] = ["candidate_sheet", "full_frame"]
    rejected_cross_view = confirm_sequence_anomalies(
        one_glyph_cross_view[-1], one_glyph_cross_view
    )
    assert not rejected_cross_view[4]["anomaly_confirmed"]

    one_glyph_ultra = [copy.deepcopy(frame) for frame in one_glyph_cross_view]
    one_glyph_ultra.append(copy.deepcopy(one_glyph_cross_view[-1]))
    for frame in one_glyph_ultra:
        frame[4]["ocr_number_score"] = 0.98
    accepted_ultra = confirm_sequence_anomalies(
        one_glyph_ultra[-1], one_glyph_ultra
    )
    assert accepted_ultra[4]["anomaly_confirmed"]

    print("arbitrary-range-sparse-anomaly-ok")


if __name__ == "__main__":
    main()
