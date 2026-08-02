"""Spatially tracked, format-constrained temporal fusion for label OCR."""

from __future__ import annotations

import os
import re
import math
from collections import Counter, defaultdict
from typing import Iterable


def _box_size(item: dict) -> tuple[float, float]:
    box = item.get("box") or []
    if len(box) < 3:
        return 32.0, 22.0
    return (
        max(1.0, float(box[1][0]) - float(box[0][0])),
        max(1.0, float(box[2][1]) - float(box[1][1])),
    )


def _logical_number(item: dict) -> str | None:
    """Return the format-constrained serial used to track a sticker."""
    for key in ("number", "ocr_number"):
        value = item.get(key)
        if value is not None:
            value = str(value)
            if len(value) == 4 and value.isdigit():
                return value
    return None


def _strong_row_prefixes(candidates: list[dict]) -> dict[int, str]:
    """Collect only repeated, pixel-derived prefix evidence per row."""
    result = {}
    by_row: dict[int, list[dict]] = defaultdict(list)
    for item in candidates:
        by_row[int(item.get("row", 0))].append(item)
    for row, items in by_row.items():
        votes: Counter[str] = Counter()
        for item in items:
            if item.get("ocr_prefix"):
                votes[str(item["ocr_prefix"])] += 1
        ranked = votes.most_common(2)
        # Scene resets discard the entire temporal window, so they need more
        # conservative evidence than row-prefix recognition itself. A visual
        # stack and one sticker crop share pixels and must not count as two
        # independent labels here.
        if ranked and ranked[0][1] >= 2 and (
            len(ranked) == 1 or ranked[0][1] > ranked[1][1]
        ):
            result[row] = ranked[0][0]
    return result


def scene_compatible(previous: list[dict], current: list[dict]) -> bool:
    """Return whether two OCR samples show substantially the same geometry."""
    if not previous or not current:
        return False

    # Prefix OCR itself can be systematically wrong for several consecutive
    # frames (for example A04 -> A00). It must not erase the evidence window
    # used to detect a physical wrong sticker. Prefix changes age naturally
    # through the bounded history; scene compatibility is geometric.

    # During straight motion every sticker can shift and scale by much more
    # than its box width.  The four-digit serial assignment is invariant, so
    # use row+serial overlap before falling back to absolute image geometry.
    previous_keys = {
        (int(item.get("row", 0)), number)
        for item in previous
        if (number := _logical_number(item))
    }
    current_keys = {
        (int(item.get("row", 0)), number)
        for item in current
        if (number := _logical_number(item))
    }
    if previous_keys and current_keys:
        overlap = len(previous_keys & current_keys) / max(
            1, min(len(previous_keys), len(current_keys))
        )
        if overlap >= 0.45:
            return True
    matches = 0
    for item in current:
        row = int(item.get("row", 0))
        center = item.get("center") or [0.0, 0.0]
        width, height = _box_size(item)
        distances = []
        for other in previous:
            if int(other.get("row", 0)) != row:
                continue
            other_center = other.get("center") or [0.0, 0.0]
            dx = abs(float(center[0]) - float(other_center[0]))
            dy = abs(float(center[1]) - float(other_center[1]))
            if dx <= max(30.0, width * 0.9) and dy <= max(24.0, height * 1.2):
                distances.append(dx + dy)
        if distances:
            matches += 1
    return matches / max(1, min(len(previous), len(current))) >= 0.55


def _build_tracks(candidate_history: list[list[dict]]) -> list[dict]:
    tracks: list[dict] = []
    for frame_index, candidates in enumerate(candidate_history):
        used_tracks: set[int] = set()
        for item_index, item in enumerate(sorted(
            candidates,
            key=lambda value: (int(value.get("row", 0)), float((value.get("center") or [0])[0])),
        )):
            row = int(item.get("row", 0))
            center = item.get("center") or [0.0, 0.0]
            width, height = _box_size(item)
            logical_number = _logical_number(item)
            choices = []
            for track_index, track in enumerate(tracks):
                if track_index in used_tracks:
                    continue
                dx = abs(float(center[0]) - track["center_x"])
                dy = abs(float(center[1]) - track["center_y"])
                if (
                    logical_number
                    and track.get("logical_number") == logical_number
                    and track["row"] == row
                    and dx <= max(320.0, min(width, track["width"]) * 8.0)
                    and dy <= max(180.0, min(height, track["height"]) * 6.0)
                ):
                    choices.append((-1.0, track_index))
                    continue
                if track["row"] != row:
                    continue
                if dx <= max(18.0, min(width, track["width"]) * 0.68) and dy <= max(16.0, min(height, track["height"]) * 0.85):
                    choices.append((dx + dy * 0.5, track_index))
            if choices:
                _, track_index = min(choices)
                track = tracks[track_index]
            else:
                track_index = len(tracks)
                track = {
                    "row": row,
                    "center_x": float(center[0]),
                    "center_y": float(center[1]),
                    "width": width,
                    "height": height,
                    "logical_number": logical_number,
                    "observations": [],
                }
                tracks.append(track)
            used_tracks.add(track_index)
            track["observations"].append((frame_index, item_index, item))
            observation_count = len(track["observations"])
            track["center_x"] += (float(center[0]) - track["center_x"]) / observation_count
            track["center_y"] += (float(center[1]) - track["center_y"]) / observation_count
            track["width"] += (width - track["width"]) / observation_count
            track["height"] += (height - track["height"]) / observation_count
            if logical_number and not track.get("logical_number"):
                track["logical_number"] = logical_number
    return tracks


def _row_prefixes(candidate_history: list[list[dict]]) -> dict[int, dict]:
    scores: dict[int, Counter[str]] = defaultdict(Counter)
    frames: dict[int, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    evidence_counts: dict[int, Counter[str]] = defaultdict(Counter)
    direct_evidence_counts: dict[int, Counter[str]] = defaultdict(Counter)
    prefix_y_values: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for frame_index, candidates in enumerate(candidate_history):
        by_row: dict[int, list[dict]] = defaultdict(list)
        for item in candidates:
            by_row[int(item.get("row", 0))].append(item)
        for row, items in by_row.items():
            row_center_y = sum(
                float((item.get("center") or [0.0, 0.0])[1]) for item in items
            ) / max(1, len(items))
            structural_prefix = next((
                item.get("prefix")
                for item in items
                if item.get("prefix_source") in ("row_sequence", "configured")
                and item.get("prefix")
            ), None)
            if structural_prefix:
                # One independent structural vote per row/frame.  It is
                # stronger than correlated OCR crops but still must recur in
                # two frames before temporal propagation.
                scores[row][structural_prefix] += 4.0
                evidence_counts[row][structural_prefix] += 1
                prefix_y_values[row][structural_prefix].append(row_center_y)
                frames[row][structural_prefix].add(frame_index)

            # Visual-stack evidence is repeated on each item; consume it once
            # per row/frame and retain each independent crop-height view.
            visual_evidence = next((
                item.get("row_visual_prefix_candidates")
                for item in items if item.get("row_visual_prefix_candidates")
            ), [])
            for evidence in visual_evidence:
                prefix = evidence.get("prefix")
                if not prefix:
                    continue
                score = max(0.05, float(evidence.get("score") or 0.0)) * 2.0
                scores[row][prefix] += score
                evidence_counts[row][prefix] += 1
                prefix_y_values[row][prefix].append(row_center_y)
                frames[row][prefix].add(frame_index)

            for item in items:
                prefix = item.get("ocr_prefix")
                if not prefix:
                    continue
                score = max(0.05, float(
                    item.get("ocr_prefix_score") or item.get("score") or 0.0
                ))
                scores[row][prefix] += score
                evidence_counts[row][prefix] += 1
                direct_evidence_counts[row][prefix] += 1
                prefix_y_values[row][prefix].append(
                    float((item.get("center") or [0.0, row_center_y])[1])
                )
                frames[row][prefix].add(frame_index)

    selected: dict[int, dict] = {}
    for row, row_scores in scores.items():
        ranked = row_scores.most_common(2)
        if not ranked:
            continue
        prefix, score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        total_score = sum(row_scores.values())
        count = evidence_counts[row][prefix]
        frame_count = len(frames[row][prefix])
        dominance = score / total_score if total_score else 0.0
        if count < 2:
            continue
        # Three agreeing per-sticker crops are independent enough to seed a
        # moving sequence immediately.  Otherwise require recurrence in two
        # frames before propagating a row prefix.
        direct_count = direct_evidence_counts[row][prefix]
        if frame_count < 2 and direct_count < 3:
            continue
        if dominance < 0.40 or (second_score and score < second_score * 1.18):
            continue
        selected[row] = {
            "prefix": prefix,
            "score": round(score, 4),
            "confidence": round(dominance, 4),
            "evidence_count": count,
            "frame_count": frame_count,
            "direct_evidence_count": direct_count,
            "center_y": round(
                sorted(prefix_y_values[row][prefix])[
                    len(prefix_y_values[row][prefix]) // 2
                ],
                1,
            ),
        }
    return selected


def _infer_adjacent_row_prefixes(
    candidate_history: list[list[dict]],
    selected: dict[int, dict],
) -> dict[int, dict]:
    """Fill unread adjacent rows from a stable dynamic prefix relationship."""
    if not candidate_history or not selected:
        return selected
    try:
        step = int(os.environ.get("THU_VR_LABEL_ROW_PREFIX_STEP", "0"))
        sequence_start = int(os.environ.get("THU_VR_LABEL_SEQUENCE_START", "-1"))
        sequence_end = int(os.environ.get("THU_VR_LABEL_SEQUENCE_END", "-1"))
    except ValueError:
        return selected
    if not step:
        return selected
    expected_count = (
        sequence_end - sequence_start + 1
        if 0 <= sequence_start <= sequence_end <= 9999 else None
    )
    if expected_count is not None and expected_count < 3:
        return selected

    # Rows on the same inventory wall share the letter and use the configured
    # numeric row step.  A row clipped by the image edge can repeatedly turn
    # A01 into D01; when another visible row supplies substantially more
    # independent per-sticker evidence for A, keep that dynamically observed
    # letter and discard the conflicting one.  Nothing here assumes that the
    # letter is A -- B/N/etc. win by the same evidence rule.
    selected_letters = {
        str(evidence.get("prefix") or "")[:1]
        for evidence in selected.values()
        if re.fullmatch(r"[A-Z][0-9]{2}", evidence.get("prefix") or "")
    }
    if len(selected_letters) > 1:
        current_letter_counts: Counter[str] = Counter()
        for item in candidate_history[-1]:
            prefix = item.get("ocr_prefix") or ""
            if re.fullmatch(r"[A-Z][0-9]{2}", prefix):
                current_letter_counts[prefix[0]] += 1
        ranked_letters = current_letter_counts.most_common(2)
        dominant_letter = None
        if ranked_letters and ranked_letters[0][1] >= 3:
            runner_count = ranked_letters[1][1] if len(ranked_letters) > 1 else 0
            if ranked_letters[0][1] >= runner_count + 2:
                dominant_letter = ranked_letters[0][0]
        if dominant_letter:
            selected = {
                row: evidence for row, evidence in selected.items()
                if str(evidence.get("prefix") or "").startswith(dominant_letter)
            }
        else:
            # Ambiguous cross-row letters are not safe to propagate.
            return {}

    latest_rows: dict[int, set[str]] = defaultdict(set)
    for item in candidate_history[-1]:
        number = _logical_number(item)
        if number:
            latest_rows[int(item.get("row", 0))].add(number)
    eligible_rows = {
        row for row, numbers in latest_rows.items()
        if len(numbers) >= (
            max(3, int(expected_count * 0.65))
            if expected_count is not None else 3
        )
    }
    if len(eligible_rows) < 2:
        return selected

    row_center_y = {
        row: sum(
            float((item.get("center") or [0.0, 0.0])[1])
            for item in candidate_history[-1]
            if int(item.get("row", 0)) == row
        ) / max(
            1,
            sum(
                int(item.get("row", 0)) == row
                for item in candidate_history[-1]
            ),
        )
        for row in eligible_rows
    }
    mapped_selected: list[tuple[int, dict]] = []
    for original_row, evidence in selected.items():
        evidence_y = float(evidence.get("center_y", row_center_y.get(original_row, 0.0)))
        target_row = min(
            eligible_rows,
            key=lambda row: abs(row_center_y[row] - evidence_y),
        )
        if abs(row_center_y[target_row] - evidence_y) <= 180.0:
            mapped_selected.append((target_row, evidence))

    models: Counter[tuple[str, int]] = Counter()
    for row, evidence in mapped_selected:
        match = re.fullmatch(r"([A-Z])([0-9]{2})", evidence.get("prefix") or "")
        if not match:
            continue
        models[(match.group(1), int(match.group(2)) - step * row)] += 1
    ranked = models.most_common(2)
    if not ranked or (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
        return selected
    (letter, constant), model_support = ranked[0]
    supporting = [
        evidence for row, evidence in mapped_selected
        if evidence.get("prefix") == f"{letter}{constant + step * row:02d}"
    ]
    if not supporting:
        return selected
    anchor_confidence = max(float(item.get("confidence") or 0.0) for item in supporting)
    anchor_frames = max(int(item.get("frame_count") or 0) for item in supporting)
    anchor_direct = max(int(item.get("direct_evidence_count") or 0) for item in supporting)
    if anchor_frames < 2 and anchor_direct < 3:
        return selected

    # Re-key stable prefixes to the current frame's vertical rows.  This is
    # what prevents a lower row that used to be ordinal row 1 from being
    # attached to a newly entered upper row.
    output: dict[int, dict] = {}
    for row, evidence in mapped_selected:
        existing = output.get(row)
        if (
            existing is None
            or float(evidence.get("confidence") or 0.0)
            > float(existing.get("confidence") or 0.0)
        ):
            output[row] = dict(evidence)
    for row in eligible_rows:
        number = constant + step * row
        if not 0 <= number <= 99:
            continue
        predicted = f"{letter}{number:02d}"
        existing = output.get(row)
        if existing and existing.get("prefix") == predicted:
            continue
        # Two adjacent rows supporting one dynamic model outweigh an isolated
        # systematic crop hallucination on a clipped row.  A genuinely new
        # row letter is retained when four or more direct sticker crops agree.
        if existing and (
            model_support < 2
            or (
                int(existing.get("direct_evidence_count") or 0) >= 4
                and float(existing.get("confidence") or 0.0) >= 0.65
            )
        ):
            continue
        output[row] = {
            "prefix": predicted,
            "score": round(anchor_confidence * 0.9, 4),
            "confidence": round(anchor_confidence * 0.9, 4),
            "evidence_count": 1,
            "frame_count": anchor_frames,
            "direct_evidence_count": 0,
            "source": "temporal_row_sequence",
            "center_y": round(row_center_y[row], 1),
        }
    return output


def _fuse_number(observations: Iterable[tuple[int, int, dict]]) -> dict | None:
    values: list[tuple[str, float]] = []
    frame_ids: set[int] = set()
    for frame_index, _, item in observations:
        number_source = item.get("number_source")
        if number_source in (
            "configured_sequence",
            "dynamic_row_sequence",
            "dynamic_sequence_inferred",
            "cross_row_sequence_inferred",
        ):
            number = item.get("number")
            structural_score = (
                0.98 if number_source == "configured_sequence" else 0.86
            )
        else:
            number = item.get("ocr_number")
            structural_score = 0.0
        if not number or len(number) != 4 or not str(number).isdigit():
            continue
        score = max(
            0.05,
            structural_score,
            float(item.get("ocr_number_score") or item.get("score") or 0.0),
        )
        values.append((str(number), score))
        frame_ids.add(frame_index)
    if not values:
        return None

    character_votes = [Counter() for _ in range(4)]
    for value, score in values:
        for index, character in enumerate(value):
            character_votes[index][character] += score
    character_consensus = "".join(
        votes.most_common(1)[0][0] for votes in character_votes
    )
    full_counts = Counter(value for value, _ in values)
    full_scores = Counter()
    for value, score in values:
        full_scores[value] += score
    repeated = [value for value, count in full_counts.items() if count >= 2]
    number = max(
        repeated,
        key=lambda value: (full_counts[value], full_scores[value]),
    ) if repeated else character_consensus
    confidences = [
        votes.most_common(1)[0][1] / max(1e-6, sum(votes.values()))
        for votes in character_votes
    ]
    confidence = sum(confidences) / len(confidences)
    exact_count = sum(value == number for value, _ in values)
    max_score = max(score for value, score in values if value == number) if exact_count else 0.0
    # Never invent a four-digit value that was not read in any frame.  Once a
    # track has multiple readings, require the selected full number to recur;
    # character-wise consensus is used to resolve noise, not to manufacture a
    # new serial from incompatible one-off strings.
    if exact_count == 0:
        return None
    if len(values) == 1 and max_score < 0.82:
        return None
    if len(values) >= 2 and exact_count < 2:
        return None
    if len(values) >= 2 and confidence < 0.56:
        return None
    return {
        "number": number,
        "confidence": round(confidence, 4),
        "evidence_count": len(values),
        "exact_count": exact_count,
        "frame_count": len(frame_ids),
    }


def fuse_candidate_history(candidate_history: list[list[dict]]) -> list[dict]:
    """Fuse dynamic prefixes and serials per spatial track across frames."""
    if not candidate_history:
        return []
    prefixes = _infer_adjacent_row_prefixes(
        candidate_history, _row_prefixes(candidate_history)
    )
    tracks = _build_tracks(candidate_history)
    minimum_observations = 1 if len(candidate_history) <= 2 else 2
    fused: list[dict] = []
    for track in tracks:
        latest = max(track["observations"], key=lambda value: value[0])[2]
        latest_center = latest.get("center") or [track["center_x"], track["center_y"]]
        prefix_choices = [
            (
                abs(float(evidence.get("center_y", latest_center[1])) - float(latest_center[1])),
                -float(evidence.get("confidence") or 0.0),
                evidence,
            )
            for evidence in prefixes.values()
            if abs(
                float(evidence.get("center_y", latest_center[1]))
                - float(latest_center[1])
            ) <= 180.0
        ]
        prefix = min(prefix_choices, key=lambda value: (value[0], value[1]))[2] if prefix_choices else None
        # A newly entering row has only one observation, while an already
        # tracked adjacent row can provide a strong dynamic row-prefix model.
        # In the site-confirmed serial-grid mode the current item's number is
        # structurally constrained, so it is safe to expose that first frame;
        # arbitrary/unconfigured scenes still require repeated observations.
        structural_first_frame = bool(
            len(track["observations"]) == 1
            and latest.get("number_source") == "configured_sequence"
            and prefix
            and prefix.get("source") == "temporal_row_sequence"
        )
        if (
            len(track["observations"]) < minimum_observations
            and not structural_first_frame
        ):
            continue
        number = _fuse_number(track["observations"])
        if not prefix or not number:
            continue
        fused.append({
            "row": int(latest.get("row", track["row"])),
            "center": [round(float(latest_center[0]), 1), round(float(latest_center[1]), 1)],
            "box": latest.get("box"),
            "prefix": prefix["prefix"],
            "number": number["number"],
            "label": f"{prefix['prefix']}-{number['number']}",
            "confidence": round(min(prefix["confidence"], number["confidence"]), 4),
            "track_observations": len(track["observations"]),
            "prefix_evidence": prefix,
            "number_evidence": number,
        })
    return sorted(fused, key=lambda item: (item["row"], item["center"][0]))


def apply_fused_to_current(current: list[dict], fused: list[dict]) -> list[dict]:
    """Attach stable fused labels to the nearest current-frame candidates."""
    used: set[int] = set()
    for stable in fused:
        choices = []
        for index, item in enumerate(current):
            if index in used or int(item.get("row", 0)) != int(stable["row"]):
                continue
            logical_number = _logical_number(item)
            if logical_number and logical_number == stable["number"]:
                choices.append((-1.0, index))
                continue
            center = item.get("center") or [0.0, 0.0]
            dx = abs(float(center[0]) - float(stable["center"][0]))
            dy = abs(float(center[1]) - float(stable["center"][1]))
            width, height = _box_size(item)
            if dx <= max(24.0, width * 0.75) and dy <= max(20.0, height):
                choices.append((dx + dy * 0.5, index))
        if not choices:
            continue
        _, index = min(choices)
        used.add(index)
        item = current[index]
        direct_label = item.get("label") if item.get("resolved") else None
        structurally_constrained_prefix = item.get("prefix_source") in (
            "row_sequence", "configured"
        )
        structurally_constrained_number = item.get("number_source") in (
            "configured_sequence",
            "ocr_sequence_anchor",
            "dynamic_sequence_inferred",
            "cross_row_sequence_inferred",
        )
        if (
            direct_label
            and direct_label != stable["label"]
            and (
                structurally_constrained_prefix
                or (structurally_constrained_number and not item.get("inferred"))
            )
        ):
            continue
        direct_inferred = bool(item.get("inferred"))
        item["direct_label"] = direct_label
        item["direct_prefix_source"] = item.get("prefix_source")
        item["direct_number_source"] = item.get("number_source")
        item["prefix"] = stable["prefix"]
        item["number"] = stable["number"]
        item["label"] = stable["label"]
        item["resolved"] = True
        item["inferred"] = direct_inferred or direct_label != stable["label"]
        item["prefix_source"] = "spatial_temporal"
        item["temporal_confidence"] = stable["confidence"]
        item["temporal_observations"] = stable["track_observations"]
    return current


def stabilize_complete_row_grids(current: list[dict]) -> list[dict]:
    """Validate full rows after fusion and fill an equal full adjacent row.

    Track voting can preserve a repeated OCR hallucination such as
    0011 -> 0211. A complete row supplies a stronger constraint: a strict +1
    sequence shares one ``number - column`` offset. Only rows with the maximum
    sticker count participate, so clipped/merged rows are never force-filled.
    """
    by_row: dict[int, list[dict]] = defaultdict(list)
    for item in current:
        by_row[int(item.get("row", 0))].append(item)
    for items in by_row.values():
        items.sort(key=lambda item: float((item.get("center") or [0.0])[0]))
    if not by_row:
        return current
    complete_count = max(len(items) for items in by_row.values())
    if complete_count < 5:
        return current

    models: dict[int, dict] = {}
    for row, items in by_row.items():
        if len(items) != complete_count:
            continue
        offsets: Counter[int] = Counter()
        prefixes: Counter[str] = Counter()
        for column, item in enumerate(items):
            number = item.get("number")
            if (
                item.get("resolved")
                and str(number or "").isdigit()
                and len(str(number)) == 4
            ):
                offsets[int(str(number)) - column] += 1
            prefix = str(item.get("prefix") or "")
            if item.get("resolved") and re.fullmatch(r"[A-Z][0-9]{2}", prefix):
                prefixes[prefix] += 1
        ranked_offsets = offsets.most_common(2)
        ranked_prefixes = prefixes.most_common(2)
        if not ranked_offsets or not ranked_prefixes:
            continue
        offset, offset_support = ranked_offsets[0]
        prefix, prefix_support = ranked_prefixes[0]
        offset_runner = ranked_offsets[1][1] if len(ranked_offsets) > 1 else 0
        prefix_runner = ranked_prefixes[1][1] if len(ranked_prefixes) > 1 else 0
        if (
            offset_support < max(4, math.floor(complete_count * 0.50) + 1)
            or offset_support <= offset_runner
            or prefix_support < max(3, math.ceil(complete_count * 0.55))
            or prefix_support <= prefix_runner
            or not 0 <= offset <= 9999
            or offset + complete_count - 1 > 9999
        ):
            continue
        models[row] = {
            "prefix": prefix,
            "offset": offset,
            "numbers": tuple(
                f"{offset + column:04d}" for column in range(complete_count)
            ),
        }

    def apply_expected(item: dict, prefix: str, number: str, source: str) -> None:
        observed_prefix = item.get("observed_prefix") or item.get("ocr_prefix")
        observed_number = item.get("observed_number") or item.get("ocr_number")
        item["expected_prefix"] = prefix
        item["expected_number"] = number
        item["expected_label"] = f"{prefix}-{number}"
        if (observed_prefix or prefix) and observed_number:
            item["observed_label"] = f"{observed_prefix or prefix}-{observed_number}"
        item["prefix"] = prefix
        item["number"] = number
        item["label"] = f"{prefix}-{number}"
        item["prefix_source"] = source
        item["number_source"] = source
        item["resolved"] = True
        conflict = bool(
            (observed_prefix and observed_prefix != prefix)
            or (observed_number and str(observed_number) != number)
        )
        item["inferred"] = bool(
            item.get("inferred") or conflict or not observed_number
        )
        if conflict and item.get("anomaly_candidate"):
            item["sequence_status"] = "suspected_wrong_label"
        elif observed_number == number:
            item["sequence_status"] = "correct"
        else:
            item["sequence_status"] = "unread_inferred"

    for row, model in models.items():
        for item, number in zip(by_row[row], model["numbers"]):
            apply_expected(item, model["prefix"], number, "temporal_row_sequence")

    if not models:
        return current
    try:
        prefix_step = int(os.environ.get("THU_VR_LABEL_ROW_PREFIX_STEP", "0"))
    except ValueError:
        prefix_step = 0
    if not prefix_step or not -9 <= prefix_step <= 9:
        return current
    serial_grids = {model["numbers"] for model in models.values()}
    if len(serial_grids) != 1:
        return current

    for row, items in by_row.items():
        if row in models or len(items) != complete_count:
            continue
        reference_row = min(models, key=lambda candidate: abs(candidate - row))
        reference = models[reference_row]
        match = re.fullmatch(r"([A-Z])([0-9]{2})", reference["prefix"])
        if not match:
            continue
        prefix_number = int(match.group(2)) + prefix_step * (row - reference_row)
        if not 0 <= prefix_number <= 99:
            continue
        prefix = f"{match.group(1)}{prefix_number:02d}"
        for item, number in zip(items, reference["numbers"]):
            apply_expected(item, prefix, number, "temporal_cross_row_sequence")

    # Keep the public label consistent with its structured fields and suppress
    # extreme track hallucinations in clipped rows. A genuine sparse wrong
    # sticker remains anomaly evidence and can be re-enabled by the stricter
    # multi-frame/full-frame confirmation step that runs next.
    proven_serials = set(next(iter(serial_grids)))
    for items in by_row.values():
        numeric_values = [
            int(str(item.get("number")))
            for item in items
            if item.get("resolved")
            and str(item.get("number") or "").isdigit()
            and len(str(item.get("number"))) == 4
        ]
        if not numeric_values:
            continue
        ordered_values = sorted(numeric_values)
        median = ordered_values[len(ordered_values) // 2]
        outlier_limit = max(50, len(items) * 4)
        for item in items:
            prefix = str(item.get("prefix") or "")
            number = str(item.get("number") or "")
            if not item.get("resolved"):
                continue
            if (
                not re.fullmatch(r"[A-Z][0-9]{2}", prefix)
                or not re.fullmatch(r"[0-9]{4}", number)
            ):
                item["resolved"] = False
                item["label"] = "未读全"
                continue
            if number not in proven_serials:
                item["resolved"] = False
                item["label"] = "未读全"
                item["sequence_status"] = "unverified_outlier"
                continue
            if len(numeric_values) >= 4 and abs(int(number) - median) > outlier_limit:
                item["resolved"] = False
                item["label"] = "未读全"
                item["sequence_status"] = "unverified_outlier"
                continue
            item["label"] = f"{prefix}-{number}"
    return current


def apply_cross_camera_row_prefix_model(
    current: list[dict],
    camera: str,
    latest_by_camera: dict[str, list[dict]],
) -> list[dict]:
    """Learn one contiguous row-prefix model across head and base views.

    The fixed robot geometry sees the upper wall rows in ``head`` and the
    continuation in ``base``. Values and letters are learned from pixels; the
    only configured fact is the generic adjacent-row step. Multiple agreeing
    physical rows are required before correcting either camera.
    """
    try:
        step = int(os.environ.get("THU_VR_LABEL_ROW_PREFIX_STEP", "0"))
    except ValueError:
        return current
    if not step or not -9 <= step <= 9:
        return current

    sources = {key: list(value) for key, value in latest_by_camera.items()}
    sources[camera] = current
    serial_grids: Counter[tuple[str, ...]] = Counter()
    for labels in sources.values():
        grouped_serials: dict[int, list[dict]] = defaultdict(list)
        for item in labels:
            grouped_serials[int(item.get("row", 0))].append(item)
        for items in grouped_serials.values():
            ordered = sorted(
                items, key=lambda item: float((item.get("center") or [0.0])[0])
            )
            numbers = [
                str(item.get("expected_number") or item.get("number") or "")
                for item in ordered
            ]
            if len(numbers) < 5 or not all(
                re.fullmatch(r"[0-9]{4}", number) for number in numbers
            ):
                continue
            integer_numbers = [int(number) for number in numbers]
            if all(
                right == left + 1
                for left, right in zip(integer_numbers, integer_numbers[1:])
            ):
                serial_grids[tuple(numbers)] += 1
    head_rows = {
        int(item.get("row", 0))
        for item in sources.get("head", [])
        if int(item.get("row", 0)) > 0
    }
    if not head_rows:
        return current
    head_row_count = max(head_rows)

    model_rows: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    observed_global_rows: set[int] = set()
    for source_camera, labels in sources.items():
        if source_camera not in ("head", "base"):
            continue
        grouped: dict[int, list[dict]] = defaultdict(list)
        for item in labels:
            grouped[int(item.get("row", 0))].append(item)
        maximum_count = max((len(items) for items in grouped.values()), default=0)
        for row, items in grouped.items():
            if len(items) < max(3, math.ceil(maximum_count * 0.40)):
                continue
            global_row = row if source_camera == "head" else head_row_count + row
            observed_global_rows.add(global_row)
            prefixes = Counter(
                str(item.get("prefix"))
                for item in items
                if item.get("resolved")
                and re.fullmatch(r"[A-Z][0-9]{2}", str(item.get("prefix") or ""))
            )
            ranked = prefixes.most_common(2)
            if not ranked:
                continue
            prefix, support = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else 0
            if support < max(2, math.ceil(len(items) * 0.40)) or support <= runner:
                continue
            match = re.fullmatch(r"([A-Z])([0-9]{2})", prefix)
            if not match:
                continue
            model = (match.group(1), int(match.group(2)) - step * global_row)
            model_rows[model].add((source_camera, row))

    ranked_models = sorted(
        (
            (len(rows), model)
            for model, rows in model_rows.items()
            if all(
                0 <= model[1] + step * global_row <= 99
                for global_row in observed_global_rows
            )
        ),
        reverse=True,
    )
    if not ranked_models or ranked_models[0][0] < 2:
        return current
    top_support, (letter, constant) = ranked_models[0]
    runner_support = ranked_models[1][0] if len(ranked_models) > 1 else 0
    if top_support <= runner_support:
        return current

    grouped_current: dict[int, list[dict]] = defaultdict(list)
    for item in current:
        grouped_current[int(item.get("row", 0))].append(item)
    maximum_count = max((len(items) for items in grouped_current.values()), default=0)
    shared_serial_grid: tuple[str, ...] | None = None
    ranked_serial_grids = serial_grids.most_common(2)
    if ranked_serial_grids:
        candidate_grid, support = ranked_serial_grids[0]
        runner_support = ranked_serial_grids[1][1] if len(ranked_serial_grids) > 1 else 0
        if support >= 2 and support > runner_support:
            shared_serial_grid = candidate_grid

    def align_partial_row(
        ordered: list[dict], grid: tuple[str, ...]
    ) -> list[str] | None:
        """Align a clipped row to a learned arbitrary +1 serial grid."""
        item_count = len(ordered)
        grid_count = len(grid)
        if not 3 <= item_count <= grid_count:
            return None
        if item_count == grid_count:
            return list(grid)
        infinity = float("inf")
        costs = [[infinity] * grid_count for _ in range(item_count)]
        parents = [[-1] * grid_count for _ in range(item_count)]
        for item_index, item in enumerate(ordered):
            observed = str(
                item.get("observed_number") or item.get("ocr_number") or ""
            )
            for grid_index, expected in enumerate(grid):
                if (
                    grid_index < item_index
                    or grid_count - grid_index < item_count - item_index
                ):
                    continue
                read_cost = 0.0 if not observed or observed == expected else 5.0
                position_cost = abs(
                    item_index / max(1, item_count - 1)
                    - grid_index / max(1, grid_count - 1)
                )
                local_cost = read_cost + position_cost
                if item_index == 0:
                    costs[item_index][grid_index] = local_cost
                    continue
                best_parent = -1
                best_cost = infinity
                for previous in range(grid_index):
                    if costs[item_index - 1][previous] < best_cost:
                        best_cost = costs[item_index - 1][previous]
                        best_parent = previous
                if best_parent >= 0:
                    costs[item_index][grid_index] = best_cost + local_cost
                    parents[item_index][grid_index] = best_parent
        final_index = min(
            range(grid_count), key=lambda index: costs[-1][index]
        )
        if costs[-1][final_index] == infinity:
            return None
        assignments = [""] * item_count
        cursor = final_index
        for item_index in range(item_count - 1, -1, -1):
            assignments[item_index] = grid[cursor]
            cursor = parents[item_index][cursor] if item_index else -1
        return assignments
    for row, items in grouped_current.items():
        if len(items) < max(3, math.ceil(maximum_count * 0.40)):
            continue
        global_row = row if camera == "head" else head_row_count + row
        prefix_number = constant + step * global_row
        if not 0 <= prefix_number <= 99:
            continue
        expected_prefix = f"{letter}{prefix_number:02d}"
        ordered_items = sorted(
            items, key=lambda item: float((item.get("center") or [0.0])[0])
        )
        row_serials = (
            align_partial_row(ordered_items, shared_serial_grid)
            if shared_serial_grid else None
        )
        for column, item in enumerate(ordered_items):
            item["expected_prefix"] = expected_prefix
            item["prefix"] = expected_prefix
            item["prefix_source"] = "cross_camera_row_sequence"
            if row_serials:
                expected_number = row_serials[column]
                observed_number = item.get("observed_number") or item.get("ocr_number")
                item["expected_number"] = expected_number
                item["number"] = expected_number
                item["number_source"] = "cross_camera_serial_grid"
                item["expected_label"] = f"{expected_prefix}-{expected_number}"
                item["label"] = item["expected_label"]
                item["resolved"] = True
                item["inferred"] = bool(
                    item.get("inferred") or observed_number != expected_number
                )
                if observed_number:
                    item["observed_label"] = f"{expected_prefix}-{observed_number}"
                if observed_number and str(observed_number) != expected_number:
                    item["number_anomaly_candidate"] = True
                    evidence_sources = sorted({
                        str(reading.get("source") or "unknown")
                        for reading in item.get("raw_texts", [])
                        if reading.get("line") != "prefix"
                    })
                    item["number_anomaly_sources"] = evidence_sources
                    item["anomaly_candidate"] = True
                    item["anomaly_evidence_sources"] = evidence_sources
                    item["sequence_status"] = "suspected_wrong_label"
                elif observed_number == expected_number:
                    item["sequence_status"] = "correct"
                else:
                    item["sequence_status"] = "unread_inferred"
                continue
            number = str(item.get("number") or "")
            if item.get("resolved") and re.fullmatch(r"[0-9]{4}", number):
                item["expected_label"] = f"{expected_prefix}-{number}"
                item["label"] = item["expected_label"]
                observed_prefix = item.get("observed_prefix") or item.get("ocr_prefix")
                observed_number = item.get("observed_number") or item.get("ocr_number")
                if observed_number:
                    item["observed_label"] = (
                        f"{observed_prefix or expected_prefix}-{observed_number}"
                    )
                if observed_prefix and observed_prefix != expected_prefix:
                    item["inferred"] = True
    return current


def confirm_sequence_anomalies(
    current: list[dict],
    candidate_history: list[list[dict]],
    minimum_frames: int = 2,
) -> list[dict]:
    """Confirm a sparse row-sequence conflict only when pixels repeat it.

    ``expected_label`` is learned from the row majority, while
    ``observed_label`` retains the per-sticker OCR reading.  Matching both
    values across frames separates a physical wrong sticker from transient
    motion blur.  This function never turns an expected value into observed
    evidence.
    """
    for item in current:
        item["anomaly_confirmed"] = False
        expected = item.get("expected_label")
        if not expected and item.get("expected_number"):
            expected_prefix = item.get("expected_prefix") or item.get("prefix")
            if expected_prefix:
                expected = f"{expected_prefix}-{item['expected_number']}"
                item["expected_label"] = expected
        observed = item.get("observed_label")
        if not observed and item.get("observed_number"):
            observed_prefix = item.get("observed_prefix") or item.get("prefix")
            if observed_prefix:
                observed = f"{observed_prefix}-{item['observed_number']}"
                item["observed_label"] = observed
        if item.get("number_anomaly_candidate") and item.get("observed_number"):
            canonical_prefix = item.get("expected_prefix") or item.get("prefix")
            if canonical_prefix:
                observed = f"{canonical_prefix}-{item['observed_number']}"
                item["observed_label"] = observed
        # Preserve a repeatable physical conflict through a blurred current
        # frame. Expected labels uniquely identify row/column positions, so
        # aggregate prior observed values at that same position and surface
        # the dominant conflict once it has appeared in at least two frames.
        historical_conflicts: dict[str, dict] = defaultdict(
            lambda: {"frames": 0, "peak": 0.0}
        )
        if expected:
            for frame in candidate_history:
                seen_in_frame: set[str] = set()
                for candidate in frame:
                    candidate_observed = candidate.get("observed_label")
                    if (
                        candidate.get("number_anomaly_candidate")
                        and candidate.get("observed_number")
                    ):
                        candidate_prefix = (
                            candidate.get("expected_prefix")
                            or candidate.get("prefix")
                        )
                        if candidate_prefix:
                            candidate_observed = (
                                f"{candidate_prefix}-{candidate['observed_number']}"
                            )
                    if (
                        not candidate.get("anomaly_candidate")
                        or candidate.get("expected_label") != expected
                        or not candidate_observed
                        or candidate_observed == expected
                    ):
                        continue
                    candidate_observed = str(candidate_observed)
                    evidence = historical_conflicts[candidate_observed]
                    evidence["peak"] = max(
                        float(evidence["peak"]),
                        float(candidate.get("ocr_number_score") or 0.0),
                        float(candidate.get("ocr_prefix_score") or 0.0),
                    )
                    if candidate_observed not in seen_in_frame:
                        evidence["frames"] += 1
                        seen_in_frame.add(candidate_observed)
        if historical_conflicts:
            historical_observed, historical = max(
                historical_conflicts.items(),
                key=lambda value: (value[1]["frames"], value[1]["peak"]),
            )
            current_frames = historical_conflicts.get(str(observed), {}).get(
                "frames", 0
            ) if observed else 0
            if historical["frames"] >= 2 and (
                not item.get("anomaly_candidate")
                or historical["frames"] >= current_frames
            ):
                observed = historical_observed
                item["observed_label"] = observed
                item["anomaly_candidate"] = True
                item["sequence_status"] = "suspected_wrong_label"
                item["inferred"] = True
        if not item.get("anomaly_candidate"):
            continue
        if not expected or not observed or expected == observed:
            item["sequence_status"] = "unverified"
            continue
        confirming_frames = 0
        evidence_sources: set[str] = set()
        peak_evidence_score = 0.0
        for frame in candidate_history:
            matches = [
                candidate for candidate in frame
                if (
                candidate.get("anomaly_candidate")
                and candidate.get("expected_label") == expected
                and candidate.get("observed_label") == observed
                )
            ]
            if matches:
                confirming_frames += 1
                for candidate in matches:
                    evidence_sources.update(candidate.get("anomaly_evidence_sources") or [])
                    peak_evidence_score = max(
                        peak_evidence_score,
                        float(candidate.get("ocr_number_score") or 0.0),
                        float(candidate.get("ocr_prefix_score") or 0.0),
                    )
        item["anomaly_observations"] = confirming_frames
        item["anomaly_evidence_sources"] = sorted(evidence_sources)
        item["anomaly_peak_score"] = round(peak_evidence_score, 4)
        item["resolved"] = True
        expected_codes = {
            str(candidate.get("expected_label"))
            for candidate in current
            if candidate.get("expected_label")
        }
        structured_displacement = False
        serial_hamming: int | None = None
        prefix_changed = False
        expected_match = re.fullmatch(r"([A-Z][0-9]{2})-([0-9]{4})", expected)
        observed_match = re.fullmatch(r"([A-Z][0-9]{2})-([0-9]{4})", observed)
        if expected_match and observed_match:
            prefix_changed = expected_match.group(1) != observed_match.group(1)
            serial_hamming = sum(
                left != right
                for left, right in zip(
                    expected_match.group(2), observed_match.group(2)
                )
            )
            structured_displacement = bool(
                observed in expected_codes
                and expected_match.group(1) == observed_match.group(1)
                and abs(
                    int(expected_match.group(2))
                    - int(observed_match.group(2))
                ) >= 3
                and serial_hamming >= 2
            )
        item["anomaly_structured_displacement"] = structured_displacement
        reciprocal = False
        for other in current:
            if other is item or not other.get("anomaly_candidate"):
                continue
            if (
                other.get("expected_label") == observed
                and other.get("observed_label") == expected
            ):
                other_peak = max(
                    float(other.get("ocr_number_score") or 0.0),
                    float(other.get("ocr_prefix_score") or 0.0),
                )
                if peak_evidence_score >= 0.90 and other_peak >= 0.90:
                    reciprocal = True
                    break
        item["anomaly_reciprocal"] = reciprocal
        # The contact sheet and recognition-only line both originate from the
        # same localized crop. A shifted candidate box can therefore make
        # both variants read a neighbouring serial. Confirmation requires
        # recurrence plus full-frame/crop agreement.
        # One-character OCR slips (0010/0020, A01/A00) are by far the most
        # common live false alarms.  Two frames are enough for a multi-digit
        # displacement or a reciprocal swap, but a one-character conflict
        # needs three exceptionally clear reads.  This keeps the normal path
        # fast without allowing the same blurry glyph to vote twice.
        multi_character_conflict = bool(
            serial_hamming is not None and serial_hamming >= 2
        )
        ultra_reliable_repeat = bool(
            confirming_frames >= 3
            and peak_evidence_score >= 0.97
            and (prefix_changed or serial_hamming == 1)
        )
        repeated_direct_evidence = bool(
            confirming_frames >= max(2, int(minimum_frames))
            and evidence_sources
            and (
                (multi_character_conflict and peak_evidence_score >= 0.90)
                or ultra_reliable_repeat
            )
        )
        if (
            reciprocal
            or (
                confirming_frames >= max(2, int(minimum_frames))
                and structured_displacement
            )
            or (
                confirming_frames >= max(2, int(minimum_frames))
                and repeated_direct_evidence
            )
        ):
            item["label"] = observed
            item["inferred"] = False
            item["anomaly_confirmed"] = True
            item["sequence_status"] = "wrong_label"
        else:
            # Keep uncertain OCR conflicts visible in metadata/annotation but
            # do not let them poison the recognized-label output.
            item["label"] = expected
            item["inferred"] = True
            item["sequence_status"] = "suspected_wrong_label"
    return current
