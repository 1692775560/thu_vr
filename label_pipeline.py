#!/usr/bin/env python3
"""Frame pipeline and temporal accumulation for sticker label reading.

Per frame: segment stickers, group them into rows, propose the row-pitch
positions that segmentation may have missed, and recognize every candidate's
two text lines in one batched call.

Across frames: keep one slot per physical sticker and keep adding pixel votes
to it.  The published label is always the accumulated pixel vote.  Row
sequence expectations are computed too, but only to flag a sticker as
out-of-sequence, never to replace what the pixels say -- a physically
misplaced sticker has to stay readable.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from label_reader import (
    INK_FLOOR,
    PREFIX_PATTERN,
    PREFIX_VARIANTS,
    SERIAL_LENGTH,
    SERIAL_VARIANTS,
    Sticker,
    accumulate,
    rank_votes,
    build_line_views,
    build_row_prefix_composites,
    cluster_rows,
    group_rows,
    locate_stickers,
    prefix_from_text,
    prefix_views_from_plate,
    propose_missing,
    propose_row_ends,
    select_variants,
    serial_from_text,
    serial_views_from_plate,
    stack_plates,
)

# A proposed (not segmented) position needs clearly stronger pixel evidence
# than a segmented sticker before it is published, because a proposal sits
# wherever the pitch says and may be empty fabric.
PROPOSAL_MIN_MARGIN = 0.55
PROPOSAL_MIN_SUPPORT = 1.2
SEGMENTED_MIN_MARGIN = 0.20
SEGMENTED_MIN_SUPPORT = 0.45
# A serial that survived being checked against its row's range and against the
# serials already taken there has passed a test no margin threshold applies, so
# it is published on a much weaker margin than an unchecked reading.
TIEBREAK_MARGIN = 0.70
TIEBREAK_MIN_MARGIN = 0.03
RANGE_SLACK = 3
# A reading with few votes and no competitor scores a perfect margin, so margin
# alone cannot separate a genuine sticker from a fold that read once.  What
# separates them is how much evidence the rest of the row gathered: a fixed
# floor cannot, because a distant row's real stickers carry less evidence than a
# near row's spurious ones.
ROW_SUPPORT_SHARE = 0.15
# A shelf row is a run of stickers that corroborate one another.  A lone reading
# in a sparse cluster has nothing backing it, and on this curtain backdrop a
# handful of folds is exactly what produces one.
ISOLATED_ROW_MAX_SLOTS = 4
ISOLATED_ROW_MAX_RESOLVED = 1
# An out-of-sequence serial with far less evidence than its row-mates is almost
# always a fold that OCR once committed to, not a sticker moved from elsewhere.
# A physically misplaced sticker still carries comparable ink and vote mass, so
# this ratio leaves those alone and only withdraws the weak impostor.
WEAK_OOS_SUPPORT_RATIO = 0.35
# Digits that PP-OCR regularly returns in place of a leading zero on these
# stickers.  Kept as alternates so row-aware re-ranking can still recover 0010
# when every per-frame crop preferred 2010 / 9010.
LEADING_ZERO_CONFUSIONS = frozenset("2689")
# Printed-ink coverage bounds on the weaker text line of a rectified plate.
# Below the floor is blank fabric; above the ceiling is a dark object such as
# the black tape markers stuck on the same wall.
MIN_INK_RATIO = 0.035
MAX_INK_RATIO = 0.50
# A row-stacked prefix read is far cleaner than one sticker's crop, so it is
# weighted above an individual reading without being allowed to be the only
# voice.
ROW_COMPOSITE_WEIGHT = 2.5
# A read off the temporally averaged, row-stacked plate is the cleanest view of
# a prefix the pipeline can build, so it outweighs the per-frame reads it is
# meant to correct.
REFINED_PREFIX_WEIGHT = 2.0
REFINED_SERIAL_WEIGHT = 1.5
# A prefix is a zone letter, a bay tens digit and a bay units digit.  Within
# one camera view the zone letter and the tens digit are effectively constant,
# because the rows on screen are neighbours on the same shelf, while the units
# digit is what tells those rows apart.  Pooling each character at the scope
# where it is actually constant beats voting on the whole string, which lets a
# single badly read sticker carry a wrong prefix such as "A22" for "A02".  Each
# pool is only trusted when it actually agrees.
LETTER_CONSENSUS_MARGIN = 0.30
TENS_CONSENSUS_MARGIN = 0.30
# Once a row agrees on its bay number this clearly, a single sticker reading a
# different one is treated as a misread rather than as truth: the number is
# printed on every sticker of the row, so one blurred crop cannot outweigh it.
ROW_BAY_SETTLED_MARGIN = 0.50
# ...unless that sticker read its own bay this strongly, which is how a sticker
# physically moved in from another bay stays readable.  Either way the
# disagreement is reported on the entry.  The bar is a share of the row's own
# evidence, not a fixed count, so it does not soften as frames accumulate.
BAY_OVERRIDE_SHARE = 0.50
# A share alone is met too easily when the whole row read weakly, so the
# overriding sticker also has to clear an absolute amount of evidence.
BAY_OVERRIDE_MIN_SUPPORT = 4.0

# Before a sticker is allowed to publish a prefix its row disagrees with, its
# own crops have to be near-unanimous and carry evidence comparable to its
# row-mates'.  A blurred sticker misreading its own row's bay looks the same at
# low confidence, so anything softer is recorded as a disagreement only.
PREFIX_TRANSPLANT_MARGIN = 0.90
PREFIX_TRANSPLANT_SUPPORT_RATIO = 0.60
# A prefix read only in a stray frame or two accumulates single-digit support,
# while a sticker read across the pass reaches tens of it.  Contradicting the
# row it sits in needs the latter: on a row too distant for any sticker to read
# its bay, there is no in-row reference to compare against, and a couple of
# noisy reads would otherwise be published as if they came from another bay.
PREFIX_TRANSPLANT_MIN_SUPPORT = 12.0


@dataclass
class Slot:
    """One physical sticker tracked across frames."""

    row: int
    center: tuple[float, float]
    box: tuple[int, int, int, int]
    proposed: bool = True
    frames: int = 0
    segmented_frames: int = 0
    digit_votes: list[dict[str, float]] = field(
        default_factory=lambda: [defaultdict(float) for _ in range(SERIAL_LENGTH)]
    )
    prefix_votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    solo_prefix_votes: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    ink_scores: list[float] = field(default_factory=list)
    plate_sum: np.ndarray | None = None
    plate_count: int = 0

    @property
    def ink(self) -> float:
        return float(np.median(self.ink_scores)) if self.ink_scores else 0.0

    def mean_plate(self) -> np.ndarray | None:
        """This sticker averaged over every frame that saw it.

        Rectification already put each observation on the same canonical grid,
        so averaging needs no registration and lifts the faint bay-number
        strokes out of the sensor noise that hides them in any single frame.
        """
        if self.plate_sum is None or not self.plate_count:
            return None
        return (self.plate_sum / self.plate_count).astype(np.uint8)

    def merge(self, sticker: Sticker) -> None:
        self.frames += 1
        self.ink_scores.extend(sticker.ink_scores)
        if sticker.plate is not None:
            if self.plate_sum is None or self.plate_sum.shape != sticker.plate.shape:
                self.plate_sum = sticker.plate.astype(np.float32)
                self.plate_count = 1
            else:
                self.plate_sum += sticker.plate
                self.plate_count += 1
        if not sticker.proposed:
            self.segmented_frames += 1
            self.proposed = False
        weight = 0.6 if self.frames > 1 else 1.0
        self.center = (
            self.center[0] * (1 - weight) + sticker.center[0] * weight,
            self.center[1] * (1 - weight) + sticker.center[1] * weight,
        )
        self.box = sticker.box
        for position in range(SERIAL_LENGTH):
            for digit, score in sticker.digit_votes[position].items():
                self.digit_votes[position][digit] += score
        for prefix, score in sticker.prefix_votes.items():
            self.prefix_votes[prefix] += score
        for prefix, score in sticker.solo_prefix_votes.items():
            self.solo_prefix_votes[prefix] += score

    def serial(self) -> tuple[str | None, float]:
        digits: list[str] = []
        margins: list[float] = []
        for votes in self.digit_votes:
            if not votes:
                return None, 0.0
            ranked = sorted(votes.items(), key=lambda entry: -entry[1])
            total = sum(votes.values())
            runner = ranked[1][1] if len(ranked) > 1 else 0.0
            digits.append(ranked[0][0])
            margins.append((ranked[0][1] - runner) / total if total else 0.0)
        return "".join(digits), round(float(min(margins)), 4)

    def support(self) -> float:
        return float(sum(self.digit_votes[0].values()))

    def own_prefix(self) -> tuple[str | None, float]:
        return rank_votes(self.prefix_votes)

    def solo_prefix(self) -> tuple[str | None, float]:
        """This sticker's own crops only, with its row's pooled votes excluded."""
        return rank_votes(self.solo_prefix_votes)

    def solo_prefix_support(self) -> float:
        return float(sum(self.solo_prefix_votes.values()))


def read_frame(
    engine,
    image: np.ndarray,
    *,
    phase: int = 0,
    cheap_serial_variants: int = 2,
    cheap_prefix_variants: int = 1,
) -> list[list[Sticker]]:
    """Locate and recognize every sticker in one frame.

    A cheap rotating pair of variants covers stickers that read cleanly; only
    the ones whose digits did not agree are re-rendered with the remaining
    variants.  Cost therefore concentrates on the distant or blurred rows.
    """
    located = locate_stickers(image)
    rows_of_boxes = group_rows(located)
    # A truncated distant row often keeps only its rightmost stickers.  The
    # densest neighbouring row tells us how far left/right the shelf actually
    # extends, so end proposals can reach the missing end instead of stopping
    # three pitches short.
    span_target = 0
    count_target = 0
    if rows_of_boxes:
        populated = [row for row in rows_of_boxes if len(row) >= 3]
        if populated:
            span_target = max(
                max(entry["box"][0] + entry["box"][2] for entry in row)
                - min(entry["box"][0] for entry in row)
                for row in populated
            )
            count_target = max(len(row) for row in populated)
    rows: list[list[Sticker]] = []
    for row_index, row_boxes in enumerate(rows_of_boxes, start=1):
        stickers = [
            Sticker(
                row=row_index,
                box=entry["box"],
                center=(
                    entry["box"][0] + entry["box"][2] / 2.0,
                    entry["box"][1] + entry["box"][3] / 2.0,
                ),
                quad=entry.get("quad"),
            )
            for entry in row_boxes
        ]
        own_span = (
            max(entry["box"][0] + entry["box"][2] for entry in row_boxes)
            - min(entry["box"][0] for entry in row_boxes)
            if row_boxes else 0
        )
        # Complete rows already have their ends; walking past them lands on
        # fabric and, worse, on the neighbouring shelf's stickers, which then
        # vote the wrong bay into this row.  Only a clearly truncated row is
        # allowed to ask for end proposals.
        end_slots = 0
        truncated = bool(
            (span_target and own_span and own_span < span_target * 0.85)
            or (count_target and len(row_boxes) <= count_target - 2)
        )
        if truncated:
            end_slots = min(6, max(2, int(round(
                (span_target - own_span) / max(row_boxes[0]["box"][2], 1)
            )) if span_target and own_span else 3))
        for entry in (
            propose_missing(row_boxes)
            + propose_row_ends(row_boxes, max_end_slots=end_slots)
        ):
            stickers.append(Sticker(
                row=row_index,
                box=entry["box"],
                center=(
                    entry["box"][0] + entry["box"][2] / 2.0,
                    entry["box"][1] + entry["box"][3] / 2.0,
                ),
                quad=entry.get("quad"),
                proposed=True,
            ))
        stickers.sort(key=lambda sticker: sticker.center[0])
        rows.append(stickers)
    if not rows:
        return []
    flat = [sticker for row in rows for sticker in row]
    cheap_serial = select_variants(SERIAL_VARIANTS, phase, cheap_serial_variants)
    cheap_prefix = select_variants(PREFIX_VARIANTS, phase, cheap_prefix_variants)
    views, metadata = build_line_views(
        image, flat, serial_table=cheap_serial, prefix_table=cheap_prefix
    )
    accumulate(metadata, engine.recognize_lines(views))

    # Only segmented stickers vote in the row-stacked prefix read.  A proposal
    # sitting on fabric or on the neighbouring shelf would otherwise tip the
    # bay number of a whole real row.
    composite_rows = [
        [sticker for sticker in row if not sticker.proposed] for row in rows
    ]
    composites, owners = build_row_prefix_composites(image, composite_rows)
    if composites:
        for row, (text, score) in zip(owners, engine.recognize_lines(composites)):
            prefix = prefix_from_text(text)
            if prefix and score >= 0.25:
                for sticker in row:
                    sticker.prefix_votes[prefix] += score * ROW_COMPOSITE_WEIGHT

    weak = [sticker for sticker in flat if sticker.needs_more_variants()]
    if weak:
        rest_serial = tuple(
            variant for variant in SERIAL_VARIANTS if variant not in cheap_serial
        )
        rest_prefix = tuple(
            variant for variant in PREFIX_VARIANTS if variant not in cheap_prefix
        )
        if rest_serial or rest_prefix:
            views, metadata = build_line_views(
                image, weak,
                serial_table=rest_serial, prefix_table=rest_prefix,
            )
            accumulate(metadata, engine.recognize_lines(views))
    # End-extension proposals that never read a serial are empty fabric (or
    # black tape) sitting on the pitch grid.  Keeping them creates tracker
    # slots that split real rows and push neighbouring ink scores through the
    # ceiling, so only proposals that actually produced digits survive.
    return [
        [
            sticker for sticker in row
            if not sticker.proposed or sticker.serial()[0] is not None
        ]
        for row in rows
        if any(not sticker.proposed or sticker.serial()[0] for sticker in row)
    ]


class LabelTracker:
    """Accumulates pixel votes per physical sticker across frames.

    Slots are matched by image position, not by which row ordinal they landed
    in: one spurious row in a single frame would otherwise renumber every row
    and break association for the whole sequence.
    """

    def __init__(self, reset_on_scene_change: bool = True) -> None:
        self.slots: list[Slot] = []
        self.reset_on_scene_change = reset_on_scene_change

    def update(self, rows: list[list[Sticker]]) -> None:
        stickers = [sticker for row in rows for sticker in row]
        if not stickers:
            return

        if self.reset_on_scene_change and self.slots:
            matched = sum(
                1 for sticker in stickers
                if self._nearest(sticker, *_tolerances(sticker)) is not None
            )
            if matched < max(2, int(len(stickers) * 0.25)):
                self.slots.clear()

        claimed: set[int] = set()
        for sticker in sorted(stickers, key=lambda item: item.center[0]):
            match = self._nearest(sticker, *_tolerances(sticker), claimed)
            if match is None:
                match = Slot(
                    row=0, center=sticker.center, box=sticker.box,
                    proposed=sticker.proposed,
                )
                self.slots.append(match)
            claimed.add(id(match))
            match.merge(sticker)

    def _nearest(
        self,
        sticker: Sticker,
        tolerance_x: float,
        tolerance_y: float,
        claimed: set[int] | None = None,
    ) -> Slot | None:
        best: Slot | None = None
        best_distance = float("inf")
        for slot in self.slots:
            if claimed is not None and id(slot) in claimed:
                continue
            dx = abs(slot.center[0] - sticker.center[0])
            dy = abs(slot.center[1] - sticker.center[1])
            if dx > tolerance_x or dy > tolerance_y:
                continue
            distance = dx + dy * 0.5
            if distance < best_distance:
                best, best_distance = slot, distance
        return best

    def _merged_slots(self) -> list[Slot]:
        """Slots with same-sticker duplicates folded together.

        A single frame occasionally segments one sticker as two overlapping
        boxes, which would otherwise create a second slot that competes with
        the real one and splits its evidence.  Folding is done on copies so
        accumulation itself stays untouched.
        """
        if len(self.slots) < 2:
            return list(self.slots)
        widths = float(np.median([slot.box[2] for slot in self.slots]))
        heights = float(np.median([slot.box[3] for slot in self.slots]))
        kept: list[Slot] = []
        for slot in sorted(self.slots, key=lambda item: -item.frames):
            duplicate = None
            for other in kept:
                if (
                    abs(other.center[0] - slot.center[0]) < widths * 0.45
                    and abs(other.center[1] - slot.center[1]) < heights * 0.70
                ):
                    duplicate = other
                    break
            if duplicate is None:
                copy = Slot(
                    row=slot.row, center=slot.center, box=slot.box,
                    proposed=slot.proposed, frames=slot.frames,
                    segmented_frames=slot.segmented_frames,
                    ink_scores=list(slot.ink_scores),
                    plate_sum=(
                        None if slot.plate_sum is None else slot.plate_sum.copy()
                    ),
                    plate_count=slot.plate_count,
                )
                for position in range(SERIAL_LENGTH):
                    copy.digit_votes[position].update(slot.digit_votes[position])
                copy.prefix_votes.update(slot.prefix_votes)
                copy.solo_prefix_votes.update(slot.solo_prefix_votes)
                kept.append(copy)
                continue
            duplicate.frames += slot.frames
            duplicate.segmented_frames += slot.segmented_frames
            duplicate.proposed = duplicate.proposed and slot.proposed
            duplicate.ink_scores.extend(slot.ink_scores)
            if slot.plate_sum is not None:
                if (
                    duplicate.plate_sum is not None
                    and duplicate.plate_sum.shape == slot.plate_sum.shape
                ):
                    duplicate.plate_sum += slot.plate_sum
                    duplicate.plate_count += slot.plate_count
                elif duplicate.plate_sum is None:
                    duplicate.plate_sum = slot.plate_sum.copy()
                    duplicate.plate_count = slot.plate_count
            for position in range(SERIAL_LENGTH):
                for digit, score in slot.digit_votes[position].items():
                    duplicate.digit_votes[position][digit] += score
            for prefix, score in slot.prefix_votes.items():
                duplicate.prefix_votes[prefix] += score
            for prefix, score in slot.solo_prefix_votes.items():
                duplicate.solo_prefix_votes[prefix] += score
        return kept

    def _assign_rows(self, slots: list[Slot]) -> dict[int, list[Slot]]:
        """Cluster slots into rows, following a row's tilt across the frame."""
        if not slots:
            return {}
        height = float(np.median([slot.box[3] for slot in slots]))
        rows = cluster_rows(slots, lambda slot: slot.center, height)
        grouped: dict[int, list[Slot]] = {}
        for index, row in enumerate(rows, start=1):
            for slot in row:
                slot.row = index
            grouped[index] = row
        return grouped

    def refine(self, engine) -> None:
        """Re-read both text lines from temporally averaged plates.

        A single frame of a distant row does not carry the light-printed bay
        number at all -- the serial below it is legible while the prefix above
        is not.  Averaging every observation of a sticker, then stacking the
        whole row's averages, recovers it, and the same average sharpens the
        serial enough to settle digits the per-frame votes split on.  This is
        deliberately a separate pass: it costs one recognition batch per row
        instead of per frame.
        """
        # Clustered over the live slots themselves, not over the merged copies,
        # so the votes land on the slots that keep accumulating.
        live = [
            slot for slot in self.slots
            if slot.plate_count and slot.ink >= INK_FLOOR
        ]
        if not live:
            return
        height = float(np.median([slot.box[3] for slot in live]))
        views: list[np.ndarray] = []
        # The trailing flag marks a view built from one sticker's own plate, as
        # opposed to its whole row stacked together.
        owners: list[tuple[list[Slot], str, float, bool]] = []
        for slots in cluster_rows(live, lambda slot: slot.center, height):
            plates = [slot.mean_plate() for slot in slots]
            # Both views are read: the row stack survives a sticker whose own
            # average is hopeless, while a sticker's own average survives a row
            # whose stack blurred the prefix by stacking misaligned plates.
            stacked = stack_plates([plate for plate in plates if plate is not None])
            if stacked is not None:
                depth = float(np.mean([slot.plate_count for slot in slots]))
                for view in prefix_views_from_plate(stacked):
                    views.append(view)
                    owners.append((slots, "prefix", depth, False))
            for slot, plate in zip(slots, plates):
                if plate is None:
                    continue
                for kind, build in (
                    ("prefix", prefix_views_from_plate),
                    ("serial", serial_views_from_plate),
                ):
                    for view in build(plate):
                        views.append(view)
                        owners.append(([slot], kind, float(slot.plate_count), True))
        if not views:
            return
        results = engine.recognize_lines(views)
        for (slots, kind, depth, solo), (text, score) in zip(owners, results):
            # An average of N observations suppresses noise as sqrt(N), so a
            # deeply averaged plate speaks proportionally louder than the
            # per-frame reads it is meant to settle.
            weight = score * np.sqrt(max(1.0, depth))
            if kind == "prefix":
                prefix = prefix_from_text(text)
                if prefix and score >= 0.25:
                    for slot in slots:
                        slot.prefix_votes[prefix] += weight * REFINED_PREFIX_WEIGHT
                        if solo:
                            slot.solo_prefix_votes[prefix] += (
                                weight * REFINED_PREFIX_WEIGHT
                            )
                continue
            serial = serial_from_text(text)
            if serial and score >= 0.25:
                for slot in slots:
                    for position, digit in enumerate(serial):
                        slot.digit_votes[position][digit] += (
                            weight * REFINED_SERIAL_WEIGHT
                        )
                    if serial[0] in LEADING_ZERO_CONFUSIONS:
                        slot.digit_votes[0]["0"] += (
                            weight * REFINED_SERIAL_WEIGHT * 0.45
                        )

    def results(self) -> list[dict]:
        """Publish one entry per slot, label decided by accumulated pixels."""
        # A slot with neither ink nor any digit evidence can never publish a
        # label, and keeping it would let a curtain fold sitting between two
        # stickers split their row -- which turns that half-row into its own
        # bay consensus.
        slots_in_view = [
            slot for slot in self._merged_slots()
            if slot.ink >= INK_FLOOR or slot.support() >= SEGMENTED_MIN_SUPPORT
        ]
        by_row = self._assign_rows(slots_in_view)
        prefix_slots = [slot for slot in slots_in_view if not slot.proposed]
        # Frame-wide zone and tens, used only where a row's own evidence is too
        # thin to stand on: a distant row often cannot read its bay number at
        # all, and its neighbours can.
        frame_letter, frame_letter_margin = _pool_prefix_vote(
            prefix_slots, lambda prefix: prefix[0]
        )
        if frame_letter_margin < LETTER_CONSENSUS_MARGIN:
            frame_letter, frame_letter_margin = None, 0.0
        frame_tens, frame_tens_margin = _pool_prefix_vote(
            prefix_slots, lambda prefix: prefix[1]
        )
        if frame_tens_margin < TENS_CONSENSUS_MARGIN:
            frame_tens, frame_tens_margin = None, 0.0

        # Zone and bay decade are settled per row, before the units digit: a
        # shelf can hold more than one zone at a time, and a row in another one
        # must not have its digits pooled together with this row's.  A row only
        # speaks for itself when most of its stickers agree; otherwise it takes
        # the frame-wide reading, which is what carries a row too distant to
        # read its own bay.
        row_zone: dict[int, tuple[str | None, float, str | None, float]] = {}
        for row, slots in by_row.items():
            own_slots = [slot for slot in slots if not slot.proposed]
            letter, letter_margin = _row_prefix_majority(
                own_slots, lambda prefix: prefix[0]
            )
            if letter is None:
                letter, letter_margin = frame_letter, frame_letter_margin
            tens, tens_margin = _row_prefix_majority(
                own_slots, lambda prefix: prefix[1]
            )
            if tens is None:
                tens, tens_margin = frame_tens, frame_tens_margin
            row_zone[row] = (letter, letter_margin, tens, tens_margin)

        # Neighbouring shelf rows carry different bay units.  Resolve units per
        # row first, then break collisions so a weak distant row cannot steal
        # the bay of the sharp row above it.
        row_rankings: dict[int, list[tuple[str, float, float]]] = {}
        row_units_claimed: dict[int, tuple[str, float]] = {}
        for row, slots in by_row.items():
            own_slots = [slot for slot in slots if not slot.proposed]
            row_rankings[row] = _pool_prefix_ranking(
                own_slots, lambda prefix: prefix[2]
            )
            units, share = _row_prefix_majority(own_slots, lambda prefix: prefix[2])
            if units is not None:
                row_units_claimed[row] = (units, share)
        row_units = _unique_row_units(row_rankings, row_units_claimed)
        units_owner = {
            units: row for row, (units, _, _) in row_units.items() if units
        }

        published: list[dict] = []
        for row, slots in sorted(by_row.items()):
            slots.sort(key=lambda slot: slot.center[0])
            letter, letter_margin, tens, tens_margin = row_zone[row]
            units, units_margin, _ = row_units.get(row, (None, 0.0, 0.0))
            row_bay = f"{tens}{units}" if tens and units else None
            # A bay assigned by cross-row uniqueness is authoritative even when
            # its margin is soft -- the alternative was a collision.
            row_bay_margin = (
                max(units_margin, ROW_BAY_SETTLED_MARGIN)
                if units else 0.0
            )
            if tens:
                row_bay_margin = min(tens_margin, row_bay_margin)
            consensus_prefix = f"{letter}{row_bay}" if letter and row_bay else None
            consensus_margin = min(letter_margin, row_bay_margin)
            row_bay_support = sum(
                weight for slot in slots
                for candidate, weight in slot.prefix_votes.items()
                if row_bay and candidate[2] == row_bay[1]
            )

            entries: list[dict] = []
            for slot in slots:
                serial, margin = slot.serial()
                own, own_margin = slot.own_prefix()
                # A sticker may disagree with its row about the units digit --
                # that is how one moved in from a neighbouring bay reads -- but
                # the tens digit stays on the row consensus, since a
                # neighbouring bay shares it.
                own_bay = (
                    f"{tens or own[1]}{own[2]}" if own else None
                )
                own_bay_support = sum(
                    weight for candidate, weight in slot.prefix_votes.items()
                    if candidate[2] == own[2]
                ) if own else 0.0
                letter_agrees = bool(own and letter and own[0] == letter)
                # A sticker whose bay digit is already owned by another row is
                # almost always a misread of this row's prefix, not a physical
                # transplant from that other bay.
                own_units_claimed_elsewhere = bool(
                    own and units_owner.get(own[2], row) != row
                )
                own_overrides = (
                    letter_agrees
                    and not own_units_claimed_elsewhere
                    and own_margin >= 0.95
                    and own_bay_support >= max(
                        BAY_OVERRIDE_MIN_SUPPORT * 1.5,
                        row_bay_support * max(BAY_OVERRIDE_SHARE, 0.65),
                    )
                    and not slot.proposed
                )
                settled = row_bay is not None and row_bay_margin >= ROW_BAY_SETTLED_MARGIN
                if (
                    own_bay
                    and letter_agrees
                    and not own_units_claimed_elsewhere
                    and (not settled or own_overrides)
                ):
                    bay, prefix_source = own_bay, "sticker"
                elif row_bay:
                    bay, prefix_source = row_bay, "row"
                else:
                    bay, prefix_source = own_bay, "sticker"
                zone = letter or (own[0] if own else None)
                prefix = f"{zone}{bay}" if zone and bay else None
                entries.append({
                    "row": row,
                    "center": [round(slot.center[0], 1), round(slot.center[1], 1)],
                    "box": list(slot.box),
                    "label": None,
                    "resolved": False,
                    "serial": serial,
                    "serial_margin": margin,
                    "serial_support": round(slot.support(), 3),
                    "prefix": prefix,
                    "prefix_source": prefix_source,
                    "prefix_margin": (
                        own_margin if prefix_source == "sticker" else consensus_margin
                    ),
                    "sticker_prefix": own,
                    "row_prefix": consensus_prefix,
                    "prefix_conflict": bool(
                        own_bay and row_bay and own_bay != row_bay
                    ),
                    "ink": round(slot.ink, 4),
                    "proposed": slot.proposed,
                    "frames": slot.frames,
                    "segmented_frames": slot.segmented_frames,
                    "serial_source": "pixels",
                })
            annotate_prefix(entries, slots)
            supports = [
                entry["serial_support"] for entry in entries if entry["serial"]
            ]
            floor = (
                float(np.median(supports)) * ROW_SUPPORT_SHARE if supports else 0.0
            )
            for entry in entries:
                decide_resolved(entry, floor)
            settle_uncertain_serials(entries, slots)
            for entry in entries:
                decide_resolved(entry, floor)
            withdraw_repeated_serials(entries)
            # A duplicate withdrawal often frees the serial the weaker slot was
            # supposed to carry (0013 taken twice leaves the real 0011 unread).
            # Re-rank only the withdrawn slots so the strong copy stays put.
            settle_uncertain_serials(entries, slots, only_unresolved=True)
            for entry in entries:
                decide_resolved(entry, floor)
            withdraw_repeated_serials(entries)
            annotate_sequence(entries)
            withdraw_weak_out_of_sequence(entries)
            withdraw_isolated_row(entries)
            withdraw_low_evidence_labels(entries)
            published.extend(entries)
        return published


def _tolerances(sticker: Sticker) -> tuple[float, float]:
    """How far this sticker may have moved since the previous frame.

    Scaled to the sticker's own size rather than the frame's median: a near row
    is both larger on screen and sweeps further between frames, and a median
    taken over the whole frame is dominated by the small distant rows.  Too
    tight a bound there is what fragments a near row into a new slot per frame,
    splitting the evidence that row needed.
    """
    return max(14.0, sticker.box[2] * 0.85), max(14.0, sticker.box[3] * 1.20)


def _pool_prefix_part(
    slots: list[Slot], part, keep=None, *, solo: bool = False
) -> dict[str, float]:
    """Vote weight per value of one prefix character, pooled over given slots.

    `keep` restricts which whole prefixes may contribute.  Pooling the three
    characters independently is what lets a faint row combine partial reads,
    but without a filter it also shreds two incompatible prefixes and
    recombines them: a row holding A04 and B12 stickers would pool a tens digit
    of 0 with a units digit of 2 and publish A02, which no sticker ever read.

    `solo` pools only what each sticker read by itself, leaving out the
    row-stack read that was broadcast to all of them.
    """
    pooled: dict[str, float] = defaultdict(float)
    for slot in slots:
        votes = slot.solo_prefix_votes if solo else slot.prefix_votes
        for prefix, weight in votes.items():
            if keep is not None and not keep(prefix):
                continue
            pooled[part(prefix)] += weight
    return pooled


def _pool_prefix_vote(
    slots: list[Slot], part, keep=None, *, solo: bool = False
) -> tuple[str | None, float]:
    """Winner and margin for one character of the prefix."""
    return rank_votes(_pool_prefix_part(slots, part, keep, solo=solo))


def _row_prefix_majority(slots: list[Slot], part) -> tuple[str | None, float]:
    """One prefix character for a row, if most of its stickers agree on it.

    Which zone or bay decade a row belongs to is a question about how many of
    its stickers claim it, so the vote is one per sticker rather than by vote
    weight.  Weight would let the row-stack read decide it: that read is a
    single observation broadcast to every sticker in the row, so it outweighs
    all of them together.

    The majority has to be of the whole row, not just of the stickers that
    managed a read, and there is no answer when it isn't reached.  On a row too
    distant to read, two or three stickers get a prefix and they tend to be
    wrong the same way, so a majority among only those would be worth nothing.
    """
    counts: dict[str, float] = defaultdict(float)
    for slot in slots:
        value, margin = rank_votes(slot.solo_prefix_votes)
        if value is not None and margin > 0.0:
            counts[part(value)] += 1.0
    winner, _ = rank_votes(counts)
    if winner is None or counts[winner] <= len(slots) / 2:
        return None, 0.0
    # Report the share of the row claiming this value, not its lead over the
    # runner-up: a strict majority already decided the question, and a narrow
    # lead must not then read as weak evidence to the caller.
    return winner, round(counts[winner] / len(slots), 4)


def _pool_prefix_ranking(
    slots: list[Slot], part, keep=None, *, solo: bool = False
) -> list[tuple[str, float, float]]:
    """All candidates for one prefix part, strongest first as (value, margin, support)."""
    pooled = _pool_prefix_part(slots, part, keep, solo=solo)
    if not pooled:
        return []
    ranked = sorted(pooled.items(), key=lambda entry: -entry[1])
    total = sum(pooled.values())
    result: list[tuple[str, float, float]] = []
    for index, (value, support) in enumerate(ranked):
        runner = ranked[index + 1][1] if index + 1 < len(ranked) else 0.0
        margin = (support - runner) / total if total else 0.0
        result.append((value, round(margin, 4), support))
    return result


def _unique_row_units(
    row_rankings: dict[int, list[tuple[str, float, float]]],
    pinned: dict[int, tuple[str, float]] | None = None,
) -> dict[int, tuple[str | None, float, float]]:
    """Ensure neighbouring rows do not claim the same bay units digit.

    Shelf rows above one another are consecutive bays.  When a blurred row's
    OCR collapses onto its sharper neighbour's bay, the collision is broken by
    giving the shared units digit to the row with more prefix evidence and
    leaving the loser on its next-best units candidate.

    A row whose own stickers agree on a units digit by majority is settled
    first and keeps it, since that outranks any weight a blurred row pools.
    """
    pinned = pinned or {}
    claimed: set[str] = set()
    resolved: dict[int, tuple[str | None, float, float]] = {}
    order = sorted(
        row_rankings.items(),
        key=lambda item: (
            0 if item[0] in pinned else 1,
            -(item[1][0][2] if item[1] else 0.0),
            -(item[1][0][1] if item[1] else 0.0),
            item[0],
        ),
    )
    for row, ranking in order:
        chosen: tuple[str | None, float, float] = (None, 0.0, 0.0)
        want = pinned.get(row)
        if want and want[0] not in claimed:
            support = next(
                (entry[2] for entry in ranking if entry[0] == want[0]), 0.0
            )
            chosen = (want[0], want[1], support)
        else:
            for value, margin, support in ranking:
                if value in claimed:
                    continue
                chosen = (value, margin, support)
                break
        if chosen[0]:
            claimed.add(chosen[0])
        resolved[row] = chosen
    return resolved


def decide_resolved(entry: dict, support_floor: float = 0.0) -> None:
    """Set whether this entry may be published, and with what label."""
    if entry["serial_source"] == "row_tiebreak":
        minimum_margin, minimum_support = TIEBREAK_MIN_MARGIN, SEGMENTED_MIN_SUPPORT
    elif entry["proposed"]:
        minimum_margin, minimum_support = PROPOSAL_MIN_MARGIN, PROPOSAL_MIN_SUPPORT
    else:
        minimum_margin, minimum_support = SEGMENTED_MIN_MARGIN, SEGMENTED_MIN_SUPPORT
    entry["resolved"] = bool(
        entry["serial"]
        and entry["prefix"]
        and PREFIX_PATTERN.fullmatch(entry["prefix"])
        and entry["serial_margin"] >= minimum_margin
        and entry["serial_support"] >= max(minimum_support, support_floor)
        and MIN_INK_RATIO <= entry["ink"] <= MAX_INK_RATIO
    )
    entry["label"] = (
        f"{entry['prefix']}-{entry['serial']}" if entry["resolved"] else None
    )


def _serial_candidates(slot: Slot, limit: int = 12) -> list[str]:
    """The serials this sticker's pixels actually voted for, strongest first.

    The leading digit keeps an extra zero option when the top vote is a known
    zero-confusion: distant 0010 stickers otherwise lock onto 2010 and the
    row-aware re-ranker has nothing legal left to pick.
    """
    per_position: list[list[tuple[str, float]]] = []
    for index, votes in enumerate(slot.digit_votes):
        if not votes:
            return []
        ranked = sorted(votes.items(), key=lambda item: -item[1])[:2]
        if (
            index == 0
            and ranked[0][0] in LEADING_ZERO_CONFUSIONS
            and "0" not in {digit for digit, _ in ranked}
        ):
            zero_weight = votes.get("0", ranked[0][1] * 0.25)
            ranked = ranked + [("0", zero_weight)]
        per_position.append(ranked)
    scored: list[tuple[float, str]] = []
    for combination in itertools.product(*per_position):
        scored.append((
            min(weight for _, weight in combination),
            "".join(digit for digit, _ in combination),
        ))
    scored.sort(reverse=True)
    return [serial for _, serial in scored[:limit]]


def _slot_supports_serial(slot: Slot, serial: str) -> bool:
    """True when every digit of serial has some pixel vote on this sticker."""
    if len(serial) != SERIAL_LENGTH:
        return False
    return all(
        serial[index] in slot.digit_votes[index]
        for index in range(SERIAL_LENGTH)
    )


def _slot_partially_supports_serial(
    slot: Slot, serial: str, *, minimum: int = 2
) -> bool:
    """True when at least ``minimum`` digits of serial have pixel votes."""
    if len(serial) != SERIAL_LENGTH:
        return False
    hits = sum(
        1 for index, digit in enumerate(serial)
        if digit in slot.digit_votes[index]
    )
    return hits >= minimum


def settle_uncertain_serials(
    entries: list[dict],
    slots: list[Slot],
    *,
    only_unresolved: bool = False,
) -> None:
    """Re-rank an inconclusive serial against the row it sits in.

    Where the pixels split between two digits, the row still carries usable
    facts: its confident readings span a range, and no two stickers share a
    serial.  A candidate the pixels already voted for that fits both beats one
    that fits neither -- this chooses among pixel evidence rather than inventing
    a value from position.

    Only inconclusive readings are re-ranked.  A confident one keeps whatever it
    says even when that lands outside its row's range, which is precisely how a
    misplaced label stays visible instead of being quietly corrected.
    """
    # A reading is only exempt from the row's checks when it is both decisive
    # and backed by as much evidence as its row-mates.  A high margin on two
    # votes means nothing competed, not that the reading is right, and letting
    # that skip the checks is how an out-of-range serial gets published.
    supports = [entry["serial_support"] for entry in entries if entry["serial"]]
    confidence_floor = float(np.median(supports)) if supports else 0.0
    # Stickers belonging to another bay are left out of every row statistic
    # below: they neither define this row's range nor occupy one of its serials.
    tentative = [
        int(entry["serial"])
        if (
            entry["resolved"]
            and entry["serial_margin"] >= TIEBREAK_MARGIN
            and entry["serial_support"] >= confidence_floor
            and entry["serial"]
            and entry.get("prefix_status") != "out_of_bay"
        )
        else None
        for entry in entries
    ]
    # Build the row range from the tight cluster of confident serials so a
    # single 2010 / 9010 impostor cannot stretch the window and then exempt
    # itself as an "in-range anchor".
    confident_values = sorted(value for value in tentative if value is not None)
    # Seed the range with every already-resolved serial too.  Restricting the
    # window to high-margin anchors alone drops a soft 0010 / 0011 at the end
    # of the row and then refuses to re-rank the neighbour that needed them.
    resolved_values = sorted({
        int(entry["serial"])
        for entry in entries
        if entry.get("resolved") and entry.get("serial")
        and entry.get("prefix_status") != "out_of_bay"
    })
    # Always widen the window with resolved serials so a soft end sticker
    # (0010 with low margin) still counts as in-range for its neighbours.
    seed = sorted(set(confident_values) | set(resolved_values))
    if len(seed) < 3:
        return
    core = seed
    if len(seed) >= 4:
        med = int(np.median(seed))
        core = [
            value for value in seed
            if abs(value - med) <= max(RANGE_SLACK * 4, 12)
        ] or seed
    low, high = min(core) - RANGE_SLACK, max(core) + RANGE_SLACK
    anchors = [
        value if (value is not None and low <= value <= high) else None
        for value in tentative
    ]
    taken = {value for value in anchors if value is not None}
    # Resolved non-anchors still occupy their serial so a re-rank cannot steal
    # it, but they do not block a neighbour from taking the open integers.
    taken.update(resolved_values)
    if len(taken) < 3:
        return
    for index, (entry, slot) in enumerate(zip(entries, slots)):
        has_votes = any(slot.digit_votes[position] for position in range(SERIAL_LENGTH))
        if not entry["serial"] and not has_votes:
            continue
        if only_unresolved and entry["resolved"]:
            continue
        # A sticker from another bay is numbered by that bay, so this row's
        # range and occupancy say nothing about what it should read.
        if entry.get("prefix_status") == "out_of_bay":
            continue
        # Confident in-range anchors keep their pixels.  Everything else --
        # unread, low-margin, or a high-margin reading outside the row -- is
        # re-ranked against candidates the pixels actually voted for.
        if anchors[index] is not None:
            continue
        # Prefer high-margin anchors for neighbour bounds, but fall back to any
        # resolved serial so a soft 0010 still brackets the open 0011 slot.
        bound_values = [
            anchors[i] if anchors[i] is not None else (
                int(entries[i]["serial"])
                if entries[i].get("resolved") and entries[i].get("serial")
                else None
            )
            for i in range(len(entries))
        ]
        left = next(
            (value for value in reversed(bound_values[:index]) if value is not None),
            None,
        )
        right = next(
            (value for value in bound_values[index + 1:] if value is not None),
            None,
        )

        def _fits(value: int) -> bool:
            if value in taken or not low <= value <= high:
                return False
            if left is not None and value <= left:
                return False
            if right is not None and value >= right:
                return False
            return True

        current = int(entry["serial"]) if entry["serial"] else None
        # A proposed sticker may already hold the correct serial with a margin
        # just below the proposal threshold.  Marking it as a tiebreak lets
        # decide_resolved publish it instead of discarding a reading that
        # already fits every row constraint.
        if current is not None and _fits(current):
            entry["serial_source"] = "row_tiebreak"
            entry.pop("withdrawn", None)
            taken.add(current)
            continue

        candidates = list(_serial_candidates(slot))
        # When neighbours leave exactly one integer open, include it if the
        # sticker's pixels have any vote mass on those digits.  This recovers
        # 0011 stuck as a duplicate 0013 without inventing digits from thin air.
        if left is not None and right is not None and right - left == 2:
            expected = f"{left + 1:04d}"
            if expected not in candidates and (
                _slot_supports_serial(slot, expected)
                or _slot_partially_supports_serial(slot, expected, minimum=2)
            ):
                candidates.append(expected)

        for candidate in candidates:
            value = int(candidate)
            if not _fits(value):
                continue
            if candidate != entry["serial"]:
                entry["serial"] = candidate
            entry["serial_source"] = "row_tiebreak"
            entry.pop("withdrawn", None)
            taken.add(value)
            break
        else:
            # Nothing the pixels voted for fits the row.  A high-margin reading
            # outside the range is still the pixel answer for a misplaced
            # sticker, so only the weak ones are marked unreadable.
            if entry["serial"] is None:
                continue
            if (
                entry["serial_margin"] >= TIEBREAK_MARGIN
                and entry["serial_support"] >= confidence_floor
            ):
                continue
            entry["serial_source"] = "row_conflict"


def withdraw_repeated_serials(entries: list[dict]) -> None:
    """Unpublish the weaker reading when one row repeats a serial.

    Two stickers in a row cannot carry the same serial, so a repeat means one
    was misread -- and publishing both adds a wrong label on top of the one it
    was mistaken for.  The weaker reading goes back to unread, which is the
    honest answer, while the repeat stays visible on both entries.

    Serials are only compared within one bay.  A sticker carried in from a
    neighbouring bay legitimately repeats a serial this row already has, since
    each bay numbers its own stickers from scratch.
    """
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for entry in entries:
        if entry["resolved"] and entry["serial"]:
            groups[(entry["prefix"], entry["serial"])].append(entry)
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(
            key=lambda entry: entry["serial_margin"] * entry["serial_support"],
            reverse=True,
        )
        for entry in group[1:]:
            entry["resolved"] = False
            entry["label"] = None
            entry["withdrawn"] = "serial_repeated_in_row"


def annotate_prefix(entries: list[dict], slots: list[Slot]) -> None:
    """Honour what each sticker's own crops say about its prefix, and flag it.

    This mirrors the serial policy.  A sticker carried in from another bay
    prints a different prefix, and that is what it has to report; the row
    consensus is only there for the stickers whose own read is too weak to
    stand alone.  The bar is deliberately high, because a blurred sticker
    misreading its own row's bay looks identical at low confidence -- which is
    why a soft disagreement is recorded but not acted on.
    """
    row_prefix = next(
        (entry["row_prefix"] for entry in entries if entry["row_prefix"]), None
    )
    agreeing = [
        slot.solo_prefix_support() for slot in slots
        if not slot.proposed and slot.solo_prefix()[0] == row_prefix
    ]
    reference = float(np.median(agreeing)) if agreeing else 0.0
    for entry, slot in zip(entries, slots):
        solo, solo_margin = slot.solo_prefix()
        support = slot.solo_prefix_support()
        entry["solo_prefix"] = solo
        entry["solo_prefix_margin"] = solo_margin
        entry["solo_prefix_support"] = round(support, 3)
        if not solo or not PREFIX_PATTERN.fullmatch(solo):
            entry["prefix_status"] = "unread"
        elif not row_prefix or solo == row_prefix:
            entry["prefix_status"] = "matches_row"
        elif (
            not slot.proposed
            and solo_margin >= PREFIX_TRANSPLANT_MARGIN
            and support >= PREFIX_TRANSPLANT_MIN_SUPPORT
            and support >= reference * PREFIX_TRANSPLANT_SUPPORT_RATIO
        ):
            entry["prefix"] = solo
            entry["prefix_source"] = "sticker_transplant"
            entry["prefix_margin"] = solo_margin
            entry["prefix_status"] = "out_of_bay"
        else:
            entry["prefix_status"] = "weak_disagreement"


def _misplaced_positions(
    values: list[int], weights: list[float]
) -> set[int]:
    """Smallest set of positions whose removal leaves the rest increasing.

    Checking each sticker against its immediate neighbours is wrong: one
    sticker moved to the wrong slot also breaks the comparison for the
    neighbours it displaced, so a single swap reports four culprits.  Keeping
    the longest increasing subsequence and flagging its complement reports
    exactly the stickers that have to move.  Ties in length are broken towards
    keeping the better-evidenced readings, so a weak misread is preferred as
    the culprit over the strong sticker next to it.
    """
    count = len(values)
    # best[i] = (length, evidence) of the strongest increasing run ending at i.
    best: list[tuple[int, float]] = [(1, weights[i]) for i in range(count)]
    parent = [-1] * count
    for index in range(count):
        for previous in range(index):
            if values[previous] >= values[index]:
                continue
            candidate = (
                best[previous][0] + 1, best[previous][1] + weights[index]
            )
            if candidate > best[index]:
                best[index] = candidate
                parent[index] = previous
    if not count:
        return set()
    tail = max(range(count), key=lambda index: best[index])
    kept: set[int] = set()
    while tail != -1:
        kept.add(tail)
        tail = parent[tail]
    return set(range(count)) - kept


def annotate_sequence(entries: list[dict]) -> None:
    """Flag out-of-sequence serials without changing them.

    Serials normally increase left to right, so a decrease or a repeat is
    worth surfacing to an operator.  Deliberate gaps between sticker groups
    make a strict +1 model unusable as a corrector, which is exactly why this
    only annotates.
    """
    for entry in entries:
        entry["sequence_status"] = "unread" if not entry["resolved"] else "unverified"
    # A sticker from another bay carries that bay's numbering, so it has no
    # place in this row's running order.  Its prefix status already reports it.
    resolved = [
        entry for entry in entries
        if entry["resolved"] and entry["serial"]
        and entry.get("prefix_status") != "out_of_bay"
    ]
    for entry in entries:
        if entry["resolved"] and entry.get("prefix_status") == "out_of_bay":
            entry["sequence_status"] = "out_of_bay"
    if len(resolved) < 3:
        return
    values = [int(entry["serial"]) for entry in resolved]
    weights = [
        entry["serial_margin"] * entry["serial_support"] for entry in resolved
    ]
    misplaced = _misplaced_positions(values, weights)
    for index, entry in enumerate(resolved):
        left = values[index - 1] if index > 0 else None
        right = values[index + 1] if index + 1 < len(values) else None
        if values.count(values[index]) > 1:
            entry["sequence_status"] = "duplicate_serial"
        elif index in misplaced:
            entry["sequence_status"] = "out_of_sequence"
        else:
            entry["sequence_status"] = "in_sequence"
        entry["neighbour_serials"] = [left, right]


def withdraw_weak_out_of_sequence(entries: list[dict]) -> None:
    """Drop weak out-of-sequence readings; keep strong misplaced stickers.

    A fold that OCR invents a serial for often lands outside the row order and
    carries a fraction of the evidence of its neighbours.  A sticker that was
    physically swapped into the wrong slot has comparable support, so it stays
    published and only flagged.
    """
    resolved = [entry for entry in entries if entry["resolved"] and entry["serial"]]
    if len(resolved) < 3:
        return
    supports = [entry["serial_support"] for entry in resolved]
    floor = float(np.median(supports)) * WEAK_OOS_SUPPORT_RATIO
    for entry in resolved:
        if entry.get("sequence_status") not in {"out_of_sequence", "duplicate_serial"}:
            continue
        # Only a high-margin reading with row-comparable support is treated as
        # a physically misplaced sticker.  Everything else that breaks order is
        # an OCR invention and is withdrawn.
        if (
            entry["serial_margin"] >= 0.85
            and entry["serial_support"] >= max(floor, float(np.median(supports)) * 0.5)
        ):
            continue
        entry["resolved"] = False
        entry["label"] = None
        entry["withdrawn"] = "weak_out_of_sequence"


def withdraw_isolated_row(entries: list[dict]) -> None:
    """Unpublish a sparse row that never gathered a real sticker majority.

    Curtain folds segment into one or two bright rectangles and occasionally
    OCR into a full label.  A genuine shelf row has many corroborating
    stickers; a fold cluster does not.
    """
    resolved = [entry for entry in entries if entry["resolved"] and entry["label"]]
    if not resolved:
        return
    single_weak = (
        len(resolved) == 1
        and resolved[0]["serial_support"] < max(SEGMENTED_MIN_SUPPORT * 8, 4.0)
    )
    short_sparse = (
        len(entries) <= ISOLATED_ROW_MAX_SLOTS
        and len(resolved) <= ISOLATED_ROW_MAX_RESOLVED
    )
    if not single_weak and not short_sparse:
        return
    # A short row is still trusted when every resolved sticker is strongly
    # supported -- that is a clipped end of a real shelf, not a fold.
    if (
        not single_weak
        and len(resolved) >= 2
        and all(
            entry["serial_support"] >= max(SEGMENTED_MIN_SUPPORT * 4, 2.0)
            and entry["serial_margin"] >= 0.55
            for entry in resolved
        )
    ):
        return
    for entry in resolved:
        entry["resolved"] = False
        entry["label"] = None
        entry["withdrawn"] = "isolated_row"


def withdraw_low_evidence_labels(entries: list[dict]) -> None:
    """Drop resolved labels whose vote mass is tiny next to their row-mates.

    Margin alone cannot catch these: a fold that OCR reads once scores a
    perfect margin because nothing competed.  Comparing support to the row
    median does, while a real sticker -- even a misplaced one -- sits near
    that median.
    """
    resolved = [entry for entry in entries if entry["resolved"] and entry["serial"]]
    if len(resolved) < 3:
        # Lone resolved survivors in a junk row are handled elsewhere.
        for entry in resolved:
            if entry["serial_support"] < SEGMENTED_MIN_SUPPORT:
                entry["resolved"] = False
                entry["label"] = None
                entry["withdrawn"] = "low_evidence"
        return
    floor = float(np.median([entry["serial_support"] for entry in resolved])) * 0.12
    floor = max(floor, SEGMENTED_MIN_SUPPORT)
    for entry in resolved:
        if entry["serial_support"] >= floor:
            continue
        # Keep a strong-margin reading only when it is not already flagged as
        # sequence-broken; that is the misplaced-sticker path.
        if (
            entry["serial_margin"] >= 0.90
            and entry.get("sequence_status") == "in_sequence"
        ):
            continue
        entry["resolved"] = False
        entry["label"] = None
        entry["withdrawn"] = "low_evidence"
