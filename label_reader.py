#!/usr/bin/env python3
"""Read inventory-sticker labels from pixels only.

Design rule: the label a sticker carries is decided by its own pixels.  Row
geometry is used to find stickers and to group them, never to choose or
overwrite a serial.  That keeps a physically misplaced sticker readable
instead of being "corrected" into the value its position implies.

Each sticker is rendered as several crops of its two text lines (deskewed,
rescaled, contrast-variant), the whole frame's crops go through one batched
recognition call, and every digit position is decided by a
confidence-weighted vote.  Votes are exposed so a temporal layer can keep
accumulating them across frames.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

STICKER_ASPECT_RATIO = 1.78
SERIAL_LENGTH = 4

# Glyph confusions seen on these labels.  Applied only where a digit is
# required; the prefix letter is never mapped to a digit.
DIGIT_FIXES = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "l": "1", "i": "1", "L": "1", "|": "1", "]": "1",
    "S": "5", "s": "5",
    "Z": "2", "z": "2",
    "B": "8",
    "G": "6", "b": "6",
    "T": "7", "?": "7",
    "A": "4",
    "g": "9", "q": "9",
})
PREFIX_DIGIT_FIXES = str.maketrans({
    "O": "0", "o": "0", "Q": "0", "D": "0", "U": "0",
    "I": "1", "l": "1", "i": "1", "L": "1", "|": "1",
    "S": "5", "s": "5", "Z": "2", "z": "2", "B": "8", "G": "6", "T": "7",
})
# A thin "1" in the bay number is regularly returned as a bracket glyph.
PREFIX_BRACKET_ONES = str.maketrans(
    {char: "1" for char in "[]{}()【】|/\\!"}
)
# The zone letter is required by the format, so a digit in its place is a
# misread of the letter it resembles -- an "A" whose crossbar faded reads as
# "4".  This is the inverse of the digit repairs, applied only at that one
# position.
PREFIX_LETTER_FIXES = str.maketrans({
    "4": "A", "8": "B", "6": "G", "5": "S",
    "0": "O", "1": "I", "2": "Z", "7": "T",
})
PREFIX_PATTERN = re.compile(r"^[A-Z][0-9]{2}$")

# Variants of the same sticker, as (rectify_margin, top, bottom, scale, mode).
# Fractions are of the rectified plate, whose extra margin shifts the sticker's
# own text lines inward.  Diversity here is what a per-digit vote consumes.
SERIAL_VARIANTS = (
    (0.06, 0.42, 0.98, 1.0, "plain"),
    (0.06, 0.46, 1.00, 1.0, "clahe"),
    (0.06, 0.40, 0.96, 1.0, "unsharp"),
    (0.16, 0.43, 0.94, 1.0, "plain"),
    (0.16, 0.47, 0.96, 1.4, "clahe"),
)
PREFIX_VARIANTS = (
    (0.06, 0.00, 0.58, 1.0, "plain"),
    (0.16, 0.05, 0.56, 1.0, "clahe"),
)


@dataclass
class Sticker:
    """One localized sticker and the pixel evidence gathered for it."""

    row: int
    box: tuple[int, int, int, int]
    center: tuple[float, float]
    quad: np.ndarray | None = None
    proposed: bool = False
    digit_votes: list[dict[str, float]] = field(
        default_factory=lambda: [defaultdict(float) for _ in range(SERIAL_LENGTH)]
    )
    prefix_votes: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    # The same votes minus anything pooled from the sticker's row.  A sticker
    # that was moved in from another bay only shows up as a disagreement if
    # its own crops are kept separate from its neighbours' consensus.
    solo_prefix_votes: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    readings: list[dict] = field(default_factory=list)
    ink_scores: list[float] = field(default_factory=list)
    # Grayscale rectified plate for this frame, on the canonical grid, so a
    # temporal layer can average observations without registering them.
    plate: np.ndarray | None = None

    @property
    def ink(self) -> float:
        return float(np.median(self.ink_scores)) if self.ink_scores else 0.0

    @property
    def width(self) -> int:
        return self.box[2]

    @property
    def height(self) -> int:
        return self.box[3]

    def serial(self) -> tuple[str | None, float]:
        """Most-voted digit at every position, plus the weakest vote margin."""
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

    def prefix(self) -> tuple[str | None, float]:
        return rank_votes(self.prefix_votes)

    def solo_prefix(self) -> tuple[str | None, float]:
        """What this sticker's own crops say, with no row pooling mixed in."""
        return rank_votes(self.solo_prefix_votes)

    def serial_support(self) -> float:
        return float(sum(self.digit_votes[0].values())) if self.digit_votes[0] else 0.0

    def needs_more_variants(self) -> bool:
        """True when the cheap variants did not agree on all four digits."""
        serial, margin = self.serial()
        return serial is None or margin < 0.85 or self.serial_support() < 0.9


def locate_stickers(image: np.ndarray) -> list[dict]:
    """Segment the bright sticker rectangles, splitting touching neighbours.

    A local-contrast mask keeps a shaded row visible, which a single global
    threshold on this curtain backdrop does not.  Each sticker is returned with
    both an axis-aligned box and the corner quad of its minimum-area rect, so
    recognition can rectify the tilted paper instead of reading it slanted.
    """
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    background = cv2.GaussianBlur(gray, (0, 0), 21)
    contrast = gray.astype(np.int16) - background.astype(np.int16)
    global_floor = int(np.clip(np.percentile(gray, 99.0), 150, 170))
    mask = (
        ((contrast > 9) & (gray > 80)) | (gray >= global_floor)
    ).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_width = max(11, int(width * 0.005))
    max_width = int(width * 0.16)
    min_height = max(7, int(height * 0.005))
    max_height = int(height * 0.07)
    blobs: list[dict] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if not (min_width <= box_width <= max_width):
            continue
        if not (min_height <= box_height <= max_height):
            continue
        if cv2.contourArea(contour) / max(1, box_width * box_height) < 0.42:
            continue
        if box_width / box_height < 0.75:
            continue
        blobs.append({
            "box": (x, y, box_width, box_height),
            "quad": order_quad(cv2.boxPoints(cv2.minAreaRect(contour))),
        })

    return _split_touching(blobs)


def order_quad(points: np.ndarray) -> np.ndarray:
    """Order four corners as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    order = np.argsort(points[:, 1])
    top = sorted(points[order[:2]], key=lambda point: point[0])
    bottom = sorted(points[order[2:]], key=lambda point: point[0])
    return np.array([top[0], top[1], bottom[1], bottom[0]], dtype=np.float32)


def _cut_quad(quad: np.ndarray, parts: int, index: int) -> np.ndarray:
    """Slice a rectified sticker quad into its nth vertical part."""
    top_left, top_right, bottom_right, bottom_left = quad
    left = index / parts
    right = (index + 1) / parts
    return np.array([
        top_left + (top_right - top_left) * left,
        top_left + (top_right - top_left) * right,
        bottom_left + (bottom_right - bottom_left) * right,
        bottom_left + (bottom_right - bottom_left) * left,
    ], dtype=np.float32)


def _split_touching(blobs: list[dict]) -> list[dict]:
    """Cut blobs that merged several stickers, using same-row median width."""
    result: list[dict] = []
    for blob in blobs:
        x, y, box_width, box_height = blob["box"]
        center_y = y + box_height / 2.0
        peers = [
            (other["box"][2], other["box"][3])
            for other in blobs
            if abs(center_y - (other["box"][1] + other["box"][3] / 2.0))
            <= max(box_height, other["box"][3]) * 1.2
            and other["box"][2] / max(1, other["box"][3])
            <= STICKER_ASPECT_RATIO * 1.45
        ]
        reference_width = float(np.median([p[0] for p in peers])) if peers else 0.0
        aspect_parts = box_width / max(box_height * STICKER_ASPECT_RATIO, 1.0)
        width_parts = box_width / reference_width if reference_width else 1.0
        parts = int(np.clip(np.floor(max(aspect_parts, width_parts) + 0.5), 1, 8))
        step = box_width / parts
        for index in range(parts):
            x1 = int(round(x + index * step))
            x2 = int(round(x + (index + 1) * step))
            if (x2 - x1) / max(1, box_height) < 0.60:
                continue
            result.append({
                "box": (x1, y, x2 - x1, box_height),
                "quad": (
                    blob["quad"] if parts == 1
                    else _cut_quad(blob["quad"], parts, index)
                ),
            })
    return sorted(result, key=lambda entry: (
        entry["box"][1] + entry["box"][3] / 2, entry["box"][0]
    ))


def cluster_rows(items: list, centre, height: float) -> list[list]:
    """Group items into shelf rows by growing a fitted line per row.

    A row is straight but tilted and spans the whole frame, so a fixed
    horizontal band loses its far end, while walking nearest neighbours breaks
    apart wherever one sticker along the way is missing or misplaced -- and a
    row broken into segments then votes on its bay number segment by segment.
    Seeding at the topmost item and re-fitting the line as members join
    survives both.
    """
    tolerance = max(height * 0.80, 12.0)
    remaining = list(items)
    rows: list[list] = []
    while remaining:
        seed = min(remaining, key=lambda item: centre(item)[1])
        members = [
            item for item in remaining
            if abs(centre(item)[1] - centre(seed)[1]) <= tolerance
        ]
        for _ in range(4):
            xs = np.array([centre(item)[0] for item in members], dtype=np.float64)
            ys = np.array([centre(item)[1] for item in members], dtype=np.float64)
            if len(members) >= 2 and xs.max() - xs.min() > height * 2:
                slope, intercept = np.polyfit(xs, ys, 1)
            else:
                slope, intercept = 0.0, float(ys.mean())
            grown = [
                item for item in remaining
                if abs(centre(item)[1] - (slope * centre(item)[0] + intercept))
                <= tolerance
            ]
            if len(grown) == len(members):
                break
            members = grown
        if not members:
            members = [seed]
        rows.append(sorted(members, key=lambda item: centre(item)[0]))
        chosen = {id(item) for item in members}
        remaining = [item for item in remaining if id(item) not in chosen]
    rows.sort(key=lambda row: float(np.median([centre(item)[1] for item in row])))
    return rows


def group_rows(stickers: list[dict]) -> list[list[dict]]:
    """Group located stickers into rows, tolerating a tilted row."""
    if not stickers:
        return []
    height = float(np.median([entry["box"][3] for entry in stickers]))
    return cluster_rows(
        stickers,
        lambda entry: (
            entry["box"][0] + entry["box"][2] / 2.0,
            entry["box"][1] + entry["box"][3] / 2.0,
        ),
        height,
    )


def row_pitch(row: list[dict]) -> float | None:
    """Within-cluster centre spacing for a segmented row.

    Cluster gaps (the empty stretch between sticker groups) are several sticker
    widths wide and would pull a plain median far above the true pitch.  Using
    only the short gaps keeps end-extension on the grid the stickers sit on.
    """
    if len(row) < 2:
        return None
    row = sorted(row, key=lambda entry: entry["box"][0])
    gaps = [
        right["box"][0] - left["box"][0]
        for left, right in zip(row, row[1:])
        if right["box"][0] - left["box"][0] > 0
    ]
    if not gaps:
        return None
    width = float(np.median([entry["box"][2] for entry in row]))
    local = [gap for gap in gaps if gap < width * 2.2]
    pitch = float(np.median(local if local else gaps))
    if pitch < width * 0.7:
        return None
    return pitch


def _row_line(row: list[dict]) -> tuple[float, float]:
    """Slope and intercept of the row's sticker centres, for tilted shelves."""
    xs = np.array(
        [entry["box"][0] + entry["box"][2] / 2.0 for entry in row], dtype=np.float64
    )
    ys = np.array(
        [entry["box"][1] + entry["box"][3] / 2.0 for entry in row], dtype=np.float64
    )
    if len(row) >= 2 and xs.max() - xs.min() > 40:
        slope, intercept = np.polyfit(xs, ys, 1)
        return float(slope), float(intercept)
    return 0.0, float(ys.mean())


def propose_missing(row: list[dict]) -> list[dict]:
    """Suggest sticker positions on the row pitch that segmentation missed.

    Only positions inside the row's own span are proposed.  A proposal is a
    place to look, not a label: it survives only if its pixels read a serial.
    """
    if len(row) < 3:
        return []
    row = sorted(row, key=lambda entry: entry["box"][0])
    gaps = [
        right["box"][0] - left["box"][0]
        for left, right in zip(row, row[1:])
        if right["box"][0] - left["box"][0] > 0
    ]
    if not gaps:
        return []
    pitch = float(np.median(gaps))
    width = float(np.median([entry["box"][2] for entry in row]))
    height = float(np.median([entry["box"][3] for entry in row]))
    if pitch < width * 0.7:
        return []
    proposals: list[dict] = []
    for left, right in zip(row, row[1:]):
        span = right["box"][0] - left["box"][0]
        slots = int(round(span / pitch))
        if slots <= 1 or slots > 6:
            continue
        for slot in range(1, slots):
            fraction = slot / slots
            x = int(round(left["box"][0] + span * fraction))
            y = int(round(
                left["box"][1] + (right["box"][1] - left["box"][1]) * fraction
            ))
            quad = (
                left["quad"] + (right["quad"] - left["quad"]) * fraction
                if left.get("quad") is not None and right.get("quad") is not None
                else None
            )
            proposals.append({
                "box": (x, y, int(round(width)), int(round(height))),
                "quad": quad,
            })
    return proposals


def propose_row_ends(row: list[dict], *, max_end_slots: int = 4) -> list[dict]:
    """Walk a few pitches past each end of a truncated row.

    Distant rows regularly lose their leftmost stickers to low contrast while
    the rest of the row segments cleanly.  Only call this for rows that are
    clearly shorter than their neighbours -- walking past a complete row lands
    on the next shelf and poisons bay-prefix votes.
    """
    if max_end_slots <= 0 or len(row) < 3:
        return []
    row = sorted(row, key=lambda entry: entry["box"][0])
    pitch = row_pitch(row)
    if pitch is None:
        return []
    width = float(np.median([entry["box"][2] for entry in row]))
    height = float(np.median([entry["box"][3] for entry in row]))
    slope, intercept = _row_line(row)
    leftmost = row[0]["box"][0] + row[0]["box"][2] / 2.0
    rightmost = row[-1]["box"][0] + row[-1]["box"][2] / 2.0
    proposals: list[dict] = []
    for step in range(1, max_end_slots + 1):
        for center_x in (leftmost - step * pitch, rightmost + step * pitch):
            center_y = slope * center_x + intercept
            proposals.append({
                "box": (
                    int(round(center_x - width / 2.0)),
                    int(round(center_y - height / 2.0)),
                    int(round(width)),
                    int(round(height)),
                ),
                "quad": None,
            })
    return proposals



def _enhance(gray: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clahe":
        return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
    if mode == "normclahe":
        stretched = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 4)).apply(stretched)
    if mode == "unsharp":
        base = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.addWeighted(
            base, 1.6, cv2.GaussianBlur(base, (0, 0), 1.4), -0.6, 0
        )
    return gray


def row_tilt(row: list[Sticker]) -> float:
    if len(row) < 3:
        return 0.0
    xs = np.array([sticker.center[0] for sticker in row], dtype=np.float64)
    ys = np.array([sticker.center[1] for sticker in row], dtype=np.float64)
    if xs.max() - xs.min() < 40:
        return 0.0
    slope = float(np.polyfit(xs, ys, 1)[0])
    return float(np.degrees(np.arctan(slope)))


RECTIFIED_HEIGHT = 132
RECTIFIED_WIDTH = int(RECTIFIED_HEIGHT * STICKER_ASPECT_RATIO)

# Candidates whose rectified plate carries less printed ink than this are not
# stickers at all -- on this curtain backdrop most of them are bright folds.
# Skipping their recognition is what keeps a wide base-camera frame affordable,
# and it also stops them from reaching the row clustering, where a fold sitting
# between two stickers would split one physical row in two.
INK_FLOOR = 0.020


def rectify(image: np.ndarray, sticker: Sticker, margin: float) -> np.ndarray | None:
    """Warp one sticker to a canonical upright rectangle.

    Rectifying each sticker separately removes its own tilt and perspective,
    which an axis-aligned crop cannot, and costs far less than rotating the
    whole frame.
    """
    quad = sticker.quad
    if quad is None:
        x, y, box_width, box_height = sticker.box
        quad = np.array([
            [x, y], [x + box_width, y],
            [x + box_width, y + box_height], [x, y + box_height],
        ], dtype=np.float32)
    center = quad.mean(axis=0)
    expanded = center + (quad - center) * (1.0 + margin)
    target = np.array([
        [0, 0], [RECTIFIED_WIDTH, 0],
        [RECTIFIED_WIDTH, RECTIFIED_HEIGHT], [0, RECTIFIED_HEIGHT],
    ], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(expanded.astype(np.float32), target)
    warped = cv2.warpPerspective(
        image, matrix, (RECTIFIED_WIDTH, RECTIFIED_HEIGHT),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE,
    )
    return warped if warped.size else None


def printed_text_score(plate: np.ndarray) -> float:
    """How much dark print a rectified plate carries, as a 0..1 score.

    A bright curtain fold segments exactly like a sticker but carries no ink,
    and a static fold reads the same garbage in every frame, so cross-frame
    agreement alone cannot reject it.  Requiring dark strokes on both text
    lines does.
    """
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    paper = float(np.percentile(gray, 88))
    if paper < 60:
        return 0.0
    ink = (gray.astype(np.float32) < paper - 42)
    height = ink.shape[0]
    top = ink[int(height * 0.10):int(height * 0.48)]
    bottom = ink[int(height * 0.50):int(height * 0.94)]
    top_ratio = float(top.mean()) if top.size else 0.0
    bottom_ratio = float(bottom.mean()) if bottom.size else 0.0
    # Both printed lines must be present; score on the weaker one.
    return float(min(top_ratio, bottom_ratio))


def select_variants(table: tuple, phase: int, count: int) -> tuple:
    """Pick a rotating subset of variants for this frame.

    Evidence accumulates across frames, so a frame does not need every
    variant.  Cycling the subset keeps per-frame cost low while still covering
    the whole variant set over a short sequence.
    """
    if count >= len(table):
        return table
    start = (phase * count) % len(table)
    return tuple(table[(start + offset) % len(table)] for offset in range(count))


def build_line_views(
    image: np.ndarray,
    stickers: list[Sticker],
    *,
    serial_table: tuple = SERIAL_VARIANTS,
    prefix_table: tuple = PREFIX_VARIANTS,
) -> tuple[list[np.ndarray], list[tuple[Sticker, str]]]:
    """Render the given recognition variants for the given stickers."""
    views: list[np.ndarray] = []
    metadata: list[tuple[Sticker, str]] = []
    for sticker in stickers:
        margins = {margin for margin, *_ in serial_table + prefix_table}
        rectified = {
            margin: rectify(image, sticker, margin) for margin in margins
        }
        reference = rectified.get(0.06)
        if reference is None:
            reference = next(
                (plate for plate in rectified.values() if plate is not None),
                None,
            )
        if reference is not None:
            ink = printed_text_score(reference)
            sticker.ink_scores.append(ink)
            if ink < INK_FLOOR:
                continue
            sticker.plate = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        for kind, table in (
            ("serial", serial_table), ("prefix", prefix_table)
        ):
            for margin, top, bottom, scale, mode in table:
                    plate = rectified.get(margin)
                    if plate is None:
                        continue
                    y1 = int(round(RECTIFIED_HEIGHT * top))
                    y2 = int(round(RECTIFIED_HEIGHT * bottom))
                    line = plate[max(0, y1):min(RECTIFIED_HEIGHT, y2)]
                    if line.size == 0 or line.shape[0] < 6:
                        continue
                    gray = _enhance(cv2.cvtColor(line, cv2.COLOR_BGR2GRAY), mode)
                    if scale != 1.0:
                        gray = cv2.resize(
                            gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC,
                        )
                    views.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
                    metadata.append((sticker, kind))
    return views, metadata


# The bay number is printed in a much lighter stroke than the serial, so it
# needs contrast pulled up rather than just rescaled.  These are the crops and
# enhancements that read it on a stacked plate, as (top, bottom, scale, mode).
STACKED_PREFIX_VARIANTS = (
    (0.14, 0.50, 2.0, "normclahe"),
    (0.14, 0.50, 2.0, "clahe"),
    (0.18, 0.46, 3.0, "normclahe"),
    (0.10, 0.52, 2.0, "unsharp"),
)


def stack_plates(plates: list[np.ndarray]) -> np.ndarray | None:
    """Average aligned plates.

    The mean beats the median here: the strokes of a light-printed bay number
    are only a few grey levels above the paper, and the median discards exactly
    the faint agreement across observations that makes them readable.
    """
    usable = [plate for plate in plates if plate is not None and plate.size]
    if not usable:
        return None
    shape = usable[0].shape
    same = [plate for plate in usable if plate.shape == shape]
    if not same:
        return None
    return np.mean(np.stack(same).astype(np.float32), axis=0).astype(np.uint8)


# A temporally averaged plate sits on the canonical grid far more precisely
# than any single frame, so the serial crop can start below the prefix line
# instead of reaching up to tolerate localisation jitter.  Reaching up is what
# lets a prefix glyph enter the serial and read as its leading digit.
STACKED_SERIAL_VARIANTS = (
    (0.50, 0.98, 2.0, "normclahe"),
    (0.52, 1.00, 2.0, "clahe"),
    (0.48, 0.96, 3.0, "unsharp"),
    (0.54, 0.98, 3.0, "normclahe"),
)


def views_from_plate(plate: np.ndarray, table: tuple) -> list[np.ndarray]:
    """Render one stacked plate's recognition variants for the given table."""
    views: list[np.ndarray] = []
    for top, bottom, scale, mode in table:
        line = plate[
            int(round(plate.shape[0] * top)):int(round(plate.shape[0] * bottom))
        ]
        if line.size == 0 or line.shape[0] < 6:
            continue
        rendered = _enhance(line, mode)
        if scale != 1.0:
            rendered = cv2.resize(
                rendered, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        views.append(cv2.cvtColor(rendered, cv2.COLOR_GRAY2BGR))
    return views


def prefix_views_from_plate(plate: np.ndarray) -> list[np.ndarray]:
    return views_from_plate(plate, STACKED_PREFIX_VARIANTS)


def serial_views_from_plate(plate: np.ndarray) -> list[np.ndarray]:
    return views_from_plate(plate, STACKED_SERIAL_VARIANTS)


def build_row_prefix_composites(
    image: np.ndarray, rows: list[list[Sticker]]
) -> tuple[list[np.ndarray], list[list[Sticker]]]:
    """Stack the prefix line shared by every sticker in a row.

    All stickers in a row carry the same printed prefix, and rectification puts
    them on a common canonical grid, so stacking needs no alignment and removes
    most of the noise that defeats a single small crop at distance.
    """
    views: list[np.ndarray] = []
    owners: list[list[Sticker]] = []
    for row in rows:
        plates = [
            cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            for plate in (rectify(image, sticker, 0.06) for sticker in row)
            if plate is not None and plate.size
        ]
        if len(plates) < 3:
            continue
        stacked = stack_plates(plates)
        if stacked is None:
            continue
        views.append(cv2.cvtColor(
            _enhance(stacked[0:int(RECTIFIED_HEIGHT * 0.58)], "normclahe"),
            cv2.COLOR_GRAY2BGR,
        ))
        owners.append(row)
    return views, owners


def serial_from_text(text: str) -> str | None:
    """Extract the four-digit serial, never by blind right-alignment.

    The serial crop sometimes catches the prefix line above it, so a reading
    can hold more than four glyphs.  Right-aligning the digits then shifts
    them silently: a dropped leading zero plus a stray edge glyph turns
    ``A04-0016`` into ``1600``.  A longer reading is therefore only split when
    a separator marks where the prefix ends; otherwise it is dropped and the
    other variants of the same sticker decide the vote.
    """
    raw = str(text)
    match = re.search(
        r"[A-Za-z][0-9A-Za-z]{2}\s*[-–—_/]\s*([0-9A-Za-z]{4})(?![0-9A-Za-z])", raw
    )
    if match:
        candidate = match.group(1).translate(DIGIT_FIXES)
        return candidate if candidate.isdigit() else None
    compact = re.sub(r"[^0-9A-Za-z]", "", raw)
    # A serial line whose first glyph reads as a letter is far more often the
    # prefix line's letter bleeding into the crop than a misread digit, so such
    # a reading is dropped instead of mapped -- "A012" would become "4012".
    if re.fullmatch(r"[A-Za-z][0-9]{3}", compact):
        return None
    digits = re.sub(r"[^0-9]", "", compact.translate(DIGIT_FIXES))
    return digits if len(digits) == SERIAL_LENGTH else None


def prefix_from_text(text: str) -> str | None:
    """Parse a shelf prefix: one zone letter plus a two-digit bay number.

    Both halves are repaired towards the shape the format demands before the
    result is accepted, because the glyphs that get confused are known: a thin
    bay "1" comes back as a bracket, and a faded zone "A" comes back as "4".
    Dropping those readings throws away most of the evidence a light-printed
    prefix ever produces.
    """
    compact = re.sub(r"[^0-9A-Za-z\[\]{}()【】|/\\!]", "", str(text)).upper()
    compact = compact.translate(PREFIX_BRACKET_ONES)
    for match in re.finditer(r"([0-9A-Z])([0-9A-Z]{2})", compact):
        letter = match.group(1)
        if letter.isdigit():
            letter = letter.translate(PREFIX_LETTER_FIXES)
        candidate = f"{letter}{match.group(2).translate(PREFIX_DIGIT_FIXES)}"
        if PREFIX_PATTERN.fullmatch(candidate):
            return candidate
    return None


def accumulate(
    metadata: list[tuple[Sticker, str]],
    results: list[tuple[str, float]],
    *,
    minimum_score: float = 0.25,
) -> None:
    """Fold one frame's recognition output into each sticker's vote tallies."""
    for (sticker, kind), (text, score) in zip(metadata, results):
        sticker.readings.append(
            {"kind": kind, "text": text, "score": round(score, 4)}
        )
        if score < minimum_score:
            continue
        if kind == "serial":
            serial = serial_from_text(text)
            if serial:
                for position, digit in enumerate(serial):
                    sticker.digit_votes[position][digit] += score
                # Distant leading zeros regularly read as 2/6/8/9.  Keep a
                # parallel vote for zero so temporal fusion can still land on
                # 0010 when every raw crop preferred 2010.
                if serial[0] in "2689":
                    sticker.digit_votes[0]["0"] += score * 0.45
        else:
            prefix = prefix_from_text(text)
            if prefix:
                sticker.prefix_votes[prefix] += score
                sticker.solo_prefix_votes[prefix] += score


def rank_votes(votes: dict[str, float]) -> tuple[str | None, float]:
    """Winner and its margin over the runner-up, as a share of all weight."""
    if not votes:
        return None, 0.0
    ranked = sorted(votes.items(), key=lambda entry: -entry[1])
    total = sum(votes.values())
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], round((ranked[0][1] - runner) / total, 4)


def row_prefix(row: list[Sticker]) -> tuple[str | None, float]:
    """Consensus prefix for a row.

    Every sticker in a row carries the same printed prefix, so pooling the
    row's votes is far more reliable than one sticker's crop, especially at
    distance.  Per-sticker disagreement is reported separately rather than
    silently resolved.
    """
    pooled: dict[str, float] = defaultdict(float)
    for sticker in row:
        for prefix, weight in sticker.prefix_votes.items():
            pooled[prefix] += weight
    if not pooled:
        return None, 0.0
    ranked = sorted(pooled.items(), key=lambda entry: -entry[1])
    total = sum(pooled.values())
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    return ranked[0][0], round((ranked[0][1] - runner) / total, 4)
