#!/usr/bin/env python3
"""Render annotated prediction videos for the pixel-first label pipeline.

Two modes:

  normal     replay a trial's frames and draw the tracker state after each
             frame, so the video shows evidence accumulating

  misplaced  swap two stickers' pixels inside the frames first, which
             reproduces a physically misplaced label, then replay the same way
  zone       repaint a stretch of the lower row into another zone, which
             reproduces stickers shelved in the wrong bay entirely

Stickers are ~35 px wide in the raw 1920x1080 frame, so the output is cropped
to the region the labels live in and upscaled; otherwise nothing is readable.

    python render_video.py --trials A5_A4-5__light --cameras head,base
    python render_video.py --mode misplaced --trials A4__light --cameras head
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

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

from ground_truth import GROUND_TRUTH, expected_labels  # noqa: E402
from label_ocr_engine import RecognitionEngine  # noqa: E402
from label_pipeline import LabelTracker, read_frame  # noqa: E402
from label_reader import group_rows, locate_stickers  # noqa: E402
from test_misplaced_labels import swap_stickers  # noqa: E402
from test_zone_letter import repaint  # noqa: E402

GREEN = (60, 220, 90)
AMBER = (40, 190, 245)
RED = (60, 60, 240)
VIOLET = (230, 90, 200)
GREY = (140, 140, 140)
WHITE = (245, 245, 245)
FONT = cv2.FONT_HERSHEY_SIMPLEX
OUT_WIDTH = 1920
OUT_BODY_HEIGHT = 1200
MAX_ZOOM = 4.0
ZONE_DEMO_PREFIX = "B12"
MISPLACED = {"out_of_sequence", "duplicate_serial"}


def flag_of(entry: dict) -> str | None:
    """Which kind of misplacement this entry carries, if any.

    A sticker in the wrong slot of its own bay and one carried in from another
    bay are different mistakes for whoever has to reshelve it, so they are
    drawn apart.
    """
    if entry.get("prefix_status") == "out_of_bay":
        return "out_of_bay"
    if entry.get("sequence_status") in MISPLACED:
        return "out_of_sequence"
    return None


def crop_bands(
    snapshots: list[list[dict]], shape: tuple[int, ...]
) -> list[tuple[int, int, int, int]]:
    """One crop per shelf row, sharing an x range, ordered top to bottom.

    Shelf rows are metres apart on the curtain, so a single crop covering both
    is mostly the empty span between them and leaves the stickers as small as
    they were in the raw frame.  Cropping each row and stacking the strips
    spends the whole output on the stickers.

    Slots that never produced a serial are left out: a fold halfway down the
    curtain would otherwise stretch a band over dead frame.
    """
    height, width = shape[:2]
    read = [
        entry for snapshot in snapshots for entry in snapshot
        if entry["resolved"] and entry["label"]
    ]
    if not read:
        read = [entry for snapshot in snapshots for entry in snapshot]
    if not read:
        return [(0, 0, width, height)]

    boxes = [entry["box"] for entry in read]
    x0 = max(0, min(box[0] for box in boxes) - max(40, int(
        (max(box[0] + box[2] for box in boxes) - min(box[0] for box in boxes))
        * 0.04
    )))
    x1 = min(width, max(box[0] + box[2] for box in boxes) + 40)

    rows: dict[int, list[tuple[int, int]]] = {}
    for entry in read:
        _, by, _, bh = entry["box"]
        rows.setdefault(entry.get("row", 0), []).append((by, by + bh))

    bands = []
    for spans in rows.values():
        top = min(span[0] for span in spans)
        bottom = max(span[1] for span in spans)
        # Headroom above for the three staggered levels of label text.
        head = max(70, int((bottom - top) * 1.6))
        top, bottom = max(0, top - head), min(height, bottom + 12)
        bands.append((x0, top, x1 - x0, bottom - top))
    bands.sort(key=lambda band: band[1])

    # Overlapping bands would show the same stickers twice; merge them.
    merged = [bands[0]]
    for band in bands[1:]:
        last = merged[-1]
        if band[1] <= last[1] + last[3]:
            bottom = max(last[1] + last[3], band[1] + band[3])
            merged[-1] = (last[0], last[1], last[2], bottom - last[1])
        else:
            merged.append(band)
    return merged


def drop_shadowed(entries: list[dict]) -> list[dict]:
    """Hide an unread slot that sits on top of a published one.

    The tracker keeps both until they merge, which is correct bookkeeping but
    draws two boxes around one sticker.
    """
    resolved = [
        entry for entry in entries if entry["resolved"] and entry["label"]
    ]
    keep = []
    for entry in entries:
        if entry in resolved:
            keep.append(entry)
            continue
        ax, ay, aw, ah = entry["box"]
        shadowed = False
        for other in resolved:
            bx, by, bw, bh = other["box"]
            overlap = (
                max(0, min(ax + aw, bx + bw) - max(ax, bx))
                * max(0, min(ay + ah, by + bh) - max(ay, by))
            )
            if overlap > 0.4 * aw * ah:
                shadowed = True
                break
        if not shadowed:
            keep.append(entry)
    return keep


def draw(
    image: np.ndarray,
    entries: list[dict],
    bands: list[tuple[int, int, int, int]],
    header: list[str],
) -> np.ndarray:
    crops = [image[y:y + h, x:x + w] for x, y, w, h in bands]
    gap = 10 if len(crops) > 1 else 0
    span_x = max(crop.shape[1] for crop in crops)
    span_y = sum(crop.shape[0] for crop in crops) + gap * (len(crops) - 1)
    scale = min(OUT_WIDTH / max(1, span_x), OUT_BODY_HEIGHT / max(1, span_y),
                MAX_ZOOM)

    resized = [
        cv2.resize(
            crop,
            (int(round(crop.shape[1] * scale)), int(round(crop.shape[0] * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
        for crop in crops
    ]
    body_h = sum(crop.shape[0] for crop in resized) + gap * (len(resized) - 1)
    body_w = max(crop.shape[1] for crop in resized)
    # Centre the strips horizontally; vertically the canvas is exactly as tall
    # as they are, so no frame is spent on nothing.
    offset_x = (OUT_WIDTH - body_w) // 2
    offset_y = 0
    stage = np.zeros((body_h, OUT_WIDTH, 3), np.uint8)
    tops: list[int] = []
    cursor = offset_y
    for crop in resized:
        tops.append(cursor)
        stage[cursor:cursor + crop.shape[0],
              offset_x:offset_x + crop.shape[1]] = crop
        cursor += crop.shape[0] + gap

    # The header sits in its own band above the strips, so it can never hide
    # the top row of stickers.
    panel = 34 * len(header) + 16
    canvas = np.vstack([np.zeros((panel, OUT_WIDTH, 3), np.uint8), stage])

    ordered = sorted(
        drop_shadowed(entries), key=lambda entry: entry["center"][0]
    )
    # Stagger by position within the sticker's own row.  Numbering across the
    # whole frame instead would interleave the rows and put two neighbours in
    # one row on the same level.
    level: dict[int, int] = {}
    levels = {}
    for entry in ordered:
        row = entry.get("row", 0)
        levels[id(entry)] = level.get(row, 0)
        level[row] = level.get(row, 0) + 1
    for entry in ordered:
        index = levels[id(entry)]
        bx, by, bw, bh = entry["box"]
        band = next(
            (
                position for position, (_, top, _, height) in enumerate(bands)
                if top <= by + bh / 2 < top + height
            ),
            None,
        )
        if band is None:
            continue
        x0, y0 = bands[band][0], bands[band][1]
        px = int((bx - x0) * scale) + offset_x
        py = int((by - y0) * scale) + tops[band] + panel
        pw, ph = int(bw * scale), int(bh * scale)
        resolved = bool(entry["resolved"] and entry["label"])
        flag = flag_of(entry) if resolved else None
        if flag == "out_of_bay":
            colour = VIOLET
        elif flag:
            colour = RED
        elif resolved:
            colour = GREEN
        elif entry.get("serial"):
            colour = AMBER
        else:
            colour = GREY
        cv2.rectangle(
            canvas, (px, py), (px + pw, py + ph), colour, 2 if resolved else 1
        )
        # The whole label, prefix included: the bay is half of what identifies a
        # book, and a sticker from another bay is only visibly wrong with it.
        text = entry.get("label") or entry.get("serial") or ""
        if not text:
            continue
        if flag:
            text += "!"
        font_scale = 0.56
        (tw, th), _ = cv2.getTextSize(text, FONT, font_scale, 2)
        tx = int(np.clip(px + pw / 2 - tw / 2, 2, canvas.shape[1] - tw - 2))
        # A whole label is wider than the sticker it belongs to, so the row is
        # written on three staggered levels instead of two.
        above = py - 8 - (index % 3) * (th + 12)
        ty = max(panel + th + 4, above)
        cv2.rectangle(
            canvas, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 5), (15, 15, 15), -1
        )
        cv2.putText(
            canvas, text, (tx, ty), FONT, font_scale, colour, 2, cv2.LINE_AA
        )

    for index, line in enumerate(header):
        cv2.putText(
            canvas, line, (18, 34 + index * 34), FONT, 0.72, WHITE, 2,
            cv2.LINE_AA,
        )
    return canvas


def load_frames(trial_dir: Path, camera: str, mode: str, swap: tuple[int, int]):
    paths = sorted(trial_dir.glob(f"{camera}_move_*.jpg"))
    if not paths:
        raise SystemExit(f"no {camera}_move_*.jpg frames in {trial_dir}")
    frames = [cv2.imread(str(path)) for path in paths]
    frames = [frame for frame in frames if frame is not None]
    if mode == "normal":
        return frames, None

    rows = group_rows(locate_stickers(frames[0]))
    if mode == "zone":
        # Every sticker recorded is in zone A, so a sticker from another zone
        # has to be painted in: the lower row's plates are reused and their text
        # lines redrawn at the same size and contrast.
        lower = sorted(
            (row for row in rows if len(row) >= 8),
            key=lambda row: sum(entry["box"][1] for entry in row) / len(row),
        )[-1]
        first, last = swap
        edited = []
        for frame in frames:
            copy = frame.copy()
            for index in range(first, min(last + 1, len(lower))):
                repaint(
                    copy, lower[index]["box"], ZONE_DEMO_PREFIX,
                    f"{10 + index:04d}",
                )
            edited.append(copy)
        return edited, (first, last)

    row = max(rows, key=len)
    left_index, right_index = swap
    if max(left_index, right_index) >= len(row):
        raise SystemExit(f"row only has {len(row)} stickers; pick lower --swap")
    left, right = row[left_index], row[right_index]
    return (
        [swap_stickers(frame, left, right) for frame in frames],
        (left_index, right_index),
    )


def render(
    engine, trial_dir: Path, camera: str, expected: set[str], output: Path,
    fps: float, mode: str, swap: tuple[int, int],
) -> dict:
    frames, swapped = load_frames(trial_dir, camera, mode, swap)
    pose, lighting = trial_dir.name.split("__")
    prefixes = "/".join(GROUND_TRUTH[pose][camera])

    tracker = LabelTracker()
    snapshots = []
    for phase, frame in enumerate(frames):
        tracker.update(read_frame(engine, frame, phase=phase))
        snapshots.append(tracker.results())
    # The refine pass re-reads temporally averaged plates; it is what settles
    # the faint far rows, so the video must end on its output.
    tracker.refine(engine)
    final_entries = tracker.results()
    snapshots.append(final_entries)

    bands = crop_bands(snapshots, frames[0].shape)
    rendered = []
    for phase, snapshot in enumerate(snapshots):
        published = {
            entry["label"] for entry in snapshot
            if entry["resolved"] and entry["label"]
        }
        hit = published & expected
        last = phase == len(snapshots) - 1
        stage = "最终（时序精修后）" if last else f"第 {phase + 1}/{len(frames)} 帧"
        header = [
            f"{pose}  {lighting}  {camera}  rows {prefixes}   {stage}",
            f"published {len(published)}/{len(expected)}   "
            f"correct {len(hit)}   wrong {len(published - expected)}",
            "绿=已读出并发布   黄=检出但证据不足,未发布   灰=未读出",
        ]
        if mode != "normal":
            in_row = sorted(
                entry["label"] for entry in snapshot
                if entry["resolved"] and flag_of(entry) == "out_of_sequence"
            )
            other_bay = sorted(
                entry["label"] for entry in snapshot
                if entry["resolved"] and flag_of(entry) == "out_of_bay"
            )
            header.append(f"排位错乱(红): {in_row or '—'}")
            header.append(f"来自别的书架(紫): {other_bay or '—'}")
        image = frames[min(phase, len(frames) - 1)]
        rendered.append(
            cv2.cvtColor(draw(image, snapshot, bands, header), cv2.COLOR_BGR2RGB)
        )
    rendered.extend([rendered[-1]] * max(4, int(round(fps * 2.5))))

    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(
        str(output), rendered, fps=fps, codec="libx264", macro_block_size=8,
        output_params=["-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p"],
    )

    published = {
        entry["label"] for entry in final_entries
        if entry["resolved"] and entry["label"]
    }
    hit = published & expected
    return {
        "trial": trial_dir.name,
        "camera": camera,
        "mode": mode,
        "output": str(
            output.relative_to(WORK) if output.is_relative_to(WORK) else output
        ),
        "frames": len(frames),
        "expected": len(expected),
        "recall": round(len(hit) / len(expected), 4) if expected else 0.0,
        "precision": round(len(hit) / len(published), 4) if published else 0.0,
        "missed": sorted(expected - published),
        "wrong": sorted(published - expected),
        "swapped_slots": swapped,
        "flagged_out_of_sequence": sorted(
            entry["label"] for entry in final_entries
            if entry["resolved"] and flag_of(entry) == "out_of_sequence"
        ),
        "flagged_out_of_bay": sorted(
            entry["label"] for entry in final_entries
            if entry["resolved"] and flag_of(entry) == "out_of_bay"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", default="A5_A4-5__light")
    parser.add_argument("--cameras", default="head,base")
    parser.add_argument("--mode", default="normal", choices=["normal", "misplaced", "zone"])
    parser.add_argument("--swap", default="3,8")
    parser.add_argument("--out-dir", default=str(WORK / "videos"))
    parser.add_argument("--fps", type=float, default=2.0)
    args = parser.parse_args()

    engine = RecognitionEngine()
    print(f"识别后端: {engine.backend}\n")
    out_dir = Path(args.out_dir)
    swap = tuple(int(value) for value in args.swap.split(","))
    summary = []

    for name in [t for t in args.trials.split(",") if t]:
        pose, _ = name.split("__")
        for camera in [c for c in args.cameras.split(",") if c]:
            if camera not in GROUND_TRUTH.get(pose, {}):
                continue
            suffix = "" if args.mode == "normal" else f"_{args.mode}"
            result = render(
                engine, WORK / "eval" / name, camera,
                expected_labels(pose, camera),
                out_dir / f"{name}_{camera}{suffix}.mp4",
                args.fps, args.mode, swap,
            )
            summary.append(result)
            print(
                f"{name} {camera}: 召回 {result['recall']:.3f} "
                f"精确 {result['precision']:.3f} -> {result['output']}",
                flush=True,
            )
            if result["missed"]:
                print(f"   漏检 {result['missed']}")
            if result["wrong"]:
                print(f"   误检 {result['wrong']}")
            if result["flagged_out_of_sequence"]:
                print(f"   标记排位错乱 {result['flagged_out_of_sequence']}")
            if result["flagged_out_of_bay"]:
                print(f"   标记来自别的书架 {result['flagged_out_of_bay']}")

    index = out_dir / "index.json"
    existing = json.loads(index.read_text()) if index.exists() else []
    keep = {(row["trial"], row["camera"], row["mode"]) for row in summary}
    existing = [
        row for row in existing
        if (row["trial"], row["camera"], row["mode"]) not in keep
    ]
    index.write_text(
        json.dumps(existing + summary, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
