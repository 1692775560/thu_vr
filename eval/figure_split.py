"""Figure for the report: how the merged-sticker split changed between v4 and v5.

Draws the same shelf row three times -- the raw connected components, the v4
split, and the v5 split -- so the under-cut is visible rather than argued.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

WORK = Path(__file__).resolve().parent
sys.path.insert(0, str(WORK.parent / "thu_vr"))

import label_reader as lr  # noqa: E402

BLUE = (255, 160, 60)
RED = (60, 60, 235)
GREEN = (80, 210, 80)


def raw_blobs(image: np.ndarray) -> list[dict]:
    """Connected components before any splitting, as locate_stickers finds them."""
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
    min_width, max_width = max(11, int(width * 0.005)), int(width * 0.16)
    min_height, max_height = max(7, int(height * 0.005)), int(height * 0.07)
    blobs = []
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
            "quad": lr.order_quad(cv2.boxPoints(cv2.minAreaRect(contour))),
        })
    return blobs


def split_v4(blobs: list[dict]) -> list[dict]:
    """The v4 rule: part count from the blob's own height, which merging inflates."""
    result = []
    for blob in blobs:
        x, y, box_width, box_height = blob["box"]
        center_y = y + box_height / 2.0
        peers = [
            (other["box"][2], other["box"][3])
            for other in blobs
            if abs(center_y - (other["box"][1] + other["box"][3] / 2.0))
            <= max(box_height, other["box"][3]) * 1.2
            and other["box"][2] / max(1, other["box"][3])
            <= lr.STICKER_ASPECT_RATIO * 1.45
        ]
        reference_width = float(np.median([p[0] for p in peers])) if peers else 0.0
        aspect_parts = box_width / max(box_height * lr.STICKER_ASPECT_RATIO, 1.0)
        width_parts = box_width / reference_width if reference_width else 1.0
        parts = int(np.clip(np.floor(max(aspect_parts, width_parts) + 0.5), 1, 8))
        step = box_width / parts
        for index in range(parts):
            x1 = int(round(x + index * step))
            x2 = int(round(x + (index + 1) * step))
            if (x2 - x1) / max(1, box_height) < 0.60:
                continue
            result.append({"box": (x1, y, x2 - x1, box_height)})
    return result


def band(image: np.ndarray, boxes: list[dict], y0: int, y1: int,
         colour: tuple[int, int, int], title: str, scale: float) -> np.ndarray:
    strip = image[y0:y1].copy()
    for entry in boxes:
        x, y, box_width, box_height = entry["box"]
        if not (y0 <= y + box_height / 2 <= y1):
            continue
        cv2.rectangle(
            strip, (x, y - y0), (x + box_width, y - y0 + box_height), colour, 1
        )
    strip = cv2.resize(strip, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_LANCZOS4)
    header = np.zeros((34, strip.shape[1], 3), np.uint8)
    cv2.putText(header, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                colour, 2, cv2.LINE_AA)
    return np.vstack([header, strip])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--y0", type=int, default=620)
    parser.add_argument("--y1", type=int, default=740)
    parser.add_argument("--x0", type=int, default=730)
    parser.add_argument("--x1", type=int, default=1290)
    parser.add_argument("--scale", type=float, default=2.6)
    args = parser.parse_args()

    image = cv2.imread(str(args.frame))
    raw = raw_blobs(image)
    v4 = split_v4(raw)
    v5 = lr.locate_stickers(image)

    def inside(entries):
        return [
            e for e in entries
            if args.y0 <= e["box"][1] + e["box"][3] / 2 <= args.y1
            and args.x0 <= e["box"][0] <= args.x1
        ]

    crop = image[:, args.x0:args.x1]
    shift = lambda entries: [
        {"box": (e["box"][0] - args.x0, *e["box"][1:])} for e in inside(entries)
    ]
    panels = [
        band(crop, shift(raw), args.y0, args.y1, BLUE,
             f"raw connected components: {len(inside(raw))}", args.scale),
        band(crop, shift(v4), args.y0, args.y1, RED,
             f"v4 split: {len(inside(v4))} boxes for 11 stickers", args.scale),
        band(crop, shift(v5), args.y0, args.y1, GREEN,
             f"v5 split: {len(inside(v5))} boxes for 11 stickers", args.scale),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), np.vstack(panels))
    print(f"原始连通块 {len(inside(raw))} 个 -> v4 拆成 {len(inside(v4))} 个 "
          f"-> v5 拆成 {len(inside(v5))} 个   已写入 {args.out}")


if __name__ == "__main__":
    main()
