#!/usr/bin/env python3
"""Pull the report's figures out of the rendered videos."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import imageio.v2 as imageio

# The data root holds eval/<trial>/<camera>_move_*.jpg and receives results/.
# It defaults to this script's own directory, which is the layout used while
# developing; point LABEL_DATA at that directory to run from anywhere else.
HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get("LABEL_DATA", HERE))

# figure name -> (video, frame index; -1 for the final settled frame)
FIGURES = {
    "best_A5_light_head": ("A5_A4-5__light_head.mp4", -1),
    "fail_A2_light_head": ("A2__light_head.mp4", -1),
    "hard_A2_dark_head_frame1": ("A2__dark_head.mp4", 0),
    "hard_A2_dark_head_final": ("A2__dark_head.mp4", -1),
    "near_A3-4_light_base": ("A3-4__light_base.mp4", -1),
    "misplaced_A5_light_head": ("A5_A4-5__light_head_misplaced.mp4", -1),
    "zone_A5_light_head": ("A5_A4-5__light_head_zone.mp4", -1),
}


def main() -> int:
    videos = Path(sys.argv[1] if len(sys.argv) > 1 else WORK / "videos_v4")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else WORK / "figures_v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = []
    for name, (video, index) in FIGURES.items():
        path = videos / video
        if not path.exists():
            missing.append(video)
            continue
        # Read one frame at a time; holding a whole clip of 1920-wide frames in
        # a list exhausts memory.
        reader = imageio.get_reader(str(path))
        frame = None
        for position, current in enumerate(reader):
            if index >= 0 and position == index:
                frame = current
                break
            frame = current
        reader.close()
        if frame is None:
            missing.append(video)
            continue
        target = out_dir / f"{name}.png"
        cv2.imwrite(str(target), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        print(f"{target.name}  {frame.shape[1]}x{frame.shape[0]}")
    for video in missing:
        print(f"缺少 {video}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
