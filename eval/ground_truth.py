#!/usr/bin/env python3
"""Per-trial ground truth, read off the frames by eye.

Every entry lists the serials actually stuck on the wall for that row in that
recording.  Rows are not assumed to be complete: in the A3 take the A03 row is
genuinely missing 0012 and 0013, and the labels were rearranged before the
later takes.  Scoring against an assumed 0010..0020 for every row is what made
the previous report show 0% and 50% for trials that were in fact correct.
"""

FULL = tuple(f"{value:04d}" for value in range(10, 21))

GROUND_TRUTH: dict[str, dict[str, dict[str, tuple[str, ...]]]] = {
    "A2": {
        "head": {"A05": FULL, "A04": FULL},
        "base": {"A02": FULL, "A01": FULL},
    },
    "A3": {
        "head": {
            "A03": ("0010", "0011", "0014", "0015", "0016",
                    "0017", "0018", "0019", "0020"),
            "A02": FULL,
        },
        "base": {"A01": FULL},
    },
    "A3-4": {
        "head": {"A04": FULL, "A03": FULL},
        "base": {"A01": FULL},
    },
    "A4": {
        "head": {"A04": FULL},
        "base": {"A01": FULL},
    },
    "A5_A4-5": {
        "head": {"A05": FULL, "A04": FULL},
        "base": {"A01": FULL},
    },
}


def expected_labels(pose: str, camera: str) -> set[str]:
    rows = GROUND_TRUTH[pose][camera]
    return {
        f"{prefix}-{serial}"
        for prefix, serials in rows.items()
        for serial in serials
    }
