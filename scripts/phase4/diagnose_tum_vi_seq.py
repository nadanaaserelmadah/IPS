#!/usr/bin/env python3
"""Print TUM-VI sync / GT path diagnostics for one sequence."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIFT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VIFT"
if str(VIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIFT_ROOT))

from src.utils.tum_vi_utils import (  # noqa: E402
    gt_path_stats,
    raw_mocap_stats,
    resolve_gt_csv,
    resolve_sequence_root,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=VIFT_ROOT / "data" / "tum_vi",
    )
    parser.add_argument("--seq", default="dataset-room1_512_16")
    parser.add_argument("--cam-stride", type=int, default=2)
    args = parser.parse_args()

    seq_dir = args.root / args.seq
    if not seq_dir.is_dir():
        print(f"ERROR: {seq_dir} not found")
        return 1

    root = resolve_sequence_root(seq_dir)
    print(f"TUM-VI diagnostics: {args.seq}")
    print(f"  sensor_root: {root}")
    print(f"  gt_csv: {resolve_gt_csv(seq_dir)}")

    print("\n  Raw mocap (before smooth/sanitize):")
    raw = raw_mocap_stats(seq_dir)
    for k, v in raw.items():
        print(f"    {k}: {v}")

    stats = gt_path_stats(seq_dir, cam_stride=args.cam_stride)
    print("\n  After smooth + sanitize (used for eval):")
    for k, v in stats.items():
        print(f"    {k}: {v}")

    ok = True
    duration = stats.get("duration_s", 0.0)
    # ~1 m/s walking → path ≈ duration; allow up to 3 m/s for fast motion
    path_limit = max(200.0, duration * 3.0)
    if stats["gt_path_len_m"] > path_limit and "room" in args.seq:
        print(
            f"WARN: GT path {stats['gt_path_len_m']:.1f} m > {path_limit:.0f} m "
            f"for duration {duration:.0f} s"
        )
        ok = False
    if stats["mean_step_m"] > 0.35:
        print("WARN: mean GT step > 0.35 m at camera rate — check sync / data.")
        ok = False
    if stats["max_step_m"] > 2.0:
        print("WARN: max GT step > 2 m between camera frames.")
        ok = False

    if ok:
        print("OK: GT stats look plausible for TUM-VI.")
    else:
        print(
            "\nFix options:\n"
            "  1) Re-run after syncing repo (mocap outlier filter is on by default)\n"
            "  2) Official calibrated TUM-VI 512_16 from TUM\n"
            "  3) Tune TUM_VI_MOCAP_MAX_JUMP_M / TUM_VI_MOCAP_MAX_SPEED_M_S"
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
