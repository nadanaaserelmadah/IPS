#!/usr/bin/env python3
"""TUM-VI GT sanity check using mocap world positions (not Euler integration)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

VIFT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VIFT"

if str(VIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(VIFT_ROOT))

from src.utils.tum_vi_utils import gt_path_stats, raw_mocap_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=VIFT_ROOT / "data" / "tum_vi",
    )
    parser.add_argument("--seq", type=str, default="dataset-room1_512_16")
    parser.add_argument("--cam-stride", type=int, default=2)
    args = parser.parse_args()

    seq_name = Path(args.seq.strip()).name
    seq_dir = args.root / seq_name
    if not seq_dir.is_dir():
        print(f"ERROR: {seq_dir} not found")
        return 1

    raw = raw_mocap_stats(seq_dir)
    stats = gt_path_stats(seq_dir, cam_stride=args.cam_stride)

    print("TUM-VI GT sanity check (mocap world positions)")
    print(f"  sequence: {seq_name}")
    print(f"  position_span_m: {raw.get('position_span_m', 'n/a')}")
    print(f"  median_mocap_speed_m_s: {raw.get('median_mocap_speed_m_s', 'n/a')}")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    duration = stats.get("duration_s", 0.0)
    path_limit = max(200.0, duration * 3.0)
    ok = (
        stats["gt_path_len_m"] <= path_limit
        and stats["mean_step_m"] <= 0.35
        and stats["max_step_m"] <= 2.0
    )
    if ok:
        print("PASS: GT trajectory looks plausible.")
        return 0
    print("WARN: GT stats out of expected range — check data or sync.")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
