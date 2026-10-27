#!/usr/bin/env python3
"""Download preprocessed KITTI IMU .mat files from Visual-Selective-VIO (required for VIFT eval)."""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/mingyuyng/Visual-Selective-VIO/main/data/imus"
# Sequences used by VIFT (skip 03 — no IMU in odometry benchmark)
DEFAULT_SEQS = ["00", "01", "02", "04", "05", "06", "07", "08", "09", "10"]
MINIMAL_SEQS = ["05", "07", "10"]


def download_seq(imus_dir: Path, seq: str) -> None:
    dest = imus_dir / f"{seq}.mat"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip {seq}.mat (exists)")
        return
    url = f"{BASE}/{seq}.mat"
    print(f"  {url} -> {dest}")
    imus_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vift-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "third_party" / "VIFT",
    )
    parser.add_argument(
        "--minimal",
        action="store_true",
        help=f"Only test sequences {MINIMAL_SEQS}",
    )
    parser.add_argument("--seq", action="append", default=None, help="Explicit sequence id(s), e.g. 05")
    args = parser.parse_args()

    if args.seq:
        seqs = args.seq
    elif args.minimal:
        seqs = MINIMAL_SEQS
    else:
        seqs = DEFAULT_SEQS

    imus_dir = args.vift_root / "data" / "kitti_data" / "imus"
    print(f"IMU target: {imus_dir}")
    for seq in seqs:
        download_seq(imus_dir, seq)
    print("Done.")


if __name__ == "__main__":
    main()
