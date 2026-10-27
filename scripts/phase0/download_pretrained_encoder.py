#!/usr/bin/env python3
"""Ensure Visual-Selective-VIO encoder weights exist under VIFT/pretrained_models/."""
from __future__ import annotations

import argparse
import os
import shutil
import urllib.request
from pathlib import Path

FILENAME = "vf_512_if_256_3e-05.model"
DRIVE_FOLDER_ID = "1KrxpvUV9Bn5SwUlrDKe76T2dqF1ooZyk"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"


def find_in_kaggle_input() -> Path | None:
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return None
    for path in inp.rglob(FILENAME):
        if path.is_file() and path.stat().st_size > 1_000_000:
            return path
    return None


def download_via_gdown(dest: Path) -> Path:
    try:
        import gdown
    except ImportError as e:
        raise SystemExit("Install gdown: pip install gdown") from e

    out_dir = dest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading folder from Google Drive to {out_dir} ...")
    gdown.download_folder(
        url=DRIVE_FOLDER_URL,
        output=str(out_dir),
        quiet=False,
        remaining_ok=True,
    )
    if dest.is_file():
        return dest
    matches = list(out_dir.rglob(FILENAME))
    if len(matches) == 1:
        if matches[0] != dest:
            shutil.copy2(matches[0], dest)
        return dest
    raise FileNotFoundError(f"{FILENAME} not found after gdown (got {matches})")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vift-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "third_party" / "VIFT",
    )
    parser.add_argument("--gdown", action="store_true", help="Download from Google Drive (needs pip install gdown)")
    args = parser.parse_args()

    dest = args.vift_root / "pretrained_models" / FILENAME
    if dest.is_file() and dest.stat().st_size > 1_000_000:
        print(f"OK: {dest} ({dest.stat().st_size // 1_000_000} MB)")
        return 0

    found = find_in_kaggle_input()
    if found:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(found, dest)
        print(f"Copied {found} -> {dest}")
        return 0

    if args.gdown:
        download_via_gdown(dest)
        print(f"OK: {dest}")
        return 0

    print(f"MISSING: {dest}")
    print("Options:")
    print("  1) Attach a Kaggle dataset containing vf_512_if_256_3e-05.model and re-run")
    print("  2) Run with --gdown (pip install gdown first)")
    print(f"  3) Manual: https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
