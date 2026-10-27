#!/usr/bin/env python3
"""
Restore checkpoints from a zip exported by export_checkpoints.py.

Usage:
  python scripts/phase2/import_checkpoints.py --zip /kaggle/input/my-ckpts/2026-05-25_03-14-02.zip
  python scripts/phase2/import_checkpoints.py --zip ./exports/checkpoints/2026-05-25_03-14-02.zip
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vift_root() -> Path:
    return ips_root() / "third_party" / "VIFT"


def main() -> int:
    parser = argparse.ArgumentParser(description="Import checkpoint zip into VIFT logs/")
    parser.add_argument("--zip", type=Path, required=True, help="Zip from export_checkpoints.py")
    args = parser.parse_args()

    zpath = args.zip.resolve()
    if not zpath.is_file():
        print(f"ERROR: Not found: {zpath}")
        return 1

    vift = vift_root()
    runs_dir = vift / "logs" / "train" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    with zipfile.ZipFile(zpath, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".ckpt"):
                continue
            # Expect: <run>/checkpoints/foo.ckpt
            parts = Path(name).parts
            if len(parts) < 3 or parts[1] != "checkpoints":
                # flat fallback
                dest = runs_dir / "imported" / "checkpoints" / Path(name).name
            else:
                run_name = parts[0]
                dest = runs_dir / run_name / "checkpoints" / Path(name).name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(zf.read(name))
            print("Restored:", dest)
            restored += 1

    if restored == 0:
        print("ERROR: No .ckpt files in zip")
        return 1

    print(f"\nDone ({restored} files). Resume with:")
    print("  python scripts/phase2/resume_euroc_train.py --list")
    print("  python scripts/phase2/resume_euroc_train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
