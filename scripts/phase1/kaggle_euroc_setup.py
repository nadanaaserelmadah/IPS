#!/usr/bin/env python3
"""Link EuRoC from Kaggle input and run GT verify on listed sequences."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def find_ips() -> Path:
    for p in [Path("/kaggle/working/IPS/IPS"), Path("/kaggle/working/IPS"), Path.cwd()]:
        if (p / "third_party" / "VIFT" / "src" / "train.py").is_file():
            return p.resolve()
    raise FileNotFoundError("IPS root not found")


def find_euroc_source(slug: str | None) -> Path | None:
    inp = Path("/kaggle/input")
    if not inp.is_dir():
        return None

    candidates: list[Path] = []
    if slug:
        candidates.extend(
            [
                inp / slug,
                inp / "datasets" / slug,
            ]
        )
        for p in inp.glob(f"datasets/*/{slug}"):
            candidates.append(p)

    for p in inp.iterdir():
        if p.is_dir():
            candidates.append(p)

    def _has_seq(root: Path, seq: str = "MH_01_easy") -> bool:
        flat = root / seq / "cam0" / "data.csv"
        mav0 = root / seq / "mav0" / "cam0" / "data.csv"
        return flat.is_file() or mav0.is_file()

    for base in candidates:
        if not base.is_dir():
            continue
        if _has_seq(base):
            return base
        if _has_seq(base / "euroc"):
            return base / "euroc"
        for sub in base.iterdir():
            if sub.is_dir() and _has_seq(sub):
                return base
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--euroc-slug", default=None)
    parser.add_argument("--verify-seq", default="MH_01_easy,V2_02_medium")
    args = parser.parse_args()

    ips = find_ips()
    vift = ips / "third_party" / "VIFT"
    target = vift / "data" / "euroc"

    src = find_euroc_source(args.euroc_slug)
    if src is None:
        print("ERROR: EuRoC not found under /kaggle/input")
        print("Attach EuRoC dataset and pass --euroc-slug if needed")
        return 1

    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    os.symlink(src.resolve(), target, target_is_directory=True)
    print(f"Linked {src} -> {target}")

    verify = ips / "scripts" / "phase1" / "verify_gt_ate.py"
    rc = 0
    for seq in args.verify_seq.split(","):
        seq = seq.strip()
        if not seq:
            continue
        code = subprocess.call(
            [sys.executable, str(verify), "--root", str(target), "--seq", seq],
        )
        if code != 0:
            rc = code

    if rc == 0:
        print("\nNext:")
        print(f"  %cd {vift}/data && python latent_euroc_caching.py")
        print(f"  %cd {vift} && python src/train.py experiment=euroc_latent_minimal trainer=gpu")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
