#!/usr/bin/env python3
"""Link TUM-VI from Kaggle input and verify GT on listed sequences."""
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


def find_tum_vi_source(slug: str | None) -> Path | None:
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
        # e.g. datasets/nadaelmadah/tum-vi when slug is tum-vi
        for p in inp.glob("datasets/*/*"):
            if p.is_dir() and p.name == slug:
                candidates.append(p)

    for p in inp.iterdir():
        if p.is_dir():
            candidates.append(p)

    vift_guess = find_ips() / "third_party" / "VIFT"
    sys.path.insert(0, str(vift_guess))
    try:
        from src.utils.tum_vi_utils import list_sequences, resolve_sequence_root
    except ImportError:
        list_sequences = None
        resolve_sequence_root = None

    def _has_tum_seq(root: Path) -> bool:
        if not root.is_dir():
            return False
        if list_sequences is not None and list_sequences(root):
            return True
        if resolve_sequence_root is not None:
            try:
                resolve_sequence_root(root)
                return True
            except FileNotFoundError:
                pass
        # fallback without imports
        for base in [root, root / "mav0"]:
            if (base / "cam0" / "data.csv").is_file():
                return True
        for sub in root.iterdir() if root.is_dir() else []:
            if sub.is_dir():
                for base in [sub, sub / sub.name, sub / "mav0", sub / sub.name / "mav0"]:
                    if base.is_dir() and (base / "cam0" / "data.csv").is_file():
                        return True
        return False

    for base in candidates:
        if not base.is_dir():
            continue
        if _has_tum_seq(base):
            return base
        if _has_tum_seq(base / "tum_vi"):
            return base / "tum_vi"
        for sub in base.iterdir():
            if sub.is_dir() and _has_tum_seq(sub):
                return base
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tum-slug", default="tum-vi", help="Kaggle input folder name")
    parser.add_argument(
        "--src",
        type=Path,
        default=None,
        help="Explicit source dir (e.g. /kaggle/input/datasets/nadaelmadah/tum-vi)",
    )
    parser.add_argument(
        "--verify-seq",
        default="dataset-room1_512_16",
        help="Comma-separated sequence names to GT-verify",
    )
    args = parser.parse_args()

    ips = find_ips()
    vift = ips / "third_party" / "VIFT"
    target = vift / "data" / "tum_vi"

    src = args.src
    if src is not None:
        src = src.resolve()
        if not src.is_dir():
            print(f"ERROR: --src not a directory: {src}")
            return 1
    else:
        src = find_tum_vi_source(args.tum_slug)
    if src is None:
        print("ERROR: TUM-VI not found under /kaggle/input")
        print("Attach dataset or pass:")
        print("  --src /kaggle/input/datasets/nadaelmadah/tum-vi")
        return 1

    if target.is_symlink():
        target.unlink()
    elif target.exists():
        shutil.rmtree(target)
    os.symlink(src, target, target_is_directory=True)
    print(f"Linked {src} -> {target}")

    verify = ips / "scripts" / "phase4" / "verify_tum_vi_gt.py"
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
        print("\nNext (zero-shot eval with EuRoC Phase 3 ckpt):")
        print(
            f"  !python {ips}/scripts/phase4/eval_all_tum_vi.py "
            f"--ckpt <P3_CKPT> --sequences dataset-room1_512_16"
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
