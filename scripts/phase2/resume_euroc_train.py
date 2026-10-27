#!/usr/bin/env python3
"""
Resume Phase 2 EuRoC training from the newest checkpoint (or a given path).

Usage:
  python scripts/phase2/resume_euroc_train.py
  python scripts/phase2/resume_euroc_train.py --ckpt path/to/last.ckpt
  python scripts/phase2/resume_euroc_train.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vift_root() -> Path:
    return ips_root() / "third_party" / "VIFT"


def list_checkpoints(vift: Path) -> list[Path]:
    runs_dir = vift / "logs" / "train" / "runs"
    if not runs_dir.is_dir():
        return []
    ckpts = list(runs_dir.glob("*/checkpoints/*.ckpt"))
    return sorted(ckpts, key=lambda p: p.stat().st_mtime, reverse=True)


def find_latest_checkpoint(vift: Path) -> Path | None:
    ckpts = list_checkpoints(vift)
    return ckpts[0] if ckpts else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Resume euroc_latent_train from checkpoint")
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=None,
        help="Checkpoint file (default: newest under logs/train/runs)",
    )
    parser.add_argument("--list", action="store_true", help="List available checkpoints and exit")
    parser.add_argument("--epochs", type=int, default=None, help="Override trainer.max_epochs")
    parser.add_argument("--test", action="store_true", help="Also run test after fit (not recommended on Kaggle)")
    args = parser.parse_args()

    vift = vift_root()
    if not vift.is_dir():
        print(f"ERROR: VIFT root not found: {vift}")
        return 1

    ckpts = list_checkpoints(vift)
    if args.list:
        if not ckpts:
            print("No checkpoints found under", vift / "logs/train/runs")
            return 1
        for p in ckpts[:20]:
            print(p)
        return 0

    ckpt = args.ckpt
    if ckpt is None:
        ckpt = find_latest_checkpoint(vift)
    if ckpt is None or not ckpt.is_file():
        print("ERROR: No checkpoint found. Train first:")
        print(f"  cd {vift}")
        print("  python src/train.py experiment=euroc_latent_train trainer=gpu train=true test=false")
        return 1

    ck = ckpt.resolve().as_posix()
    print(f"Resuming from: {ck}")

    cmd = [
        sys.executable,
        "src/train.py",
        "experiment=euroc_latent_train",
        "trainer=gpu",
        f"ckpt_path='{ck}'",
        "train=true",
        f"test={'true' if args.test else 'false'}",
    ]
    if args.epochs is not None:
        cmd.append(f"trainer.max_epochs={args.epochs}")

    env = {**os.environ, "PROJECT_ROOT": str(vift)}
    return subprocess.call(cmd, cwd=vift, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
