#!/usr/bin/env python3
"""
Train exactly until epoch N, then stop (for Kaggle: 5 epochs → export → new session → +5).

Examples:
  # Session 1: epochs 0 → 5
  python scripts/phase2/train_one_chunk.py --until-epoch 5 --export

  # Session 2: after import checkpoint dataset
  python scripts/phase2/train_one_chunk.py --until-epoch 10 --export \
    --ckpt /kaggle/working/.../last.ckpt

  # Phase 3 joint fine-tune
  python scripts/phase2/train_one_chunk.py --until-epoch 5 --export \
    --experiment euroc_joint_finetune
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


def latest_ckpt(vift: Path) -> Path | None:
    ckpts = list((vift / "logs/train/runs").glob("*/checkpoints/last.ckpt"))
    return max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train until cumulative epoch N, then stop (one Kaggle session)"
    )
    parser.add_argument(
        "--until-epoch",
        type=int,
        required=True,
        help="Lightning trainer.max_epochs (cumulative: 5, then 10, then 15, ...)",
    )
    parser.add_argument("--ckpt", type=Path, default=None, help="Resume from this .ckpt")
    parser.add_argument(
        "--experiment",
        default="euroc_latent_train",
        choices=[
            "euroc_latent_train",
            "euroc_joint_finetune",
            "tum_vi_joint_finetune",
            "euroc_joint_phase5",
        ],
    )
    parser.add_argument(
        "--phase2-ckpt",
        type=Path,
        default=None,
        help="Init fusion from Phase 2 (joint fine-tune only, first session)",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Zip last.ckpt to /kaggle/working/ after training",
    )
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument(
        "--init-weights",
        action="store_true",
        help="Warm-start from ckpt (weights only, epoch 0); use when init from EuRoC/other run",
    )
    args = parser.parse_args()

    vift = vift_root()
    ips = ips_root()
    env = {**os.environ, "PROJECT_ROOT": str(vift)}

    cmd = [
        sys.executable,
        "src/train.py",
        f"experiment={args.experiment}",
        "trainer=gpu",
        "train=true",
        "test=false",
        f"trainer.max_epochs={args.until_epoch}",
        f"model.scheduler.T_0={args.until_epoch}",
    ]
    if args.lr is not None:
        cmd.append(f"model.optimizer.lr={args.lr}")

    ckpt = args.ckpt
    if ckpt is None and args.until_epoch > 5:
        ckpt = latest_ckpt(vift)
    if ckpt is not None and ckpt.is_file():
        cmd.append(f"ckpt_path='{ckpt.resolve().as_posix()}'")
        print(f"Resume from: {ckpt}")

    init_weights = args.init_weights
    if ckpt is not None and ckpt.is_file() and not init_weights:
        # Cross-run init (e.g. EuRoC Phase 3 → TUM fine-tune): avoid epoch-49 vs max_epochs=5 clash.
        ckpt_posix = ckpt.resolve().as_posix()
        if "logs/train/runs" not in ckpt_posix.replace("\\", "/"):
            init_weights = True
            print("External checkpoint — warm-start (init_ckpt_weights=true)")
    if init_weights:
        cmd.append("init_ckpt_weights=true")

    if args.phase2_ckpt is not None:
        p2 = args.phase2_ckpt.resolve().as_posix()
        cmd.append(f"model.phase2_ckpt='{p2}'")

    print(f"Train until epoch {args.until_epoch} ({args.experiment})")
    rc = subprocess.call(cmd, cwd=vift, env=env)
    if rc != 0:
        return rc

    if args.export:
        export = [
            sys.executable,
            str(ips / "scripts/phase2/export_checkpoints.py"),
            "--latest-only",
            "--download-link",
        ]
        print("\nExporting checkpoint for download / Kaggle Dataset...")
        subprocess.call(export, cwd=ips, env=env)

    ckpt = latest_ckpt(vift)
    if ckpt:
        print(f"\nDone. Next session: import zip, then --until-epoch {args.until_epoch + 5} --export")
        print(f"Checkpoint: {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
