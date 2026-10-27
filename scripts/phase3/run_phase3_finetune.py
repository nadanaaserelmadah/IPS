#!/usr/bin/env python3
"""
Phase 3B: joint encoder + fusion fine-tune on raw EuRoC.

Usage:
  python scripts/phase3/run_phase3_finetune.py train
  python scripts/phase3/run_phase3_finetune.py train --phase2-ckpt path/to/last.ckpt
  python scripts/phase3/run_phase3_finetune.py test --ckpt path/to/last.ckpt
  python scripts/phase3/run_phase3_finetune.py resume
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
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["train", "test", "resume"])
    parser.add_argument("--phase2-ckpt", type=Path, default=None, help="Phase 2 fusion weights")
    parser.add_argument("--ckpt", type=Path, default=None, help="Phase 3 Lightning ckpt")
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    vift = vift_root()
    env = {**os.environ, "PROJECT_ROOT": str(vift)}
    cmd = [
        sys.executable,
        "src/train.py",
        "experiment=euroc_joint_finetune",
        "trainer=gpu",
        f"trainer.max_epochs={args.epochs}",
    ]

    if args.step == "train":
        cmd += ["train=true", "test=false"]
        if args.phase2_ckpt:
            p2 = args.phase2_ckpt.resolve().as_posix()
            cmd.append(f"model.phase2_ckpt='{p2}'")
    elif args.step == "resume":
        ckpt = args.ckpt or latest_ckpt(vift)
        if ckpt is None:
            print("ERROR: no checkpoint to resume")
            return 1
        ck = ckpt.resolve().as_posix()
        cmd += ["train=true", "test=false", f"ckpt_path='{ck}'"]
    else:
        ckpt = args.ckpt or latest_ckpt(vift)
        if ckpt is None:
            print("ERROR: no checkpoint for test")
            return 1
        ck = ckpt.resolve().as_posix()
        cmd += ["train=false", "test=true", f"ckpt_path='{ck}'"]

    print("$", " ".join(str(c) for c in cmd))
    return subprocess.call(cmd, cwd=vift, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
