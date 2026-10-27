#!/usr/bin/env python3
"""Experiment 1: KITTI latent fusion with 6D rotation (window 11, no RPMG)."""
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


def run_train_py(extra: list[str]) -> int:
    vift = vift_root()
    cmd = [sys.executable, "src/train.py", *extra]
    print("$", " ".join(cmd))
    env = {**os.environ, "PROJECT_ROOT": str(vift)}
    return subprocess.call(cmd, cwd=vift, env=env)


def _require_short_latents() -> int:
    root = vift_root() / "data" / "kitti_latent_data"
    train = root / "train_10"
    val = root / "val_10"
    if not val.is_dir():
        print(f"ERROR: missing {val}")
        return 1
    if train.is_symlink():
        print(f"ERROR: {train} is a symlink -> {train.resolve()}")
        print("Do not train on the experiment-0 dummy. On Kaggle:")
        print("  1) unlink train_10")
        print("  2) if val_10 has ~13000 windows, it is TRAIN data: mv val_10 train_10")
        print("  3) recache val: python scripts/phase0/cache_kitti_latents.py --split val --sequence-length 11")
        return 1
    if not train.is_dir():
        print(f"ERROR: missing {train}")
        print("  python scripts/phase0/cache_kitti_latents.py --split train --sequence-length 11")
        return 1
    n_train = len(list(train.glob("*_gt.npy")))
    n_val = len(list(val.glob("*_gt.npy")))
    print(f"latents train_10 gt={n_train} val_10 gt={n_val}")
    if n_train < 8000:
        print("ERROR: train_10 is empty or not the paper train cache (~13200 windows).")
        print("Do not start train. Recache train into a real folder (not a symlink):")
        print("  python scripts/phase0/cache_kitti_latents.py --split train --sequence-length 11")
        print("Needs KITTI sequences 00,01,02,04,06,09 + IMU. Takes ~1h on GPU.")
        return 1
    return 0


def cmd_cache() -> int:
    script = ips_root() / "scripts" / "phase0" / "cache_kitti_latents.py"
    rc = subprocess.call(
        [sys.executable, str(script), "--split", "train", "--sequence-length", "11"]
    )
    if rc != 0:
        return rc
    return subprocess.call(
        [sys.executable, str(script), "--split", "val", "--sequence-length", "11"]
    )


def cmd_train(args: argparse.Namespace) -> int:
    rc = _require_short_latents()
    if rc != 0:
        return rc
    return run_train_py(
        [
            "experiment=latent_kitti_arch_rot6d",
            "train=true",
            "test=false",
            "trainer=gpu",
            f"trainer.max_epochs={args.max_epochs}",
            f"data.batch_size={args.batch_size}",
        ]
    )


def cmd_eval(args: argparse.Namespace) -> int:
    ck = args.ckpt.resolve().as_posix()
    return run_train_py(
        [
            "experiment=latent_kitti_arch_rot6d",
            "train=false",
            "test=true",
            "trainer=gpu",
            f"ckpt_path={ck}",
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cache")
    tr = sub.add_parser("train")
    tr.add_argument("--max-epochs", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=128)
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "cache":
        return cmd_cache()
    if args.cmd == "train":
        return cmd_train(args)
    return cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
