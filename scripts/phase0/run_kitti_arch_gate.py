#!/usr/bin/env python3
"""Experiment 2: KITTI latent fusion with vis/IMU gates (Euler 6-D, window 11)."""
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
    if train.is_symlink() or not train.is_dir() or not val.is_dir():
        print("ERROR: need real train_10 and val_10 directories (not a symlink).")
        return 1
    n_train = len(list(train.glob("*_gt.npy")))
    n_val = len(list(val.glob("*_gt.npy")))
    print(f"latents train_10 gt={n_train} val_10 gt={n_val}")
    if n_train < 8000 or n_val < 1000:
        print("ERROR: unexpected cache sizes (expect ~13200 train, ~5030 val).")
        return 1
    if train.resolve() == val.resolve():
        print("ERROR: train_10 and val_10 resolve to the same folder.")
        return 1
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    rc = _require_short_latents()
    if rc != 0:
        return rc
    return run_train_py(
        [
            "experiment=latent_kitti_arch_gate",
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
            "experiment=latent_kitti_arch_gate",
            "train=false",
            "test=true",
            "trainer=gpu",
            f"ckpt_path={ck}",
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train")
    tr.add_argument("--max-epochs", type=int, default=200)
    tr.add_argument("--batch-size", type=int, default=128)
    ev = sub.add_parser("eval")
    ev.add_argument("--ckpt", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "train":
        return cmd_train(args)
    return cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
