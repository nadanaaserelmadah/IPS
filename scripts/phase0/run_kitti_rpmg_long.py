#!/usr/bin/env python3
"""
KITTI frozen fusion with RPMG + long windows (path to close VIFT 1.96% t_rel).

Does NOT jointly train the encoder. Recache latents at window 65, then train fusion.

Usage:
  python scripts/phase0/run_kitti_rpmg_long.py cache
  python scripts/phase0/run_kitti_rpmg_long.py train --max-epochs 200
  python scripts/phase0/run_kitti_rpmg_long.py eval --ckpt /path/to/last.ckpt
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


def run_train_py(extra: list[str]) -> int:
    vift = vift_root()
    cmd = [sys.executable, "src/train.py", *extra]
    print("$", " ".join(cmd))
    env = {**os.environ, "PROJECT_ROOT": str(vift)}
    return subprocess.call(cmd, cwd=vift, env=env)


def cmd_cache() -> int:
    script = ips_root() / "scripts" / "phase0" / "cache_kitti_latents.py"
    rc = subprocess.call([sys.executable, str(script), "--split", "train", "--sequence-length", "65"])
    if rc != 0:
        return rc
    return subprocess.call([sys.executable, str(script), "--split", "val", "--sequence-length", "65"])


def _require_long_latents() -> int:
    root = vift_root() / "data" / "kitti_latent_data"
    missing = [p for p in (root / "train_64", root / "val_64") if not p.is_dir()]
    if not missing:
        return 0
    print("ERROR: long-window latents not found:")
    for p in missing:
        print(f"  missing {p}")
    print("Run cache first (needs kitti_data + pretrained encoder + GPU):")
    print("  python scripts/phase0/run_kitti_rpmg_long.py cache")
    print("This writes data/kitti_latent_data/train_64 and val_64.")
    print("Do not start train until both folders exist.")
    return 1


def cmd_train(args: argparse.Namespace) -> int:
    rc = _require_long_latents()
    if rc != 0:
        return rc
    return run_train_py(
        [
            "experiment=latent_kitti_rpmg_long",
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
            "experiment=latent_kitti_rpmg_long",
            "train=false",
            "test=true",
            "trainer=gpu",
            f"ckpt_path='{ck}'",
        ]
    )


def main() -> int:
    p = argparse.ArgumentParser(description="KITTI RPMG + long-window frozen fusion")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("cache", help="Build kitti_latent_data/train_64 and val_64")
    t = sub.add_parser("train")
    t.add_argument("--max-epochs", type=int, default=200)
    t.add_argument("--batch-size", type=int, default=128)
    e = sub.add_parser("eval")
    e.add_argument("--ckpt", type=Path, required=True)
    args = p.parse_args()
    if args.command == "cache":
        return cmd_cache()
    if args.command == "train":
        return cmd_train(args)
    if not args.ckpt.is_file():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        return 1
    return cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
