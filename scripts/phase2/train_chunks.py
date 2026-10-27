#!/usr/bin/env python3
"""
Train in short chunks with auto-resume (helps when Kaggle sessions die mid-run).

Each chunk runs up to --chunk-epochs, then saves; next chunk resumes from last.ckpt.

Usage:
  python scripts/phase2/train_chunks.py --total-epochs 100 --chunk-epochs 10
  python scripts/phase2/train_chunks.py --total-epochs 100 --chunk-epochs 10 --export-each
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


def run(cmd: list[str], cwd: Path, env: dict) -> int:
    print("$", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=cwd, env=env)


def latest_ckpt(vift: Path) -> Path | None:
    ckpts = list((vift / "logs/train/runs").glob("*/checkpoints/last.ckpt"))
    return max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunked euroc_latent_train with resume")
    parser.add_argument("--total-epochs", type=int, default=100)
    parser.add_argument("--chunk-epochs", type=int, default=10, help="Epochs per Kaggle cell/session")
    parser.add_argument("--ckpt", type=Path, default=None, help="Start from this checkpoint")
    parser.add_argument("--export-each", action="store_true", help="Zip checkpoints after each chunk")
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    vift = vift_root()
    ips = ips_root()
    env = {**os.environ, "PROJECT_ROOT": str(vift)}
    ckpt = args.ckpt

    chunks = (args.total_epochs + args.chunk_epochs - 1) // args.chunk_epochs
    target = args.total_epochs

    for i in range(chunks):
        # Aim for cumulative max_epochs = min((i+1)*chunk, total)
        max_ep = min((i + 1) * args.chunk_epochs, target)
        cmd = [
            sys.executable,
            "src/train.py",
            "experiment=euroc_latent_train",
            "trainer=gpu",
            "train=true",
            "test=false",
            f"trainer.max_epochs={max_ep}",
            f"model.scheduler.T_0={target}",
        ]
        if args.lr is not None:
            cmd.append(f"model.optimizer.lr={args.lr}")
        if ckpt is not None:
            cmd.append(f"ckpt_path='{ckpt.resolve().as_posix()}'")

        print(f"\n=== Chunk {i + 1}/{chunks}: max_epochs={max_ep} ===", flush=True)
        rc = run(cmd, vift, env)
        if rc != 0:
            print(f"Chunk failed (exit {rc}). Resume with:", flush=True)
            print(f"  python scripts/phase2/train_chunks.py --total-epochs {target} "
                  f"--chunk-epochs {args.chunk_epochs} --ckpt <last.ckpt>", flush=True)
            return rc

        ckpt = latest_ckpt(vift)
        if ckpt is None:
            print("ERROR: no last.ckpt after chunk")
            return 1
        print(f"Saved checkpoint: {ckpt}", flush=True)

        if args.export_each:
            run(
                [sys.executable, str(ips / "scripts/phase2/export_checkpoints.py"), "--latest-only"],
                ips,
                env,
            )

    print(f"\nDone: trained to max_epochs={target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
