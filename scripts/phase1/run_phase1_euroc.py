#!/usr/bin/env python3
"""
EuRoC Phase 1 orchestrator: verify GT -> cache latents -> train fusion model.

Usage:
  python scripts/phase1/run_phase1_euroc.py verify
  python scripts/phase1/run_phase1_euroc.py cache
  python scripts/phase1/run_phase1_euroc.py train
  python scripts/phase1/run_phase1_euroc.py all
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


def run(cmd: list[str], cwd: Path | None = None, env: dict | None = None) -> int:
    e = {**os.environ, **(env or {})}
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd or ips_root(), env=e)


def step_verify(sequences: list[str], euroc_root: Path) -> int:
    py = sys.executable
    script = ips_root() / "scripts" / "phase1" / "verify_gt_ate.py"
    rc = 0
    for seq in sequences:
        code = run(
            [py, str(script), "--root", str(euroc_root), "--seq", seq],
        )
        if code != 0:
            rc = code
    return rc


def step_cache(vift: Path, minimal: bool) -> int:
    env = {"PROJECT_ROOT": str(vift)}
    if minimal:
        env["EUROC_TRAIN_SEQS"] = "MH_01_easy"
        env["EUROC_VAL_SEQS"] = "V2_02_medium"
    data = vift / "data"
    if run([sys.executable, "latent_euroc_caching.py"], cwd=data, env=env) != 0:
        return 1
    return run([sys.executable, "latent_euroc_val_caching.py"], cwd=data, env=env)


def step_train(vift: Path, epochs: int, ckpt: str | None) -> int:
    env = {"PROJECT_ROOT": str(vift)}
    cmd = [
        sys.executable,
        "src/train.py",
        "experiment=euroc_latent_minimal",
        "trainer=gpu",
        f"trainer.max_epochs={epochs}",
    ]
    if ckpt:
        cmd += ["train=false", "test=true", f"ckpt_path={ckpt}"]
    return run(cmd, cwd=vift, env=env)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "step",
        choices=["verify", "cache", "train", "all"],
        help="Pipeline step",
    )
    parser.add_argument(
        "--euroc-root",
        type=Path,
        default=None,
        help="EuRoC root (default: VIFT/data/euroc)",
    )
    parser.add_argument("--seq", default="MH_01_easy,V2_02_medium")
    parser.add_argument("--minimal", action="store_true", help="Cache/train on fewer sequences")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--ckpt", default=None, help="Test-only checkpoint path")
    args = parser.parse_args()

    vift = vift_root()
    euroc = args.euroc_root or (vift / "data" / "euroc")
    seqs = [s.strip() for s in args.seq.split(",") if s.strip()]

    if args.step in ("verify", "all"):
        if not euroc.is_dir():
            print(f"ERROR: EuRoC root missing: {euroc}")
            return 1
        if step_verify(seqs, euroc) != 0:
            return 1

    if args.step in ("cache", "all"):
        if step_cache(vift, args.minimal) != 0:
            return 1

    if args.step in ("train", "all"):
        if step_train(vift, args.epochs, args.ckpt) != 0:
            return 1

    print("Phase 1 step(s) finished OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
