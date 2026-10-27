#!/usr/bin/env python3
"""
Phase 3C entrypoint after 50-epoch joint fine-tune: baselines + full eval.

Usage:
  python scripts/phase3/run_phase3c.py --ckpt path/to/last.ckpt
  python scripts/phase3/run_phase3c.py --ckpt path/to/last.ckpt --skip-full
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(py: str, args: list[str]) -> int:
    cmd = [sys.executable, str(ips_root() / "scripts" / "phase3" / py), *args]
    print("\n>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3C: GT oracle + eval all EuRoC")
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--skip-full", action="store_true", help="Only test split (default run)")
    parser.add_argument(
        "--experiment",
        default="euroc_joint_finetune",
        choices=["euroc_joint_finetune", "euroc_latent_eval", "euroc_latent_train"],
    )
    args = parser.parse_args()

    ck = str(args.ckpt.resolve())

    print("=== Step 1/3: GT oracle baselines (should be ~0 m ATE) ===")
    if run("gt_oracle_baselines.py", ["--split", "all"]) != 0:
        return 1

    print("\n=== Step 2/3: Test split (MH_05 + V2_02) ===")
    if (
        run(
            "eval_all_euroc.py",
            ["--ckpt", ck, "--split", "test", "--experiment", args.experiment],
        )
        != 0
    ):
        return 1

    if not args.skip_full:
        print("\n=== Step 3/3: All available sequences (slow) ===")
        if (
            run(
                "eval_all_euroc.py",
                ["--ckpt", ck, "--split", "all", "--experiment", args.experiment],
            )
            != 0
        ):
            return 1
    else:
        print("\n(Skipped full eval — use without --skip-full for all sequences)")

    print("\nDone. See docs/PHASE3C.md for baselines table + indoor IPS next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
