#!/usr/bin/env python3
"""
Phase 5 runner: baseline eval, EuRoC refresh train, resume, EuRoC/TUM eval.

Usage (from IPS root or anywhere):
  python scripts/phase5/run_phase5.py baseline --ckpt /path/to/tum_ft_epoch_024.ckpt
  python scripts/phase5/run_phase5.py train --ckpt /path/to/tum_ft_epoch_024.ckpt
  python scripts/phase5/run_phase5.py resume --ckpt /path/to/last.ckpt
  python scripts/phase5/run_phase5.py eval-euroc --ckpt /path/to/ckpt --alignment se3
  python scripts/phase5/run_phase5.py eval-tum --ckpt /path/to/ckpt --split test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vift_root() -> Path:
    return ips_root() / "third_party" / "VIFT"


def hydra_seq_list(seqs: list[str]) -> str:
    return "[" + ",".join(seqs) + "]"


def run_train_py(extra: list[str], env: dict | None = None) -> int:
    vift = vift_root()
    cmd = [sys.executable, "src/train.py", *extra]
    print("$", " ".join(cmd))
    full_env = {**os.environ, "PROJECT_ROOT": str(vift), **(env or {})}
    return subprocess.call(cmd, cwd=vift, env=full_env)


def cmd_baseline(args: argparse.Namespace) -> int:
    """Re-baseline TUM-FT (or any ckpt) on EuRoC test with explicit alignment."""
    ck = args.ckpt.resolve().as_posix()
    overrides = [
        "experiment=euroc_joint_phase5_eval",
        "train=false",
        "test=true",
        "trainer=gpu",
        "logger=csv",
        f"ckpt_path='{ck}'",
        f"model.metrics_calculator.primary_alignment={args.alignment}",
        f"model.tester.eval_mode={args.eval_mode}",
        f"model.tester.use_history_in_eval={'true' if args.eval_mode == 'streaming_history' else 'false'}",
        "model.freeze_encoder_epochs=0",
    ]
    if args.sequences:
        overrides.append(f"model.tester.sequences={hydra_seq_list(args.sequences.split(','))}")
    return run_train_py(overrides)


def cmd_train(args: argparse.Namespace) -> int:
    """Warm-start Phase 5 EuRoC training from TUM-FT weights."""
    ck = args.ckpt.resolve().as_posix()
    overrides = [
        "experiment=euroc_joint_phase5",
        "train=true",
        "test=false",
        "trainer=gpu",
        f"ckpt_path='{ck}'",
        "init_ckpt_weights=true",
        f"trainer.max_epochs={args.max_epochs}",
        f"model.scheduler.T_0={args.max_epochs}",
        f"data.batch_size={args.batch_size}",
    ]
    if args.eval_mode == "streaming_history":
        overrides.append("model.tester.eval_mode=streaming_history")
        overrides.append("model.tester.use_history_in_eval=true")
    return run_train_py(overrides)


def cmd_resume(args: argparse.Namespace) -> int:
    """Resume a Phase 5 run (full Lightning state)."""
    ck = args.ckpt.resolve().as_posix()
    overrides = [
        "experiment=euroc_joint_phase5",
        "train=true",
        "test=false",
        "trainer=gpu",
        f"ckpt_path='{ck}'",
        "init_ckpt_weights=false",
        f"trainer.max_epochs={args.max_epochs}",
        f"data.batch_size={args.batch_size}",
    ]
    return run_train_py(overrides)


def cmd_eval_euroc(args: argparse.Namespace) -> int:
    # Delegate to updated eval_all_euroc with Phase 5 experiment
    script = ips_root() / "scripts" / "phase3" / "eval_all_euroc.py"
    cmd = [
        sys.executable,
        str(script),
        "--ckpt",
        str(args.ckpt),
        "--split",
        args.split,
        "--experiment",
        "euroc_joint_phase5_eval",
        "--eval-mode",
        args.eval_mode,
        "--alignment",
        args.alignment,
    ]
    if args.sequences:
        cmd.extend(["--sequences", args.sequences])
    if args.export:
        cmd.extend(["--export", str(args.export)])
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_eval_tum(args: argparse.Namespace) -> int:
    script = ips_root() / "scripts" / "phase4" / "eval_all_tum_vi.py"
    cmd = [
        sys.executable,
        str(script),
        "--ckpt",
        str(args.ckpt),
        "--split",
        args.split,
        "--experiment",
        "tum_vi_joint_eval",
        "--eval-mode",
        args.eval_mode,
        "--alignment",
        args.alignment,
    ]
    if args.sequences:
        cmd.extend(["--sequences", args.sequences])
    if args.export:
        cmd.extend(["--export", str(args.export)])
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Phase 5 IPS runner")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--ckpt", type=Path, required=True)
        sp.add_argument(
            "--eval-mode",
            default="window_last",
            choices=["window_last", "streaming_history"],
        )
        sp.add_argument(
            "--alignment",
            default="se3",
            choices=["origin", "se3", "sim3"],
            help="Primary ATE alignment for Phase 5 metrics",
        )
        sp.add_argument("--sequences", default=None, help="Comma-separated override")
        sp.add_argument("--export", type=Path, default=None, help="JSON/CSV summary path")

    b = sub.add_parser("baseline", help="Eval TUM-FT / ckpt on EuRoC before training")
    add_common(b)

    t = sub.add_parser("train", help="Warm-start Phase 5 EuRoC training from TUM-FT")
    add_common(t)
    t.add_argument("--max-epochs", type=int, default=100)
    t.add_argument("--batch-size", type=int, default=4)

    r = sub.add_parser("resume", help="Resume Phase 5 training")
    add_common(r)
    r.add_argument("--max-epochs", type=int, default=100)
    r.add_argument("--batch-size", type=int, default=4)

    e = sub.add_parser("eval-euroc", help="Evaluate on EuRoC")
    add_common(e)
    e.add_argument("--split", default="test", choices=["test", "train", "val", "all"])

    u = sub.add_parser("eval-tum", help="Evaluate on TUM-VI (forgetting check)")
    add_common(u)
    u.add_argument("--split", default="test", choices=["test", "train", "val", "all"])

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.ckpt.is_file():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        return 1
    dispatch = {
        "baseline": cmd_baseline,
        "train": cmd_train,
        "resume": cmd_resume,
        "eval-euroc": cmd_eval_euroc,
        "eval-tum": cmd_eval_tum,
    }
    return dispatch[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
