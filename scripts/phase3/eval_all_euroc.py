#!/usr/bin/env python3
"""
Evaluate a checkpoint on EuRoC sequence(s).

Usage:
  python scripts/phase3/eval_all_euroc.py --ckpt path/to/last.ckpt
  python scripts/phase3/eval_all_euroc.py --ckpt path/to/last.ckpt --split all
  python scripts/phase3/eval_all_euroc.py --ckpt path/to/last.ckpt --eval-mode streaming_history --alignment se3
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "VIFT"))
from src.utils.euroc_utils import (  # noqa: E402
    EUROC_TEST_SEQS_DEFAULT,
    EUROC_TRAIN_SEQS_DEFAULT,
    EUROC_VAL_SEQS_DEFAULT,
    filter_euroc_sequences,
    list_sequences,
)


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vift_root() -> Path:
    return ips_root() / "third_party" / "VIFT"


def latest_ckpt(vift: Path) -> Path | None:
    ckpts = list((vift / "logs/train/runs").glob("*/checkpoints/last.ckpt"))
    return max(ckpts, key=lambda p: p.stat().st_mtime) if ckpts else None


def sequences_for_split(euroc_root: Path, split: str) -> list[str]:
    if split == "all":
        return list_sequences(euroc_root)
    env_key = {
        "train": "EUROC_TRAIN_SEQS",
        "val": "EUROC_VAL_SEQS",
        "test": "EUROC_TEST_SEQS",
    }[split]
    default = {
        "train": EUROC_TRAIN_SEQS_DEFAULT,
        "val": EUROC_VAL_SEQS_DEFAULT,
        "test": EUROC_TEST_SEQS_DEFAULT,
    }[split]
    requested = os.environ.get(env_key, default).split(",")
    return filter_euroc_sequences(euroc_root, requested)


def hydra_seq_list(seqs: list[str]) -> str:
    return "[" + ",".join(seqs) + "]"


def write_export(path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta.keys()))
            w.writeheader()
            w.writerow(meta)
    else:
        path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote summary: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="EuRoC eval on one or more sequences")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument(
        "--split",
        choices=["test", "train", "val", "all"],
        default="test",
        help="test=MH_05+V2_02 (default); all=every sequence on disk",
    )
    parser.add_argument(
        "--experiment",
        default="euroc_joint_finetune",
        choices=[
            "euroc_joint_finetune",
            "euroc_joint_phase5_eval",
            "euroc_joint_phase5",
            "euroc_latent_eval",
            "euroc_latent_train",
        ],
    )
    parser.add_argument("--sequences", default=None, help="Comma list override")
    parser.add_argument(
        "--eval-mode",
        default="window_last",
        choices=["window_last", "streaming_history"],
    )
    parser.add_argument(
        "--use-history",
        action="store_true",
        help="Alias for --eval-mode streaming_history",
    )
    parser.add_argument(
        "--alignment",
        default="origin",
        choices=["origin", "se3", "sim3"],
        help="Primary ATE alignment (Phase 5 configs default to se3)",
    )
    parser.add_argument("--export", type=Path, default=None, help="Write JSON/CSV run summary")
    args = parser.parse_args()

    vift = vift_root()
    euroc_root = vift / "data" / "euroc"
    if not euroc_root.is_dir():
        print(f"ERROR: EuRoC not found: {euroc_root}")
        return 1

    ckpt = args.ckpt or latest_ckpt(vift)
    if ckpt is None or not ckpt.is_file():
        print("ERROR: --ckpt required (no last.ckpt found)")
        return 1

    if args.sequences:
        seqs = filter_euroc_sequences(euroc_root, args.sequences.split(","))
    else:
        seqs = sequences_for_split(euroc_root, args.split)

    experiment = args.experiment
    if experiment == "euroc_latent_train":
        print("NOTE: remapping euroc_latent_train -> euroc_latent_eval")
        experiment = "euroc_latent_eval"

    eval_mode = "streaming_history" if args.use_history else args.eval_mode
    use_hist = "true" if eval_mode == "streaming_history" else "false"

    print(f"Experiment: {experiment}")
    print(f"Checkpoint: {ckpt}")
    print(f"Sequences ({len(seqs)}): {seqs}")
    print(f"eval_mode={eval_mode} alignment={args.alignment}")

    env = {**os.environ, "PROJECT_ROOT": str(vift), "EUROC_TEST_SEQS": ",".join(seqs)}
    ck = ckpt.resolve().as_posix()
    cmd = [
        sys.executable,
        "src/train.py",
        f"experiment={experiment}",
        "train=false",
        "test=true",
        "trainer=gpu",
        "logger=csv",
        f"ckpt_path='{ck}'",
        f"model.tester.sequences={hydra_seq_list(seqs)}",
        f"model.tester.use_history_in_eval={use_hist}",
        f"model.tester.eval_mode={eval_mode}",
        f"model.metrics_calculator.primary_alignment={args.alignment}",
    ]
    print("$", " ".join(cmd))
    code = subprocess.call(cmd, cwd=vift, env=env)

    if args.export is not None:
        write_export(
            args.export,
            {
                "dataset": "euroc",
                "checkpoint": str(ckpt),
                "experiment": experiment,
                "split": args.split,
                "sequences": ",".join(seqs),
                "eval_mode": eval_mode,
                "alignment": args.alignment,
                "exit_code": code,
            },
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
