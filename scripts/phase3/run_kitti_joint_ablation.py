#!/usr/bin/env python3
"""
KITTI frozen-vs-joint ablation runner.

Compares trainable encoder+fusion (this script) against the frozen-encoder
latent replication (mean t_rel 2.10% on seq 05/07/10).

Usage:
  python scripts/phase3/run_kitti_joint_ablation.py train --max-epochs 50
  python scripts/phase3/run_kitti_joint_ablation.py train --max-epochs 50 \\
      --fusion-ckpt /path/to/latent_kitti_last.ckpt
  python scripts/phase3/run_kitti_joint_ablation.py resume --ckpt /path/to/last.ckpt
  python scripts/phase3/run_kitti_joint_ablation.py eval --ckpt /path/to/last.ckpt

Checkpoints: every 2 epochs (last.ckpt). ZIP to /kaggle/working every 3 epochs
(via configs/callbacks/kitti_joint_checkpoint.yaml).
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


def _model_train_overrides(args: argparse.Namespace) -> list[str]:
    return [
        f"model.freeze_encoder_epochs={args.freeze_encoder_epochs}",
        f"model.freeze_fusion_epochs={args.freeze_fusion_epochs}",
        f"model.fusion_lr={args.fusion_lr}",
        f"model.encoder_lr_scale={args.encoder_lr_scale}",
        f"model.encoder_heads_only={str(args.encoder_heads_only).lower()}",
    ]


def cmd_train(args: argparse.Namespace) -> int:
    overrides = [
        "experiment=kitti_joint_finetune",
        "train=true",
        "test=false",
        "trainer=gpu",
        f"trainer.max_epochs={args.max_epochs}",
        f"model.scheduler.T_0={min(25, args.max_epochs)}",
        f"data.batch_size={args.batch_size}",
        *_model_train_overrides(args),
    ]
    if args.fusion_ckpt is not None:
        ck = args.fusion_ckpt.resolve().as_posix()
        overrides.append(f"model.phase2_ckpt='{ck}'")
    return run_train_py(overrides)


def _load_resume_hparams(ckpt_path: Path) -> dict:
    """Best-effort hparams; never requires importing training ``src``."""
    # Last stable KITTI recipe (frozen fusion + encoder heads only).
    fallback = {
        "freeze_encoder_epochs": 0,
        "freeze_fusion_epochs": 999,
        "fusion_lr": 1e-6,
        "encoder_lr_scale": 1.0,
        "encoder_heads_only": True,
        "epoch": "unknown",
    }
    try:
        import pickle
        import zipfile

        class _SkipUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "builtins":
                    return getattr(__import__("builtins"), name)
                if module == "collections" and name == "OrderedDict":
                    from collections import OrderedDict

                    return OrderedDict

                class _Dummy:
                    def __init__(self, *args, **kwargs):
                        pass

                    def __setstate__(self, state):
                        if isinstance(state, dict):
                            self.__dict__.update(state)

                _Dummy.__name__ = name
                return _Dummy

        hp: dict = {}
        epoch = "unknown"
        with zipfile.ZipFile(ckpt_path) as zf:
            names = [n for n in zf.namelist() if n.endswith("data.pkl")]
            raw = zf.read(names[0]) if names else zf.read(zf.namelist()[0])
        obj = _SkipUnpickler(__import__("io").BytesIO(raw)).load()
        if isinstance(obj, dict):
            epoch = obj.get("epoch", epoch)
            cand = obj.get("hyper_parameters") or {}
            if isinstance(cand, dict):
                hp = cand
            elif hasattr(cand, "__dict__"):
                hp = dict(cand.__dict__)
        out = {
            "freeze_encoder_epochs": int(hp.get("freeze_encoder_epochs", fallback["freeze_encoder_epochs"])),
            "freeze_fusion_epochs": int(hp.get("freeze_fusion_epochs", fallback["freeze_fusion_epochs"])),
            "fusion_lr": float(hp.get("fusion_lr", fallback["fusion_lr"])),
            "encoder_lr_scale": float(hp.get("encoder_lr_scale", fallback["encoder_lr_scale"])),
            "encoder_heads_only": bool(hp.get("encoder_heads_only", fallback["encoder_heads_only"])),
            "epoch": epoch,
        }
    except Exception as exc:
        print(f"WARN: could not parse ckpt hparams ({exc}); using heads-only defaults")
        out = fallback
    print(
        "Resume hparams "
        f"(epoch {out['epoch']}): freeze_encoder={out['freeze_encoder_epochs']} "
        f"freeze_fusion={out['freeze_fusion_epochs']} heads_only={out['encoder_heads_only']} "
        f"fusion_lr={out['fusion_lr']} enc_lr_scale={out['encoder_lr_scale']}"
    )
    return out


def cmd_resume(args: argparse.Namespace) -> int:
    ck = args.ckpt.resolve()
    saved = _load_resume_hparams(ck)
    if args.freeze_encoder_epochs is None:
        args.freeze_encoder_epochs = saved["freeze_encoder_epochs"]
    if args.freeze_fusion_epochs is None:
        args.freeze_fusion_epochs = saved["freeze_fusion_epochs"]
    if args.fusion_lr is None:
        args.fusion_lr = saved["fusion_lr"]
    if args.encoder_lr_scale is None:
        args.encoder_lr_scale = saved["encoder_lr_scale"]
    if not args.encoder_heads_only:
        args.encoder_heads_only = saved["encoder_heads_only"]
    overrides = [
        "experiment=kitti_joint_finetune",
        "train=true",
        "test=false",
        "trainer=gpu",
        f"ckpt_path='{ck.as_posix()}'",
        "init_ckpt_weights=false",
        f"trainer.max_epochs={args.max_epochs}",
        f"data.batch_size={args.batch_size}",
        *_model_train_overrides(args),
    ]
    return run_train_py(overrides)


def cmd_eval(args: argparse.Namespace) -> int:
    ck = args.ckpt.resolve().as_posix()
    hist = "true" if args.history else "false"
    mode = "streaming_history" if args.history else "window_last"
    overrides = [
        "experiment=kitti_joint_eval",
        "train=false",
        "test=true",
        "trainer=gpu",
        f"ckpt_path='{ck}'",
        f"model.tester.use_history_in_eval={hist}",
        f"model.tester.eval_mode={mode}",
    ]
    return run_train_py(overrides)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="KITTI frozen vs joint ablation")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Joint fine-tune on raw KITTI")
    t.add_argument("--max-epochs", type=int, default=50)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--freeze-encoder-epochs", type=int, default=5)
    t.add_argument(
        "--freeze-fusion-epochs",
        type=int,
        default=0,
        help="Keep fusion frozen this many epochs (use 999 to freeze for whole run)",
    )
    t.add_argument(
        "--fusion-lr",
        type=float,
        default=1e-4,
        help="Fusion AdamW lr (use 1e-5 if fine-tuning a warm-started fusion)",
    )
    t.add_argument(
        "--encoder-lr-scale",
        type=float,
        default=0.1,
        help="Encoder lr = fusion_lr * scale",
    )
    t.add_argument(
        "--encoder-heads-only",
        action="store_true",
        help="Freeze CNN/IMU conv; train visual_head + inertial proj only",
    )
    t.add_argument(
        "--fusion-ckpt",
        type=Path,
        default=None,
        help="Optional frozen KITTI latent last.ckpt to warm-start fusion only",
    )

    r = sub.add_parser("resume", help="Resume joint KITTI training")
    r.add_argument("--ckpt", type=Path, required=True)
    r.add_argument("--max-epochs", type=int, default=50)
    r.add_argument("--batch-size", type=int, default=4)
    r.add_argument("--freeze-encoder-epochs", type=int, default=None)
    r.add_argument(
        "--freeze-fusion-epochs",
        type=int,
        default=None,
        help="Default: restore from checkpoint hyper_parameters",
    )
    r.add_argument("--fusion-lr", type=float, default=None)
    r.add_argument("--encoder-lr-scale", type=float, default=None)
    r.add_argument(
        "--encoder-heads-only",
        action="store_true",
        help="Force heads-only; default is whatever the checkpoint used",
    )

    e = sub.add_parser("eval", help="Eval on KITTI 05/07/10 (t_rel / r_rel)")
    e.add_argument("--ckpt", type=Path, required=True)
    e.add_argument(
        "--no-history",
        action="store_true",
        help="Use window_last instead of streaming_history",
    )
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        if args.fusion_ckpt is not None and not args.fusion_ckpt.is_file():
            print(f"ERROR: fusion ckpt not found: {args.fusion_ckpt}")
            return 1
        return cmd_train(args)
    if args.command == "resume":
        if not args.ckpt.is_file():
            print(f"ERROR: checkpoint not found: {args.ckpt}")
            return 1
        return cmd_resume(args)
    args.history = not args.no_history
    if not args.ckpt.is_file():
        print(f"ERROR: checkpoint not found: {args.ckpt}")
        return 1
    return cmd_eval(args)


if __name__ == "__main__":
    raise SystemExit(main())
