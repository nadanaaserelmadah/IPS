#!/usr/bin/env python3
"""
Minimal KITTI evaluation for IPS Phase 0.

Subcommands:
  check       — verify kitti_data layout, IMU, pretrained encoder
  gt-sanity   — KITTI metric on GT poses (expect ~0 trel/rrel); needs poses only
  cache-val   — run latent_val_caching_minimal.py (GPU)
  train-test  — Hydra experiment kitti_eval_minimal (short train + full tester)

Paper targets (VIFT Table 1, w. RPMG): seq05 trel=2.02 rrel=0.53; seq07 1.75/0.47; seq10 2.11/0.39
Short training will NOT hit these; use gt-sanity to verify metrics, then compare order-of-magnitude after longer train.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Paper reference (trel %, rrel deg/m) — VIFT + RPMG
PAPER = {
    "05": (2.02, 0.53),
    "07": (1.75, 0.47),
    "10": (2.11, 0.39),
}


def vift_root_from_arg(path: Path | None) -> Path:
    if path is not None:
        return path.resolve()
    return Path(__file__).resolve().parents[2] / "third_party" / "VIFT"


def kitti_root(vift: Path) -> Path:
    return vift / "data" / "kitti_data"


def check_layout(vift: Path, seqs: list[str]) -> bool:
    root = kitti_root(vift)
    ok = True
    pretrained = vift / "pretrained_models" / "vf_512_if_256_3e-05.model"
    if not pretrained.is_file():
        print(f"MISSING encoder: {pretrained}")
        ok = False
    else:
        print(f"OK encoder: {pretrained}")

    imus = root / "imus"
    for seq in seqs:
        pose = root / "poses" / f"{seq}.txt"
        imu = imus / f"{seq}.mat"
        img_glob = list((root / "sequences" / seq / "image_2").glob("*.png"))[:1]
        if not pose.is_file():
            print(f"MISSING pose: {pose}")
            ok = False
        if not imu.is_file():
            print(f"MISSING imu:  {imu}  (run: python scripts/phase0/download_kitti_imus.py --minimal)")
            ok = False
        if not img_glob:
            print(f"MISSING images: {root / 'sequences' / seq / 'image_2'}")
            ok = False
        if pose.is_file() and imu.is_file() and img_glob:
            print(f"OK seq {seq}: pose + imu + images")

    latent_val = vift / "data" / "kitti_latent_data" / "val_10"
    if (latent_val / "0.npy").is_file():
        n = len(list(latent_val.glob("*.npy"))) // 4
        print(f"OK latents: {latent_val} (~{n} windows)")
    else:
        print(f"MISSING latents: {latent_val} (run: cache-val)")
    return ok


def gt_sanity(vift: Path, seqs: list[str]) -> int:
    sys.path.insert(0, str(vift))
    import numpy as np
    from src.utils.kitti_eval import kitti_eval
    from src.utils.kitti_utils import path_accu, read_pose_from_text

    root = kitti_root(vift)
    print("GT sanity: pose_est == pose_gt should give trel,rrel ~ 0\n")
    all_ok = True
    for seq in seqs:
        _, poses_rel = read_pose_from_text(str(root / "poses" / f"{seq}.txt"))
        pose_est = np.asarray(poses_rel, dtype=np.float64)
        pose_gt = pose_est.copy()
        _, _, t_rel, r_rel, t_rmse, r_rmse, _ = kitti_eval(pose_est, pose_gt)
        # kitti_eval already scales t_rel (*100) and r_rel (to deg/m *100)
        print(f"  seq {seq}: trel={t_rel:.4f}  rrel={r_rel:.4f}  (expect ~0)")
        if t_rel > 0.01 or r_rel > 0.01:
            all_ok = False
    if all_ok:
        print("\nPASS: metric pipeline sane on ground truth.")
        return 0
    print("\nWARN: non-zero GT error — check kitti_eval / pose format.")
    return 1


def cache_val(vift: Path, seqs: list[str]) -> int:
    env = {**os.environ, "KITTI_VAL_SEQS": ",".join(seqs)}
    data_dir = vift / "data"
    cmd = [sys.executable, "latent_val_caching_minimal.py"]
    print("Running:", " ".join(cmd), f"(KITTI_VAL_SEQS={env['KITTI_VAL_SEQS']})")
    return subprocess.call(cmd, cwd=data_dir, env=env)


def train_test(vift: Path, extra: list[str]) -> int:
    cmd = [
        sys.executable,
        "src/train.py",
        "experiment=kitti_eval_minimal",
        "trainer=gpu",
    ] + extra
    env = {**os.environ, "PROJECT_ROOT": str(vift)}
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=vift, env=env)


def print_paper_table():
    print("\nPaper reference (VIFT Table 1, w. RPMG):")
    print("  seq | trel (%) | rrel (deg/m)")
    for seq, (t, r) in PAPER.items():
        print(f"  {seq} | {t:8.2f} | {r:11.2f}")
    print("\nAfter train-test, compare logged test/{seq}_t_rel and test/{seq}_r_rel.")
    print("Within ~10–20% of paper only after full 200-epoch train on train_10 latents.\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vift-root", type=Path, default=None)
    parser.add_argument("--seq", default="05,07,10", help="Comma-separated sequence ids")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="Verify files on disk")
    sub.add_parser("gt-sanity", help="Metrics on GT poses (no model)")
    sub.add_parser("cache-val", help="Cache val latents with pretrained encoder")
    p_train = sub.add_parser("train-test", help="Short Hydra train + KITTILatentTester")
    p_train.add_argument("--ckpt", type=str, default=None, help="Skip train; test with checkpoint")
    p_train.add_argument("--epochs", type=int, default=None)

    args, extra = parser.parse_known_args()
    vift = vift_root_from_arg(args.vift_root)
    seqs = [s.strip() for s in args.seq.split(",") if s.strip()]
    ckpt = getattr(args, "ckpt", None)
    epochs = getattr(args, "epochs", None)

    if args.cmd == "check":
        print_paper_table()
        return 0 if check_layout(vift, seqs) else 1
    if args.cmd == "gt-sanity":
        return gt_sanity(vift, seqs)
    if args.cmd == "cache-val":
        return cache_val(vift, seqs)
    if args.cmd == "train-test":
        print_paper_table()
        hydra_extra = list(extra)
        if ckpt:
            hydra_extra += ["train=false", "test=true", f"ckpt_path={ckpt}"]
        if epochs is not None:
            hydra_extra += [f"trainer.max_epochs={epochs}"]
        return train_test(vift, hydra_extra)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
