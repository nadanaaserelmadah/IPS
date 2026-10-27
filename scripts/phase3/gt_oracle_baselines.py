#!/usr/bin/env python3
"""
GT oracle ATE per EuRoC sequence (integrate GT relatives → should be ~0 m).

Usage:
  python scripts/phase3/gt_oracle_baselines.py
  python scripts/phase3/gt_oracle_baselines.py --split all
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_euroc_utils():
    path = ips_root() / "third_party" / "VIFT" / "src" / "utils" / "euroc_utils.py"
    spec = importlib.util.spec_from_file_location("euroc_utils", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="EuRoC root (default: VIFT/data/euroc)",
    )
    parser.add_argument("--split", choices=["test", "train", "val", "all"], default="all")
    args = parser.parse_args()

    eu = load_euroc_utils()
    vift = ips_root() / "third_party" / "VIFT"
    root = args.root or (vift / "data" / "euroc")

    if args.split == "all":
        seqs = eu.list_sequences(root)
    else:
        import os

        defaults = {
            "train": eu.EUROC_TRAIN_SEQS_DEFAULT,
            "val": eu.EUROC_VAL_SEQS_DEFAULT,
            "test": eu.EUROC_TEST_SEQS_DEFAULT,
        }
        seqs = eu.filter_euroc_sequences(root, defaults[args.split].split(","))

    import numpy as np

    vift = ips_root() / "third_party" / "VIFT"
    sys.path.insert(0, str(vift))
    from src.metrics.euroc_metrics_calculator import ate_rmse_m

    print("EuRoC GT oracle ATE (upper bound; should be ~0)")
    print(f"{'sequence':<22} {'ate_m':>10} {'path_len_m':>12} {'windows':>8}")
    print("-" * 56)

    rows = []
    for seq in seqs:
        seq_dir = root / seq
        samples, _ = eu.build_sequence_samples(
            seq_dir, sequence_length=11, cam_stride=2, imu_freq=10
        )
        if not samples:
            print(f"{seq:<22} {'—':>10} {'—':>12} {'0':>8}")
            continue

        gt_deltas = []
        for s in samples:
            g = np.asarray(s["gts"], dtype=np.float64)
            gt_deltas.append(g[-1] if g.ndim == 2 else g)

        gt_deltas = np.vstack(gt_deltas)
        est = gt_deltas.copy()
        est_xyz = np.array([T[:3, 3] for T in eu.path_accu(est)])
        gt_xyz = np.array([T[:3, 3] for T in eu.path_accu(gt_deltas)])
        ate = ate_rmse_m(est_xyz, gt_xyz)
        path_len = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
        rows.append((seq, ate, path_len, len(samples)))
        print(f"{seq:<22} {ate:10.6f} {path_len:12.2f} {len(samples):8d}")

    if rows:
        mean_ate = sum(r[1] for r in rows) / len(rows)
        print("-" * 56)
        print(f"{'MEAN':<22} {mean_ate:10.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
