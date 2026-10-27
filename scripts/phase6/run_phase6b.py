#!/usr/bin/env python3
"""
Phase 6-B: gravity-aware IMU preintegration + offline/velocity fusion (no BA yet).

Usage:
  python scripts/phase6/run_phase6b.py \\
    --poses-dir /path/to/poses \\
    --euroc-root /path/to/EuRoC \\
    --sequences MH_05_difficult,V2_02_medium \\
    --out-dir /kaggle/working/phase6_b
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def ips_root() -> Path:
    return Path(__file__).resolve().parents[2]


def vift_root() -> Path:
    return ips_root() / "third_party" / "VIFT"


def _ensure_path() -> None:
    vift = vift_root()
    if str(vift) not in sys.path:
        sys.path.insert(0, str(vift))


def first_edge_start_from_cam(
    seq_dir: Path, t_ends, *, seq_len: int, cam_stride: int
) -> int | None:
    import numpy as np
    from src.utils.euroc_utils import load_cam_entries

    cam_ts, _ = load_cam_entries(seq_dir)
    cam_ts = cam_ts[::cam_stride]
    idx_end0 = seq_len - 1
    idx_start0 = seq_len - 2
    if idx_end0 >= len(cam_ts) or idx_start0 < 0:
        return None
    if int(cam_ts[idx_end0]) != int(t_ends[0]):
        matches = np.where(cam_ts == int(t_ends[0]))[0]
        if len(matches) == 0:
            return None
        j = int(matches[0])
        if j < 1:
            return None
        return int(cam_ts[j - 1])
    return int(cam_ts[idx_start0])


def ate_pair(est_xyz, gt_xyz, alignment: str) -> float:
    from src.metrics.euroc_metrics_calculator import ate_rmse_m

    return float(ate_rmse_m(est_xyz, gt_xyz, alignment=alignment))


def process_sequence(
    seq: str,
    poses_dir: Path,
    euroc_root: Path,
    out_dir: Path,
    *,
    seq_len: int,
    cam_stride: int,
    imu_rot_weight: float,
    imu_pos_weight: float,
    estimate_biases: bool,
    smooth: bool,
    smooth_window: int,
) -> dict:
    import numpy as np
    from src.phase6.backend_6b import refine_trajectory_6b
    from src.phase6.window_smoother import edge_times_from_window_ends, relative_poses_to_xyz
    from src.utils.euroc_utils import load_imu, path_accu

    est_path = poses_dir / f"{seq}_estimated_poses.npy"
    if not est_path.is_file():
        raise FileNotFoundError(f"Missing {est_path}")

    rel = np.load(est_path)
    ts_path = poses_dir / f"{seq}_timestamps.npy"
    gt_rel_path = poses_dir / f"{seq}_gt_poses.npy"
    seq_dir = euroc_root / seq
    imu = load_imu(seq_dir)

    if not ts_path.is_file():
        raise FileNotFoundError(
            f"Phase 6-B requires {ts_path.name} (timestamps). Re-run Phase 5 eval with pose dumps."
        )

    t_ends = np.load(ts_path)
    first_start = first_edge_start_from_cam(
        seq_dir, t_ends, seq_len=seq_len, cam_stride=cam_stride
    )
    edges = edge_times_from_window_ends(t_ends, first_start_ns=first_start)

    refined = refine_trajectory_6b(
        rel,
        imu,
        edges,
        imu_rot_weight=imu_rot_weight,
        imu_pos_weight=imu_pos_weight,
        estimate_biases=estimate_biases,
        apply_xyz_smooth=smooth,
        smooth_window=smooth_window,
    )
    rel_f = refined["relative_poses"]
    xyz_f = refined["xyz"]
    xyz_raw = relative_poses_to_xyz(rel)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{seq}_refined_poses.npy", rel_f)
    np.save(out_dir / f"{seq}_refined_xyz.npy", xyz_f)

    row: dict = {
        "sequence": seq,
        "n_edges": int(len(rel)),
        "path_len_raw_m": float(np.sum(np.linalg.norm(np.diff(xyz_raw, axis=0), axis=1))),
        "path_len_refined_m": refined["path_len_m"],
        "imu_rot_weight": imu_rot_weight,
        "imu_pos_weight": imu_pos_weight,
        "gravity_norm": float(np.linalg.norm(refined["gravity_world"])),
        "bias_gyro_norm": float(np.linalg.norm(refined["bias_gyro"])),
        "bias_accel_norm": float(np.linalg.norm(refined["bias_accel"])),
    }

    gt_xyz = None
    if gt_rel_path.is_file():
        gt_rel = np.load(gt_rel_path)
        gt_mats = path_accu(gt_rel)
        gt_xyz = np.array([T[:3, 3] for T in gt_mats], dtype=np.float64)

    if gt_xyz is not None:
        for align in ("origin", "se3", "sim3"):
            row[f"ate_{align}_raw_m"] = ate_pair(xyz_raw, gt_xyz, align)
            row[f"ate_{align}_refined_m"] = ate_pair(xyz_f, gt_xyz, align)

    print(
        f"{seq}: path {row['path_len_raw_m']:.2f}m -> {row['path_len_refined_m']:.2f}m"
        + (
            f" | SE3 {row.get('ate_se3_raw_m', float('nan')):.3f} -> "
            f"{row.get('ate_se3_refined_m', float('nan')):.3f}"
            if "ate_se3_raw_m" in row
            else ""
        )
        + f" | |g|={row['gravity_norm']:.3f}"
    )
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 6-B gravity-aware IMU backend")
    p.add_argument("--poses-dir", type=Path, required=True)
    p.add_argument("--euroc-root", type=Path, required=True)
    p.add_argument("--sequences", type=str, default="MH_05_difficult,V2_02_medium")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=11)
    p.add_argument("--cam-stride", type=int, default=2)
    p.add_argument("--imu-rot-weight", type=float, default=0.25)
    p.add_argument("--imu-pos-weight", type=float, default=0.35)
    p.add_argument("--no-biases", action="store_true")
    p.add_argument("--smooth", action="store_true")
    p.add_argument("--smooth-window", type=int, default=5)
    args = p.parse_args()

    _ensure_path()
    seqs = [s.strip() for s in args.sequences.split(",") if s.strip()]
    rows = []
    for seq in seqs:
        rows.append(
            process_sequence(
                seq,
                args.poses_dir.resolve(),
                args.euroc_root.resolve(),
                args.out_dir.resolve(),
                seq_len=args.seq_len,
                cam_stride=args.cam_stride,
                imu_rot_weight=args.imu_rot_weight,
                imu_pos_weight=args.imu_pos_weight,
                estimate_biases=not args.no_biases,
                smooth=args.smooth,
                smooth_window=args.smooth_window,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / "phase6b_summary.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = args.out_dir / "phase6b_summary.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {summary}")
    print(f"Wrote {csv_path}")
    print(
        "NOTE: Phase 6-B has gravity+bias preintegration but NO sliding-window BA. "
        "That is Phase 6-C. Do not claim Adaptive-VIO-level results yet."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
