#!/usr/bin/env python3
"""
Phase 6-E: ORB triangulation + reprojection BA (Level-B constraints).

Requires: pip install opencv-python-headless

Usage:
  python scripts/phase6/run_phase6e.py \\
    --poses-dir /path/to/poses \\
    --euroc-root /path/to/EuRoC \\
    --sequences MH_05_difficult,V2_02_medium \\
    --out-dir /kaggle/working/phase6_e
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


def first_edge_start_from_cam(seq_dir: Path, t_ends, *, seq_len: int, cam_stride: int):
    import numpy as np
    from src.utils.euroc_utils import load_cam_entries

    cam_ts, _ = load_cam_entries(seq_dir)
    cam_ts = cam_ts[::cam_stride]
    idx_end0, idx_start0 = seq_len - 1, seq_len - 2
    if idx_end0 >= len(cam_ts) or idx_start0 < 0:
        return None
    if int(cam_ts[idx_end0]) != int(t_ends[0]):
        matches = np.where(cam_ts == int(t_ends[0]))[0]
        if len(matches) == 0:
            return None
        j = int(matches[0])
        return int(cam_ts[j - 1]) if j >= 1 else None
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
    keyframe_step: int,
    max_landmarks: int,
    max_per_pair: int,
    min_loop_sep: int,
    max_loop_dist_m: float,
    max_loop_pairs: int,
    odom_pos_sigma: float,
    sigma_pix: float,
    max_nfev: int,
) -> dict:
    import numpy as np
    from src.phase6.landmark_ba_6e import refine_trajectory_6e
    from src.phase6.window_smoother import relative_poses_to_xyz
    from src.utils.euroc_utils import path_accu

    est_path = poses_dir / f"{seq}_estimated_poses.npy"
    ts_path = poses_dir / f"{seq}_timestamps.npy"
    if not est_path.is_file():
        raise FileNotFoundError(est_path)
    if not ts_path.is_file():
        raise FileNotFoundError(ts_path)

    rel = np.load(est_path)
    t_ends = np.load(ts_path)
    seq_dir = euroc_root / seq
    first_start = first_edge_start_from_cam(
        seq_dir, t_ends, seq_len=seq_len, cam_stride=cam_stride
    )
    if first_start is None:
        gaps = np.diff(t_ends.astype(np.float64))
        med = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
        first_start = int(t_ends[0] - med)

    print(f"{seq}: Phase 6-E landmark BA on {len(rel) + 1} poses ...")
    refined = refine_trajectory_6e(
        rel,
        seq_dir,
        t_ends,
        int(first_start),
        cam_stride=cam_stride,
        keyframe_step=keyframe_step,
        max_landmarks=max_landmarks,
        max_per_pair=max_per_pair,
        min_loop_sep=min_loop_sep,
        max_loop_dist_m=max_loop_dist_m,
        max_loop_pairs=max_loop_pairs,
        max_nfev=max_nfev,
        sigma_odom_pos=odom_pos_sigma,
        sigma_pix=sigma_pix,
    )
    xyz_raw = relative_poses_to_xyz(rel)
    xyz_f = refined["xyz"]
    rel_f = refined["relative_poses"]

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{seq}_refined_poses.npy", rel_f)
    np.save(out_dir / f"{seq}_refined_xyz.npy", xyz_f)

    row = {
        "sequence": seq,
        "n_edges": int(len(rel)),
        "n_landmarks": int(refined["n_landmarks"]),
        "n_observations": int(refined["n_observations"]),
        "path_len_raw_m": float(np.sum(np.linalg.norm(np.diff(xyz_raw, axis=0), axis=1))),
        "path_len_refined_m": refined["path_len_m"],
    }
    gt_rel_path = poses_dir / f"{seq}_gt_poses.npy"
    if gt_rel_path.is_file():
        gt_xyz = np.array(
            [T[:3, 3] for T in path_accu(np.load(gt_rel_path))], dtype=np.float64
        )
        for align in ("origin", "se3", "sim3"):
            row[f"ate_{align}_raw_m"] = ate_pair(xyz_raw, gt_xyz, align)
            row[f"ate_{align}_refined_m"] = ate_pair(xyz_f, gt_xyz, align)

    print(
        f"{seq}: landmarks={row['n_landmarks']} obs={row['n_observations']} | "
        f"path {row['path_len_raw_m']:.2f}->{row['path_len_refined_m']:.2f}"
        + (
            f" | SE3 {row.get('ate_se3_raw_m', float('nan')):.3f}->{row.get('ate_se3_refined_m', float('nan')):.3f}"
            if "ate_se3_raw_m" in row
            else ""
        )
    )
    return row


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 6-E landmark reprojection BA")
    p.add_argument("--poses-dir", type=Path, required=True)
    p.add_argument("--euroc-root", type=Path, required=True)
    p.add_argument("--sequences", type=str, default="MH_05_difficult,V2_02_medium")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--seq-len", type=int, default=11)
    p.add_argument("--cam-stride", type=int, default=2)
    p.add_argument("--keyframe-step", type=int, default=10)
    p.add_argument("--max-landmarks", type=int, default=2500)
    p.add_argument("--max-per-pair", type=int, default=40)
    p.add_argument("--min-loop-sep", type=int, default=80)
    p.add_argument("--max-loop-dist-m", type=float, default=3.0)
    p.add_argument("--max-loop-pairs", type=int, default=20)
    p.add_argument("--odom-pos-sigma", type=float, default=0.12)
    p.add_argument("--sigma-pix", type=float, default=1.5)
    p.add_argument("--max-nfev", type=int, default=25)
    args = p.parse_args()

    _ensure_path()
    try:
        import cv2  # noqa: F401
    except ImportError:
        print("ERROR: opencv not installed. Run: pip install opencv-python-headless")
        return 1

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
                keyframe_step=args.keyframe_step,
                max_landmarks=args.max_landmarks,
                max_per_pair=args.max_per_pair,
                min_loop_sep=args.min_loop_sep,
                max_loop_dist_m=args.max_loop_dist_m,
                max_loop_pairs=args.max_loop_pairs,
                odom_pos_sigma=args.odom_pos_sigma,
                sigma_pix=args.sigma_pix,
                max_nfev=args.max_nfev,
            )
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = args.out_dir / "phase6e_summary.json"
    summary.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    csv_path = args.out_dir / "phase6e_summary.csv"
    if rows:
        keys = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {summary}")
    print(
        "NOTE: 6-E adds triangulated ORB landmarks + reprojection BA. "
        "Still not full ORB-SLAM3 / Adaptive-VIO."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
