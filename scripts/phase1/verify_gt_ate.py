"""
Phase 1 sanity check: integrate EuRoC GT relative poses and measure reconstruction error.
Expected: max position error ~0 (machine precision) when using GT relatives from GT absolutes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import importlib.util

import numpy as np

VIFT_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "VIFT"
sys.path.insert(0, str(VIFT_ROOT))  # build_sequence_samples imports src.utils.imu_canonical


def _load_euroc_utils():
    """Load euroc_utils without importing src.utils (pulls in torch/lightning)."""
    path = VIFT_ROOT / "src" / "utils" / "euroc_utils.py"
    spec = importlib.util.spec_from_file_location("euroc_utils", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


eu = _load_euroc_utils()
build_sequence_samples = eu.build_sequence_samples
interpolate_gt_pose = eu.interpolate_gt_pose
load_cam_entries = eu.load_cam_entries
load_gt_poses = eu.load_gt_poses
get_relative_pose_6DoF = eu.get_relative_pose_6DoF
path_accu = eu.path_accu


def ate_rmse(est_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """RMSE after aligning trajectories by first pose (translation only)."""
    offset = gt_xyz[0] - est_xyz[0]
    est_aligned = est_xyz + offset
    return float(np.sqrt(np.mean(np.sum((est_aligned - gt_xyz) ** 2, axis=1))))


def verify_sequence(seq_dir: Path, cam_stride: int = 2) -> dict:
    timestamps, _ = load_cam_entries(seq_dir)
    gt = load_gt_poses(seq_dir)
    timestamps = timestamps[::cam_stride]

    poses_abs = [interpolate_gt_pose(gt, int(t)) for t in timestamps]
    poses_rel = np.array(
        [get_relative_pose_6DoF(poses_abs[i], poses_abs[i + 1]) for i in range(len(poses_abs) - 1)]
    )
    traj = path_accu(poses_rel)
    w0 = poses_abs[0]
    # Integrated poses are in frame 0; map back to world: W_k = W_0 @ T_k
    est_xyz = np.array([(w0 @ T)[:3, 3] for T in traj])
    gt_xyz = np.array([T[:3, 3] for T in poses_abs])

    max_pos_err = float(np.max(np.linalg.norm(est_xyz - gt_xyz, axis=1)))
    # The round trip above holds for any invertible matrices, so also check that GT
    # "rotations" are rotations and relative translations are frame displacements.
    rot_orth_err = float(max(np.abs(T[:3, :3].T @ T[:3, :3] - np.eye(3)).max() for T in poses_abs))
    path_rel = float(np.sum(np.linalg.norm(poses_rel[:, 3:6], axis=1)))
    path_abs = float(np.sum(np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)))
    rmse = ate_rmse(est_xyz, gt_xyz)

    # Window-level check via dataset builder
    samples, _ = build_sequence_samples(seq_dir, sequence_length=11, cam_stride=cam_stride)
    n = min(100, len(samples))
    gts = np.stack([samples[i]["gts"] for i in range(n)], axis=0)
    mean_trans = float(np.mean(np.linalg.norm(gts[:, :, 3:6], axis=-1)))

    return {
        "sequence": seq_dir.name,
        "frames": len(timestamps),
        "windows": len(samples),
        "max_pos_err_m": max_pos_err,
        "ate_rmse_aligned_m": rmse,
        "mean_rel_trans_m": mean_trans,
        "rot_orthonormality_err": rot_orth_err,
        "path_rel_m": path_rel,
        "path_abs_m": path_abs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=VIFT_ROOT / "data" / "euroc",
        help="EuRoC root containing MH_01_easy/ etc.",
    )
    parser.add_argument(
        "--seq",
        type=str,
        default="MH_01_easy",
        help="Sequence folder name only (e.g. MH_01_easy), not a path to data.csv",
    )
    args = parser.parse_args()

    seq_name = Path(args.seq.strip()).name
    if seq_name.endswith(".csv"):
        raise SystemExit(
            "ERROR: --seq must be a sequence folder name (e.g. MH_01_easy), not a file path.\n"
            "Example:\n"
            "  python verify_gt_ate.py --root .../data/euroc --seq MH_01_easy"
        )

    seq_dir = args.root / seq_name
    if not seq_dir.is_dir():
        print(f"ERROR: {seq_dir} not found. Run scripts/phase1/download_euroc.ps1 first.")
        sys.exit(1)

    stats = verify_sequence(seq_dir)
    print("EuRoC GT integration sanity check")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    if stats["max_pos_err_m"] > 1e-4:
        print("WARN: max position error larger than expected (check pose chain).")
        sys.exit(2)
    if stats["rot_orthonormality_err"] > 1e-6:
        print("FAIL: GT rotation matrices are not orthonormal (quaternion conversion).")
        sys.exit(3)
    if abs(stats["path_rel_m"] - stats["path_abs_m"]) > 1e-6 * max(1.0, stats["path_abs_m"]):
        print("FAIL: sum of relative translations != path of absolute positions.")
        sys.exit(4)
    print("PASS: GT pose integration is consistent.")


if __name__ == "__main__":
    main()
