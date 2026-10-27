"""Fuse network relative poses with IMU preintegration (Phase 6-A scaffold)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.phase6.imu_preintegration import (
    preintegrate_segment,
    relative_rotation_geodesic_blend,
)
from src.utils.euroc_utils import euler_from_matrix, euler_to_rotation_matrix, path_accu


def fuse_relative_poses_with_imu(
    rel_poses: np.ndarray,
    imu: np.ndarray,
    edge_times_ns: Sequence[tuple[int, int]],
    *,
    imu_rot_weight: float = 0.35,
    scale_fuse: bool = True,
    scale_clip: tuple[float, float] = (0.25, 4.0),
) -> np.ndarray:
    """
    Fuse each relative 6-DoF network edge with IMU preintegration.

    Parameters
    ----------
    rel_poses:
        (N, 6) network edges [rx, ry, rz, tx, ty, tz].
    imu:
        EuRoC IMU table [t, gyro, accel].
    edge_times_ns:
        Length-N list of (t_i, t_{i+1}) camera timestamps in nanoseconds.
    imu_rot_weight:
        Blend weight toward IMU ΔR on SO(3).
    scale_fuse:
        If True, rescale translation magnitude using ||Δp_imu|| vs ||Δp_net||
        with a clipped per-edge ratio (scaffold heuristic — gravity not removed).
    """
    rel = np.asarray(rel_poses, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[1] != 6:
        raise ValueError(f"rel_poses must be (N,6), got {rel.shape}")
    if len(edge_times_ns) != len(rel):
        raise ValueError("edge_times_ns length must match number of relative poses")

    out = rel.copy()
    for i, (t0, t1) in enumerate(edge_times_ns):
        pre = preintegrate_segment(imu, int(t0), int(t1))
        R_net = euler_to_rotation_matrix(rel[i, :3])
        R_fused = relative_rotation_geodesic_blend(R_net, pre.delta_R, imu_rot_weight)
        out[i, :3] = euler_from_matrix(R_fused)

        t_net = rel[i, 3:]
        n_net = float(np.linalg.norm(t_net))
        n_imu = float(np.linalg.norm(pre.delta_p))
        if scale_fuse and n_net > 1e-6 and n_imu > 1e-6 and pre.dt > 1e-3:
            ratio = float(np.clip(n_imu / n_net, scale_clip[0], scale_clip[1]))
            out[i, 3:] = t_net * ratio
        else:
            out[i, 3:] = t_net
    return out


def smooth_absolute_xyz(xyz: np.ndarray, *, window: int = 5) -> np.ndarray:
    """Centered moving-average on absolute positions (optional light smoother)."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if window <= 1 or len(xyz) < window:
        return xyz.copy()
    pad = window // 2
    padded = np.pad(xyz, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.stack(
        [np.convolve(padded[:, j], kernel, mode="valid") for j in range(3)],
        axis=1,
    )


def relative_poses_to_xyz(rel_poses: np.ndarray) -> np.ndarray:
    """Integrate relative 6-DoF to absolute xyz (including identity start)."""
    mats = path_accu(np.asarray(rel_poses, dtype=np.float64))
    return np.array([T[:3, 3] for T in mats], dtype=np.float64)


def rescale_rel_translations_to_xyz_path(
    rel_poses: np.ndarray, xyz_abs: np.ndarray
) -> np.ndarray:
    """
    Keep fused rotations; set each edge translation magnitude from consecutive xyz.

    ``xyz_abs`` must have length N+1 for N relative poses (path_accu convention).
    """
    rel = np.asarray(rel_poses, dtype=np.float64).copy()
    xyz = np.asarray(xyz_abs, dtype=np.float64)
    if len(xyz) != len(rel) + 1:
        raise ValueError(
            f"xyz length {len(xyz)} must be len(rel)+1={len(rel)+1}"
        )
    for i in range(len(rel)):
        n = float(np.linalg.norm(xyz[i + 1] - xyz[i]))
        t0 = rel[i, 3:]
        n0 = float(np.linalg.norm(t0))
        if n0 > 1e-9 and n > 0:
            rel[i, 3:] = t0 * (n / n0)
    return rel


def edge_times_from_window_ends(
    t_ends_ns: np.ndarray,
    *,
    first_start_ns: int | None = None,
) -> list[tuple[int, int]]:
    """
    Build per-edge (t0, t1) from Phase-5/joint tester ``timestamps`` (= window t_end).

    Overlapping last-edge eval forms a chain: edge k ends at t_ends[k] and
    (for k>0) starts at t_ends[k-1]. The first start must be supplied or inferred
    from the median gap.
    """
    t = np.asarray(t_ends_ns, dtype=np.int64).reshape(-1)
    if len(t) == 0:
        return []
    if first_start_ns is None:
        if len(t) >= 2:
            gaps = np.diff(t.astype(np.float64))
            med = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 0.0
            first_start_ns = int(t[0] - med) if med > 0 else int(t[0] - 1)
        else:
            first_start_ns = int(t[0] - 1)
    edges: list[tuple[int, int]] = [(int(first_start_ns), int(t[0]))]
    for k in range(1, len(t)):
        edges.append((int(t[k - 1]), int(t[k])))
    return edges


def refine_trajectory_scaffold(
    rel_poses: np.ndarray,
    imu: np.ndarray,
    edge_times_ns: Sequence[tuple[int, int]],
    *,
    imu_rot_weight: float = 0.35,
    scale_fuse: bool = True,
    apply_xyz_smooth: bool = True,
    smooth_window: int = 5,
) -> dict:
    """
    Full Phase 6-A pass:
      network relatives -> IMU-fused relatives -> optional xyz smooth -> relatives out.
    """
    fused = fuse_relative_poses_with_imu(
        rel_poses,
        imu,
        edge_times_ns,
        imu_rot_weight=imu_rot_weight,
        scale_fuse=scale_fuse,
    )
    xyz = relative_poses_to_xyz(fused)
    if apply_xyz_smooth:
        xyz_s = smooth_absolute_xyz(xyz, window=smooth_window)
        fused = rescale_rel_translations_to_xyz_path(fused, xyz_s)
        xyz = relative_poses_to_xyz(fused)
    return {
        "relative_poses": fused,
        "xyz": xyz,
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))),
    }
