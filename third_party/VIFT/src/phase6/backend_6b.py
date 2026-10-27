"""Phase 6-B offline backend: gravity-aware IMU propagation + network fusion (no BA)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.phase6.imu_preintegration import relative_rotation_geodesic_blend
from src.phase6.imu_preintegration_se3 import (
    BiasState,
    estimate_accel_bias,
    estimate_gravity_in_world,
    estimate_gyro_bias,
    preintegrate_segment_se3,
)
from src.phase6.window_smoother import relative_poses_to_xyz, smooth_absolute_xyz
from src.utils.euroc_utils import (
    euler_from_matrix,
    euler_to_rotation_matrix,
    path_accu,
)


def _abs_rotations_from_rel(rel_poses: np.ndarray) -> list[np.ndarray]:
    mats = path_accu(np.asarray(rel_poses, dtype=np.float64))
    return [T[:3, :3].copy() for T in mats]


def absolute_se3_to_relative(Rs: list[np.ndarray], ps: np.ndarray) -> np.ndarray:
    """Convert absolute R,p chain to relative 6-DoF edges."""
    n = len(Rs) - 1
    rel = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        R_rel = Rs[i].T @ Rs[i + 1]
        t_body = Rs[i].T @ (ps[i + 1] - ps[i])
        rel[i, :3] = euler_from_matrix(R_rel)
        rel[i, 3:] = t_body
    return rel


def refine_trajectory_6b(
    rel_poses: np.ndarray,
    imu: np.ndarray,
    edge_times_ns: Sequence[tuple[int, int]],
    *,
    imu_rot_weight: float = 0.25,
    imu_pos_weight: float = 0.35,
    estimate_biases: bool = True,
    g_mag: float = 9.81007,
    apply_xyz_smooth: bool = False,
    smooth_window: int = 5,
) -> dict:
    """
    Phase 6-B pass (offline, no factor-graph BA):

    1. Estimate gyro/accel bias and gravity in the network world frame.
    2. Preintegrate each edge with biases (body-frame ΔR, Δv, Δp).
    3. Blend rotations on SO(3).
    4. Propagate p,v with gravity; fuse network position step with IMU step.
    """
    rel = np.asarray(rel_poses, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[1] != 6:
        raise ValueError(f"rel_poses must be (N,6), got {rel.shape}")
    if len(edge_times_ns) != len(rel):
        raise ValueError("edge_times_ns length must match relative poses")

    # Initial absolute orientations from network (for gravity / bias estimation).
    R_abs_net = _abs_rotations_from_rel(rel)
    edges = [(int(a), int(b)) for a, b in edge_times_ns]

    bg = estimate_gyro_bias(imu) if estimate_biases else np.zeros(3)
    g_world = estimate_gravity_in_world(
        R_abs_net, imu, edges, g_mag=g_mag
    )
    ba = (
        estimate_accel_bias(R_abs_net, imu, edges, g_world)
        if estimate_biases
        else np.zeros(3)
    )
    bias = BiasState(bg, ba)

    w_rot = float(np.clip(imu_rot_weight, 0.0, 1.0))
    w_pos = float(np.clip(imu_pos_weight, 0.0, 1.0))

    n = len(rel)
    Rs: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    ps = np.zeros((n + 1, 3), dtype=np.float64)
    vs = np.zeros((n + 1, 3), dtype=np.float64)

    # Seed velocity from first network step / dt (optional).
    if n >= 1:
        pre0 = preintegrate_segment_se3(imu, edges[0][0], edges[0][1], bias)
        if pre0.dt > 1e-3:
            t0 = rel[0, 3:]
            vs[0] = Rs[0] @ (t0 / pre0.dt)

    for i in range(n):
        t0, t1 = edges[i]
        pre = preintegrate_segment_se3(imu, t0, t1, bias)
        R_net = euler_to_rotation_matrix(rel[i, :3])
        R_rel = relative_rotation_geodesic_blend(R_net, pre.delta_R, w_rot)
        Rs.append(Rs[i] @ R_rel)

        dt = max(pre.dt, 1e-6)
        # Network step in world using fused rotation at i
        p_net = ps[i] + Rs[i] @ rel[i, 3:]
        # IMU step with gravity
        p_imu = (
            ps[i]
            + vs[i] * dt
            + 0.5 * g_world * dt * dt
            + Rs[i] @ pre.delta_p
        )
        v_imu = vs[i] + g_world * dt + Rs[i] @ pre.delta_v

        ps[i + 1] = (1.0 - w_pos) * p_net + w_pos * p_imu
        # Velocity: blend IMU with finite-difference of fused positions
        v_fd = (ps[i + 1] - ps[i]) / dt
        vs[i + 1] = 0.5 * v_imu + 0.5 * v_fd

    if apply_xyz_smooth:
        ps = smooth_absolute_xyz(ps, window=smooth_window)

    fused_rel = absolute_se3_to_relative(Rs, ps)
    xyz = relative_poses_to_xyz(fused_rel)
    return {
        "relative_poses": fused_rel,
        "xyz": xyz,
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))),
        "gravity_world": g_world,
        "bias_gyro": bg,
        "bias_accel": ba,
        "imu_rot_weight": w_rot,
        "imu_pos_weight": w_pos,
    }
