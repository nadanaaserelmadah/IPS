"""Phase 6-D: pose-graph optimization with odometry + visual/loop constraints."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from src.phase6.imu_preintegration import so3_exp, so3_log
from src.phase6.loop_closure import RelativeConstraint, build_visual_constraints
from src.phase6.window_smoother import relative_poses_to_xyz
from src.utils.euroc_utils import euler_from_matrix, euler_to_rotation_matrix, path_accu


def _abs_Rp(rel: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    mats = path_accu(np.asarray(rel, dtype=np.float64))
    Rs = [T[:3, :3].copy() for T in mats]
    ps = np.array([T[:3, 3] for T in mats], dtype=np.float64)
    return Rs, ps


def absolute_se3_to_relative(Rs: list[np.ndarray], ps: np.ndarray) -> np.ndarray:
    n = len(Rs) - 1
    rel = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        R_rel = Rs[i].T @ Rs[i + 1]
        t_body = Rs[i].T @ (ps[i + 1] - ps[i])
        rel[i, :3] = euler_from_matrix(R_rel)
        rel[i, 3:] = t_body
    return rel


def _pack(dtheta: np.ndarray, dp: np.ndarray) -> np.ndarray:
    return np.concatenate([dtheta.reshape(-1), dp.reshape(-1)])


def _unpack(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    dtheta = x[: 3 * n].reshape(n, 3)
    dp = x[3 * n :].reshape(n, 3)
    return dtheta, dp


def _translation_residual(
    t_est: np.ndarray,
    t_meas: np.ndarray,
    sigma_pos: float,
    *,
    up_to_scale: bool,
) -> np.ndarray:
    """
    SE(3) position residual, or Sim(3)-style direction-only residual.

    ``up_to_scale`` eliminates the unknown 2-view scale by projecting onto the
    measured direction (optimal scalar ``s = t_est · t_dir``), so loops cannot
    lock the drifted network path length.
    """
    t_meas = np.asarray(t_meas, dtype=np.float64).reshape(3)
    t_est = np.asarray(t_est, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(t_meas))
    if n < 1e-12:
        return t_est / sigma_pos
    t_dir = t_meas / n
    if float(np.dot(t_est, t_dir)) < 0.0:
        t_dir = -t_dir
    if up_to_scale:
        s = float(np.dot(t_est, t_dir))
        return (t_est - s * t_dir) / sigma_pos
    return (t_est - t_dir * n) / sigma_pos


def pose_graph_residuals(
    x: np.ndarray,
    *,
    R0: list[np.ndarray],
    p0: np.ndarray,
    rel_odom: np.ndarray,
    extras: list[RelativeConstraint],
    sigma_odom_rot: float,
    sigma_odom_pos: float,
    sigma_loop_rot: float,
    sigma_loop_pos: float,
    sigma_vis_rot: float,
    sigma_vis_pos: float,
    loops_rotation_only: bool = False,
    loops_sim3: bool = False,
    visual_sim3: bool = False,
) -> np.ndarray:
    n = len(R0)
    dtheta, dp = _unpack(x, n)
    Rs = [R0[i] @ so3_exp(dtheta[i]) for i in range(n)]
    ps = p0 + dp
    res: list[np.ndarray] = []

    # Gauge: fix first pose
    res.append(dtheta[0] / 1e-3)
    res.append(dp[0] / 1e-3)

    # Sequential odometry from network relatives
    for k in range(n - 1):
        Ri, Rj = Rs[k], Rs[k + 1]
        R_net = euler_to_rotation_matrix(rel_odom[k, :3])
        t_net = rel_odom[k, 3:]
        res.append(so3_log(R_net.T @ (Ri.T @ Rj)) / sigma_odom_rot)
        res.append((Ri.T @ (ps[k + 1] - ps[k]) - t_net) / sigma_odom_pos)

    # Visual / loop constraints
    for c in extras:
        i, j = c.i, c.j
        if i < 0 or j >= n or i >= j:
            continue
        Ri, Rj = Rs[i], Rs[j]
        is_loop = c.kind == "loop"
        if is_loop:
            sr, sp = sigma_loop_rot, sigma_loop_pos
        else:
            sr, sp = sigma_vis_rot, sigma_vis_pos
        R_meas = c.R
        t_meas = c.t
        t_est = Ri.T @ (ps[j] - ps[i])
        res.append(so3_log(R_meas.T @ (Ri.T @ Rj)) / sr)
        # Skip loop translation entirely (rotation-only mode)
        if is_loop and loops_rotation_only:
            continue
        up_to_scale = (is_loop and loops_sim3) or ((not is_loop) and visual_sim3)
        res.append(
            _translation_residual(t_est, t_meas, sp, up_to_scale=up_to_scale)
        )

    return np.concatenate(res)


def optimize_pose_graph(
    rel_odom: np.ndarray,
    extras: list[RelativeConstraint],
    *,
    max_nfev: int = 40,
    sigma_odom_rot: float = 0.05,
    sigma_odom_pos: float = 0.08,
    sigma_loop_rot: float = 0.08,
    sigma_loop_pos: float = 0.25,
    sigma_vis_rot: float = 0.1,
    sigma_vis_pos: float = 0.2,
    loops_rotation_only: bool = False,
    loops_sim3: bool = False,
    visual_sim3: bool = False,
) -> tuple[list[np.ndarray], np.ndarray]:
    R0, p0 = _abs_Rp(rel_odom)
    n = len(R0)
    x0 = np.zeros(6 * n, dtype=np.float64)

    def fun(x: np.ndarray) -> np.ndarray:
        return pose_graph_residuals(
            x,
            R0=R0,
            p0=p0,
            rel_odom=rel_odom,
            extras=extras,
            sigma_odom_rot=sigma_odom_rot,
            sigma_odom_pos=sigma_odom_pos,
            sigma_loop_rot=sigma_loop_rot,
            sigma_loop_pos=sigma_loop_pos,
            sigma_vis_rot=sigma_vis_rot,
            sigma_vis_pos=sigma_vis_pos,
            loops_rotation_only=loops_rotation_only,
            loops_sim3=loops_sim3,
            visual_sim3=visual_sim3,
        )

    sol = least_squares(
        fun,
        x0,
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        verbose=0,
    )
    dtheta, dp = _unpack(sol.x, n)
    Rs = [R0[i] @ so3_exp(dtheta[i]) for i in range(n)]
    ps = p0 + dp
    return Rs, ps


def refine_trajectory_6d(
    rel_poses: np.ndarray,
    seq_dir,
    t_ends_ns: np.ndarray,
    first_start_ns: int,
    *,
    cam_stride: int = 2,
    keyframe_step: int = 10,
    min_loop_sep: int = 80,
    max_loop_dist_m: float = 2.5,
    max_loops: int = 40,
    add_visual_odom: bool = True,
    loops_rotation_only: bool = False,
    loops_sim3: bool = False,
    visual_sim3: bool = False,
    max_nfev: int = 40,
    loop_pos_sigma: float = 0.25,
    odom_pos_sigma: float = 0.08,
) -> dict:
    """
    Phase 6-D: detect ORB loops/visual edges, then pose-graph optimize.

    ``loops_sim3`` / ``visual_sim3``: direction-only translation residuals
    (free scale) — stronger than locking ``||p_j-p_i||`` from the drifted estimate.
    """
    from pathlib import Path

    if loops_rotation_only and loops_sim3:
        raise ValueError("Use either loops_rotation_only or loops_sim3, not both")

    rel = np.asarray(rel_poses, dtype=np.float64)
    xyz0 = relative_poses_to_xyz(rel)
    extras = build_visual_constraints(
        Path(seq_dir),
        xyz0,
        np.asarray(t_ends_ns),
        int(first_start_ns),
        cam_stride=cam_stride,
        keyframe_step=keyframe_step,
        min_loop_sep=min_loop_sep,
        max_loop_dist_m=max_loop_dist_m,
        max_loops=max_loops,
        add_visual_odom=add_visual_odom,
    )
    n_loop = sum(c.kind == "loop" for c in extras)
    n_vis = sum(c.kind == "visual_odom" for c in extras)

    Rs, ps = optimize_pose_graph(
        rel,
        extras,
        max_nfev=max_nfev,
        sigma_loop_pos=loop_pos_sigma,
        sigma_odom_pos=odom_pos_sigma,
        loops_rotation_only=loops_rotation_only,
        loops_sim3=loops_sim3,
        visual_sim3=visual_sim3,
    )
    fused = absolute_se3_to_relative(Rs, ps)
    xyz = relative_poses_to_xyz(fused)
    return {
        "relative_poses": fused,
        "xyz": xyz,
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))),
        "n_loop_constraints": n_loop,
        "n_visual_odom_constraints": n_vis,
        "n_extra_constraints": len(extras),
        "loops_rotation_only": bool(loops_rotation_only),
        "loops_sim3": bool(loops_sim3),
        "visual_sim3": bool(visual_sim3),
        "add_visual_odom": bool(add_visual_odom),
    }
