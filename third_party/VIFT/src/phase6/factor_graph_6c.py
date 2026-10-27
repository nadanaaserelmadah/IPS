"""Phase 6-C: sliding-window pose-graph with network + IMU factors (scipy LS)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares

from src.phase6.imu_preintegration import so3_exp, so3_log
from src.phase6.imu_preintegration_se3 import (
    BiasState,
    PreintegrationSE3,
    estimate_accel_bias,
    estimate_gravity_in_world,
    estimate_gyro_bias,
    preintegrate_segment_se3,
)
from src.phase6.window_smoother import relative_poses_to_xyz
from src.utils.euroc_utils import euler_from_matrix, euler_to_rotation_matrix, path_accu


@dataclass
class NoiseModel:
    """Diagonal std-devs for residuals (smaller = stronger factor)."""

    sigma_imu_rot: float = 0.02  # rad
    sigma_imu_vel: float = 0.5  # m/s
    sigma_imu_pos: float = 0.3  # m
    sigma_net_rot: float = 0.05  # rad
    sigma_net_pos: float = 0.08  # m
    sigma_anchor: float = 1e-3


def _abs_Rp_from_rel(rel: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    mats = path_accu(np.asarray(rel, dtype=np.float64))
    Rs = [T[:3, :3].copy() for T in mats]
    ps = np.array([T[:3, 3] for T in mats], dtype=np.float64)
    return Rs, ps


def _finite_diff_velocities(ps: np.ndarray, dts: np.ndarray) -> np.ndarray:
    """Initialize velocities from position finite differences."""
    n = len(ps)
    vs = np.zeros_like(ps)
    for i in range(n - 1):
        dt = max(float(dts[i]), 1e-3)
        vs[i] = (ps[i + 1] - ps[i]) / dt
    if n >= 2:
        vs[-1] = vs[-2]
    return vs


def absolute_se3_to_relative(Rs: list[np.ndarray], ps: np.ndarray) -> np.ndarray:
    n = len(Rs) - 1
    rel = np.zeros((n, 6), dtype=np.float64)
    for i in range(n):
        R_rel = Rs[i].T @ Rs[i + 1]
        t_body = Rs[i].T @ (ps[i + 1] - ps[i])
        rel[i, :3] = euler_from_matrix(R_rel)
        rel[i, 3:] = t_body
    return rel


def _pack(dtheta: np.ndarray, dp: np.ndarray, dv: np.ndarray) -> np.ndarray:
    # (W,3) each → (9W,)
    return np.concatenate([dtheta.reshape(-1), dp.reshape(-1), dv.reshape(-1)])


def _unpack(x: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtheta = x[0 : 3 * w].reshape(w, 3)
    dp = x[3 * w : 6 * w].reshape(w, 3)
    dv = x[6 * w : 9 * w].reshape(w, 3)
    return dtheta, dp, dv


def _states_from_delta(
    R0: list[np.ndarray],
    p0: np.ndarray,
    v0: np.ndarray,
    dtheta: np.ndarray,
    dp: np.ndarray,
    dv: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    w = len(R0)
    Rs = [R0[i] @ so3_exp(dtheta[i]) for i in range(w)]
    ps = p0 + dp
    vs = v0 + dv
    return Rs, ps, vs


def window_residuals(
    x: np.ndarray,
    *,
    R0: list[np.ndarray],
    p0: np.ndarray,
    v0: np.ndarray,
    rel_edges: np.ndarray,
    preints: list[PreintegrationSE3],
    g_world: np.ndarray,
    noise: NoiseModel,
    n_anchor: int = 1,
) -> np.ndarray:
    """
    Residuals for one window of W poses / W-1 edges.

    ``rel_edges`` shape (W-1, 6), ``preints`` length W-1.
    First ``n_anchor`` frames are soft-anchored to the linearization point.
    """
    w = len(R0)
    dtheta, dp, dv = _unpack(x, w)
    Rs, ps, vs = _states_from_delta(R0, p0, v0, dtheta, dp, dv)
    res: list[np.ndarray] = []

    # Anchor priors on the oldest frames (gauge + continuity).
    for i in range(min(n_anchor, w)):
        res.append(dtheta[i] / noise.sigma_anchor)
        res.append(dp[i] / noise.sigma_anchor)
        res.append(dv[i] / (noise.sigma_anchor * 10.0))

    for k in range(w - 1):
        Ri, Rj = Rs[k], Rs[k + 1]
        pi, pj = ps[k], ps[k + 1]
        vi, vj = vs[k], vs[k + 1]
        pre = preints[k]
        dt = max(pre.dt, 1e-4)
        R_net = euler_to_rotation_matrix(rel_edges[k, :3])
        t_net = rel_edges[k, 3:]

        # --- Network factors ---
        r_net_rot = so3_log(R_net.T @ (Ri.T @ Rj)) / noise.sigma_net_rot
        r_net_pos = (Ri.T @ (pj - pi) - t_net) / noise.sigma_net_pos
        res.append(r_net_rot)
        res.append(r_net_pos)

        # --- IMU factors (gravity outside preintegration) ---
        r_imu_rot = so3_log(pre.delta_R.T @ (Ri.T @ Rj)) / noise.sigma_imu_rot
        v_pred = vi + g_world * dt + Ri @ pre.delta_v
        p_pred = pi + vi * dt + 0.5 * g_world * dt * dt + Ri @ pre.delta_p
        r_imu_vel = (vj - v_pred) / noise.sigma_imu_vel
        r_imu_pos = (pj - p_pred) / noise.sigma_imu_pos
        res.append(r_imu_rot)
        res.append(r_imu_vel)
        res.append(r_imu_pos)

    return np.concatenate(res)


def optimize_window(
    R0: list[np.ndarray],
    p0: np.ndarray,
    v0: np.ndarray,
    rel_edges: np.ndarray,
    preints: list[PreintegrationSE3],
    g_world: np.ndarray,
    *,
    noise: NoiseModel | None = None,
    n_anchor: int = 1,
    max_nfev: int = 30,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Local BA on one window; returns optimized R, p, v."""
    noise = noise or NoiseModel()
    w = len(R0)
    x0 = np.zeros(9 * w, dtype=np.float64)

    def fun(x: np.ndarray) -> np.ndarray:
        return window_residuals(
            x,
            R0=R0,
            p0=p0,
            v0=v0,
            rel_edges=rel_edges,
            preints=preints,
            g_world=g_world,
            noise=noise,
            n_anchor=n_anchor,
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
    dtheta, dp, dv = _unpack(sol.x, w)
    return _states_from_delta(R0, p0, v0, dtheta, dp, dv)


def refine_trajectory_6c(
    rel_poses: np.ndarray,
    imu: np.ndarray,
    edge_times_ns: Sequence[tuple[int, int]],
    *,
    window_size: int = 30,
    stride: int = 10,
    noise: NoiseModel | None = None,
    max_nfev: int = 25,
    estimate_biases: bool = True,
    g_mag: float = 9.81007,
    imu_prior_weight: float = 1.0,
) -> dict:
    """
    Sliding-window factor-graph refine (Phase 6-C).

    Parameters
    ----------
    window_size:
        Number of poses in each window (>= 3).
    stride:
        How many poses to advance after each solve (poses before stride stay fixed).
    imu_prior_weight:
        Scales IMU residual strength (1.0 = default NoiseModel; <1 weakens IMU).
    """
    rel = np.asarray(rel_poses, dtype=np.float64)
    if rel.ndim != 2 or rel.shape[1] != 6:
        raise ValueError(f"rel_poses must be (N,6), got {rel.shape}")
    n_edges = len(rel)
    n_poses = n_edges + 1
    if len(edge_times_ns) != n_edges:
        raise ValueError("edge_times_ns length mismatch")
    if window_size < 3:
        raise ValueError("window_size must be >= 3")
    if stride < 1:
        raise ValueError("stride must be >= 1")

    edges = [(int(a), int(b)) for a, b in edge_times_ns]
    R_init, p_init = _abs_Rp_from_rel(rel)

    bg = estimate_gyro_bias(imu) if estimate_biases else np.zeros(3)
    g_world = estimate_gravity_in_world(R_init, imu, edges, g_mag=g_mag)
    ba = (
        estimate_accel_bias(R_init, imu, edges, g_world)
        if estimate_biases
        else np.zeros(3)
    )
    bias = BiasState(bg, ba)

    # Precompute all edge preintegrations once.
    preints_all: list[PreintegrationSE3] = [
        preintegrate_segment_se3(imu, t0, t1, bias) for t0, t1 in edges
    ]
    dts = np.array([max(p.dt, 1e-3) for p in preints_all], dtype=np.float64)
    v_init = _finite_diff_velocities(p_init, dts)

    noise = noise or NoiseModel()
    # Soften / strengthen IMU via imu_prior_weight
    w_imu = float(max(imu_prior_weight, 1e-3))
    noise = NoiseModel(
        sigma_imu_rot=noise.sigma_imu_rot / w_imu,
        sigma_imu_vel=noise.sigma_imu_vel / w_imu,
        sigma_imu_pos=noise.sigma_imu_pos / w_imu,
        sigma_net_rot=noise.sigma_net_rot,
        sigma_net_pos=noise.sigma_net_pos,
        sigma_anchor=noise.sigma_anchor,
    )

    Rs = [R.copy() for R in R_init]
    ps = p_init.copy()
    vs = v_init.copy()

    # Window starts: stride along the trajectory; always cover the tail.
    max_start = max(0, n_poses - window_size)
    starts = list(range(0, max_start + 1, stride))
    if not starts or starts[-1] != max_start:
        starts.append(max_start)
    # Deduplicate while preserving order
    seen = set()
    starts_u = []
    for s in starts:
        if s not in seen:
            seen.add(s)
            starts_u.append(s)

    n_windows = 0
    for start in starts_u:
        end = start + window_size
        R0 = [Rs[i].copy() for i in range(start, end)]
        p0 = ps[start:end].copy()
        v0 = vs[start:end].copy()
        rel_w = rel[start : end - 1].copy()
        pre_w = preints_all[start : end - 1]
        n_anchor = 1 if start == 0 else min(max(stride, 1), window_size // 3)

        R_opt, p_opt, v_opt = optimize_window(
            R0,
            p0,
            v0,
            rel_w,
            pre_w,
            g_world,
            noise=noise,
            n_anchor=n_anchor,
            max_nfev=max_nfev,
        )
        for j in range(window_size):
            Rs[start + j] = R_opt[j]
            ps[start + j] = p_opt[j]
            vs[start + j] = v_opt[j]
        n_windows += 1

    fused_rel = absolute_se3_to_relative(Rs, ps)
    xyz = relative_poses_to_xyz(fused_rel)
    return {
        "relative_poses": fused_rel,
        "xyz": xyz,
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))),
        "gravity_world": g_world,
        "bias_gyro": bg,
        "bias_accel": ba,
        "n_windows": n_windows,
        "window_size": window_size,
        "stride": stride,
    }
