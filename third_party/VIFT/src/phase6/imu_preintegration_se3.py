"""Phase 6-B: body-frame IMU preintegration with bias (Forster-style midpoint)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.phase6.imu_preintegration import so3_exp, so3_log


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


@dataclass
class BiasState:
    gyro: np.ndarray  # (3,)
    accel: np.ndarray  # (3,)

    @staticmethod
    def zeros() -> "BiasState":
        return BiasState(np.zeros(3), np.zeros(3))


@dataclass
class PreintegrationSE3:
    """
    Preintegrated measurements between frames i and j, expressed in body frame i.

    Gravity is *not* included here; apply it when composing in world:
      R_j = R_i @ delta_R
      v_j = v_i + g * dt + R_i @ delta_v
      p_j = p_i + v_i * dt + 0.5 * g * dt^2 + R_i @ delta_p
    """

    delta_R: np.ndarray  # (3, 3)
    delta_v: np.ndarray  # (3,)
    delta_p: np.ndarray  # (3,)
    dt: float
    n_samples: int
    bias: BiasState


def preintegrate_segment_se3(
    imu: np.ndarray,
    t_start_ns: int,
    t_end_ns: int,
    bias: BiasState | None = None,
    *,
    gyro_cols: slice = slice(1, 4),
    accel_cols: slice = slice(4, 7),
) -> PreintegrationSE3:
    """
    Mid-point IMU preintegration in the starting body frame (bias-corrected).

    ``imu`` rows: [t_ns, wx, wy, wz, ax, ay, az, ...]
    """
    bias = bias or BiasState.zeros()
    bg = np.asarray(bias.gyro, dtype=np.float64).reshape(3)
    ba = np.asarray(bias.accel, dtype=np.float64).reshape(3)
    imu = np.asarray(imu, dtype=np.float64)

    if t_end_ns <= t_start_ns:
        return PreintegrationSE3(np.eye(3), np.zeros(3), np.zeros(3), 0.0, 0, bias)

    mask = (imu[:, 0] >= t_start_ns) & (imu[:, 0] <= t_end_ns)
    seg = imu[mask]
    if len(seg) < 2:
        dt = (t_end_ns - t_start_ns) * 1e-9
        return PreintegrationSE3(np.eye(3), np.zeros(3), np.zeros(3), float(dt), int(len(seg)), bias)

    # Discrete preintegration (Qin/Forster midpoint): accumulate ΔR, Δv, Δp in frame i.
    dR = np.eye(3, dtype=np.float64)
    dv = np.zeros(3, dtype=np.float64)
    dp = np.zeros(3, dtype=np.float64)

    for k in range(len(seg) - 1):
        t0 = seg[k, 0]
        t1 = seg[k + 1, 0]
        dt = max((t1 - t0) * 1e-9, 1e-6)
        w = 0.5 * (seg[k, gyro_cols] + seg[k + 1, gyro_cols]) - bg
        a = 0.5 * (seg[k, accel_cols] + seg[k + 1, accel_cols]) - ba
        # Order: position/velocity with current ΔR, then advance ΔR.
        dp = dp + dv * dt + 0.5 * (dR @ a) * dt * dt
        dv = dv + dR @ a * dt
        dR = dR @ so3_exp(w * dt)

    dt_total = (seg[-1, 0] - seg[0, 0]) * 1e-9
    return PreintegrationSE3(dR, dv, dp, float(dt_total), int(len(seg)), bias)


def estimate_gyro_bias(
    imu: np.ndarray,
    *,
    n_samples: int = 200,
    gyro_cols: slice = slice(1, 4),
    still_gyro_norm: float = 0.05,
) -> np.ndarray:
    """Rough gyro bias = mean ω on early low-motion samples."""
    imu = np.asarray(imu, dtype=np.float64)
    if len(imu) == 0:
        return np.zeros(3)
    head = imu[: max(n_samples, 1)]
    w = head[:, gyro_cols]
    norms = np.linalg.norm(w, axis=1)
    still = w[norms < still_gyro_norm]
    if len(still) < 5:
        still = w
    return still.mean(axis=0)


def estimate_gravity_in_world(
    abs_R: list[np.ndarray],
    imu: np.ndarray,
    edge_times_ns: list[tuple[int, int]],
    *,
    g_mag: float = 9.81007,
    accel_cols: slice = slice(4, 7),
    gyro_cols: slice = slice(1, 4),
    still_gyro_norm: float = 0.08,
) -> np.ndarray:
    """
    Estimate gravity vector in the network/world frame.

    When nearly static: a ≈ -R^T g  ⇒  R @ a ≈ -g.
    Uses IMU samples near edge starts with small gyro norm.
    """
    imu = np.asarray(imu, dtype=np.float64)
    samples = []
    for i, (t0, _t1) in enumerate(edge_times_ns):
        if i >= len(abs_R):
            break
        # Take a short IMU slice around t0
        mask = (imu[:, 0] >= t0 - 5_000_000) & (imu[:, 0] <= t0 + 20_000_000)
        seg = imu[mask]
        if len(seg) == 0:
            continue
        w_n = np.linalg.norm(seg[:, gyro_cols], axis=1)
        a = seg[w_n < still_gyro_norm][:, accel_cols]
        if len(a) == 0:
            continue
        a_mean = a.mean(axis=0)
        samples.append(abs_R[i] @ a_mean)

    if not samples:
        # Fallback: average first-second accel mapped by R0
        t0 = int(imu[0, 0])
        mask = imu[:, 0] <= t0 + 1_000_000_000
        a0 = imu[mask][:50, accel_cols].mean(axis=0) if np.any(mask) else imu[0, accel_cols]
        g_est = -(abs_R[0] @ a0)
    else:
        g_est = -np.mean(np.stack(samples, axis=0), axis=0)

    n = float(np.linalg.norm(g_est))
    if n < 1e-6:
        return np.array([0.0, 0.0, -g_mag], dtype=np.float64)
    return (g_est / n) * g_mag


def estimate_accel_bias(
    abs_R: list[np.ndarray],
    imu: np.ndarray,
    edge_times_ns: list[tuple[int, int]],
    g_world: np.ndarray,
    *,
    accel_cols: slice = slice(4, 7),
    gyro_cols: slice = slice(1, 4),
    still_gyro_norm: float = 0.08,
) -> np.ndarray:
    """When static: a ≈ -R^T g + b_a  ⇒  b_a ≈ a + R^T g."""
    imu = np.asarray(imu, dtype=np.float64)
    g_world = np.asarray(g_world, dtype=np.float64).reshape(3)
    samples = []
    for i, (t0, _t1) in enumerate(edge_times_ns):
        if i >= len(abs_R):
            break
        mask = (imu[:, 0] >= t0 - 5_000_000) & (imu[:, 0] <= t0 + 20_000_000)
        seg = imu[mask]
        if len(seg) == 0:
            continue
        w_n = np.linalg.norm(seg[:, gyro_cols], axis=1)
        quiet = seg[w_n < still_gyro_norm]
        if len(quiet) == 0:
            continue
        a_mean = quiet[:, accel_cols].mean(axis=0)
        samples.append(a_mean + abs_R[i].T @ g_world)
    if not samples:
        return np.zeros(3)
    return np.mean(np.stack(samples, axis=0), axis=0)
