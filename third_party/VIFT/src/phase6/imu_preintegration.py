"""Simplified IMU preintegration for Phase 6-A scaffold (bias-free)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)


def so3_exp(phi: np.ndarray) -> np.ndarray:
    """Rodrigues: so(3) vector -> SO(3)."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(phi))
    if theta < 1e-12:
        return np.eye(3) + _skew(phi)
    k = phi / theta
    K = _skew(k)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


def so3_log(R: np.ndarray) -> np.ndarray:
    """SO(3) -> so(3) vector."""
    R = np.asarray(R, dtype=np.float64)
    cos_theta = (np.trace(R) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < 1e-12:
        return np.zeros(3, dtype=np.float64)
    w = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=np.float64,
    )
    return (theta / (2.0 * np.sin(theta))) * w


@dataclass
class PreintegrationResult:
    delta_R: np.ndarray  # (3, 3) body rotation from i to j
    delta_v: np.ndarray  # (3,) approx, no gravity compensation
    delta_p: np.ndarray  # (3,) approx double-integrated accel
    dt: float
    n_samples: int


def preintegrate_segment(
    imu: np.ndarray,
    t_start_ns: int,
    t_end_ns: int,
    *,
    gyro_cols: slice = slice(1, 4),
    accel_cols: slice = slice(4, 7),
) -> PreintegrationResult:
    """
    Bias-free mid-point-style preintegration on a raw EuRoC IMU table.

    ``imu`` rows: [t_ns, wx, wy, wz, ax, ay, az, ...]

    Notes:
    - Suitable as a Phase 6-A scaffold, not production VINS-quality preintegration.
    - Gravity is *not* removed (needs known world orientation); use delta_R primarily.
    - delta_p / delta_v are rough scale cues for fusion, not metric truth alone.
    """
    imu = np.asarray(imu, dtype=np.float64)
    if t_end_ns <= t_start_ns:
        return PreintegrationResult(np.eye(3), np.zeros(3), np.zeros(3), 0.0, 0)

    mask = (imu[:, 0] >= t_start_ns) & (imu[:, 0] <= t_end_ns)
    seg = imu[mask]
    if len(seg) < 2:
        dt = (t_end_ns - t_start_ns) * 1e-9
        return PreintegrationResult(np.eye(3), np.zeros(3), np.zeros(3), dt, int(len(seg)))

    R = np.eye(3, dtype=np.float64)
    v = np.zeros(3, dtype=np.float64)
    p = np.zeros(3, dtype=np.float64)

    for k in range(len(seg) - 1):
        t0 = seg[k, 0]
        t1 = seg[k + 1, 0]
        dt = max((t1 - t0) * 1e-9, 1e-6)
        w = 0.5 * (seg[k, gyro_cols] + seg[k + 1, gyro_cols])
        a = 0.5 * (seg[k, accel_cols] + seg[k + 1, accel_cols])
        # Integrate in the evolving body/world-mixed frame (scaffold).
        # Order: p with the old v and current R, then v, then advance R.
        p = p + v * dt + 0.5 * (R @ a) * dt * dt
        v = v + R @ a * dt
        R = R @ so3_exp(w * dt)

    dt_total = (seg[-1, 0] - seg[0, 0]) * 1e-9
    return PreintegrationResult(R, v, p, float(dt_total), int(len(seg)))


def relative_rotation_geodesic_blend(
    R_net: np.ndarray, R_imu: np.ndarray, imu_weight: float
) -> np.ndarray:
    """
    Blend two rotations on SO(3): R = R_net * Exp(w * Log(R_net^T R_imu)).

    ``imu_weight`` in [0, 1]; 0 = network only, 1 = IMU only.
    """
    w = float(np.clip(imu_weight, 0.0, 1.0))
    if w <= 0.0:
        return np.asarray(R_net, dtype=np.float64)
    if w >= 1.0:
        return np.asarray(R_imu, dtype=np.float64)
    dR = np.asarray(R_net, dtype=np.float64).T @ np.asarray(R_imu, dtype=np.float64)
    return np.asarray(R_net, dtype=np.float64) @ so3_exp(w * so3_log(dR))
