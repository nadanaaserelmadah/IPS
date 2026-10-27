"""Canonical IMU + camera-frame poses for VIFT's KITTI-pretrained inertial encoder.

EuRoC / TUM-VI CSVs are gyro-first in the IMU/body frame. Visual-Selective-VIO
was trained on KITTI OXTS windows with accelerometer-first channels. This module
reorders channels and rotates IMU vectors into the camera frame via cam0 T_BS.

Disable with IPS_IMU_LEGACY=1 to reproduce the mismatched (A0) loader.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

G_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)


def imu_canonical_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return bool(explicit)
    return os.environ.get("IPS_IMU_LEGACY", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def gyro_first_to_accel_first(imu6: np.ndarray) -> np.ndarray:
    """(…, 6) [gx,gy,gz,ax,ay,az] → [ax,ay,az,gx,gy,gz]."""
    imu6 = np.asarray(imu6, dtype=np.float64)
    return np.concatenate([imu6[..., 3:6], imu6[..., 0:3]], axis=-1)


def parse_kalibr_T_BS(text: str) -> np.ndarray | None:
    """Parse a 4×4 T_BS block from Kalibr / EuRoC sensor.yaml text."""
    match = re.search(
        r"T_BS\s*:.*?data\s*:\s*\[([^\]]+)\]",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    nums = [float(x) for x in match.group(1).replace("\n", " ").split(",") if x.strip()]
    if len(nums) != 16:
        return None
    return np.asarray(nums, dtype=np.float64).reshape(4, 4)


def load_T_BS(seq_dir: Path, cam: str = "cam0") -> np.ndarray:
    """Body-from-camera T_BS (EuRoC: p_body = T_BS p_cam). Identity if missing."""
    seq_dir = Path(seq_dir)
    candidates = [
        seq_dir / cam / "sensor.yaml",
        seq_dir / "mav0" / cam / "sensor.yaml",
        seq_dir / seq_dir.name / "mav0" / cam / "sensor.yaml",
        seq_dir / seq_dir.name / cam / "sensor.yaml",
    ]
    for yaml_path in candidates:
        if yaml_path.is_file():
            parsed = parse_kalibr_T_BS(yaml_path.read_text(encoding="utf-8", errors="ignore"))
            if parsed is not None:
                return parsed
    # TUM-VI exports ship no cam0/sensor.yaml; the calibration is Kalibr's dso/camchain.yaml.
    for chain in (seq_dir / "dso" / "camchain.yaml", seq_dir / seq_dir.name / "dso" / "camchain.yaml",
                  seq_dir / "camchain.yaml"):
        if chain.is_file():
            parsed = parse_camchain_T_BS(chain.read_text(encoding="utf-8", errors="ignore"), cam=cam)
            if parsed is not None:
                return parsed
    return np.eye(4, dtype=np.float64)


def parse_camchain_T_BS(text: str, cam: str = "cam0") -> np.ndarray | None:
    """Body(IMU)-from-camera T_BS = inv(T_cam_imu) from a Kalibr camchain.yaml."""
    import yaml

    block = (yaml.safe_load(text) or {}).get(cam) or {}
    t_cam_imu = block.get("T_cam_imu")
    if t_cam_imu is None:
        return None
    t_cam_imu = np.asarray(t_cam_imu, dtype=np.float64)
    if t_cam_imu.shape != (4, 4):
        return None
    return np.linalg.inv(t_cam_imu)


def R_cam_from_body(T_BS: np.ndarray) -> np.ndarray:
    """Rotate body/IMU vectors into the camera frame: v_cam = R v_body."""
    r_bs = np.asarray(T_BS, dtype=np.float64)[:3, :3]
    return r_bs.T.copy()


def rotate_accel_gyro(imu_accel_first: np.ndarray, r_cam_from_body: np.ndarray) -> np.ndarray:
    """Rotate (…, 6) [a, ω] with v_cam = R v_body."""
    imu = np.asarray(imu_accel_first, dtype=np.float64)
    r = np.asarray(r_cam_from_body, dtype=np.float64).reshape(3, 3)
    a = np.einsum("ij,...j->...i", r, imu[..., :3])
    w = np.einsum("ij,...j->...i", r, imu[..., 3:6])
    return np.concatenate([a, w], axis=-1)


def canonicalize_imu_table(
    imu_table: np.ndarray,
    seq_dir: Path | None = None,
    *,
    cam: str = "cam0",
    T_BS: np.ndarray | None = None,
    enabled: bool | None = None,
) -> np.ndarray:
    """
    ``imu_table`` is (N, 7) [t, gx,gy,gz, ax,ay,az] as on disk.

    Returns the same layout timestamp-first, but 6-vector is KITTI-like
    [ax,ay,az, wx,wy,wz] in the camera frame when ``enabled``.
    """
    imu_table = np.asarray(imu_table, dtype=np.float64)
    if not imu_canonical_enabled(enabled) or imu_table.size == 0:
        return imu_table
    out = imu_table.copy()
    acc_first = gyro_first_to_accel_first(out[:, 1:7])
    if T_BS is None:
        T_BS = load_T_BS(seq_dir, cam=cam) if seq_dir is not None else np.eye(4)
    out[:, 1:7] = rotate_accel_gyro(acc_first, R_cam_from_body(T_BS))
    return out


def body_pose_to_camera(T_WB: np.ndarray, T_BS: np.ndarray) -> np.ndarray:
    """T_WC = T_WB @ T_BS (EuRoC body = IMU)."""
    return np.asarray(T_WB, dtype=np.float64) @ np.asarray(T_BS, dtype=np.float64)


def gravity_in_camera(R_WC: np.ndarray, g_world: np.ndarray = G_WORLD) -> np.ndarray:
    return np.asarray(R_WC, dtype=np.float64)[:3, :3].T @ np.asarray(g_world, dtype=np.float64)


def velocity_world_to_camera(R_WC: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    return np.asarray(R_WC, dtype=np.float64)[:3, :3].T @ np.asarray(v_world, dtype=np.float64)
