"""EuRoC MAV dataset loading, time sync, and pose utilities."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import math

import numpy as np

_EPS = np.finfo(float).eps * 4.0


def load_image_rgb(path) -> np.ndarray:
    """(H, W, 3) uint8. 16-bit grayscale PNGs (TUM-VI ``*_512_16``) are scaled to 8 bits;
    PIL's convert("RGB") clips them instead, which turns every TUM-VI frame white."""
    from PIL import Image

    im = Image.open(path)
    if im.mode.startswith("I"):
        a = (np.asarray(im).astype(np.int64) >> 8).clip(0, 255).astype(np.uint8)
        return np.repeat(a[..., None], 3, axis=2)
    return np.asarray(im.convert("RGB"))


def euler_from_matrix(matrix: np.ndarray) -> np.ndarray:
    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    cy = math.sqrt(M[0, 0] * M[0, 0] + M[1, 0] * M[1, 0])
    ay = math.atan2(-M[2, 0], cy)
    if ay < -math.pi / 2 + _EPS and ay > -math.pi / 2 - _EPS:
        ax, az = 0.0, math.atan2(-M[1, 2], -M[0, 2])
    elif ay < math.pi / 2 + _EPS and ay > math.pi / 2 - _EPS:
        ax, az = 0.0, math.atan2(M[1, 2], M[0, 2])
    else:
        ax = math.atan2(M[2, 1], M[2, 2])
        az = math.atan2(M[1, 0], M[0, 0])
    return np.array([ax, ay, az])


def get_relative_pose(rt1: np.ndarray, rt2: np.ndarray) -> np.ndarray:
    return np.linalg.inv(rt1) @ rt2


def get_relative_pose_6DoF(rt1: np.ndarray, rt2: np.ndarray) -> np.ndarray:
    rt_rel = get_relative_pose(rt1, rt2)
    x, y, z = euler_from_matrix(rt_rel[:3, :3])
    return np.concatenate(([x, y, z], rt_rel[:3, 3]))


def rotation_error(rt1: np.ndarray, rt2: np.ndarray) -> float:
    pose_error = get_relative_pose(rt1, rt2)
    d = 0.5 * (pose_error[0, 0] + pose_error[1, 1] + pose_error[2, 2] - 1.0)
    return float(np.arccos(max(min(d, 1.0), -1.0)))


def euler_to_rotation_matrix(theta: np.ndarray) -> np.ndarray:
    rx = np.array(
        [
            [1, 0, 0],
            [0, np.cos(theta[0]), -np.sin(theta[0])],
            [0, np.sin(theta[0]), np.cos(theta[0])],
        ]
    )
    ry = np.array(
        [
            [np.cos(theta[1]), 0, np.sin(theta[1])],
            [0, 1, 0],
            [-np.sin(theta[1]), 0, np.cos(theta[1])],
        ]
    )
    rz = np.array(
        [
            [np.cos(theta[2]), -np.sin(theta[2]), 0],
            [np.sin(theta[2]), np.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    return rz @ ry @ rx


def pose_6dof_to_matrix(pose: np.ndarray) -> np.ndarray:
    r = euler_to_rotation_matrix(pose[:3])
    t = pose[3:].reshape(3, 1)
    rt = np.concatenate((r, t), axis=1)
    return np.concatenate((rt, np.array([[0, 0, 0, 1]])), axis=0)


def path_accu(relative_6dof: np.ndarray) -> list:
    answer = [np.eye(4)]
    for i in range(relative_6dof.shape[0]):
        rt_rel = pose_6dof_to_matrix(relative_6dof[i])
        answer.append(answer[-1] @ rt_rel)
    return answer


def _read_csv_numeric(path: Path) -> np.ndarray:
    """Load EuRoC-style CSV (skips # comment lines)."""
    rows: List[List[float]] = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            rows.append([float(x) for x in row])
    return np.asarray(rows, dtype=np.float64)


_GT_DIR_CANDIDATES = (
    "state_groundtruth_estimate0",  # official EuRoC / evo
    "state_groundtruth_estimate_0",  # IPS synthetic layout
)


def resolve_sequence_root(seq_dir: Path) -> Path:
    """
    Return directory containing cam0/imu0/GT.

    Supports:
      MH_01_easy/cam0/...           (IPS flat layout)
      MH_01_easy/mav0/cam0/...      (official EuRoC MAV folder)
    """
    seq_dir = Path(seq_dir)
    if (seq_dir / "cam0" / "data.csv").is_file():
        return seq_dir
    mav0 = seq_dir / "mav0"
    if (mav0 / "cam0" / "data.csv").is_file():
        return mav0
    raise FileNotFoundError(
        f"No EuRoC sensors under {seq_dir}. Expected cam0/data.csv or mav0/cam0/data.csv"
    )


def resolve_gt_csv(seq_dir: Path, gt_dir: str | None = None) -> Path:
    """
    Locate ground-truth data.csv for a EuRoC sequence.

    Official releases use ``state_groundtruth_estimate0`` (no underscore before 0).
    IPS synthetic data uses ``state_groundtruth_estimate_0``.
    """
    seq_dir = Path(seq_dir)
    if gt_dir is not None:
        root = resolve_sequence_root(seq_dir)
        path = root / gt_dir / "data.csv"
        if path.is_file():
            return path
        raise FileNotFoundError(f"EuRoC GT not found: {path}")

    bases: List[Path] = []
    root = resolve_sequence_root(seq_dir)
    bases.append(root)
    if root.name == "mav0":
        bases.append(root.parent)

    for base in bases:
        for name in _GT_DIR_CANDIDATES:
            path = base / name / "data.csv"
            if path.is_file():
                return path

    dirs = sorted(
        {
            p.name
            for base in bases
            if base.is_dir()
            for p in base.iterdir()
            if p.is_dir()
        }
    )
    raise FileNotFoundError(
        f"No EuRoC ground truth under {seq_dir}. "
        f"Looked for {_GT_DIR_CANDIDATES} in {[str(b) for b in bases]}. "
        f"Subdirectories found: {dirs}"
    )


def load_cam_entries(seq_dir: Path, cam: str = "cam0") -> Tuple[np.ndarray, List[Path]]:
    """Return (timestamps_ns, image_paths) sorted by time."""
    seq_dir = resolve_sequence_root(seq_dir)
    cam_dir = seq_dir / cam / "data"
    csv_path = seq_dir / cam / "data.csv"
    timestamps: List[int] = []
    names: List[str] = []
    with open(csv_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = [p.strip() for p in line.strip().split(",")]
            timestamps.append(int(float(parts[0])))
            names.append(parts[1])
    timestamps_arr = np.asarray(timestamps, dtype=np.int64)
    paths = [cam_dir / n for n in names]
    order = np.argsort(timestamps_arr)
    return timestamps_arr[order], [paths[i] for i in order]


def load_imu(seq_dir: Path, imu: str = "imu0") -> np.ndarray:
    """
    Returns array (N, 7): [timestamp_ns, gx, gy, gz, ax, ay, az].
    """
    seq_dir = resolve_sequence_root(seq_dir)
    return _read_csv_numeric(seq_dir / imu / "data.csv")


def load_gt_poses(seq_dir: Path, gt: str | None = None) -> np.ndarray:
    """
    Returns (N, 8): [timestamp_ns, px, py, pz, qw, qx, qy, qz].

    If ``gt`` is None, auto-detects official ``state_groundtruth_estimate0`` or IPS
    ``state_groundtruth_estimate_0``. Extra CSV columns (velocity, biases) are ignored
    here; see ``load_gt_table``.
    """
    data = _read_csv_numeric(resolve_gt_csv(seq_dir, gt_dir=gt))
    return data[:, :8]


def load_gt_table(seq_dir: Path, gt: str | None = None) -> np.ndarray:
    """Full GT CSV including velocity when the EuRoC state estimate provides it."""
    return _read_csv_numeric(resolve_gt_csv(seq_dir, gt_dir=gt))


def quat_to_rotation_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Hamilton unit quaternion (w, x, y, z) -> 3x3 rotation (EuRoC q_RS, TUM-VI q_RS)."""
    n = qw * qw + qx * qx + qy * qy + qz * qz
    s = 2.0 / n if n > 0 else 0.0
    wx, wy, wz = s * qw * qx, s * qw * qy, s * qw * qz
    xx, xy, xz = s * qx * qx, s * qx * qy, s * qx * qz
    yy, yz, zz = s * qy * qy, s * qy * qz, s * qz * qz
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_matrix_from_gt_row(row: np.ndarray) -> np.ndarray:
    """4x4 world pose from GT [t, p, q]."""
    R = quat_to_rotation_matrix(row[4], row[5], row[6], row[7])
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = row[1:4]
    return T


def interpolate_gt_pose(gt: np.ndarray, t_ns: int) -> np.ndarray:
    """Interpolate 4x4 pose at timestamp t_ns (linear p, slerp q)."""
    ts = gt[:, 0]
    if t_ns <= ts[0]:
        return pose_matrix_from_gt_row(gt[0])
    if t_ns >= ts[-1]:
        return pose_matrix_from_gt_row(gt[-1])
    idx = np.searchsorted(ts, t_ns)
    t0, t1 = ts[idx - 1], ts[idx]
    a = (t_ns - t0) / (t1 - t0)
    p = (1 - a) * gt[idx - 1, 1:4] + a * gt[idx, 1:4]
    # SLERP for quaternion (qw, qx, qy, qz)
    q0 = gt[idx - 1, 4:8]
    q1 = gt[idx, 4:8]
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = q0 + a * (q1 - q0)
    q /= np.linalg.norm(q)
    row = np.concatenate([[t_ns], p, q])
    return pose_matrix_from_gt_row(row)


def interpolate_gt_velocity_world(gt: np.ndarray, t_ns: int) -> np.ndarray:
    """World-frame velocity: GT columns 8:11 when present, else finite-difference of p."""
    ts = gt[:, 0]
    if gt.shape[1] >= 11:
        if t_ns <= ts[0]:
            return np.asarray(gt[0, 8:11], dtype=np.float64)
        if t_ns >= ts[-1]:
            return np.asarray(gt[-1, 8:11], dtype=np.float64)
        idx = np.searchsorted(ts, t_ns)
        t0, t1 = ts[idx - 1], ts[idx]
        a = (t_ns - t0) / max(t1 - t0, 1.0)
        return (1 - a) * gt[idx - 1, 8:11] + a * gt[idx, 8:11]
    pose = interpolate_gt_pose(gt, t_ns)
    dt_ns = 10_000_000
    pose2 = interpolate_gt_pose(gt, int(t_ns + dt_ns))
    return (pose2[:3, 3] - pose[:3, 3]) / (dt_ns * 1e-9)


def resample_imu_window(
    imu: np.ndarray, t_start: int, t_end: int, n_samples: int
) -> np.ndarray:
    """
    Resample IMU to n_samples evenly spaced in [t_start, t_end].
    Returns (n_samples, 6) gyro+accel.
    """
    mask = (imu[:, 0] >= t_start) & (imu[:, 0] <= t_end)
    seg = imu[mask]
    if seg.shape[0] < 2:
        if seg.shape[0] == 1:
            return np.tile(seg[0, 1:7], (n_samples, 1))
        return np.zeros((n_samples, 6), dtype=np.float64)

    target_ts = np.linspace(t_start, t_end, n_samples)
    out = np.zeros((n_samples, 6), dtype=np.float64)
    for j in range(6):
        out[:, j] = np.interp(target_ts, seg[:, 0], seg[:, 1 + j])
    return out


def build_sequence_samples(
    seq_dir: Path,
    sequence_length: int = 11,
    cam_stride: int = 2,
    imu_freq: int = 10,
    cam: str = "cam0",
    imu_sensor: str = "imu0",
    gt_name: str | None = None,
    canonical_imu: bool | None = None,
    camera_frame_gt: bool | None = None,
    max_step_m: float | None = None,
) -> Tuple[List[dict], List[float]]:
    """
    Build sliding-window samples compatible with KITTI/VIFT format.

    Each sample: imgs, imus, gts (relative 6DoF), rot (segment rotation metric).
    IMU is canonicalized (accel-first, camera frame) unless IPS_IMU_LEGACY=1.
    """
    from src.utils.imu_canonical import (
        body_pose_to_camera,
        canonicalize_imu_table,
        gravity_in_camera,
        imu_canonical_enabled,
        load_T_BS,
        velocity_world_to_camera,
    )

    timestamps, image_paths = load_cam_entries(seq_dir, cam=cam)
    imu = canonicalize_imu_table(
        load_imu(seq_dir, imu=imu_sensor), seq_dir, cam=cam, enabled=canonical_imu
    )
    gt = load_gt_table(seq_dir, gt=gt_name)
    align = imu_canonical_enabled(camera_frame_gt if camera_frame_gt is not None else canonical_imu)
    t_bs = load_T_BS(seq_dir, cam=cam) if align else np.eye(4)
    if max_step_m is None:
        max_step_m = float(os.environ.get("EUROC_MAX_STEP_M", "2.0"))

    timestamps = timestamps[::cam_stride]
    image_paths = image_paths[::cam_stride]

    poses_abs: List[np.ndarray] = []
    vel_cam: List[np.ndarray] = []
    g_cam: List[np.ndarray] = []
    for t in timestamps:
        t_wb = interpolate_gt_pose(gt, int(t))
        t_wc = body_pose_to_camera(t_wb, t_bs) if align else t_wb
        poses_abs.append(t_wc)
        v_w = interpolate_gt_velocity_world(gt, int(t))
        vel_cam.append(velocity_world_to_camera(t_wc, v_w) if align else v_w)
        g_cam.append(gravity_in_camera(t_wc) if align else np.array([0.0, 0.0, -9.81]))

    poses_rel: List[np.ndarray] = []
    for i in range(1, len(poses_abs)):
        poses_rel.append(get_relative_pose_6DoF(poses_abs[i - 1], poses_abs[i]))

    imu_per_step = imu_freq
    samples: List[dict] = []
    weights: List[float] = []

    for i in range(len(image_paths) - sequence_length + 1):
        skip = False
        for j in range(i, i + sequence_length - 1):
            dp = float(np.linalg.norm(poses_abs[j + 1][:3, 3] - poses_abs[j][:3, 3]))
            if dp > max_step_m:
                skip = True
                break
        if skip:
            continue
        img_samples = image_paths[i : i + sequence_length]
        t_start = int(timestamps[i])
        t_end = int(timestamps[i + sequence_length - 1])
        n_imu = (sequence_length - 1) * imu_per_step + 1
        imu_samples = resample_imu_window(imu, t_start, t_end, n_imu)
        pose_rel_samples = poses_rel[i : i + sequence_length - 1]
        vel_samples = np.stack(vel_cam[i : i + sequence_length - 1], axis=0)
        g_samples = np.stack(g_cam[i : i + sequence_length - 1], axis=0)
        segment_rot = rotation_error(poses_abs[i], poses_abs[i + sequence_length - 1])
        gt_xyz_end = np.asarray(poses_abs[i + sequence_length - 1][:3, 3], dtype=np.float64)
        samples.append(
            {
                "imgs": img_samples,
                "imus": imu_samples,
                "gts": pose_rel_samples,
                "rot": segment_rot,
                "t_start": t_start,
                "t_end": t_end,
                "gt_xyz_end": gt_xyz_end,
                "vel_cam": vel_samples,
                "g_cam": g_samples,
            }
        )
        weights.append(1.0)

    return samples, weights


def list_sequences(root: Union[str, Path]) -> List[str]:
    """Return sequence folder names under EuRoC root."""
    root = Path(root)
    out: List[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            resolve_sequence_root(p)
            out.append(p.name)
        except FileNotFoundError:
            continue
    return sorted(out)


# Common 9-sequence Kaggle mirror (no V1_01_easy / V1_02_medium).
EUROC_TRAIN_SEQS_DEFAULT = "MH_01_easy,MH_02_easy,V2_01_easy"
EUROC_VAL_SEQS_DEFAULT = "MH_05_difficult,V2_02_medium,V1_03_difficult"
EUROC_TEST_SEQS_DEFAULT = "MH_05_difficult,V2_02_medium"

# Phase 5 leakage-free split (val != test; V2_03 excluded until GT diagnostics pass).
EUROC_PHASE5_TRAIN_SEQS = (
    "MH_01_easy,MH_02_easy,MH_03_medium,MH_04_difficult,V2_01_easy"
)
EUROC_PHASE5_VAL_SEQS = "V1_03_difficult"
EUROC_PHASE5_TEST_SEQS = "MH_05_difficult,V2_02_medium"

# SS-VIO / DB-VIO protocol (test MH_03, V1_03, V2_03). Recache / skip V2_03 if GT path is implausible.
EUROC_DBVIO_TRAIN_SEQS = (
    "MH_01_easy,MH_02_easy,MH_04_difficult,MH_05_difficult,V1_01_easy,V1_02_medium,V2_01_easy"
)
EUROC_DBVIO_VAL_SEQS = "V2_02_medium"
EUROC_DBVIO_TEST_SEQS = "MH_03_medium,V1_03_difficult,V2_03_difficult"


def filter_euroc_sequences(
    root: Union[str, Path],
    requested: Sequence[str],
    *,
    strict: bool = False,
) -> List[str]:
    """Keep only sequences present under ``root``; warn or raise on missing names."""
    root = Path(root)
    available = set(list_sequences(root))
    req = [str(s).strip() for s in requested if str(s).strip()]
    found = [s for s in req if s in available]
    missing = [s for s in req if s not in available]
    if missing:
        msg = (
            f"Skipping missing EuRoC sequences: {missing}. "
            f"Available: {sorted(available)}"
        )
        if strict:
            raise FileNotFoundError(msg)
        print(f"WARN: {msg}")
    if not found:
        raise FileNotFoundError(
            f"No requested EuRoC sequences under {root}. "
            f"Requested: {req}. Available: {sorted(available)}"
        )
    return found


def integrate_relative_poses(relative_6dof: np.ndarray) -> List[np.ndarray]:
    """Chain relative 6DoF poses into absolute 4x4 matrices (starts at identity)."""
    from src.utils.kitti_utils import path_accu

    traj = path_accu(relative_6dof)
    return traj
