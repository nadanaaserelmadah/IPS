"""TUM-VI dataset loading, time sync, and pose utilities."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

from src.utils.euroc_utils import (
    _read_csv_numeric,
    get_relative_pose_6DoF,
    interpolate_gt_pose,
    resample_imu_window,
    rotation_error,
)

_GT_DIR_CANDIDATES = (
    "state_groundtruth_estimate0",  # calibrated EuRoC-style export (preferred)
    "state_groundtruth_estimate_0",
    "mocap0",
)


def resolve_sequence_root(seq_dir: Path) -> Path:
    """
    Return directory containing cam0/imu0/GT.

    Supports:
      dataset-room1/cam0/...
      dataset-room1/mav0/cam0/...          (EuRoC-style)
      dataset-room1/dataset-room1/mav0/... (Kaggle double-nested zip)
    """
    seq_dir = Path(seq_dir)

    def _sensor_root(base: Path) -> Path | None:
        if (base / "cam0" / "data.csv").is_file():
            return base
        mav0 = base / "mav0"
        if (mav0 / "cam0" / "data.csv").is_file():
            return mav0
        return None

    bases: List[Path] = [seq_dir]
    inner = seq_dir / seq_dir.name
    if inner.is_dir():
        bases.append(inner)

    for base in bases:
        hit = _sensor_root(base)
        if hit is not None:
            return hit

    raise FileNotFoundError(
        f"No TUM-VI sensors under {seq_dir}. "
        f"Expected cam0/data.csv or .../mav0/cam0/data.csv "
        f"(also supports {seq_dir.name}/{seq_dir.name}/mav0/)"
    )


def _offset_search_dirs(seq_dir: Path) -> List[Path]:
    """Directories that may hold cam–IMU offset calibration files."""
    seq_dir = Path(seq_dir)
    dirs: List[Path] = [seq_dir]
    inner = seq_dir / seq_dir.name
    if inner.is_dir():
        dirs.append(inner)
    try:
        root = resolve_sequence_root(seq_dir)
        dirs.append(root)
        if root.name == "mav0":
            dirs.append(root.parent)
    except FileNotFoundError:
        pass
    seen: set[Path] = set()
    out: List[Path] = []
    for d in dirs:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            out.append(d)
    return out


def _read_cam_entries_from_root(root: Path, cam: str = "cam0") -> Tuple[np.ndarray, List[Path]]:
    """Load camera timestamps and image paths from an already-resolved sensor root."""
    cam_dir = root / cam / "data"
    csv_path = root / cam / "data.csv"
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


def resolve_gt_csv(seq_dir: Path, gt_dir: str | None = None) -> Path:
    seq_dir = Path(seq_dir)
    root = resolve_sequence_root(seq_dir)
    if gt_dir is not None:
        path = root / gt_dir / "data.csv"
        if path.is_file():
            return path
        raise FileNotFoundError(f"TUM-VI GT not found: {path}")

    for name in _GT_DIR_CANDIDATES:
        path = root / name / "data.csv"
        if path.is_file():
            return path

    dirs = sorted(p.name for p in root.iterdir() if p.is_dir())
    raise FileNotFoundError(
        f"No TUM-VI ground truth under {seq_dir}. "
        f"Looked for {_GT_DIR_CANDIDATES}. Subdirectories: {dirs}"
    )


def _parse_yaml_float_key(path: Path, key: str) -> float | None:
    """Read ``key: value`` from Kalibr-style YAML (seconds)."""
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or ":" not in line:
            continue
        k, v = line.split(":", 1)
        if k.strip() != key:
            continue
        val = v.strip().split("#")[0].strip()
        if val:
            return float(val)
    return None


def load_cam_imu_offset_ns(seq_dir: Path) -> int:
    """
    Camera–IMU time offset in nanoseconds (TUM-VI / Kalibr calibration).

    ``t_imu = t_cam + timeshift_cam_imu``  →  add shift (seconds) to camera timestamps.
    """
    env = os.environ.get("TUM_VI_CAM_IMU_OFFSET_NS")
    if env is not None:
        return int(env)

    for root in _offset_search_dirs(seq_dir):
        for name in (
            "cam_to_imu_time_offset_ns.txt",
            "cam_imu_timeoffset_ns.txt",
            "cam_imu_offset_ns.txt",
        ):
            path = root / name
            if path.is_file():
                return int(path.read_text().strip().split()[0])

        for yaml_name in ("camchain.yaml", "imu-cam.yaml", "cam_imu.yaml"):
            yaml_path = root / yaml_name
            for key in ("timeshift_cam_imu", "cam_over_imu_timeshift", "cam_over_imu_offset"):
                val = _parse_yaml_float_key(yaml_path, key)
                if val is not None:
                    return int(val * 1e9)
    return 0


def load_mocap_cam_offset_ns(seq_dir: Path) -> int:
    """Optional shift applied to mocap timestamps (ns) for cam–mocap alignment."""
    env = os.environ.get("TUM_VI_MOCAP_CAM_OFFSET_NS")
    if env is not None:
        return int(env)
    return 0


def load_cam_entries(seq_dir: Path, cam: str = "cam0") -> Tuple[np.ndarray, List[Path]]:
    """Return (timestamps_ns, image_paths) with optional cam–IMU offset applied."""
    root = resolve_sequence_root(seq_dir)
    timestamps, paths = _read_cam_entries_from_root(root, cam=cam)
    offset = load_cam_imu_offset_ns(seq_dir)
    if offset != 0:
        timestamps = timestamps + offset
    return timestamps, paths


def load_imu(seq_dir: Path, imu: str = "imu0") -> np.ndarray:
    """Returns (N, 7): [timestamp_ns, gx, gy, gz, ax, ay, az]."""
    seq_dir = resolve_sequence_root(seq_dir)
    return _read_csv_numeric(seq_dir / imu / "data.csv")


def _prepare_mocap_gt(gt: np.ndarray) -> np.ndarray:
    """Sort by time and drop duplicate timestamps."""
    order = np.argsort(gt[:, 0])
    gt = gt[order]
    _, uniq = np.unique(gt[:, 0], return_index=True)
    return gt[np.sort(uniq)]


def _smooth_mocap_positions(gt: np.ndarray) -> np.ndarray:
    """Moving-average on position (raw MoCap marker noise)."""
    window = int(os.environ.get("TUM_VI_MOCAP_SMOOTH", "15"))
    if window <= 1 or len(gt) < window:
        return gt
    out = gt.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    for j in range(1, 4):
        out[:, j] = np.convolve(gt[:, j], kernel, mode="same")
    return out


def _sanitize_mocap_gt(
    gt: np.ndarray,
    max_speed_m_s: float = 3.0,
    max_jump_m: float = 1.0,
) -> np.ndarray:
    """
    Drop mocap rows with impossible jumps (marker loss / raw MoCap glitches).

    TUM-VI raw mocap can contain 100+ m spikes; this stabilizes GT integration.
    """
    if len(gt) < 2:
        return gt
    env_jump = os.environ.get("TUM_VI_MOCAP_MAX_JUMP_M")
    if env_jump is not None:
        max_jump_m = float(env_jump)
    env_speed = os.environ.get("TUM_VI_MOCAP_MAX_SPEED_M_S")
    if env_speed is not None:
        max_speed_m_s = float(env_speed)

    keep = [0]
    removed = 0
    for i in range(1, len(gt)):
        prev = keep[-1]
        dt = (gt[i, 0] - gt[prev, 0]) / 1e9
        if dt <= 0:
            removed += 1
            continue
        dp = float(np.linalg.norm(gt[i, 1:4] - gt[prev, 1:4]))
        speed = dp / dt
        if dp > max_jump_m or speed > max_speed_m_s:
            removed += 1
            continue
        keep.append(i)
    out = gt[keep]
    if removed > 0:
        print(
            f"WARN: TUM-VI mocap sanitization removed {removed}/{len(gt) - 1} "
            f"outlier rows (max_jump={max_jump_m} m, max_speed={max_speed_m_s} m/s)."
        )
    return out


def load_gt_poses(seq_dir: Path, gt: str | None = None) -> np.ndarray:
    """Returns (N, 8): [timestamp_ns, px, py, pz, qw, qx, qy, qz]."""
    data = _read_csv_numeric(resolve_gt_csv(seq_dir, gt_dir=gt))
    gt_arr = _prepare_mocap_gt(data[:, :8].copy())
    mocap_shift = load_mocap_cam_offset_ns(seq_dir)
    if mocap_shift != 0:
        gt_arr[:, 0] = gt_arr[:, 0] + mocap_shift
    gt_arr = _smooth_mocap_positions(gt_arr)
    if os.environ.get("TUM_VI_SKIP_MOCAP_SANITIZE", "").lower() not in ("1", "true", "yes"):
        gt_arr = _sanitize_mocap_gt(gt_arr)
    return gt_arr


def raw_mocap_stats(seq_dir: Path) -> dict:
    """Stats on unprocessed mocap CSV (before smooth/sanitize)."""
    data = _read_csv_numeric(resolve_gt_csv(seq_dir))
    gt = _prepare_mocap_gt(data[:, :8].copy())
    if len(gt) < 2:
        return {"n_mocap": len(gt)}
    dps = np.linalg.norm(np.diff(gt[:, 1:4], axis=0), axis=1)
    dts = np.diff(gt[:, 0]) / 1e9
    speeds = dps / np.maximum(dts, 1e-9)
    span = np.linalg.norm(gt[:, 1:4].max(axis=0) - gt[:, 1:4].min(axis=0))
    return {
        "n_mocap": len(gt),
        "gt_source": resolve_gt_csv(seq_dir).parent.name,
        "median_mocap_step_m": float(np.median(dps)),
        "max_mocap_step_m": float(np.max(dps)),
        "median_mocap_speed_m_s": float(np.median(speeds)),
        "max_mocap_speed_m_s": float(np.max(speeds)),
        "position_span_m": float(span),
    }


def _filter_cam_to_gt_range(
    timestamps: np.ndarray, image_paths: List[Path], gt: np.ndarray
) -> Tuple[np.ndarray, List[Path]]:
    """Keep camera frames whose timestamps fall inside mocap/GT span."""
    t0, t1 = gt[0, 0], gt[-1, 0]
    mask = (timestamps >= t0) & (timestamps <= t1)
    n_drop = int((~mask).sum())
    if n_drop > 0:
        print(
            f"WARN: dropping {n_drop}/{len(timestamps)} camera frames "
            f"outside GT time [{t0}, {t1}]"
        )
    if not mask.any():
        raise RuntimeError(
            "No camera timestamps overlap mocap GT. "
            "Use calibrated TUM-VI extract or set TUM_VI_MOCAP_CAM_OFFSET_NS."
        )
    idx = np.where(mask)[0]
    return timestamps[idx], [image_paths[i] for i in idx]


def gt_path_stats(seq_dir: Path, cam_stride: int = 2) -> dict:
    """Diagnostic: GT path length from mocap **world positions** (not Euler 6DoF norms)."""
    timestamps, image_paths = load_cam_entries(seq_dir)
    gt = load_gt_poses(seq_dir)
    timestamps, image_paths = _filter_cam_to_gt_range(timestamps, image_paths, gt)
    timestamps = timestamps[::cam_stride]
    duration_s = float((timestamps[-1] - timestamps[0]) / 1e9) if len(timestamps) > 1 else 0.0

    gt_xyz = np.array(
        [interpolate_gt_pose(gt, int(t))[:3, 3] for t in timestamps], dtype=np.float64
    )
    steps = np.linalg.norm(np.diff(gt_xyz, axis=0), axis=1)
    path_len = float(steps.sum())
    cam_off = load_cam_imu_offset_ns(seq_dir)
    return {
        "n_cam": len(timestamps),
        "duration_s": duration_s,
        "cam_t_ns": (int(timestamps[0]), int(timestamps[-1])),
        "gt_t_ns": (int(gt[0, 0]), int(gt[-1, 0])),
        "cam_imu_offset_ns": cam_off,
        "gt_path_len_m": path_len,
        "mean_step_m": float(steps.mean()) if len(steps) else 0.0,
        "max_step_m": float(steps.max()) if len(steps) else 0.0,
        "cam_in_gt_span_frac": 1.0,
    }


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
    max_mocap_gap_ms: float | None = None,
) -> Tuple[List[dict], List[float]]:
    """
    Build sliding-window samples (VIFT-compatible).

    TUM-VI ``*_512_16`` sequences are ~16 Hz; ``cam_stride=2`` → ~8 Hz windows.
    IMU is canonicalized unless IPS_IMU_LEGACY=1.

    Mocap only covers the mocap room: windows whose span has a mocap gap longer than
    ``max_mocap_gap_ms`` (env TUM_VI_MAX_MOCAP_GAP_MS, default 50) are dropped, since
    their interpolated GT is a straight line across the gap (near-zero motion).
    """
    from src.utils.euroc_utils import interpolate_gt_velocity_world
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
    gt = load_gt_poses(seq_dir, gt=gt_name)
    align = imu_canonical_enabled(camera_frame_gt if camera_frame_gt is not None else canonical_imu)
    t_bs = load_T_BS(seq_dir, cam=cam) if align else np.eye(4)
    if max_step_m is None:
        max_step_m = float(os.environ.get("TUM_VI_MOCAP_MAX_JUMP_M", "2.0"))
    if max_mocap_gap_ms is None:
        max_mocap_gap_ms = float(os.environ.get("TUM_VI_MAX_MOCAP_GAP_MS", "50"))
    gt_ts = gt[:, 0]
    max_gap_ns = max_mocap_gap_ms * 1e6

    timestamps, image_paths = _filter_cam_to_gt_range(timestamps, image_paths, gt)
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
        t_start = int(timestamps[i])
        t_end = int(timestamps[i + sequence_length - 1])
        # mocap rows bracketing [t_start, t_end]; every gap between them must be short
        a = int(np.searchsorted(gt_ts, t_start, side="right")) - 1
        b = int(np.searchsorted(gt_ts, t_end, side="left"))
        if a < 0 or b >= len(gt_ts) or np.diff(gt_ts[a : b + 1]).max() > max_gap_ns:
            continue
        img_samples = image_paths[i : i + sequence_length]
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
    """Return sequence folder names under TUM-VI root."""
    root = Path(root)
    out: List[str] = []
    if not root.is_dir():
        return out
    for p in root.iterdir():
        if not p.is_dir():
            continue
        try:
            resolve_sequence_root(p)
            out.append(p.name)
        except FileNotFoundError:
            continue
    return sorted(out)


# Common TUM-VI 512_16 splits (subset — skip missing on partial mirrors).
TUM_VI_TRAIN_SEQS_DEFAULT = (
    "dataset-room1_512_16,dataset-room2_512_16,dataset-corridor1_512_16"
)
TUM_VI_VAL_SEQS_DEFAULT = "dataset-corridor2_512_16,dataset-room3_512_16"
TUM_VI_TEST_SEQS_DEFAULT = "dataset-magistrale1_512_16,dataset-slides1_512_16"


def filter_tum_vi_sequences(
    root: Union[str, Path],
    requested: Sequence[str],
    *,
    strict: bool = False,
) -> List[str]:
    """Keep only sequences present under ``root``."""
    root = Path(root)
    available = set(list_sequences(root))
    req = [str(s).strip() for s in requested if str(s).strip()]
    found = [s for s in req if s in available]
    missing = [s for s in req if s not in available]
    if missing:
        msg = (
            f"Skipping missing TUM-VI sequences: {missing}. "
            f"Available: {sorted(available)}"
        )
        if strict:
            raise FileNotFoundError(msg)
        print(f"WARN: {msg}")
    if not found:
        raise FileNotFoundError(
            f"No requested TUM-VI sequences under {root}. "
            f"Requested: {req}. Available: {sorted(available)}"
        )
    return found
