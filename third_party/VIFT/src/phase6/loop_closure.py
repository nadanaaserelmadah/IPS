"""Phase 6-D: ORB loop detection + essential-matrix relative pose (optional landmarks cue)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Phase 6-D requires OpenCV. Install: pip install opencv-python-headless"
    ) from exc

from src.phase6.imu_preintegration import so3_exp, so3_log


@dataclass
class RelativeConstraint:
    """SE(3) relative pose from i to j expressed in frame i: R_ij, t_ij."""

    i: int
    j: int
    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)
    n_inliers: int
    kind: str  # "loop" | "visual_odom" | "spatial"


def load_euroc_cam0_K(seq_dir: Path) -> np.ndarray:
    """
    Load cam0 intrinsics from sensor.yaml if present; else EuRoC-ish defaults.
    """
    from src.utils.euroc_utils import resolve_sequence_root

    root = resolve_sequence_root(seq_dir)
    yaml_path = root / "cam0" / "sensor.yaml"
    # Defaults close to EuRoC cam0
    fx = fy = 458.654
    cx, cy = 367.215, 248.375
    if yaml_path.is_file():
        text = yaml_path.read_text(encoding="utf-8", errors="ignore")
        # Very small parser for "intrinsics: [fx, fy, cx, cy]"
        import re

        m = re.search(
            r"intrinsics:\s*\[\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)\s*,\s*([0-9.eE+-]+)",
            text,
        )
        if m:
            fx, fy, cx, cy = map(float, m.groups())
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _orb_match_pose(
    img_a: np.ndarray,
    img_b: np.ndarray,
    K: np.ndarray,
    *,
    max_features: int = 1500,
    ratio: float = 0.75,
    min_matches: int = 30,
    min_inliers: int = 20,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    """Return (R, t_unit, n_inliers) for a→b, or None."""
    if img_a.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    if img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=max_features)
    kpa, desa = orb.detectAndCompute(img_a, None)
    kpb, desb = orb.detectAndCompute(img_b, None)
    if desa is None or desb is None or len(kpa) < min_matches or len(kpb) < min_matches:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    knn = bf.knnMatch(desa, desb, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    if len(good) < min_matches:
        return None

    pts_a = np.float32([kpa[m.queryIdx].pt for m in good])
    pts_b = np.float32([kpb[m.trainIdx].pt for m in good])
    E, mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
    )
    if E is None or mask is None:
        return None
    inl = int(mask.sum())
    if inl < min_inliers:
        return None
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    inl2 = int(mask_pose.sum()) if mask_pose is not None else inl
    t = t.reshape(3).astype(np.float64)
    n = float(np.linalg.norm(t))
    if n < 1e-9:
        return None
    t = t / n  # unit translation (scale unknown from 2-view)
    return np.asarray(R, dtype=np.float64), t, max(inl2, inl)


def map_pose_index_to_image(
    pose_idx: int,
    t_ends_ns: np.ndarray,
    first_start_ns: int,
    cam_ts: np.ndarray,
    cam_paths: list[Path],
) -> Optional[Path]:
    """Associate path_accu pose index with a camera image path."""
    if pose_idx <= 0:
        t_query = int(first_start_ns)
    else:
        t_query = int(t_ends_ns[pose_idx - 1])
    # Nearest camera timestamp
    j = int(np.argmin(np.abs(cam_ts.astype(np.int64) - t_query)))
    if abs(int(cam_ts[j]) - t_query) > 50_000_000:  # 50 ms
        return None
    return cam_paths[j]


def detect_spatial_loop_candidates(
    xyz: np.ndarray,
    *,
    min_sep: int = 80,
    max_dist_m: float = 2.0,
    max_candidates_per_pose: int = 2,
) -> list[tuple[int, int]]:
    """Pose pairs (i,j) close in space but far in time/index."""
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    pairs: list[tuple[int, int]] = []
    for j in range(min_sep, n):
        d = np.linalg.norm(xyz[: j - min_sep + 1] - xyz[j], axis=1)
        idxs = np.where(d < max_dist_m)[0]
        if len(idxs) == 0:
            continue
        # Prefer nearest
        order = np.argsort(d[idxs])[:max_candidates_per_pose]
        for ii in idxs[order]:
            pairs.append((int(ii), int(j)))
    # Unique
    return sorted(set(pairs))


def build_visual_constraints(
    seq_dir: Path,
    xyz: np.ndarray,
    t_ends_ns: np.ndarray,
    first_start_ns: int,
    *,
    cam_stride: int = 2,
    keyframe_step: int = 10,
    min_loop_sep: int = 80,
    max_loop_dist_m: float = 2.5,
    max_loops: int = 40,
    add_visual_odom: bool = True,
) -> list[RelativeConstraint]:
    """
    Build loop (+ optional keyframe visual-odom) constraints via ORB+EssentialMat.

    Translation from 2-view is **up-to-scale**; callers should only use direction
    or scale it using the current pose-graph distance ||p_j - p_i||.
    """
    from src.utils.euroc_utils import load_cam_entries

    cam_ts, cam_paths = load_cam_entries(seq_dir)
    cam_ts = cam_ts[::cam_stride]
    cam_paths = cam_paths[::cam_stride]
    K = load_euroc_cam0_K(seq_dir)
    n = len(xyz)
    constraints: list[RelativeConstraint] = []

    def try_pair(i: int, j: int, kind: str) -> None:
        pa = map_pose_index_to_image(i, t_ends_ns, first_start_ns, cam_ts, cam_paths)
        pb = map_pose_index_to_image(j, t_ends_ns, first_start_ns, cam_ts, cam_paths)
        if pa is None or pb is None:
            return
        if not pa.is_file() or not pb.is_file():
            return
        img_a = cv2.imread(str(pa), cv2.IMREAD_GRAYSCALE)
        img_b = cv2.imread(str(pb), cv2.IMREAD_GRAYSCALE)
        if img_a is None or img_b is None:
            return
        got = _orb_match_pose(img_a, img_b, K)
        if got is None:
            return
        R, t_unit, inl = got
        # Scale translation with current estimate separation
        scale = float(np.linalg.norm(xyz[j] - xyz[i]))
        if scale < 1e-3:
            scale = 1.0
        # Align t_unit hemisphere with current Δp direction in frame i is done in optimizer
        constraints.append(RelativeConstraint(i, j, R, t_unit * scale, inl, kind))

    # Visual odometry between keyframes (landmark-match proxy)
    if add_visual_odom:
        kfs = list(range(0, n, keyframe_step))
        if kfs[-1] != n - 1:
            kfs.append(n - 1)
        for a, b in zip(kfs[:-1], kfs[1:]):
            if b - a < 2:
                continue
            try_pair(a, b, "visual_odom")

    # Loop candidates
    cands = detect_spatial_loop_candidates(
        xyz, min_sep=min_loop_sep, max_dist_m=max_loop_dist_m
    )
    # Rank by distance
    scored = []
    for i, j in cands:
        scored.append((float(np.linalg.norm(xyz[i] - xyz[j])), i, j))
    scored.sort()
    for _, i, j in scored[: max_loops * 3]:
        before = len(constraints)
        try_pair(i, j, "loop")
        if len(constraints) > before and sum(c.kind == "loop" for c in constraints) >= max_loops:
            break

    return constraints
