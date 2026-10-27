"""Phase 6-E: ORB triangulation + reprojection BA (Level-B geometric constraints).

Stronger than 6-D essential/pose-graph-only: landmarks in 3D with pixel residuals.
Still not full ORB-SLAM3 (no covisibility map / local mapping thread / DBoW).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Phase 6-E requires OpenCV. Install: pip install opencv-python-headless"
    ) from exc

from src.phase6.imu_preintegration import so3_exp, so3_log
from src.phase6.loop_closure import (
    detect_spatial_loop_candidates,
    load_euroc_cam0_K,
    map_pose_index_to_image,
)
from src.phase6.pose_graph_6d import absolute_se3_to_relative
from src.phase6.window_smoother import relative_poses_to_xyz
from src.utils.euroc_utils import euler_to_rotation_matrix, load_cam_entries, path_accu


@dataclass
class LandmarkTrack:
    """World landmark with multi-view pixel observations."""

    Xw: np.ndarray  # (3,)
    pose_idxs: list[int] = field(default_factory=list)
    uvs: list[np.ndarray] = field(default_factory=list)  # each (2,)


def _abs_Rp(rel: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    mats = path_accu(np.asarray(rel, dtype=np.float64))
    Rs = [T[:3, :3].copy() for T in mats]
    ps = np.array([T[:3, 3] for T in mats], dtype=np.float64)
    return Rs, ps


def _projection_matrix(K: np.ndarray, R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """3x4 projection: π(X) = K R^T (X - p) for world-from-camera (R,p)."""
    Rt = R.T
    return K @ np.hstack([Rt, (-Rt @ p.reshape(3, 1))])


def project_point(
    K: np.ndarray, R: np.ndarray, p: np.ndarray, Xw: np.ndarray
) -> Optional[np.ndarray]:
    xc = R.T @ (Xw - p)
    if float(xc[2]) < 0.05:
        return None
    uv = K @ xc
    return (uv[:2] / uv[2]).astype(np.float64)


def _orb_match_points(
    img_a: np.ndarray,
    img_b: np.ndarray,
    K: np.ndarray,
    *,
    max_features: int = 1200,
    ratio: float = 0.75,
    min_matches: int = 25,
    min_inliers: int = 18,
) -> Optional[tuple[np.ndarray, np.ndarray, int]]:
    """Return inlier pixel matches (N,2) each in images a,b."""
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
    mask = mask.ravel().astype(bool)
    if int(mask.sum()) < min_inliers:
        return None
    return pts_a[mask], pts_b[mask], int(mask.sum())


def triangulate_pair(
    K: np.ndarray,
    Ri: np.ndarray,
    pi: np.ndarray,
    Rj: np.ndarray,
    pj: np.ndarray,
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    *,
    max_reproj_px: float = 3.0,
    min_depth: float = 0.3,
    max_depth: float = 80.0,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Triangulate matches; return list of (Xw, uv_i, uv_j) passing cheirality/reproj gates.
    """
    pts_i = np.asarray(pts_i, dtype=np.float64).reshape(-1, 2)
    pts_j = np.asarray(pts_j, dtype=np.float64).reshape(-1, 2)
    Pi = _projection_matrix(K, Ri, pi)
    Pj = _projection_matrix(K, Rj, pj)
    Xh = cv2.triangulatePoints(
        Pi.astype(np.float64),
        Pj.astype(np.float64),
        pts_i.T.copy(),
        pts_j.T.copy(),
    )
    w = Xh[3].copy()
    w[np.abs(w) < 1e-12] = 1e-12
    X = (Xh[:3] / w).T
    out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for k in range(len(X)):
        Xw = np.asarray(X[k], dtype=np.float64).reshape(3)
        uvi = project_point(K, Ri, pi, Xw)
        uvj = project_point(K, Rj, pj, Xw)
        if uvi is None or uvj is None:
            continue
        di = float((Ri.T @ (Xw - pi))[2])
        dj = float((Rj.T @ (Xw - pj))[2])
        if di < min_depth or dj < min_depth or di > max_depth or dj > max_depth:
            continue
        if float(np.linalg.norm(uvi - pts_i[k])) > max_reproj_px:
            continue
        if float(np.linalg.norm(uvj - pts_j[k])) > max_reproj_px:
            continue
        out.append((Xw, pts_i[k].copy(), pts_j[k].copy()))
    return out


def _load_gray(
    pose_idx: int,
    t_ends_ns: np.ndarray,
    first_start_ns: int,
    cam_ts: np.ndarray,
    cam_paths: list[Path],
) -> Optional[np.ndarray]:
    path = map_pose_index_to_image(
        pose_idx, t_ends_ns, first_start_ns, cam_ts, cam_paths
    )
    if path is None or not path.is_file():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def build_landmarks(
    seq_dir: Path,
    Rs: list[np.ndarray],
    ps: np.ndarray,
    t_ends_ns: np.ndarray,
    first_start_ns: int,
    *,
    cam_stride: int = 2,
    keyframe_step: int = 10,
    max_landmarks: int = 2500,
    max_per_pair: int = 40,
    min_loop_sep: int = 80,
    max_loop_dist_m: float = 3.0,
    max_loop_pairs: int = 20,
) -> list[LandmarkTrack]:
    """Triangulate ORB landmarks on consecutive keyframes + spatial loop pairs."""
    cam_ts, cam_paths = load_cam_entries(seq_dir)
    cam_ts = cam_ts[::cam_stride]
    cam_paths = cam_paths[::cam_stride]
    K = load_euroc_cam0_K(seq_dir)
    n = len(Rs)
    kfs = list(range(0, n, keyframe_step))
    if kfs[-1] != n - 1:
        kfs.append(n - 1)

    landmarks: list[LandmarkTrack] = []

    def add_pair(i: int, j: int) -> None:
        if len(landmarks) >= max_landmarks:
            return
        img_i = _load_gray(i, t_ends_ns, first_start_ns, cam_ts, cam_paths)
        img_j = _load_gray(j, t_ends_ns, first_start_ns, cam_ts, cam_paths)
        if img_i is None or img_j is None:
            return
        got = _orb_match_points(img_i, img_j, K)
        if got is None:
            return
        pts_i, pts_j, _ = got
        tris = triangulate_pair(K, Rs[i], ps[i], Rs[j], ps[j], pts_i, pts_j)
        # Prefer closer / better distributed: take up to max_per_pair
        if len(tris) > max_per_pair:
            # Keep mid-depth band
            depths = [float((Rs[i].T @ (X - ps[i]))[2]) for X, _, _ in tris]
            order = np.argsort(np.abs(np.asarray(depths) - np.median(depths)))
            tris = [tris[int(k)] for k in order[:max_per_pair]]
        for Xw, uvi, uvj in tris:
            if len(landmarks) >= max_landmarks:
                break
            landmarks.append(
                LandmarkTrack(Xw=Xw, pose_idxs=[i, j], uvs=[uvi, uvj])
            )

    for a, b in zip(kfs[:-1], kfs[1:]):
        if b - a < 2:
            continue
        add_pair(a, b)

    xyz = ps
    cands = detect_spatial_loop_candidates(
        xyz, min_sep=min_loop_sep, max_dist_m=max_loop_dist_m
    )
    scored = sorted(
        ((float(np.linalg.norm(xyz[i] - xyz[j])), i, j) for i, j in cands)
    )
    n_loop_pairs = 0
    for _, i, j in scored:
        if n_loop_pairs >= max_loop_pairs or len(landmarks) >= max_landmarks:
            break
        before = len(landmarks)
        add_pair(i, j)
        if len(landmarks) > before:
            n_loop_pairs += 1

    return landmarks


def _pack(dtheta: np.ndarray, dp: np.ndarray, dX: np.ndarray) -> np.ndarray:
    return np.concatenate([dtheta.reshape(-1), dp.reshape(-1), dX.reshape(-1)])


def _unpack(
    x: np.ndarray, n_poses: int, n_lm: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dtheta = x[: 3 * n_poses].reshape(n_poses, 3)
    dp = x[3 * n_poses : 6 * n_poses].reshape(n_poses, 3)
    dX = x[6 * n_poses :].reshape(n_lm, 3)
    return dtheta, dp, dX


def landmark_ba_residuals(
    x: np.ndarray,
    *,
    R0: list[np.ndarray],
    p0: np.ndarray,
    X0: np.ndarray,
    rel_odom: np.ndarray,
    landmarks: list[LandmarkTrack],
    K: np.ndarray,
    sigma_odom_rot: float,
    sigma_odom_pos: float,
    sigma_pix: float,
    sigma_lm_prior: float,
) -> np.ndarray:
    n = len(R0)
    n_lm = len(landmarks)
    dtheta, dp, dX = _unpack(x, n, n_lm)
    Rs = [R0[i] @ so3_exp(dtheta[i]) for i in range(n)]
    ps = p0 + dp
    Xs = X0 + dX
    res: list[np.ndarray] = []

    # Gauge
    res.append(dtheta[0] / 1e-3)
    res.append(dp[0] / 1e-3)

    # Network odometry
    for k in range(n - 1):
        Ri, Rj = Rs[k], Rs[k + 1]
        R_net = euler_to_rotation_matrix(rel_odom[k, :3])
        t_net = rel_odom[k, 3:]
        res.append(so3_log(R_net.T @ (Ri.T @ Rj)) / sigma_odom_rot)
        res.append((Ri.T @ (ps[k + 1] - ps[k]) - t_net) / sigma_odom_pos)

    # Soft prior on landmark motion (keeps triangulation seed stable)
    res.append((dX / sigma_lm_prior).reshape(-1))

    # Reprojection
    for li, lm in enumerate(landmarks):
        Xw = Xs[li]
        for pose_idx, uv in zip(lm.pose_idxs, lm.uvs):
            pred = project_point(K, Rs[pose_idx], ps[pose_idx], Xw)
            if pred is None:
                res.append(np.array([10.0, 10.0]) / sigma_pix)
            else:
                res.append((pred - uv) / sigma_pix)

    return np.concatenate(res)


def optimize_landmark_ba(
    rel_odom: np.ndarray,
    landmarks: list[LandmarkTrack],
    K: np.ndarray,
    *,
    max_nfev: int = 25,
    sigma_odom_rot: float = 0.05,
    sigma_odom_pos: float = 0.12,
    sigma_pix: float = 1.5,
    sigma_lm_prior: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    R0, p0 = _abs_Rp(rel_odom)
    n = len(R0)
    n_lm = len(landmarks)
    if n_lm == 0:
        return R0, p0
    X0 = np.stack([lm.Xw for lm in landmarks], axis=0)
    x0 = np.zeros(6 * n + 3 * n_lm, dtype=np.float64)

    def fun(x: np.ndarray) -> np.ndarray:
        return landmark_ba_residuals(
            x,
            R0=R0,
            p0=p0,
            X0=X0,
            rel_odom=rel_odom,
            landmarks=landmarks,
            K=K,
            sigma_odom_rot=sigma_odom_rot,
            sigma_odom_pos=sigma_odom_pos,
            sigma_pix=sigma_pix,
            sigma_lm_prior=sigma_lm_prior,
        )

    sol = least_squares(
        fun,
        x0,
        method="trf",
        max_nfev=max_nfev,
        ftol=1e-5,
        xtol=1e-5,
        gtol=1e-5,
        verbose=0,
    )
    dtheta, dp, dX = _unpack(sol.x, n, n_lm)
    Rs = [R0[i] @ so3_exp(dtheta[i]) for i in range(n)]
    ps = p0 + dp
    # Write back refined landmark seeds (for diagnostics)
    for li, lm in enumerate(landmarks):
        lm.Xw = X0[li] + dX[li]
    return Rs, ps


def refine_trajectory_6e(
    rel_poses: np.ndarray,
    seq_dir,
    t_ends_ns: np.ndarray,
    first_start_ns: int,
    *,
    cam_stride: int = 2,
    keyframe_step: int = 10,
    max_landmarks: int = 2500,
    max_per_pair: int = 40,
    min_loop_sep: int = 80,
    max_loop_dist_m: float = 3.0,
    max_loop_pairs: int = 20,
    max_nfev: int = 25,
    sigma_odom_pos: float = 0.12,
    sigma_pix: float = 1.5,
) -> dict:
    """
    Phase 6-E Level-B: triangulate ORB landmarks, then pose+landmark BA with reprojection.
    """
    from pathlib import Path

    rel = np.asarray(rel_poses, dtype=np.float64)
    R0, p0 = _abs_Rp(rel)
    print(
        f"  6-E: building landmarks on {len(R0)} poses "
        f"(kf_step={keyframe_step}, max_lm={max_landmarks}) ..."
    )
    landmarks = build_landmarks(
        Path(seq_dir),
        R0,
        p0,
        np.asarray(t_ends_ns),
        int(first_start_ns),
        cam_stride=cam_stride,
        keyframe_step=keyframe_step,
        max_landmarks=max_landmarks,
        max_per_pair=max_per_pair,
        min_loop_sep=min_loop_sep,
        max_loop_dist_m=max_loop_dist_m,
        max_loop_pairs=max_loop_pairs,
    )
    K = load_euroc_cam0_K(Path(seq_dir))
    print(f"  6-E: landmarks={len(landmarks)}; optimizing ...")
    Rs, ps = optimize_landmark_ba(
        rel,
        landmarks,
        K,
        max_nfev=max_nfev,
        sigma_odom_pos=sigma_odom_pos,
        sigma_pix=sigma_pix,
    )
    fused = absolute_se3_to_relative(Rs, ps)
    xyz = relative_poses_to_xyz(fused)
    return {
        "relative_poses": fused,
        "xyz": xyz,
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))),
        "n_landmarks": len(landmarks),
        "n_observations": int(sum(len(lm.pose_idxs) for lm in landmarks)),
    }
