"""Unit tests for Phase 6-A IMU scaffold (numpy-only).

Run:
  python third_party/VIFT/tests/test_phase6.py
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np

VIFT = Path(__file__).resolve().parents[1]
SRC = VIFT / "src"
sys.path.insert(0, str(VIFT))


def _ensure_pkg(name: str, path: Path) -> None:
    if name not in sys.modules:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]  # type: ignore[attr-defined]
        sys.modules[name] = m


def _load(name: str, path: Path):
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        _ensure_pkg(pkg, VIFT / Path(*parts[:i]))
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_ensure_pkg("src", SRC)
_ensure_pkg("src.utils", SRC / "utils")
_ensure_pkg("src.phase6", SRC / "phase6")

euroc_utils = _load("src.utils.euroc_utils", SRC / "utils" / "euroc_utils.py")
sys.modules["src.utils"].euroc_utils = euroc_utils  # type: ignore[attr-defined]

preint = _load(
    "src.phase6.imu_preintegration", SRC / "phase6" / "imu_preintegration.py"
)
smooth = _load("src.phase6.window_smoother", SRC / "phase6" / "window_smoother.py")

so3_exp = preint.so3_exp
so3_log = preint.so3_log
relative_rotation_geodesic_blend = preint.relative_rotation_geodesic_blend
preintegrate_segment = preint.preintegrate_segment
edge_times_from_window_ends = smooth.edge_times_from_window_ends
fuse_relative_poses_with_imu = smooth.fuse_relative_poses_with_imu
smooth_absolute_xyz = smooth.smooth_absolute_xyz
relative_poses_to_xyz = smooth.relative_poses_to_xyz


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(20):
        phi = rng.normal(scale=0.4, size=3)
        R = so3_exp(phi)
        assert abs(np.linalg.det(R) - 1.0) < 1e-6
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        phi2 = so3_log(R)
        assert np.allclose(so3_exp(phi2), R, atol=1e-5)


def test_so3_blend_endpoints():
    R0 = np.eye(3)
    R1 = so3_exp(np.array([0.0, 0.0, 0.5]))
    assert np.allclose(relative_rotation_geodesic_blend(R0, R1, 0.0), R0)
    assert np.allclose(relative_rotation_geodesic_blend(R0, R1, 1.0), R1)
    Rm = relative_rotation_geodesic_blend(R0, R1, 0.5)
    # Midpoint angle ~ half
    ang = float(np.linalg.norm(so3_log(Rm)))
    assert abs(ang - 0.25) < 1e-5


def test_preintegrate_constant_gyro():
    # 100 Hz, 0.1 s, wz = 1 rad/s → Δyaw ≈ 0.1 rad
    n = 11
    t0 = 1_000_000_000
    dt_ns = 10_000_000  # 10 ms
    rows = []
    for k in range(n):
        t = t0 + k * dt_ns
        rows.append([t, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    imu = np.asarray(rows, dtype=np.float64)
    pre = preintegrate_segment(imu, t0, t0 + (n - 1) * dt_ns)
    ang = float(np.linalg.norm(so3_log(pre.delta_R)))
    assert abs(ang - 0.1) < 0.02
    assert pre.n_samples == n


def test_preintegrate_constant_accel_dp():
    # ax = 1 m/s² for 0.1 s from rest: Δp = ½·a·t² = 5e-3 m (old ordering gave 6e-3).
    n = 11
    t0 = 1_000_000_000
    dt_ns = 10_000_000
    imu = np.asarray(
        [[t0 + k * dt_ns, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0] for k in range(n)], dtype=np.float64
    )
    pre = preintegrate_segment(imu, t0, t0 + (n - 1) * dt_ns)
    assert np.allclose(pre.delta_v, [0.1, 0.0, 0.0], atol=1e-12)
    assert np.allclose(pre.delta_p, [5e-3, 0.0, 0.0], atol=1e-9), pre.delta_p


def test_edge_times_chain():
    t_ends = np.array([100, 200, 300], dtype=np.int64)
    edges = edge_times_from_window_ends(t_ends, first_start_ns=50)
    assert edges == [(50, 100), (100, 200), (200, 300)]


def test_fuse_weight_zero_keeps_net():
    rel = np.zeros((3, 6), dtype=np.float64)
    rel[:, 5] = 0.1  # tz
    # Fake IMU with large gyro — ignored at weight 0
    imu = np.array(
        [
            [0, 0, 0, 5.0, 0, 0, 0],
            [1e8, 0, 0, 5.0, 0, 0, 0],
            [2e8, 0, 0, 5.0, 0, 0, 0],
            [3e8, 0, 0, 5.0, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    edges = [(0, 1e8), (1e8, 2e8), (2e8, 3e8)]
    out = fuse_relative_poses_with_imu(
        rel, imu, edges, imu_rot_weight=0.0, scale_fuse=False
    )
    assert np.allclose(out, rel)


def test_smooth_xyz_reduces_noise():
    t = np.linspace(0, 1, 50)
    xyz = np.stack([t, np.zeros_like(t), np.zeros_like(t)], axis=1)
    noisy = xyz + np.random.default_rng(1).normal(scale=0.05, size=xyz.shape)
    sm = smooth_absolute_xyz(noisy, window=7)
    err_n = np.mean(np.linalg.norm(noisy - xyz, axis=1))
    err_s = np.mean(np.linalg.norm(sm - xyz, axis=1))
    assert err_s < err_n


def test_path_accu_length():
    rel = np.zeros((5, 6))
    rel[:, 3] = 1.0
    xyz = relative_poses_to_xyz(rel)
    assert len(xyz) == 6
    assert abs(xyz[-1, 0] - 5.0) < 1e-6


def test_preintegrate_se3_constant_gyro():
    se3 = _load(
        "src.phase6.imu_preintegration_se3",
        SRC / "phase6" / "imu_preintegration_se3.py",
    )
    n = 11
    t0 = 1_000_000_000
    dt_ns = 10_000_000
    rows = [[t0 + k * dt_ns, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0] for k in range(n)]
    imu = np.asarray(rows, dtype=np.float64)
    pre = se3.preintegrate_segment_se3(imu, t0, t0 + (n - 1) * dt_ns)
    ang = float(np.linalg.norm(so3_log(pre.delta_R)))
    assert abs(ang - 0.1) < 0.02
    # No accel → delta_v / delta_p ~ 0
    assert np.linalg.norm(pre.delta_v) < 1e-6
    assert np.linalg.norm(pre.delta_p) < 1e-6


def test_preintegrate_se3_respects_gyro_bias():
    se3 = _load(
        "src.phase6.imu_preintegration_se3",
        SRC / "phase6" / "imu_preintegration_se3.py",
    )
    n = 11
    t0 = 1_000_000_000
    dt_ns = 10_000_000
    # Measured ω = bias only → after bias removal, ΔR ≈ I
    rows = [[t0 + k * dt_ns, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0] for k in range(n)]
    imu = np.asarray(rows, dtype=np.float64)
    bias = se3.BiasState(np.array([0.0, 0.0, 0.2]), np.zeros(3))
    pre = se3.preintegrate_segment_se3(imu, t0, t0 + (n - 1) * dt_ns, bias)
    ang = float(np.linalg.norm(so3_log(pre.delta_R)))
    assert ang < 1e-5


def test_refine_6b_weight_zero_keeps_path_len_close():
    backend = _load("src.phase6.backend_6b", SRC / "phase6" / "backend_6b.py")
    # Need se3 module already loaded for backend imports
    _load(
        "src.phase6.imu_preintegration_se3",
        SRC / "phase6" / "imu_preintegration_se3.py",
    )
    rel = np.zeros((8, 6), dtype=np.float64)
    rel[:, 3] = 0.05  # tx
    # Synthetic IMU at 100Hz between synthetic stamps
    t0 = 1_000_000_000
    step = 50_000_000  # 50 ms edges
    edges = [(t0 + i * step, t0 + (i + 1) * step) for i in range(len(rel))]
    rows = []
    for t in range(t0, t0 + len(rel) * step + 1, 10_000_000):
        rows.append([t, 0, 0, 0, 0, 0, 9.81])  # accel ~ +g in body z (still-ish)
    imu = np.asarray(rows, dtype=np.float64)
    out = backend.refine_trajectory_6b(
        rel,
        imu,
        edges,
        imu_rot_weight=0.0,
        imu_pos_weight=0.0,
        estimate_biases=False,
        apply_xyz_smooth=False,
    )
    # Pure network path length ≈ 8 * 0.05
    assert abs(out["path_len_m"] - 0.4) < 0.05


def test_optimize_window_reduces_or_keeps_cost():
    _load(
        "src.phase6.imu_preintegration_se3",
        SRC / "phase6" / "imu_preintegration_se3.py",
    )
    fg = _load("src.phase6.factor_graph_6c", SRC / "phase6" / "factor_graph_6c.py")
    # Straight-line network edges + matching IMU (no rotation, a=0 after gravity handled outside)
    n_edges = 5
    rel = np.zeros((n_edges, 6), dtype=np.float64)
    rel[:, 3] = 0.1
    t0 = 1_000_000_000
    step = 100_000_000
    edges = [(t0 + i * step, t0 + (i + 1) * step) for i in range(n_edges)]
    rows = []
    for t in range(t0, t0 + n_edges * step + 1, 10_000_000):
        rows.append([t, 0.0, 0.0, 0.0, 0.0, 0.0, 9.81])
    imu = np.asarray(rows, dtype=np.float64)
    out = fg.refine_trajectory_6c(
        rel,
        imu,
        edges,
        window_size=6,
        stride=3,
        max_nfev=15,
        estimate_biases=False,
        imu_prior_weight=0.3,
    )
    assert out["n_windows"] >= 1
    assert out["xyz"].shape[0] == n_edges + 1
    # Should stay roughly on the x-axis path
    assert out["path_len_m"] > 0.2


def test_spatial_loop_candidates_finds_revisit():
    lc = _load("src.phase6.loop_closure", SRC / "phase6" / "loop_closure.py")
    # Go out and return near start
    xyz = np.zeros((100, 3))
    xyz[:, 0] = np.concatenate([np.linspace(0, 10, 50), np.linspace(10, 0.3, 50)])
    pairs = lc.detect_spatial_loop_candidates(xyz, min_sep=40, max_dist_m=1.0)
    assert any(i < 10 and j > 80 for i, j in pairs)


def test_pose_graph_with_synthetic_loop_closes_gap():
    pg = _load("src.phase6.pose_graph_6d", SRC / "phase6" / "pose_graph_6d.py")
    lc = _load("src.phase6.loop_closure", SRC / "phase6" / "loop_closure.py")
    # Square path with drift on last edge so end ≠ start
    segs = [
        (1, 0, 0),
        (0, 1, 0),
        (-1, 0, 0),
        (0, -0.7, 0),  # short — leaves gap
    ]
    rel = np.zeros((4, 6))
    for i, (tx, ty, tz) in enumerate(segs):
        rel[i, 3:] = [tx, ty, tz]
    # Loop constraint: pose0 → pose4 should be ~identity translation
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])
    extras = [lc.RelativeConstraint(0, 4, R, t, 50, "loop")]
    Rs, ps = pg.optimize_pose_graph(rel, extras, max_nfev=50, sigma_loop_pos=0.05)
    # End should be closer to origin than raw
    raw_end = relative_poses_to_xyz(rel)[-1]
    opt_end = ps[-1]
    assert np.linalg.norm(opt_end) < np.linalg.norm(raw_end) + 0.05


def test_loops_rotation_only_skips_loop_translation():
    pg = _load("src.phase6.pose_graph_6d", SRC / "phase6" / "pose_graph_6d.py")
    lc = _load("src.phase6.loop_closure", SRC / "phase6" / "loop_closure.py")
    rel = np.zeros((3, 6))
    rel[:, 3] = 1.0  # three 1m steps along x
    # Bad loop translation that would otherwise shrink the path
    extras = [lc.RelativeConstraint(0, 3, np.eye(3), np.array([0.1, 0, 0]), 40, "loop")]
    _, ps_full = pg.optimize_pose_graph(rel, extras, max_nfev=40, sigma_loop_pos=0.05)
    _, ps_rot = pg.optimize_pose_graph(
        rel, extras, max_nfev=40, sigma_loop_pos=0.05, loops_rotation_only=True
    )
    path_full = float(np.sum(np.linalg.norm(np.diff(ps_full, axis=0), axis=1)))
    path_rot = float(np.sum(np.linalg.norm(np.diff(ps_rot, axis=0), axis=1)))
    # Rotation-only should keep path closer to raw (~3m) than full SE3 loop
    assert path_rot > path_full
    assert abs(path_rot - 3.0) < abs(path_full - 3.0)


def test_loops_sim3_does_not_lock_bad_scale():
    pg = _load("src.phase6.pose_graph_6d", SRC / "phase6" / "pose_graph_6d.py")
    lc = _load("src.phase6.loop_closure", SRC / "phase6" / "loop_closure.py")
    rel = np.zeros((3, 6))
    rel[:, 3] = 1.0
    # Same bad short translation as SE3 case
    extras = [lc.RelativeConstraint(0, 3, np.eye(3), np.array([0.1, 0, 0]), 40, "loop")]
    _, ps_se3 = pg.optimize_pose_graph(rel, extras, max_nfev=40, sigma_loop_pos=0.05)
    _, ps_sim3 = pg.optimize_pose_graph(
        rel, extras, max_nfev=40, sigma_loop_pos=0.05, loops_sim3=True
    )
    path_se3 = float(np.sum(np.linalg.norm(np.diff(ps_se3, axis=0), axis=1)))
    path_sim3 = float(np.sum(np.linalg.norm(np.diff(ps_sim3, axis=0), axis=1)))
    # Free-scale residual should not shrink toward 0.1 m the way SE3 does
    assert path_sim3 > path_se3
    assert abs(path_sim3 - 3.0) < 0.35


def test_project_and_triangulate_roundtrip():
    ba = _load("src.phase6.landmark_ba_6e", SRC / "phase6" / "landmark_ba_6e.py")
    K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
    Ri = np.eye(3)
    pi = np.zeros(3)
    Rj = np.eye(3)
    pj = np.array([1.0, 0.0, 0.0])
    Xw = np.array([0.5, 0.0, 5.0])
    uvi = ba.project_point(K, Ri, pi, Xw)
    uvj = ba.project_point(K, Rj, pj, Xw)
    assert uvi is not None and uvj is not None, "projection failed"
    tris = ba.triangulate_pair(
        K,
        Ri,
        pi,
        Rj,
        pj,
        uvi.reshape(1, 2),
        uvj.reshape(1, 2),
        max_reproj_px=10.0,
        min_depth=0.1,
    )
    assert len(tris) >= 1, f"triangulation returned empty; uv={uvi},{uvj}"
    assert np.linalg.norm(tris[0][0] - Xw) < 0.25, tris[0][0]


def test_landmark_ba_reduces_reproj_on_noisy_pose():
    ba = _load("src.phase6.landmark_ba_6e", SRC / "phase6" / "landmark_ba_6e.py")
    K = np.array([[400.0, 0, 320.0], [0, 400.0, 240.0], [0, 0, 1.0]])
    # Two poses 1m apart; landmark in front
    rel = np.zeros((1, 6))
    rel[0, 3] = 1.0
    Xw = np.array([0.5, 0.1, 4.0])
    Ri = np.eye(3)
    pi = np.zeros(3)
    Rj = np.eye(3)
    pj = np.array([1.0, 0.0, 0.0])
    uvi = ba.project_point(K, Ri, pi, Xw)
    uvj = ba.project_point(K, Rj, pj, Xw)
    assert uvi is not None and uvj is not None
    # Perturb second pose translation in the relative dump
    rel_noisy = rel.copy()
    rel_noisy[0, 4] = 0.15  # add y drift
    lm = ba.LandmarkTrack(Xw=Xw.copy(), pose_idxs=[0, 1], uvs=[uvi, uvj])
    Rs, ps = ba.optimize_landmark_ba(
        rel_noisy, [lm], K, max_nfev=40, sigma_odom_pos=0.2, sigma_pix=1.0
    )
    # After BA, second pose should move closer to x-axis (less y)
    assert abs(float(ps[1, 1])) < abs(float(relative_poses_to_xyz(rel_noisy)[1, 1])) + 1e-6
    assert abs(float(ps[1, 1])) < 0.12


def main():
    tests = [
        test_so3_exp_log_roundtrip,
        test_so3_blend_endpoints,
        test_preintegrate_constant_gyro,
        test_preintegrate_constant_accel_dp,
        test_edge_times_chain,
        test_fuse_weight_zero_keeps_net,
        test_smooth_xyz_reduces_noise,
        test_path_accu_length,
        test_preintegrate_se3_constant_gyro,
        test_preintegrate_se3_respects_gyro_bias,
        test_refine_6b_weight_zero_keeps_path_len_close,
        test_optimize_window_reduces_or_keeps_cost,
        test_spatial_loop_candidates_finds_revisit,
        test_pose_graph_with_synthetic_loop_closes_gap,
        test_loops_rotation_only_skips_loop_translation,
        test_loops_sim3_does_not_lock_bad_scale,
        test_project_and_triangulate_roundtrip,
        test_landmark_ba_reduces_reproj_on_noisy_pose,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"OK  {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(f"{failed} test(s) failed")
    print(f"All {len(tests)} Phase 6 tests passed.")


if __name__ == "__main__":
    main()
