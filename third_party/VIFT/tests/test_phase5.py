"""Unit tests for Phase 5 metrics, loss, and streaming history.

Run:
  python third_party/VIFT/tests/test_phase5.py
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


def _load(name: str, path: Path):
    """Load a module by file path without executing package __init__ side effects."""
    # Ensure parent packages exist as namespaces so relative imports can resolve
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            m = types.ModuleType(pkg)
            m.__path__ = [str(VIFT / Path(*parts[:i]))]  # type: ignore[attr-defined]
            sys.modules[pkg] = m
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Stub lightning-dependent base so metrics calculator can import
base_mod = types.ModuleType("src.metrics.base_metrics_calculator")


class _Base:
    pass


base_mod.BaseMetricsCalculator = _Base
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules["src"].__path__ = [str(SRC)]  # type: ignore[attr-defined]
sys.modules.setdefault("src.metrics", types.ModuleType("src.metrics"))
sys.modules["src.metrics"].__path__ = [str(SRC / "metrics")]  # type: ignore[attr-defined]
sys.modules["src.metrics.base_metrics_calculator"] = base_mod

euroc_utils = _load("src.utils.euroc_utils", SRC / "utils" / "euroc_utils.py")
# Ensure package path for utils without importing hydra-heavy __init__
sys.modules.setdefault("src.utils", types.ModuleType("src.utils"))
sys.modules["src.utils"].__path__ = [str(SRC / "utils")]  # type: ignore[attr-defined]
sys.modules["src.utils"].euroc_utils = euroc_utils  # type: ignore[attr-defined]

metrics = _load(
    "src.metrics.euroc_metrics_calculator",
    SRC / "metrics" / "euroc_metrics_calculator.py",
)
infer = _load(
    "src.testers.sliding_window_infer",
    SRC / "testers" / "sliding_window_infer.py",
)

align_trajectory_se3 = metrics.align_trajectory_se3
align_trajectory_sim3 = metrics.align_trajectory_sim3
ate_rmse_m = metrics.ate_rmse_m
validate_trajectory = metrics.validate_trajectory
path_accu = euroc_utils.path_accu
resolve_eval_mode = infer.resolve_eval_mode
fuse_sliding_window = infer.fuse_sliding_window
EVAL_MODE_WINDOW_LAST = infer.EVAL_MODE_WINDOW_LAST
EVAL_MODE_STREAMING_HISTORY = infer.EVAL_MODE_STREAMING_HISTORY

try:
    import torch

    HAS_TORCH = True
except Exception:
    HAS_TORCH = False
    torch = None  # type: ignore


def test_resolve_eval_mode():
    assert resolve_eval_mode(False) == EVAL_MODE_WINDOW_LAST
    assert resolve_eval_mode(True) == EVAL_MODE_STREAMING_HISTORY
    assert resolve_eval_mode(False, "streaming_history") == EVAL_MODE_STREAMING_HISTORY
    assert resolve_eval_mode(True, "window_last") == EVAL_MODE_WINDOW_LAST


def test_se3_alignment_recovers_rigid_transform():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(50, 3))
    angle = 0.3
    c, s = math.cos(angle), math.sin(angle)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t = np.array([1.5, -2.0, 0.25])
    est = (r @ gt.T).T + t
    aligned = align_trajectory_se3(est, gt)
    err = np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1)))
    assert err < 1e-6, err


def test_sim3_alignment_recovers_scale():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(40, 3))
    scale = 2.5
    est = gt * scale + np.array([0.1, -0.2, 0.3])
    aligned = align_trajectory_sim3(est, gt)
    err = np.sqrt(np.mean(np.sum((aligned - gt) ** 2, axis=1)))
    assert err < 1e-5, err


def test_ate_origin_vs_se3():
    gt = np.cumsum(np.ones((20, 3)) * 0.1, axis=0)
    angle = 0.5
    c, s = math.cos(angle), math.sin(angle)
    r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    est = (r @ gt.T).T
    origin = ate_rmse_m(est, gt, alignment="origin")
    se3 = ate_rmse_m(est, gt, alignment="se3")
    assert se3 < 1e-5
    assert origin > se3


def test_metrics_report_sim3_scale_and_frame_free_rpe():
    # Curved est path in the camera-0 frame; world GT is the same shape rotated
    # 90° about z, shifted, and 2× longer (est under-scaled by half).
    rel = np.zeros((30, 6), dtype=np.float64)
    rel[:, 2] = 0.1
    rel[:, 3] = 0.1
    est_xyz = np.array([m[:3, 3] for m in path_accu(rel)])
    rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    gt_xyz = 2.0 * (rz @ est_xyz.T).T + np.array([3.0, -1.0, 0.5])
    gt_rel = rel.copy()
    gt_rel[:, 3:] *= 2.0

    assert abs(metrics.sim3_scale(est_xyz, gt_xyz) - 2.0) < 1e-6

    calc = metrics.EuroCMetricsCalculator(primary_alignment="se3")
    out = calc.calculate_metrics(
        {"seq": {"estimated_poses": rel, "gt_poses": gt_rel, "gt_xyz": gt_xyz}}
    )
    assert abs(out["seq_sim3_scale"] - 2.0) < 1e-6, out["seq_sim3_scale"]
    assert out["seq_ate_sim3_m"] < 1e-6
    # After SE(3) alignment each 0.1 m est step misses its 0.2 m GT step by 0.1 m.
    # Comparing raw camera-frame vs world-frame steps would give √5·0.1 ≈ 0.224 m.
    assert abs(out["seq_rpe_t_m"] - 0.1) < 1e-6, out["seq_rpe_t_m"]


def test_zero_motion_floor_equals_ate_of_stationary_estimate():
    # A curved GT and an estimate that never moves: SE(3) and Sim(3) ATE both equal the
    # GT spread about its centroid, which the calculator reports as the zero-motion floor.
    gt_rel = np.zeros((40, 6), dtype=np.float64)
    gt_rel[:, 2] = 0.15
    gt_rel[:, 3] = 0.2
    gt_xyz = np.array([m[:3, 3] for m in path_accu(gt_rel)]) + np.array([5.0, -2.0, 1.0])
    floor = metrics.zero_motion_ate_m(gt_xyz)
    expected = np.sqrt(np.mean(np.sum((gt_xyz - gt_xyz.mean(0)) ** 2, axis=1)))
    assert abs(floor - expected) < 1e-12 and floor > 0.5
    still = np.zeros_like(gt_rel)
    out = metrics.EuroCMetricsCalculator(primary_alignment="se3").calculate_metrics(
        {"seq": {"estimated_poses": still, "gt_poses": gt_rel, "gt_xyz": gt_xyz}}
    )
    assert abs(out["seq_ate_zero_motion_m"] - floor) < 1e-9, out["seq_ate_zero_motion_m"]
    assert abs(out["seq_ate_se3_m"] - floor) < 1e-9, (out["seq_ate_se3_m"], floor)
    assert abs(out["seq_ate_sim3_m"] - floor) < 1e-9, (out["seq_ate_sim3_m"], floor)
    # The floor is reported, never averaged into the primary metric.
    assert out["mean_ate_m"] == out["seq_ate_se3_m"]


def test_path_accu_identity():
    rel = np.zeros((5, 6), dtype=np.float64)
    rel[:, 3] = 1.0
    mats = path_accu(rel)
    assert len(mats) == 6
    xyz = np.array([m[:3, 3] for m in mats])
    assert np.allclose(xyz[-1], [5.0, 0, 0], atol=1e-6)


def test_validate_trajectory_flags_blowup():
    gt = np.zeros((10, 3))
    gt[:, 0] = np.linspace(0, 1, 10)
    est = gt.copy()
    est[-1] = 1e6
    report = validate_trajectory(est, gt, max_step_m=5.0)
    assert not report["valid"]
    assert "est_max_step" in report["reasons"] or "est_abs_coord" in report["reasons"]


def test_window_count_inclusive():
    n_frames = 11
    seq_len = 11
    assert n_frames - seq_len + 1 == 1
    assert n_frames - seq_len == 0


def test_streaming_history_advances_once():
    if not HAS_TORCH:
        print("SKIP test_streaming_history_advances_once (no torch)")
        return

    class _DummyFusion(torch.nn.Module):
        def forward(self, batch, gt):
            latents, _, _ = batch
            b, t, _ = latents.shape
            return torch.zeros(b, t, 6, device=latents.device)

    fusion = _DummyFusion()
    b, t, d = 1, 10, 8
    hist = torch.arange(b * t * d, dtype=torch.float32).view(b, t, d)
    latents = hist.clone()
    latents[:, -1, :] = 999.0
    pose, hist2 = fuse_sliding_window(
        fusion, latents, hist, torch.zeros(t, 6), eval_mode=EVAL_MODE_STREAMING_HISTORY
    )
    assert pose.shape == (1, t, 6)
    assert torch.allclose(hist2[:, :-1, :], hist[:, 1:, :])
    assert torch.allclose(hist2[:, -1, :], latents[:, -1, :])


def test_window_last_resets_history():
    if not HAS_TORCH:
        print("SKIP test_window_last_resets_history (no torch)")
        return

    class _DummyFusion(torch.nn.Module):
        def forward(self, batch, gt):
            latents, _, _ = batch
            b, t, _ = latents.shape
            return torch.zeros(b, t, 6, device=latents.device)

    fusion = _DummyFusion()
    hist = torch.ones(1, 10, 4)
    latents = torch.full((1, 10, 4), 2.0)
    pose, hist2 = fuse_sliding_window(
        fusion, latents, hist, torch.zeros(10, 6), eval_mode=EVAL_MODE_WINDOW_LAST
    )
    assert torch.allclose(hist2, latents)


def test_multihorizon_loss_zero_at_gt():
    if not HAS_TORCH:
        print("SKIP test_multihorizon_loss_zero_at_gt (no torch)")
        return
    # Avoid package import side effects: load weighted_loss by path
    kitti = _load("src.utils.kitti_utils", SRC / "utils" / "kitti_utils.py")
    sys.modules["src.utils"].kitti_utils = kitti  # type: ignore[attr-defined]
    # Provide utils alias expected by weighted_loss
    utils_pkg = types.ModuleType("utils")
    utils_pkg.kitti_utils = kitti  # type: ignore[attr-defined]
    sys.modules["utils"] = utils_pkg
    sys.modules["utils.kitti_utils"] = kitti
    wl = _load("src.metrics.weighted_loss", SRC / "metrics" / "weighted_loss.py")
    crit = wl.MultiHorizonPoseLoss(
        translation_weight=1.0,
        rotation_weight=10.0,
        horizon_weight=0.1,
        temporal_weight=0.05,
        horizons=(2, 5),
    )
    gts = torch.randn(2, 10, 6) * 0.01
    loss = crit(gts, gts)
    assert torch.isfinite(loss)
    assert float(loss) < 1e-5, float(loss)


def test_multihorizon_loss_backward():
    if not HAS_TORCH:
        print("SKIP test_multihorizon_loss_backward (no torch)")
        return
    if "src.metrics.weighted_loss" not in sys.modules:
        test_multihorizon_loss_zero_at_gt()
    wl = sys.modules["src.metrics.weighted_loss"]
    poses = torch.randn(2, 10, 6, requires_grad=True) * 0.02
    gts = torch.randn(2, 10, 6) * 0.02
    crit = wl.MultiHorizonPoseLoss(horizons=(2, 5, 10))
    loss = crit(poses, gts)
    loss.backward()
    assert poses.grad is not None
    assert torch.isfinite(poses.grad).all()


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    if failed:
        raise SystemExit(1)
    print(f"All {len(tests)} tests passed (torch={'yes' if HAS_TORCH else 'skipped'}).")
