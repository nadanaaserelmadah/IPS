"""GT quaternion -> rotation conversion, and GT relative poses that depend on it.

Regression for a bug where quat_to_rotation_matrix used s*qw, s*qx, s*qy in place of
s*qw*qx, s*qw*qy, s*qw*qz: GT "rotations" were not rotations (det 5 for the identity), so
relative poses, path lengths and vel/g state targets were all distorted.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np

VIFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIFT))
sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules["src"].__path__ = [str(VIFT / "src")]  # type: ignore[attr-defined]


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    k = axis / np.linalg.norm(axis)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * kx @ kx


def test_quat_matches_axis_angle():
    from src.utils.euroc_utils import quat_to_rotation_matrix

    np.testing.assert_allclose(quat_to_rotation_matrix(1.0, 0.0, 0.0, 0.0), np.eye(3), atol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(50):
        axis, angle = rng.normal(size=3), rng.uniform(-np.pi, np.pi)
        k = axis / np.linalg.norm(axis)
        q = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * k]) * rng.uniform(0.5, 2.0)
        R = quat_to_rotation_matrix(*q)
        np.testing.assert_allclose(R, _rodrigues(axis, angle), atol=1e-10)
        np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-10)


def test_relative_translation_is_displacement():
    """|t_rel| between interpolated GT poses equals the world displacement of the frame."""
    from src.utils.euroc_utils import get_relative_pose_6DoF, interpolate_gt_pose

    rng = np.random.default_rng(1)
    n = 200
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gt = np.column_stack([np.arange(n) * 5_000_000.0, np.cumsum(rng.normal(size=(n, 3)) * 0.01, 0), q])
    ts = np.arange(10, n - 10, 7) * 5_000_000 + 1_234_567
    poses = [interpolate_gt_pose(gt, int(t)) for t in ts]
    for a, b in zip(poses, poses[1:]):
        rel = get_relative_pose_6DoF(a, b)
        assert abs(np.linalg.norm(rel[3:]) - np.linalg.norm(b[:3, 3] - a[:3, 3])) < 1e-9
        assert abs(np.linalg.det(a[:3, :3]) - 1.0) < 1e-9


def test_real_euroc_gyro_matches_gt_rotation():
    """Optional: EUROC_REAL_ROOT=<dir with MH_05_difficult/mav0 csv files> (images not needed)."""
    root = os.environ.get("EUROC_REAL_ROOT")
    if not root or not (Path(root) / "MH_05_difficult").is_dir():
        print("skip: EUROC_REAL_ROOT not set")
        return
    import torch

    from src.utils.euroc_utils import build_sequence_samples, euler_to_rotation_matrix
    from src.utils.preintegration_torch import preintegrate_accel_first

    samples, _ = build_sequence_samples(Path(root) / "MH_05_difficult")
    errs = []
    for s in samples[::25]:
        im = np.asarray(s["imus"])
        win = torch.tensor(np.stack([im[k * 10 : k * 10 + 11] for k in range(10)])[None], dtype=torch.float64)
        dR = preintegrate_accel_first(win)[0][0].numpy()
        for k, rel in enumerate(np.asarray(s["gts"])):
            r_gt = euler_to_rotation_matrix(rel[:3])
            errs.append(np.arccos(np.clip((np.trace(dR[k].T @ r_gt) - 1) / 2, -1, 1)))
        assert abs(np.linalg.norm(s["g_cam"][0]) - 9.81) < 0.05
    assert np.degrees(np.mean(errs)) < 1.0, np.degrees(np.mean(errs))


if __name__ == "__main__":
    test_quat_matches_axis_angle()
    test_relative_translation_is_displacement()
    test_real_euroc_gyro_matches_gt_rotation()
    print("ok")
