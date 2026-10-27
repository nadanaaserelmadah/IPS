"""Unit tests for EuRoC horizontal-flip mirroring and back-to-back window chunks (TBPTT)."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

VIFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIFT))
SRC = VIFT / "src"

sys.modules.setdefault("src", types.ModuleType("src"))
sys.modules["src"].__path__ = [str(SRC)]  # type: ignore[attr-defined]

M3 = np.diag([-1.0, 1.0, 1.0])  # camera-frame mirror: x right -> left
LOCAL_SEQ = VIFT / "data/euroc/MH_01_easy"


def _rand_rot(rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return q * np.sign(np.linalg.det(q))


def _pose(rng: np.random.Generator) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = _rand_rot(rng)
    t[:3, 3] = rng.normal(size=3)
    return t


def _mirror_pose(t: np.ndarray) -> np.ndarray:
    """World mirrored in x (keeps gravity along z) seen by a camera mirrored in x."""
    out = np.eye(4)
    out[:3, :3] = M3 @ t[:3, :3] @ M3
    out[:3, 3] = M3 @ t[:3, 3]
    return out


def test_hflip_imu_preintegration_is_mirrored():
    from src.data.components.EuRoC_dataset import _IMU_FLIP
    from src.utils.preintegration_torch import preintegrate_accel_first

    rng = np.random.default_rng(0)
    imu = rng.normal(size=(101, 6)) * np.array([2.0, 2.0, 2.0, 0.8, 0.8, 0.8])
    imu[:, 1] += 9.81  # gravity mostly along camera y (down), like EuRoC cam0
    win = torch.tensor(imu[None, None], dtype=torch.float64)
    win_f = torch.tensor((imu * _IMU_FLIP)[None, None], dtype=torch.float64)
    dR, dv, dp, _ = (x[0, 0].numpy() for x in preintegrate_accel_first(win))
    dRf, dvf, dpf, _ = (x[0, 0].numpy() for x in preintegrate_accel_first(win_f))
    np.testing.assert_allclose(dRf, M3 @ dR @ M3, atol=1e-10)
    np.testing.assert_allclose(dvf, M3 @ dv, atol=1e-10)
    np.testing.assert_allclose(dpf, M3 @ dp, atol=1e-10)


def test_hflip_pose_and_state_gt_are_mirrored():
    from src.data.components.EuRoC_dataset import _POSE_FLIP, _STATE_FLIP
    from src.utils.euroc_utils import get_relative_pose_6DoF
    from src.utils.imu_canonical import gravity_in_camera, velocity_world_to_camera

    rng = np.random.default_rng(1)
    for _ in range(20):
        t1, t2 = _pose(rng), _pose(rng)
        v_w = rng.normal(size=3)
        rel = get_relative_pose_6DoF(t1, t2)
        rel_m = get_relative_pose_6DoF(_mirror_pose(t1), _mirror_pose(t2))
        np.testing.assert_allclose(rel_m, rel * _POSE_FLIP, atol=1e-9)
        state = np.concatenate([velocity_world_to_camera(t1, v_w), gravity_in_camera(t1)])
        t1m = _mirror_pose(t1)
        state_m = np.concatenate([velocity_world_to_camera(t1m, M3 @ v_w), gravity_in_camera(t1m)])
        np.testing.assert_allclose(state_m, state * _STATE_FLIP, atol=1e-9)


def _local_dataset(**kw):
    from src.data.components.EuRoC_dataset import EuRoC
    from src.utils.custom_transform import Compose, Resize, ToTensor

    return EuRoC(
        root=LOCAL_SEQ.parent, sequences=["MH_01_easy"], cam_stride=1, include_state_gt=True,
        transform=Compose([ToTensor(), Resize((64, 128))]), **kw,
    )


def test_hflip_window_on_real_euroc():
    """Flipped window: images mirrored; IMU gyro still matches GT rotation after the flip."""
    if not LOCAL_SEQ.is_dir():
        print("skip: no local MH_01_easy")
        return
    from src.utils.euroc_utils import euler_to_rotation_matrix
    from src.utils.preintegration_torch import preintegrate_accel_first

    plain, flipped = _local_dataset(), _local_dataset(hflip_p=1.0)
    (x0, i0, _, _), g0 = plain[0]
    (x1, i1, _, _), g1 = flipped[0]
    torch.testing.assert_close(x1, x0.flip(-1))

    def rot_err(imus: torch.Tensor, gts: torch.Tensor) -> float:
        win = torch.stack([imus[k * 10 : k * 10 + 11] for k in range(gts.shape[0])])[None].double()
        dR = preintegrate_accel_first(win)[0][0].numpy()
        errs = []
        for k in range(gts.shape[0]):
            r_gt = euler_to_rotation_matrix(gts[k, :3].double().numpy())
            c = (np.trace(dR[k].T @ r_gt) - 1.0) / 2.0
            errs.append(np.arccos(np.clip(c, -1.0, 1.0)))
        return float(np.mean(errs))

    e0, e1 = rot_err(i0, g0), rot_err(i1, g1)
    assert e0 < 0.01, e0  # gyro preintegration agrees with GT before the flip...
    assert abs(e1 - e0) < 1e-6, (e0, e1)  # ...and identically after it


def test_chunks_are_back_to_back():
    if not LOCAL_SEQ.is_dir():
        print("skip: no local MH_01_easy")
        return
    ds = _local_dataset(chunk_windows=3)
    assert len(ds) > 0
    for chunk in ds.chunks:
        for a, b in zip(chunk, chunk[1:]):
            assert ds.samples[b]["t_start"] == ds.samples[a]["t_end"]
    (x, i, r, w), g = ds[0]
    assert x.shape[:2] == (3, 11) and i.shape == (3, 101, 6) and g.shape == (3, 10, 12)
    assert r.shape == (3,) and w.shape == (3,)


if __name__ == "__main__":
    test_hflip_imu_preintegration_is_mirrored()
    test_hflip_pose_and_state_gt_are_mirrored()
    test_hflip_window_on_real_euroc()
    test_chunks_are_back_to_back()
    print("ok")
