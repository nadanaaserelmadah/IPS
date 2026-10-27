"""TUM-VI GT: camera extrinsics from Kalibr camchain.yaml, and windows dropped in mocap gaps.

Regression for (1) T_BS silently falling back to identity on TUM-VI exports without
cam0/sensor.yaml (IMU and GT left in the IMU frame while EuRoC uses the camera frame) and
(2) windows inside mocap gaps whose interpolated GT is near-zero motion (84% of corridor1).
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

CAMCHAIN = """cam0:
  T_cam_imu:
  - [-0.9995250378696743, 0.029615343885863205, -0.008522328211654736, 0.04727988224914392]
  - [0.0075019185074052044, -0.03439736061393144, -0.9993800792498829, -0.047443232143367084]
  - [-0.02989013031643309, -0.998969345370175, 0.03415885127385616, -0.0681999605066297]
  - [0.0, 0.0, 0.0, 1.0]
  camera_model: pinhole
"""


def test_camchain_T_BS(tmp_path: Path | None = None):
    from src.utils.imu_canonical import load_T_BS, parse_camchain_T_BS

    t_bs = parse_camchain_T_BS(CAMCHAIN)
    import yaml

    t_cam_imu = np.array(yaml.safe_load(CAMCHAIN)["cam0"]["T_cam_imu"])
    np.testing.assert_allclose(t_bs @ t_cam_imu, np.eye(4), atol=1e-12)
    root = Path(tmp_path or os.environ.get("TMPDIR", "/tmp")) / "tumvi_camchain_test" / "dataset-x"
    (root / "dso").mkdir(parents=True, exist_ok=True)
    (root / "dso" / "camchain.yaml").write_text(CAMCHAIN)
    np.testing.assert_allclose(load_T_BS(root), t_bs, atol=1e-12)


def test_real_corridor1():
    """Optional: TUM_VI_REAL_ROOT=<dir with dataset-corridor1_512_16/{dso,mav0}> (no images needed)."""
    root = os.environ.get("TUM_VI_REAL_ROOT")
    seq = Path(root or "") / "dataset-corridor1_512_16"
    if not root or not seq.is_dir():
        print("skip: TUM_VI_REAL_ROOT not set")
        return
    import torch

    from src.utils.euroc_utils import euler_to_rotation_matrix
    from src.utils.imu_canonical import load_T_BS
    from src.utils.preintegration_torch import preintegrate_accel_first
    from src.utils.tum_vi_utils import build_sequence_samples

    t_bs = load_T_BS(seq)
    assert not np.allclose(t_bs, np.eye(4))
    t_ci = np.linalg.inv(t_bs)[:3, 3]  # IMU origin in the camera frame
    samples, _ = build_sequence_samples(seq)
    all_samples, _ = build_sequence_samples(seq, max_mocap_gap_ms=1e12)
    assert 0 < len(samples) < 0.5 * len(all_samples), (len(samples), len(all_samples))
    rot, dpe, step = [], [], []
    for s in samples[::10]:
        im = np.asarray(s["imus"])
        dts = (s["t_end"] - s["t_start"]) / 1e9 / 10
        win = torch.tensor(np.stack([im[k * 10 : k * 10 + 11] for k in range(10)])[None], dtype=torch.float64)
        dR, _, dp, _ = (x[0].numpy() for x in preintegrate_accel_first(win, dt=dts / 10))
        for k, rel in enumerate(np.asarray(s["gts"])):
            r_gt = euler_to_rotation_matrix(rel[:3])
            rot.append(np.degrees(np.arccos(np.clip((np.trace(dR[k].T @ r_gt) - 1) / 2, -1, 1))))
            # IMU-point physics plus the camera lever arm (I - R_rel) t_cam_imu (~10 cm on TUM-VI)
            p = s["vel_cam"][k] * dts + 0.5 * s["g_cam"][k] * dts**2 + dp[k] + (np.eye(3) - r_gt) @ t_ci
            dpe.append(np.linalg.norm(p - rel[3:]))
            step.append(np.linalg.norm(rel[3:]))
    assert np.median(step) > 0.02, np.median(step)  # real motion, not gap interpolation
    assert np.median(rot) < 0.5, np.median(rot)  # IMU and GT agree in the camera frame
    assert np.median(dpe) < 0.003, np.median(dpe)  # dp physics closes to mm


if __name__ == "__main__":
    test_camchain_T_BS()
    test_real_corridor1()
    print("ok")
