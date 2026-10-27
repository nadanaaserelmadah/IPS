"""Unit tests for A1 IMU canonicalization and PI-VIFT fusion."""

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


def test_gyro_first_to_accel_first():
    from src.utils.imu_canonical import gyro_first_to_accel_first

    x = np.array([[1, 2, 3, 4, 5, 6]], dtype=np.float64)
    y = gyro_first_to_accel_first(x)
    np.testing.assert_array_equal(y, [[4, 5, 6, 1, 2, 3]])


def test_parse_t_bs_and_rotate_gravity():
    from src.utils.imu_canonical import (
        parse_kalibr_T_BS,
        R_cam_from_body,
        gyro_first_to_accel_first,
        rotate_accel_gyro,
    )

    yaml = """
sensor_type: camera
T_BS:
  cols: 4
  rows: 4
  data: [0.0, -1.0, 0.0, 0.0,
         1.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0,
         0.0, 0.0, 0.0, 1.0]
"""
    t_bs = parse_kalibr_T_BS(yaml)
    assert t_bs is not None
    r = R_cam_from_body(t_bs)
    # body accel [0,0,9.81] after 90° about z in T_BS
    imu = np.array([[0, 0, 0, 0, 0, 9.81]])
    acc_first = gyro_first_to_accel_first(imu)
    cam = rotate_accel_gyro(acc_first, r)
    assert cam.shape == (1, 6)
    assert abs(np.linalg.norm(cam[0, :3]) - 9.81) < 1e-6


def test_legacy_flag_skips_reorder():
    import os
    from src.utils.imu_canonical import canonicalize_imu_table

    table = np.array([[1.0, 0.1, 0.2, 0.3, 0.0, 0.0, 9.81]])
    os.environ["IPS_IMU_LEGACY"] = "1"
    try:
        out = canonicalize_imu_table(table, enabled=None)
        np.testing.assert_array_equal(out, table)
    finally:
        os.environ.pop("IPS_IMU_LEGACY", None)


def test_preint_zero_motion():
    from src.utils.preintegration_torch import preintegrate_accel_first

    imu = torch.zeros(2, 3, 11, 6)
    dR, dv, dp, dt = preintegrate_accel_first(imu, dt=0.01)
    assert dR.shape == (2, 3, 3, 3)
    assert torch.allclose(dR, torch.eye(3).expand_as(dR), atol=1e-5)
    assert torch.allclose(dv, torch.zeros_like(dv), atol=1e-5)
    assert torch.allclose(dt, torch.full((2, 3), 0.1), atol=1e-5)


def test_preint_constant_accel_dp():
    # a = 1 m/s² along x for 0.1 s from rest: Δv = a·t, Δp = ½·a·t² exactly.
    # Updating Δv before Δp (old ordering) gives Δp = 6e-3 instead of 5e-3.
    from src.utils.preintegration_torch import preintegrate_accel_first

    imu = torch.zeros(1, 1, 11, 6, dtype=torch.float64)
    imu[..., 0] = 1.0
    _, dv, dp, _ = preintegrate_accel_first(imu, dt=0.01)
    assert torch.allclose(dv[0, 0], torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64), atol=1e-12)
    assert torch.allclose(dp[0, 0], torch.tensor([5e-3, 0.0, 0.0], dtype=torch.float64), atol=1e-9), dp


def test_preint_rotating_accel_matches_closed_form():
    # Body accel 1 m/s² along x while yawing at ω: a_world = [cos ωt, sin ωt, 0].
    from src.utils.preintegration_torch import preintegrate_accel_first

    om, t_end = 2.0, 0.1
    imu = torch.zeros(1, 1, 11, 6, dtype=torch.float64)
    imu[..., 0] = 1.0
    imu[..., 5] = om
    _, _, dp, _ = preintegrate_accel_first(imu, dt=0.01)
    p_true = torch.tensor(
        [(1 - np.cos(om * t_end)) / om**2, (t_end - np.sin(om * t_end) / om) / om, 0.0],
        dtype=torch.float64,
    )
    # Midpoint discretisation error ≈ 5e-5; the old ordering was off by ≈ 1e-3.
    assert (dp[0, 0] - p_true).norm() < 2e-4, dp


def test_preint_biases_are_subtracted():
    from src.utils.preintegration_torch import preintegrate_accel_first

    imu = torch.zeros(2, 3, 11, 6, dtype=torch.float64)
    imu[..., 0] = 0.3
    imu[..., 5] = 0.2
    ba = torch.tensor([[0.3, 0.0, 0.0]] * 2, dtype=torch.float64)
    bg = torch.tensor([[0.0, 0.0, 0.2]] * 2, dtype=torch.float64)
    for gb, ab in ((bg, ba), (bg[:, None].expand(2, 3, 3), ba[:, None].expand(2, 3, 3))):
        dR, dv, dp, _ = preintegrate_accel_first(imu, dt=0.01, gyro_bias=gb, accel_bias=ab)
        assert torch.allclose(dR, torch.eye(3, dtype=torch.float64).expand_as(dR), atol=1e-12)
        assert torch.allclose(dv, torch.zeros_like(dv), atol=1e-12)
        assert torch.allclose(dp, torch.zeros_like(dp), atol=1e-12)


def test_rotate_into_next_frame():
    from src.models.components.pi_vift import _rotate_into_next

    # Frame k+1 is frame k yawed +90°: x_k = [1,0,0] reads [0,-1,0] in frame k+1.
    rz = torch.tensor([[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]])
    out = _rotate_into_next(rz, torch.tensor([[1.0, 0.0, 0.0]]))
    assert torch.allclose(out, torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6)


def _rx(theta: float) -> torch.Tensor:
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=torch.float64)


def _imu_rolling_in_place(omega: float, still: int = 10, n: int = 101) -> torch.Tensor:
    """Accel-first IMU of a camera that stays put and rolls about x after `still` samples."""
    g_up = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)
    rows = []
    for i in range(n):
        r = _rx(omega * max(i - still, 0) * 0.01)
        gyro = torch.tensor([omega if i >= still else 0.0, 0.0, 0.0], dtype=torch.float64)
        rows.append(torch.cat((r.T @ g_up, gyro)))
    return torch.stack(rows).unsqueeze(0)


def _zero_residual_fusion():
    from src.models.components.pi_vift import PIViftFusion

    net = PIViftFusion().double()
    for m in (net.backbone.fc2[-1], net.warmup_v, net.warmup_g):
        torch.nn.init.zeros_(m.weight)
        torch.nn.init.zeros_(m.bias)
    return net.eval()


def test_pivift_physics_stationary_while_rotating():
    # With zero residuals the physics branch alone must keep a rolling-but-static
    # camera at rest. Old preintegration ordering → |Δp| ≈ 1.3e-2 m; carrying v/g
    # without rotating them between steps → |Δp| ≈ 0.3 m, |v| ≈ 2.7 m/s.
    torch.manual_seed(0)
    net = _zero_residual_fusion()
    omega = 1.0
    with torch.no_grad():
        poses, aux = net.forward_pi(torch.randn(1, 10, 768, dtype=torch.float64), _imu_rolling_in_place(omega))
    assert poses[0, :, 3:].norm(dim=-1).max() < 1e-3, poses[0, :, 3:]
    assert aux["vel"][0].norm(dim=-1).max() < 1e-2, aux["vel"][0]
    g_down = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64)
    g_expected = torch.stack([_rx(omega * 0.1 * max(k - 1, 0)).T @ g_down for k in range(10)])
    assert (aux["g"][0] - g_expected).norm(dim=-1).max() < 0.05, aux["g"][0]


def test_pivift_biases_stay_bounded():
    from src.models.components.pi_vift import _BA_MAX, _BG_MAX

    torch.manual_seed(0)
    net = _zero_residual_fusion()
    with torch.no_grad():
        net.backbone.fc2[-1].bias[12:18] = 1e3  # saturate δb_g, δb_a
        carry = {
            "v": torch.zeros(1, 3, dtype=torch.float64),
            "g": torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.float64),
            "bg": torch.full((1, 3), 5.0, dtype=torch.float64),
            "ba": torch.full((1, 3), 5.0, dtype=torch.float64),
        }
        poses, aux = net.forward_pi(
            torch.randn(1, 10, 768, dtype=torch.float64), _imu_rolling_in_place(0.0), carry
        )
    assert torch.isfinite(poses).all()
    assert aux["carry"]["bg"].norm() <= _BG_MAX + 1e-9
    assert aux["carry"]["ba"].norm() <= _BA_MAX + 1e-9


def test_pivift_fusion_shapes():
    from src.models.components.pi_vift import PIViftFusion

    net = PIViftFusion()
    latents = torch.randn(2, 10, 768)
    imus = torch.randn(2, 101, 6)
    poses, aux = net.forward_pi(latents, imus)
    assert poses.shape == (2, 10, 6)
    assert aux["vel"].shape == (2, 10, 3)
    assert aux["g"].shape == (2, 10, 3)
    assert torch.isfinite(poses).all()


def test_pivift_loss_with_state():
    from src.metrics.weighted_loss import PIViftLoss

    loss_fn = PIViftLoss(velocity_weight=0.1, gravity_weight=0.05)
    poses = torch.zeros(2, 10, 6)
    gts = torch.zeros(2, 10, 12)
    gts[..., 9:12] = torch.tensor([0.0, 0.0, -9.81])
    aux = {"vel": torch.zeros(2, 10, 3), "g": gts[..., 9:12].clone()}
    val = loss_fn(poses, gts, aux=aux)
    assert torch.isfinite(val)


if __name__ == "__main__":
    test_gyro_first_to_accel_first()
    test_parse_t_bs_and_rotate_gravity()
    test_legacy_flag_skips_reorder()
    test_preint_zero_motion()
    test_preint_constant_accel_dp()
    test_preint_rotating_accel_matches_closed_form()
    test_preint_biases_are_subtracted()
    test_rotate_into_next_frame()
    test_pivift_physics_stationary_while_rotating()
    test_pivift_biases_stay_bounded()
    test_pivift_fusion_shapes()
    test_pivift_loss_with_state()
    print("ok")
