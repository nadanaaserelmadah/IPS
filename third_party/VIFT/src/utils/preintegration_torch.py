"""Differentiable midpoint IMU preintegration (accel-first, camera frame)."""

from __future__ import annotations

import torch


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """(…, 3) → (…, 3, 3) Rodrigues."""
    theta = torch.linalg.norm(phi, dim=-1, keepdim=True).clamp(min=1e-12)
    k = phi / theta
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    zeros = torch.zeros_like(kx)
    k_hat = torch.stack(
        (
            torch.stack((zeros, -kz, ky), dim=-1),
            torch.stack((kz, zeros, -kx), dim=-1),
            torch.stack((-ky, kx, zeros), dim=-1),
        ),
        dim=-2,
    )
    eye = torch.eye(3, device=phi.device, dtype=phi.dtype).expand(phi.shape[:-1] + (3, 3))
    s = torch.sin(theta)[..., None]
    c = (1.0 - torch.cos(theta))[..., None]
    return eye + s * k_hat + c * (k_hat @ k_hat)


def _bias_as_bts3(bias: torch.Tensor) -> torch.Tensor:
    """Broadcast a (3,), (B,3) or (B,T,3) bias against (B,T,S,3) samples."""
    if bias.ndim == 2:
        return bias[:, None, None, :]
    if bias.ndim == 3:
        return bias[:, :, None, :]
    return bias


def preintegrate_accel_first(
    imu: torch.Tensor,
    *,
    dt: float | torch.Tensor = 0.01,
    gyro_bias: torch.Tensor | None = None,
    accel_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mid-point preintegration on (B, T, S, 6) windows [ax,ay,az, wx,wy,wz].

    Returns ΔR (B,T,3,3), Δv (B,T,3), Δp (B,T,3), Δt (B,T) in the starting frame,
    gravity *not* applied (Forster / VINS convention).
    """
    b, t, s, c = imu.shape
    if c != 6 or s < 2:
        raise ValueError(f"Expected (B,T,S,6) with S>=2, got {tuple(imu.shape)}")
    a = imu[..., :3]
    w = imu[..., 3:6]
    if accel_bias is not None:
        a = a - _bias_as_bts3(accel_bias)
    if gyro_bias is not None:
        w = w - _bias_as_bts3(gyro_bias)
    if not torch.is_tensor(dt):
        dt = imu.new_full((b, t, s - 1), float(dt))
    elif dt.ndim == 0:
        dt = dt.expand(b, t, s - 1)
    elif dt.ndim == 2:
        dt = dt.unsqueeze(-1).expand(b, t, s - 1)

    dR = torch.eye(3, device=imu.device, dtype=imu.dtype).expand(b, t, 3, 3).clone()
    dv = imu.new_zeros(b, t, 3)
    dp = imu.new_zeros(b, t, 3)
    for k in range(s - 1):
        w_mid = 0.5 * (w[:, :, k] + w[:, :, k + 1])
        a_mid = 0.5 * (a[:, :, k] + a[:, :, k + 1])
        dtk = dt[:, :, k].unsqueeze(-1)
        # Order: Δp with the old Δv and current ΔR, then Δv, then advance ΔR.
        ra = torch.matmul(dR, a_mid.unsqueeze(-1)).squeeze(-1)
        dp = dp + dv * dtk + 0.5 * ra * dtk * dtk
        dv = dv + ra * dtk
        dR = dR @ so3_exp(w_mid * dtk)
    dt_total = dt.sum(dim=-1)
    return dR, dv, dp, dt_total
