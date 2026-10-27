"""Preintegration-informed VIFT (PI-VIFT): physics residuals + carried inertial state."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.euroc_joint_net import EurocJointNet
from src.models.components.pose_transformer import PoseTransformer
from src.utils.preintegration_torch import preintegrate_accel_first, so3_exp
from src.utils.rotation6d import matrix_to_euler_zyx, matrix_to_rotation_6d

_G = 9.81
_V_MAX = 8.0  # m/s, EuRoC MAV
_DP_MAX = 1.5  # m per ~0.1 s edge
# Bias random walk per step (tanh-bounded) and caps; EuRoC ADIS16448 biases are ~1e-2.
_BG_STEP = 1e-3  # rad/s
_BG_MAX = 0.1  # rad/s
_BA_STEP = 1e-2  # m/s^2
_BA_MAX = 0.5  # m/s^2


def _clamp_norm(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    n = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x * (max_norm / n).clamp(max=1.0)


def _rotate_into_next(dR: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Express a frame-k vector in frame k+1, where dR = R_{k,k+1} (B,3,3)."""
    return torch.matmul(dR.transpose(-1, -2), x.unsqueeze(-1)).squeeze(-1)


def _slice_imu_windows(imus: torch.Tensor, seq_len: int, imu_per_step: int = 10) -> torch.Tensor:
    """(B, (T-1)*10+1, 6) → (B, T, 11, 6) overlapping IMU windows."""
    slices = [imus[:, i * imu_per_step : i * imu_per_step + imu_per_step + 1, :] for i in range(seq_len)]
    return torch.stack(slices, dim=1)


class PIViftFusion(nn.Module):
    """Causal transformer that predicts corrections on top of IMU preintegration."""

    def __init__(
        self,
        v_f_len: int = 512,
        i_f_len: int = 256,
        preint_dim: int = 128,
        state_dim: int = 128,
        embedding_dim: int = 768,
        num_layers: int = 4,
        nhead: int = 6,
        dim_feedforward: int = 128,
        dropout: float = 0.0,
        imu_per_step: int = 10,
        predict_corrections: bool = True,
    ):
        super().__init__()
        self.imu_per_step = int(imu_per_step)
        self.predict_corrections = bool(predict_corrections)
        self.preint_mlp = nn.Sequential(
            nn.Linear(13, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, preint_dim),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(12, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, state_dim),
        )
        self.warmup_v = nn.Linear(v_f_len + i_f_len, 3)
        self.warmup_g = nn.Linear(v_f_len + i_f_len, 3)
        token_dim = v_f_len + i_f_len + preint_dim + state_dim
        self.backbone = PoseTransformer(
            input_dim=token_dim,
            embedding_dim=embedding_dim,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        # Replace Euler head with residual heads (δθ, δp, δv, δg, δb_g, δb_a)
        self.backbone.fc2 = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embedding_dim, 18),
        )

    def _pack_preint(self, dR, dv, dp, dt) -> torch.Tensor:
        d6 = matrix_to_rotation_6d(dR)
        return torch.cat((d6, dv, dp, dt.unsqueeze(-1)), dim=-1)

    def forward_pi(
        self,
        latents: torch.Tensor,
        imus: torch.Tensor,
        carry: Optional[dict] = None,
    ) -> tuple[torch.Tensor, dict]:
        b, t, _ = latents.shape
        windows = _slice_imu_windows(imus, t, self.imu_per_step)
        # Specific force at rest is ≈ −g. Seed gravity from the first IMU window,
        # not a random warmup direction (that double-counts g and explodes scale).
        mean_a = windows[:, 0, :, :3].mean(dim=1)
        g_imu = -F.normalize(mean_a, dim=-1) * _G
        # A carry without "v" (eval_mode=streaming_gb) keeps g/biases but re-estimates v.
        if carry is None or "v" not in carry:
            v = torch.tanh(self.warmup_v(latents[:, 0])) * _V_MAX
        else:
            v = _clamp_norm(carry["v"], _V_MAX)
        if carry is None:
            g = F.normalize(self.warmup_g(latents[:, 0]) + g_imu, dim=-1) * _G
            bg = latents.new_zeros(b, 3)
            ba = latents.new_zeros(b, 3)
        else:
            g = F.normalize(carry["g"], dim=-1) * _G
            bg = _clamp_norm(carry["bg"], _BG_MAX)
            ba = _clamp_norm(carry["ba"], _BA_MAX)

        # dt per sample ≈ 0.01 s at 100 Hz equivalent (10 IMU / 0.1 s frame).
        # Token preintegration uses the window-start biases; the per-step physics
        # below re-integrates with the evolving bias estimate.
        dR, dv, dp, dt = preintegrate_accel_first(windows, dt=0.01, gyro_bias=bg, accel_bias=ba)

        state_in = torch.cat((v, g, bg, ba), dim=-1)
        sigma = self.state_mlp(state_in).unsqueeze(1).expand(-1, t, -1)
        p_tok = self.preint_mlp(self._pack_preint(dR, dv, dp, dt))
        tokens = torch.cat((latents, p_tok, sigma), dim=-1)
        raw = self.backbone((tokens, None, None), None)
        dtheta, dp_res = raw[..., :3], raw[..., 3:6]
        dv_res, dg = raw[..., 6:9], raw[..., 9:12]
        dbg, dba = raw[..., 12:15], raw[..., 15:18]

        poses = []
        vels = []
        gravs = []
        # v_k and g_k live in camera frame k (same frame as vel_cam / g_cam GT);
        # after each step they are rotated into frame k+1 by the predicted ΔR.
        v_k, g_k, bg_k, ba_k = v, g, bg, ba
        carry_next = None
        for k in range(t):
            dtk = dt[:, k].unsqueeze(-1)
            if k == 0:
                dR_k, dv_k, dp_k = dR[:, 0], dv[:, 0], dp[:, 0]
            else:
                dR_b, dv_b, dp_b, _ = preintegrate_accel_first(
                    windows[:, k : k + 1], dt=0.01, gyro_bias=bg_k, accel_bias=ba_k
                )
                dR_k, dv_k, dp_k = dR_b[:, 0], dv_b[:, 0], dp_b[:, 0]
            if self.predict_corrections:
                dR_out = dR_k @ so3_exp(dtheta[:, k])
                dp_out = v_k * dtk + 0.5 * g_k * dtk * dtk + dp_k + dp_res[:, k]
            else:
                dR_out = so3_exp(dtheta[:, k])
                dp_out = dp_res[:, k]
            dp_out = _clamp_norm(dp_out, _DP_MAX)
            euler = matrix_to_euler_zyx(dR_out)
            poses.append(torch.cat((euler, dp_out), dim=-1))
            vels.append(v_k)
            gravs.append(g_k)
            v_next = v_k + g_k * dtk + dv_k + dv_res[:, k]
            g_next = g_k + 0.05 * dg[:, k]
            v_k = _clamp_norm(_rotate_into_next(dR_out, v_next), _V_MAX)
            g_k = F.normalize(_rotate_into_next(dR_out, g_next), dim=-1) * _G
            bg_k = _clamp_norm(bg_k + _BG_STEP * torch.tanh(dbg[:, k]), _BG_MAX)
            ba_k = _clamp_norm(ba_k + _BA_STEP * torch.tanh(dba[:, k]), _BA_MAX)
            if k == 0:
                # State at frame 1 = frame 0 of the next stride-1 window.
                carry_next = {"v": v_k, "g": g_k, "bg": bg_k, "ba": ba_k}
        poses_t = torch.stack(poses, dim=1)
        aux = {
            "vel": torch.stack(vels, dim=1),
            "g": torch.stack(gravs, dim=1),
            "carry": {"v" : v_k, "g": g_k, "bg": bg_k, "ba": ba_k},
            "carry_next": carry_next,
        }
        return poses_t, aux

    def forward(self, batch, gt):
        raise RuntimeError("Use PIViftNet.forward / forward_pi; raw fusion needs IMU windows.")


class PIViftNet(EurocJointNet):
    """EurocJointNet encoder + PI-VIFT fusion."""

    def __init__(
        self,
        img_w: int = 512,
        img_h: int = 256,
        v_f_len: int = 512,
        i_f_len: int = 256,
        imu_dropout: float = 0.1,
        predict_corrections: bool = True,
        **fusion_kwargs,
    ):
        fusion = PIViftFusion(
            v_f_len=v_f_len,
            i_f_len=i_f_len,
            predict_corrections=predict_corrections,
            **fusion_kwargs,
        )
        super().__init__(
            img_w=img_w,
            img_h=img_h,
            v_f_len=v_f_len,
            i_f_len=i_f_len,
            imu_dropout=imu_dropout,
            fusion=fusion,
        )

    def forward(self, imgs: torch.Tensor, imus: torch.Tensor, carry: Optional[dict] = None) -> torch.Tensor:
        poses, aux = self.forward_with_aux(imgs, imus, carry)
        self._last_aux = aux
        return poses

    def forward_with_aux(
        self, imgs: torch.Tensor, imus: torch.Tensor, carry: Optional[dict] = None
    ) -> tuple[torch.Tensor, dict]:
        latents = self.encode(imgs, imus)
        return self.fusion.forward_pi(latents, imus, carry)

    def infer_window(
        self,
        imgs: torch.Tensor,
        imus: torch.Tensor,
        carry: Optional[dict] = None,
        eval_mode: str = "window_last",
    ) -> tuple[torch.Tensor, Optional[dict]]:
        """Eval helper: optionally carry inertial state across overlapping windows.

        The tester slides windows by one frame, so the carry is the state after the
        first step (frame 1), not the end-of-window state (frame T).
        """
        poses, aux = self.forward_with_aux(imgs, imus, carry)
        if eval_mode == "streaming_history":
            next_carry = aux["carry_next"]
        elif eval_mode == "streaming_gb":
            # Carried v diverges without TBPTT training; keep only slow states.
            next_carry = {k: aux["carry_next"][k] for k in ("g", "bg", "ba")}
        else:
            next_carry = None
        return poses, next_carry
