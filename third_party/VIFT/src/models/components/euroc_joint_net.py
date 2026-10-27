"""Encoder + fusion Transformer on raw images/IMU (Phase 3 fine-tune)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.models.components.pose_transformer import PoseTransformer
from src.models.components.vsvio import Encoder


class EurocJointNet(nn.Module):
    """
    End-to-end path: RGB + IMU -> Encoder latents -> PoseTransformer -> relative 6DoF.

    Compatible with EuRoC dataset batches (not cached latents).
    """

    def __init__(
        self,
        img_w: int = 512,
        img_h: int = 256,
        v_f_len: int = 512,
        i_f_len: int = 256,
        imu_dropout: float = 0.1,
        fusion: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.encoder = Encoder(
            _EncoderOpt(img_w, img_h, v_f_len, i_f_len, imu_dropout)
        )
        self.fusion = fusion or PoseTransformer(
            input_dim=v_f_len + i_f_len,
            embedding_dim=768,
            num_layers=4,
            nhead=6,
            dim_feedforward=128,
            dropout=0.0,
        )

    def encode(self, imgs: torch.Tensor, imus: torch.Tensor) -> torch.Tensor:
        # Frozen encoder parts stay in eval() so BatchNorm / Dropout do not drift
        # the latent distribution seen by a warm-started fusion.
        encoder_trainable = any(p.requires_grad for p in self.encoder.parameters())
        self._sync_encoder_frozen_eval()
        if not encoder_trainable:
            with torch.no_grad():
                feat_v, feat_i = self.encoder(imgs, imus)
        else:
            feat_v, feat_i = self.encoder(imgs, imus)
        return torch.cat((feat_v, feat_i), dim=2)

    def _sync_encoder_frozen_eval(self) -> None:
        """Eval any encoder submodule whose parameters are all frozen."""
        enc = self.encoder
        if not any(p.requires_grad for p in enc.parameters()):
            enc.eval()
            return
        for name, child in enc.named_children():
            if name == "visual_head":
                continue
            if name == "inertial_encoder":
                if not any(p.requires_grad for p in child.encoder_conv.parameters()):
                    child.encoder_conv.eval()
                continue
            if not any(p.requires_grad for p in child.parameters()):
                child.eval()

    def forward(self, imgs: torch.Tensor, imus: torch.Tensor) -> torch.Tensor:
        latents = self.encode(imgs, imus)
        return self.fusion((latents, None, None), None)

    def load_pretrained_encoder(self, path: str | Path) -> None:
        """Load Visual-Selective-VIO / VIFT encoder weights.

        Checkpoint keys are often prefixed with ``Feature_net.`` (wrapper used in
        latent caching / KITTI latent eval). Matching bare ``Encoder`` keys only
        used to silently load *zero* tensors and leave a random encoder — which
        makes fusion warm-start look broken (~80–100% t_rel).
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained encoder not found: {path}")
        weights = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(weights, dict):
            raise RuntimeError(f"Expected state dict in {path}, got {type(weights)}")

        remapped: dict = {}
        for k, v in weights.items():
            if k.startswith("Feature_net."):
                remapped[k[len("Feature_net.") :]] = v
            else:
                remapped[k] = v

        enc_state = self.encoder.state_dict()
        matched = {k: v for k, v in remapped.items() if k in enc_state}
        missing = [k for k in enc_state if k not in matched]
        if missing:
            raise RuntimeError(
                f"Encoder weights incomplete from {path}: "
                f"matched {len(matched)}/{len(enc_state)}; "
                f"missing e.g. {missing[:5]}"
            )
        self.encoder.load_state_dict(matched, strict=True)
        print(f"Loaded pretrained encoder: {len(matched)} tensors from {path}")

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = True

    def freeze_encoder_backbone(self) -> None:
        """Freeze CNN + IMU conv; train only visual_head and inertial proj."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder.visual_head.parameters():
            p.requires_grad = True
        for p in self.encoder.inertial_encoder.proj.parameters():
            p.requires_grad = True
        n_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in self.encoder.parameters())
        print(
            f"Encoder backbone frozen; trainable heads "
            f"{n_train}/{n_all} params (visual_head + inertial_encoder.proj)"
        )

    def freeze_fusion(self) -> None:
        for p in self.fusion.parameters():
            p.requires_grad = False

    def unfreeze_fusion(self) -> None:
        for p in self.fusion.parameters():
            p.requires_grad = True


class _EncoderOpt:
    def __init__(self, img_w, img_h, v_f_len, i_f_len, imu_dropout):
        self.img_w = img_w
        self.img_h = img_h
        self.v_f_len = v_f_len
        self.i_f_len = i_f_len
        self.imu_dropout = imu_dropout
        self.seq_len = 11
