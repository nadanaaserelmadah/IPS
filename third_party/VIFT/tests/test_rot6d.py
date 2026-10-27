"""Tests for experiment-1 6D rotation head.

Run from VIFT root:
  python tests/test_rot6d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

VIFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIFT))

from src.metrics.weighted_loss import GeodesicRot6DPoseLoss
from src.models.components.pose_transformer import PoseTransformer, PoseTransformerRot6D
from src.utils.rotation6d import (
    matrix_to_rotation_6d,
    rot6d_xyz_to_pose6,
    rotation_6d_to_matrix,
)


def test_roundtrip_identity():
    r = torch.eye(3).unsqueeze(0)
    d6 = matrix_to_rotation_6d(r)
    r2 = rotation_6d_to_matrix(d6)
    err = (r - r2).abs().max().item()
    assert err < 1e-5, err


def test_loss_zero_on_gt():
    b, t = 2, 11
    ident = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).expand(b, t, 6)
    xyz = torch.zeros(b, t, 3)
    pred = torch.cat([ident, xyz], dim=-1)
    gts = torch.zeros(b, t, 6)
    loss = GeodesicRot6DPoseLoss()(pred, gts)
    assert loss.item() < 1e-3, loss.item()


def test_heads_differ():
    euler = PoseTransformer(input_dim=768, embedding_dim=768, num_layers=4, nhead=6, dim_feedforward=128, dropout=0.0)
    rot6 = PoseTransformerRot6D(input_dim=768, embedding_dim=768, num_layers=4, nhead=6, dim_feedforward=128, dropout=0.0)
    x = torch.randn(2, 11, 768)
    y6 = euler((x, None, None), None)
    y9 = rot6((x, None, None), None)
    assert y6.shape == (2, 11, 6)
    assert y9.shape == (2, 11, 9)
    p6 = rot6d_xyz_to_pose6(y9)
    assert p6.shape == (2, 11, 6)


if __name__ == "__main__":
    test_roundtrip_identity()
    test_loss_zero_on_gt()
    test_heads_differ()
    print("test_rot6d ok")
