"""Tests for experiment-2 gated fusion.

  python tests/test_gate.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

VIFT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIFT))

from src.models.components.pose_transformer import GatedPoseTransformer, PoseTransformer


def test_gate_output_is_euler6():
    net = GatedPoseTransformer(
        input_dim=768,
        embedding_dim=768,
        num_layers=2,
        nhead=6,
        dim_feedforward=128,
        dropout=0.0,
    )
    x = torch.randn(2, 11, 768)
    y = net((x, None, None), None)
    assert y.shape == (2, 11, 6)


def test_gate_init_near_one():
    net = GatedPoseTransformer(input_dim=768, embedding_dim=32, num_layers=1, nhead=4, dim_feedforward=64, dropout=0.0)
    z = torch.zeros(1, 3, 768)
    g = torch.sigmoid(net.gate(z))
    assert torch.all(g > 0.8)


def test_not_same_class_as_paper():
    assert GatedPoseTransformer is not PoseTransformer


if __name__ == "__main__":
    test_gate_output_is_euler6()
    test_gate_init_near_one()
    test_not_same_class_as_paper()
    print("test_gate ok")
