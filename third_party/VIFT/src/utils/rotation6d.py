"""Continuity-friendly rotation (Zhou et al., CVPR 2019) for IPS experiment 1.

Network output is 9-D: 6-D rotation + xyz. KITTI GT / t_rel eval stay 6-D Euler+xyz.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """(…, 6) → (…, 3, 3) with columns from Gram–Schmidt (PyTorch3D convention)."""
    a1 = d6[..., :3]
    a2 = d6[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-6)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-6)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """(…, 3, 3) → (…, 6) first two columns."""
    return matrix[..., :2].transpose(-1, -2).reshape(*matrix.shape[:-2], 6)


def matrix_to_euler_zyx(matrix: torch.Tensor) -> torch.Tensor:
    """Match ``kitti_utils.euler_from_matrix`` (ZYX / roll-pitch-yaw as used in IPS)."""
    m = matrix[..., :3, :3]
    cy = torch.sqrt(m[..., 0, 0] * m[..., 0, 0] + m[..., 1, 0] * m[..., 1, 0]).clamp(min=1e-8)
    ay = torch.atan2(-m[..., 2, 0], cy)
    ax = torch.atan2(m[..., 2, 1], m[..., 2, 2])
    az = torch.atan2(m[..., 1, 0], m[..., 0, 0])
    return torch.stack((ax, ay, az), dim=-1)


def rot6d_xyz_to_pose6(raw: torch.Tensor) -> torch.Tensor:
    """(…, 9) 6D+xyz → (…, 6) Euler+xyz for KITTI ``path_accu`` / ``t_rel``."""
    r = rotation_6d_to_matrix(raw[..., :6])
    euler = matrix_to_euler_zyx(r)
    return torch.cat((euler, raw[..., 6:9]), dim=-1)


def pose6_to_rot6d_xyz(pose6: torch.Tensor, etr) -> torch.Tensor:
    """(…, 6) Euler+xyz → (…, 9) using the same Euler convention as training GT."""
    r = etr(pose6[..., :3])
    d6 = matrix_to_rotation_6d(r)
    return torch.cat((d6, pose6[..., 3:6]), dim=-1)
