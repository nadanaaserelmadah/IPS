"""Placeholder batch so Lightning runs test hooks (real eval is in custom tester)."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class EvalPlaceholderDataset(Dataset):
    """One dummy latent sample — ``test_step`` is unused; ``on_test_epoch_end`` runs ATE."""

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        z = torch.zeros(10, 768)
        rot = torch.tensor(0.0)
        w = torch.tensor(1.0)
        gt = torch.zeros(9, 6)
        return (z, rot, w), gt


# Backward-compatible alias (older configs imported EmptyDataset).
EmptyDataset = EvalPlaceholderDataset
