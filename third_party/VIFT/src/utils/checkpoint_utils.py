"""Checkpoint helpers for cross-run warm-start (e.g. EuRoC → TUM fine-tune)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch
from lightning import LightningModule

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_lightning_ckpt_weights(
    model: LightningModule,
    ckpt_path: Union[str, Path],
) -> None:
    """Load ``state_dict`` from a Lightning ``.ckpt`` without restoring epoch/optimizer."""
    path = Path(ckpt_path)
    log.info(f"Warm-start: loading model weights from {path} (epoch 0, fresh optimizer)")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    if not isinstance(state, dict):
        raise RuntimeError(f"No state_dict in checkpoint: {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        log.warning(f"Warm-start missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        log.warning(f"Warm-start unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
