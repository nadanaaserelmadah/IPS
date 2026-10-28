"""TUM-VI dataset for visual-inertial odometry (VIFT-compatible)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.tum_vi_utils import build_sequence_samples, filter_tum_vi_sequences
from src.utils.euroc_utils import load_image_rgb  # noqa: E402


class TUMVI(Dataset):
    """
    Monocular TUM-VI loader with camera–IMU time sync.

    Output matches EuRoC / KITTI:
      (imgs, imus, rot, weight), gts
    """

    def __init__(
        self,
        root: str | Path,
        sequences: Sequence[str],
        sequence_length: int = 11,
        cam_stride: int = 2,
        imu_freq: int = 10,
        transform=None,
        uniform_weights: bool = True,
        include_state_gt: bool = False,
    ):
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.cam_stride = cam_stride
        self.imu_freq = imu_freq
        self.transform = transform
        self.sequences = list(sequences)
        self.uniform_weights = uniform_weights
        self.include_state_gt = bool(include_state_gt)
        self.samples: List[dict] = []
        self.weights: List[float] = []
        self._build_index()

    def _build_index(self) -> None:
        sequences = filter_tum_vi_sequences(self.root, self.sequences, strict=False)
        for seq in sequences:
            seq_dir = self.root / seq
            samples, weights = build_sequence_samples(
                seq_dir,
                sequence_length=self.sequence_length,
                cam_stride=self.cam_stride,
                imu_freq=self.imu_freq,
            )
            self.samples.extend(samples)
            if self.uniform_weights:
                self.weights.extend([1.0] * len(samples))
            else:
                self.weights.extend(weights)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        imgs = [load_image_rgb(img) for img in sample["imgs"]]

        if self.transform is not None:
            imgs, imus, gts = self.transform(
                imgs, np.copy(sample["imus"]), np.copy(sample["gts"])
            )
        else:
            imus = np.copy(sample["imus"])
            gts = np.copy(sample["gts"]).astype(np.float32)

        rot = np.float32(sample["rot"])
        weight = np.float32(self.weights[index])
        if self.include_state_gt:
            gts = np.concatenate(
                [
                    np.asarray(gts, dtype=np.float32),
                    np.asarray(sample["vel_cam"], dtype=np.float32),
                    np.asarray(sample["g_cam"], dtype=np.float32),
                ],
                axis=-1,
            )
        else:
            gts = np.asarray(gts, dtype=np.float32)
        return (
            imgs.to(torch.float),
            torch.from_numpy(imus).to(torch.float),
            rot,
            weight,
        ), torch.from_numpy(gts).to(torch.float)

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        lines = [
            f"Dataset {self.__class__.__name__}",
            f"  root: {self.root}",
            f"  sequences: {self.sequences}",
            f"  segments: {len(self)}",
            f"  seq_len: {self.sequence_length}, cam_stride: {self.cam_stride}, imu_freq: {self.imu_freq}",
        ]
        return "\n".join(lines)
