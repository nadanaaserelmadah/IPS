"""EuRoC MAV dataset for visual-inertial odometry (VIFT-compatible)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.utils.euroc_utils import build_sequence_samples, list_sequences
from src.utils.euroc_utils import load_image_rgb  # noqa: E402

# Mirror x in the camera frame (x right, y down, z forward). Proper vectors flip x;
# angular velocity is a pseudovector, so gyro y/z flip; R -> M R M flips Euler y/z.
_IMU_FLIP = np.array([-1.0, 1.0, 1.0, 1.0, -1.0, -1.0])
_POSE_FLIP = np.array([1.0, -1.0, -1.0, -1.0, 1.0, 1.0])
_STATE_FLIP = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, 1.0])  # vel_cam, g_cam


def hflip_window(imgs: torch.Tensor, imus: np.ndarray, gts: np.ndarray):
    """Horizontally mirror one window: imgs (T, C, H, W), imus (N, 6) accel-first
    camera frame, gts (T-1, 6) or (T-1, 12) with vel_cam + g_cam appended."""
    imus = np.asarray(imus) * _IMU_FLIP
    flip = _POSE_FLIP if gts.shape[-1] == 6 else np.concatenate([_POSE_FLIP, _STATE_FLIP])
    gts = (np.asarray(gts) * flip).astype(np.float32)
    return imgs.flip(-1), imus, gts


class EuRoC(Dataset):
    """
    Monocular EuRoC loader with camera–IMU time sync.

    Matches KITTI dataset output:
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
        chunk_windows: int = 1,
        chunk_start_stride: Optional[int] = None,
        hflip_p: float = 0.0,
    ):
        """
        chunk_windows > 1 returns K back-to-back windows per item (window k+1 starts at
        window k's last frame) for truncated BPTT over a carried inertial state; every
        tensor gets a leading K axis. Chunks start every ``chunk_start_stride`` windows
        (default K, so an epoch covers about as many windows as the unchunked loader).

        hflip_p mirrors a window horizontally (camera x -> -x): images, accel x,
        gyro y/z, and in the GT roll-pitch-yaw y/z, t_x, v_x, g_x. Assumes the
        canonical camera-frame IMU (default; not IPS_IMU_LEGACY=1).
        """
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.cam_stride = cam_stride
        self.imu_freq = imu_freq
        self.transform = transform
        self.sequences = list(sequences)
        self.uniform_weights = uniform_weights
        self.include_state_gt = bool(include_state_gt)
        self.chunk_windows = int(chunk_windows)
        self.chunk_start_stride = int(chunk_start_stride or chunk_windows)
        self.hflip_p = float(hflip_p)
        self.samples: List[dict] = []
        self.weights: List[float] = []
        self.chunks: List[List[int]] = []
        self._build_index()

    def _build_index(self) -> None:
        available = set(list_sequences(self.root))
        for seq in self.sequences:
            if seq not in available:
                raise FileNotFoundError(
                    f"Sequence '{seq}' not found under {self.root}. "
                    f"Available: {sorted(available)}"
                )
            seq_dir = self.root / seq
            samples, weights = build_sequence_samples(
                seq_dir,
                sequence_length=self.sequence_length,
                cam_stride=self.cam_stride,
                imu_freq=self.imu_freq,
            )
            offset = len(self.samples)
            self.samples.extend(samples)
            if self.uniform_weights:
                self.weights.extend([1.0] * len(samples))
            else:
                self.weights.extend(weights)
            if self.chunk_windows > 1:
                self.chunks.extend(self._sequence_chunks(samples, offset))

    def _sequence_chunks(self, samples: List[dict], offset: int) -> List[List[int]]:
        """Full-length chains of back-to-back windows (skipped windows break a chain)."""
        by_start = {int(s["t_start"]): i for i, s in enumerate(samples)}
        chunks = []
        for i in range(0, len(samples), self.chunk_start_stride):
            chain = [i]
            while len(chain) < self.chunk_windows:
                nxt = by_start.get(int(samples[chain[-1]]["t_end"]))
                if nxt is None:
                    break
                chain.append(nxt)
            if len(chain) == self.chunk_windows:
                chunks.append([offset + j for j in chain])
        return chunks

    def __getitem__(self, index: int):
        if self.chunk_windows <= 1:
            return self._get_window(index)
        items = [self._get_window(i) for i in self.chunks[index]]
        inputs = [it[0] for it in items]
        return (
            torch.stack([x[0] for x in inputs]),
            torch.stack([x[1] for x in inputs]),
            np.stack([x[2] for x in inputs]),
            np.stack([x[3] for x in inputs]),
        ), torch.stack([it[1] for it in items])

    def _get_window(self, index: int):
        sample = self.samples[index]
        # cam0 is grayscale; VIFT Encoder stacks consecutive RGB pairs -> 6 channels
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
        if self.hflip_p > 0.0 and np.random.random() < self.hflip_p:
            imgs, imus, gts = hflip_window(imgs, imus, gts)
        return (
            imgs.to(torch.float),
            torch.from_numpy(imus).to(torch.float),
            rot,
            weight,
        ), torch.from_numpy(gts).to(torch.float)

    def __len__(self) -> int:
        return len(self.chunks) if self.chunk_windows > 1 else len(self.samples)

    def __repr__(self) -> str:
        lines = [
            f"Dataset {self.__class__.__name__}",
            f"  root: {self.root}",
            f"  sequences: {self.sequences}",
            f"  segments: {len(self)}",
            f"  seq_len: {self.sequence_length}, cam_stride: {self.cam_stride}, imu_freq: {self.imu_freq}",
        ]
        return "\n".join(lines)
