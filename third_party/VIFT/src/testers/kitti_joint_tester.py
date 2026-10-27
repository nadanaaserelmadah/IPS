"""KITTI eval for joint encoder+fusion (raw images), comparable to latent KITTI replication."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.models.components.euroc_joint_net import EurocJointNet
from src.testers.base_tester import BaseTester
from src.testers.sliding_window_infer import (
    fuse_sliding_window,
    resolve_eval_mode,
)
from src.utils.kitti_eval import data_partition


class KittiJointTester(BaseTester):
    """
    Evaluate ``EurocJointNet`` on KITTI raw images.

    Uses the same non-overlapping ``data_partition`` windows as the latent KITTI
    tester so metrics stay comparable to the frozen-encoder replication (2.10%).
    Default eval mode is ``streaming_history`` to match ``latent_kitti_vio_paper``.
    """

    def __init__(
        self,
        val_seqs: List[str],
        data_dir: str,
        seq_len: int = 11,
        folder: str = "sequences",
        img_w: int = 512,
        img_h: int = 256,
        device: str = "cuda",
        use_history_in_eval: bool = True,
        eval_mode: Optional[str] = None,
    ):
        super().__init__()
        self.val_seq = list(val_seqs)
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.folder = folder
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        self.use_history_in_eval = use_history_in_eval
        self.eval_mode = resolve_eval_mode(use_history_in_eval, eval_mode)

        @dataclass
        class Args:
            val_seq: list
            data_dir: str
            seq_len: int
            folder: str
            img_w: int
            img_h: int

        self.args = Args(
            self.val_seq, self.data_dir, self.seq_len, self.folder, self.img_w, self.img_h
        )
        self.dataloader = [data_partition(self.args, seq) for seq in self.val_seq]

    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        if not isinstance(model, EurocJointNet):
            raise TypeError(f"Expected EurocJointNet, got {type(model)}")
        results: Dict[str, Any] = {}
        for i, seq in enumerate(self.val_seq):
            print(
                f"Testing KITTI joint {i + 1}/{len(self.val_seq)}: {seq} "
                f"(mode={self.eval_mode})"
            )
            pose_est = self._test_one_path(model, self.dataloader[i])
            results[seq] = {
                "estimated_poses": pose_est,
                "gt_poses": self.dataloader[i].poses_rel,
                "eval_mode": self.eval_mode,
            }
        return results

    def _test_one_path(self, net: EurocJointNet, df: data_partition) -> np.ndarray:
        pose_list = []
        hist = None
        for image_seq, imu_seq, gt_seq in tqdm(
            df, total=len(df), desc="KITTI joint", leave=True, mininterval=5.0
        ):
            x_in = image_seq.unsqueeze(0).to(self.device)
            i_in = imu_seq.unsqueeze(0).to(self.device)
            gt_t = torch.as_tensor(gt_seq, dtype=torch.float32, device=self.device)
            if gt_t.ndim == 2:
                gt_t = gt_t.unsqueeze(0)

            with torch.inference_mode():
                latents = net.encode(x_in, i_in)
                # Match latent KITTI history behavior when streaming:
                # advance through each step of the (non-overlapping) window.
                if self.eval_mode == "streaming_history" and hist is not None:
                    results = torch.zeros(
                        latents.shape[0], latents.shape[1], 6, device=latents.device
                    )
                    for idx in range(latents.shape[1]):
                        hist = torch.roll(hist, -1, dims=1).clone()
                        hist[:, -1, :] = latents[:, idx, :]
                        pose = net.fusion((hist, None, None), gt_t)
                        results[:, idx, :] = pose[:, -1, :]
                    pose = results
                else:
                    pose, hist = fuse_sliding_window(
                        net.fusion, latents, hist, gt_t, eval_mode="window_last"
                    )
                    if self.eval_mode == "streaming_history":
                        hist = latents

            pose_list.append(pose[0].detach().cpu().numpy())
        return np.vstack(pose_list)

    def save_results(self, results: Dict[str, Any], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for seq_name, seq_data in results.items():
            np.save(
                os.path.join(save_dir, f"{seq_name}_estimated_poses.npy"),
                seq_data["estimated_poses"],
            )
            np.save(
                os.path.join(save_dir, f"{seq_name}_gt_poses.npy"),
                seq_data["gt_poses"],
            )
