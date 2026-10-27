"""EuRoC eval for Phase 3 joint model (encoder + fusion in one module)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.components.euroc_joint_net import EurocJointNet
from src.testers.base_tester import BaseTester
from src.testers.sliding_window_infer import (
    fuse_sliding_window,
    last_edge_numpy,
    resolve_eval_mode,
)
from src.utils.euroc_utils import (
    EUROC_TEST_SEQS_DEFAULT,
    build_sequence_samples,
    filter_euroc_sequences,
)
from src.utils import custom_transform


class EurocJointTester(BaseTester):
    """Sliding-window EuRoC eval using ``EurocJointNet`` (no separate wrapper)."""

    def __init__(
        self,
        sequences: List[str],
        data_dir: str,
        seq_len: int = 11,
        cam_stride: int = 2,
        imu_freq: int = 10,
        img_w: int = 512,
        img_h: int = 256,
        wrapper_weights_path: str = "",
        device: str = "cuda",
        v_f_len: int = 512,
        i_f_len: int = 256,
        use_history_in_eval: bool = False,
        eval_mode: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        env_seqs = os.environ.get("EUROC_TEST_SEQS")
        if env_seqs:
            requested = [s.strip() for s in env_seqs.split(",") if s.strip()]
        else:
            requested = list(sequences) if sequences else EUROC_TEST_SEQS_DEFAULT.split(",")
        self.sequences = filter_euroc_sequences(self.data_dir, requested)
        self.seq_len = seq_len
        self.cam_stride = cam_stride
        self.imu_freq = imu_freq
        self.device = device
        self.use_history_in_eval = use_history_in_eval
        self.eval_mode = resolve_eval_mode(use_history_in_eval, eval_mode)
        self.transform = custom_transform.Compose(
            [custom_transform.ToTensor(), custom_transform.Resize((img_h, img_w))]
        )

    def _load_window_tensors(self, sample: dict):
        imgs = [np.asarray(Image.open(p).convert("RGB")) for p in sample["imgs"]]
        imgs, imus, gts = self.transform(
            imgs, np.copy(sample["imus"]), np.copy(sample["gts"])
        )
        if not isinstance(imgs, torch.Tensor):
            imgs = torch.from_numpy(np.asarray(imgs)).float()
        if not isinstance(imus, torch.Tensor):
            imus = torch.from_numpy(np.asarray(imus, dtype=np.float32))
        if not isinstance(gts, torch.Tensor):
            gts = torch.from_numpy(np.asarray(gts, dtype=np.float32))
        return imgs.unsqueeze(0), imus.unsqueeze(0), gts

    def test(self, model: torch.nn.Module):
        results = {}
        for i, seq in enumerate(self.sequences):
            print(f"Testing EuRoC sequence {i + 1}/{len(self.sequences)}: {seq}")
            pose_est, pose_gt, gt_xyz, timestamps = self.test_one_sequence(model, seq)
            results[seq] = {
                "estimated_poses": pose_est,
                "gt_poses": pose_gt,
                "gt_xyz": gt_xyz,
                "timestamps": timestamps,
                "eval_mode": self.eval_mode,
            }
        return results

    def save_results(self, results, save_dir: str) -> None:
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
            if seq_data.get("gt_xyz") is not None:
                np.save(
                    os.path.join(save_dir, f"{seq_name}_gt_xyz.npy"),
                    seq_data["gt_xyz"],
                )
            if seq_data.get("timestamps") is not None:
                np.save(
                    os.path.join(save_dir, f"{seq_name}_timestamps.npy"),
                    seq_data["timestamps"],
                )

    def test_one_sequence(self, net: torch.nn.Module, seq_name: str) -> tuple:
        if not isinstance(net, EurocJointNet):
            raise TypeError(f"Expected EurocJointNet, got {type(net)}")

        seq_dir = self.data_dir / seq_name
        samples, _ = build_sequence_samples(
            seq_dir,
            sequence_length=self.seq_len,
            cam_stride=self.cam_stride,
            imu_freq=self.imu_freq,
        )
        if not samples:
            raise RuntimeError(
                f"No sliding windows for {seq_name} "
                f"(seq_len={self.seq_len}, cam_stride={self.cam_stride})"
            )

        pose_list = []
        gt_list = []
        gt_xyz_list = []
        ts_list = []
        hist = None
        n = len(samples)
        print(f"  {seq_name}: {n} windows (joint eval; mode={self.eval_mode})")

        for sample in tqdm(samples, desc=f"EuRoC joint {seq_name}", leave=True, mininterval=5.0):
            x_in, i_in, gt_seq = self._load_window_tensors(sample)
            x_in = x_in.to(self.device)
            i_in = i_in.to(self.device)
            gt_seq = gt_seq.to(self.device)

            with torch.inference_mode():
                if hasattr(net, "infer_window"):
                    pose, hist = net.infer_window(
                        x_in, i_in, hist, eval_mode=self.eval_mode
                    )
                else:
                    latents = net.encode(x_in, i_in)
                    pose, hist = fuse_sliding_window(
                        net.fusion, latents, hist, gt_seq, eval_mode=self.eval_mode
                    )

            p, g = last_edge_numpy(pose, gt_seq)
            pose_list.append(p)
            gt_list.append(g)
            gt_xyz_list.append(np.asarray(sample["gt_xyz_end"], dtype=np.float64))
            ts_list.append(int(sample["t_end"]))

        return (
            np.vstack(pose_list),
            np.vstack(gt_list),
            np.asarray(gt_xyz_list),
            np.asarray(ts_list, dtype=np.int64),
        )
