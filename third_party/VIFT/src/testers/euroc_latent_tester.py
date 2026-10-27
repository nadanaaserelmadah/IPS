"""EuRoC sequence evaluation with pretrained encoder + fusion network (latent pipeline)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.models.components.vsvio import Encoder
from src.utils import custom_transform
from src.utils.euroc_utils import (
    EUROC_TEST_SEQS_DEFAULT,
    build_sequence_samples,
    filter_euroc_sequences,
)

from src.testers.sliding_window_infer import (
    fuse_sliding_window,
    last_edge_numpy,
    resolve_eval_mode,
)

from .base_tester import BaseTester


class WrapperModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Feature_net = Encoder(params)

    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return torch.cat((feat_v, feat_i), 2)


class EuroCLatentTester(BaseTester):
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
        use_history_in_eval: bool = True,
        eval_mode: str | None = None,
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
        self.img_w = img_w
        self.img_h = img_h
        self.device = device
        self.use_history_in_eval = use_history_in_eval
        self.eval_mode = resolve_eval_mode(use_history_in_eval, eval_mode)

        params = ObjFromDict(
            {
                "img_w": img_w,
                "img_h": img_h,
                "v_f_len": v_f_len,
                "i_f_len": i_f_len,
                "imu_dropout": 0.1,
                "seq_len": seq_len,
            }
        )
        self.wrapper_model = WrapperModel(params)
        self._load_wrapper(wrapper_weights_path)
        self.wrapper_model.eval().to(device)

        self.transform = custom_transform.Compose(
            [custom_transform.ToTensor(), custom_transform.Resize((img_h, img_w))]
        )

    def _load_wrapper(self, weights_path: str) -> None:
        if not weights_path or not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Pretrained encoder not found: {weights_path}")
        pretrained_w = torch.load(weights_path, map_location="cpu", weights_only=False)
        model_dict = self.wrapper_model.state_dict()
        update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
        assert len(update_dict) == len(model_dict), "Encoder weights incomplete"
        model_dict.update(update_dict)
        self.wrapper_model.load_state_dict(model_dict)
        print(f"Loaded encoder from {weights_path}")

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

    def test_one_sequence(self, net: torch.nn.Module, seq_name: str) -> tuple[np.ndarray, np.ndarray]:
        seq_dir = self.data_dir / seq_name
        samples, _ = build_sequence_samples(
            seq_dir,
            sequence_length=self.seq_len,
            cam_stride=self.cam_stride,
            imu_freq=self.imu_freq,
        )
        pose_list: List[np.ndarray] = []
        gt_list: List[np.ndarray] = []
        hist = None
        n = len(samples)
        print(
            f"  {seq_name}: {n} sliding windows "
            f"(mode={self.eval_mode}; expect minutes to hours)"
        )

        for sample in tqdm(samples, desc=f"EuRoC {seq_name}", leave=True, mininterval=5.0):
            x_in, i_in, gt_seq = self._load_window_tensors(sample)
            x_in = x_in.to(self.device)
            i_in = i_in.to(self.device)
            gt_seq = gt_seq.to(self.device)

            with torch.inference_mode():
                latents = self.wrapper_model(x_in, i_in)
                pose, hist = fuse_sliding_window(
                    net, latents, hist, gt_seq, eval_mode=self.eval_mode
                )

            # One relative pose per sliding window (stride 1).
            p, g = last_edge_numpy(pose, gt_seq)
            pose_list.append(p)
            gt_list.append(g)

        pose_est = np.vstack(pose_list)
        pose_gt = np.vstack(gt_list)
        return pose_est, pose_gt

    def test(self, model: torch.nn.Module) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for i, seq in enumerate(self.sequences):
            print(f"Testing EuRoC sequence {i + 1}/{len(self.sequences)}: {seq}")
            pose_est, pose_gt = self.test_one_sequence(model, seq)
            results[seq] = {"estimated_poses": pose_est, "gt_poses": pose_gt}
        return results

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


class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)
