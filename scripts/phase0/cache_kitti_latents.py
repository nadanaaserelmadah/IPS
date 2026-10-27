#!/usr/bin/env python3
"""Cache frozen-encoder KITTI latents (window 11 → train_10, or 65 → train_64)."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

IPS = Path(__file__).resolve().parents[2]
VIFT = IPS / "third_party" / "VIFT"
IMU_BASE = "https://raw.githubusercontent.com/mingyuyng/Visual-Selective-VIO/main/data/imus"
# VIFT paper train split (no 03; 08 optional). Default omits 08 so a minimal IMU set still works.
PAPER_TRAIN = ["00", "01", "02", "04", "06", "09"]
PAPER_VAL = ["05", "07", "10"]


class _Obj:
    def __init__(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)


def _ensure_imus(seqs: list[str]) -> None:
    import urllib.request

    imus = Path("kitti_data") / "imus"
    imus.mkdir(parents=True, exist_ok=True)
    for seq in seqs:
        dest = imus / f"{seq}.mat"
        if dest.is_file() and dest.stat().st_size > 0:
            continue
        url = f"{IMU_BASE}/{seq}.mat"
        print(f"Downloading IMU {seq} -> {dest}")
        urllib.request.urlretrieve(url, dest)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--sequence-length", type=int, default=65, help="Visual window; 11→*_10, 65→*_64")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    os.chdir(VIFT / "data")
    sys.path.insert(0, str(VIFT))

    from src.data.components.KITTI_dataset import KITTI
    from src.models.components.vsvio import Encoder
    from src.utils import custom_transform

    transform = custom_transform.Compose(
        [custom_transform.ToTensor(), custom_transform.Resize((256, 512))]
    )
    if args.split == "train":
        seqs = PAPER_TRAIN
        sub = f"train_{args.sequence_length - 1}"
    else:
        seqs = PAPER_VAL
        sub = f"val_{args.sequence_length - 1}"

    save_dir = Path("kitti_latent_data") / sub
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    if save_dir.is_symlink():
        print(f"Removing symlink {save_dir} -> {save_dir.resolve()} so this split gets a real folder")
        save_dir.unlink()
    save_dir.mkdir(parents=True, exist_ok=True)
    _ensure_imus(seqs)
    ds = KITTI("kitti_data", train_seqs=seqs, transform=transform, sequence_length=args.sequence_length)
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    params = _Obj(
        {
            "img_w": 512,
            "img_h": 256,
            "v_f_len": 512,
            "i_f_len": 256,
            "imu_dropout": 0.1,
            "seq_len": 11,
        }
    )
    from torch import nn

    class Wrapper(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.Feature_net = Encoder(params)

        def forward(self, imgs, imus):
            return self.Feature_net(imgs, imus)

    model = Wrapper()
    wpath = VIFT / "pretrained_models" / "vf_512_if_256_3e-05.model"
    pretrained = torch.load(wpath, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    upd = {k: v for k, v in pretrained.items() if k in sd}
    if len(upd) != len(sd):
        raise RuntimeError(f"Encoder load incomplete: {len(upd)}/{len(sd)}")
    sd.update(upd)
    model.load_state_dict(sd)
    for p_ in model.Feature_net.parameters():
        p_.requires_grad = False
    model.eval()
    model.to(args.device)

    with torch.no_grad():
        for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.to(args.device).float()
            imus = imus.to(args.device).float()
            feat_v, feat_i = model(imgs, imus)
            latent = torch.cat((feat_v, feat_i), 2).squeeze(0).cpu().numpy()
            np.save(save_dir / f"{i}.npy", latent)
            np.save(save_dir / f"{i}_gt.npy", gts.cpu().numpy())
            np.save(save_dir / f"{i}_rot.npy", rot.cpu().numpy())
            np.save(save_dir / f"{i}_w.npy", w.cpu().numpy())
    print(f"Wrote {len(loader)} windows to {save_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
