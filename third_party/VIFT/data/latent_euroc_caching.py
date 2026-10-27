"""Cache visual-inertial latents for EuRoC training sequences."""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from tqdm import tqdm

from src.data.components.EuRoC_dataset import EuRoC
from src.models.components.vsvio import Encoder
from src.utils import custom_transform
from src.utils.euroc_utils import EUROC_TRAIN_SEQS_DEFAULT, filter_euroc_sequences

EUROC_ROOT = Path("euroc")
_requested = os.environ.get("EUROC_TRAIN_SEQS", EUROC_TRAIN_SEQS_DEFAULT).split(",")
TRAIN_SEQS = filter_euroc_sequences(EUROC_ROOT, _requested)
SAVE_DIR = os.environ.get("EUROC_LATENT_TRAIN_DIR", "euroc_latent_data/train")
PRETRAINED = Path(
    os.environ.get(
        "VIFT_PRETRAINED",
        Path(__file__).resolve().parent.parent / "pretrained_models" / "vf_512_if_256_3e-05.model",
    )
)


class ObjFromDict:
    def __init__(self, dictionary):
        for k, v in dictionary.items():
            setattr(self, k, v)


class FeatureEncodingModel(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.Feature_net = Encoder(params)

    def forward(self, imgs, imus):
        feat_v, feat_i = self.Feature_net(imgs, imus)
        return feat_v, feat_i


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = custom_transform.Compose(
        [
            custom_transform.ToTensor(),
            custom_transform.Resize((256, 512)),
        ]
    )
    dataset = EuRoC(
        root="euroc",
        sequences=TRAIN_SEQS,
        transform=transform,
        sequence_length=11,
        cam_stride=2,
        imu_freq=10,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    params = ObjFromDict(
        {
            "img_w": 512,
            "img_h": 256,
            "v_f_len": 512,
            "i_f_len": 256,
            "imu_dropout": 0.1,
            "seq_len": 11,
        }
    )
    model = FeatureEncodingModel(params)
    if not PRETRAINED.is_file():
        raise FileNotFoundError(f"Pretrained encoder missing: {PRETRAINED}")
    pretrained_w = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
    model_dict = model.state_dict()
    update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
    assert len(update_dict) == len(model_dict), "Pretrained weights incomplete"
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)
    for p in model.Feature_net.parameters():
        p.requires_grad = False
    model.eval().to(device)

    os.makedirs(SAVE_DIR, exist_ok=True)
    with torch.no_grad():
        for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
            imgs = imgs.to(device).float()
            imus = imus.to(device).float()
            feat_v, feat_i = model(imgs, imus)
            latent = torch.cat((feat_v, feat_i), 2).squeeze(0)
            np.save(os.path.join(SAVE_DIR, f"{i}.npy"), latent.cpu().numpy())
            np.save(os.path.join(SAVE_DIR, f"{i}_gt.npy"), gts.cpu().numpy())
            np.save(os.path.join(SAVE_DIR, f"{i}_rot.npy"), rot.cpu().numpy())
            np.save(os.path.join(SAVE_DIR, f"{i}_w.npy"), w.cpu().numpy())


if __name__ == "__main__":
    main()
