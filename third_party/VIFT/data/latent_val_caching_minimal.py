# Minimal validation latent cache. From data/:
#   set KITTI_VAL_SEQS=05   (optional; default 05,07,10)
#   python latent_val_caching_minimal.py
import os
import sys
from pathlib import Path

sys.path.insert(0, "../")

PRETRAINED = Path(
    os.environ.get(
        "VIFT_PRETRAINED",
        Path(__file__).resolve().parent.parent / "pretrained_models" / "vf_512_if_256_3e-05.model",
    )
)
if not PRETRAINED.is_file():
    raise FileNotFoundError(
        f"Pretrained encoder not found: {PRETRAINED}\n"
        "Kaggle: attach a dataset with vf_512_if_256_3e-05.model or run:\n"
        "  pip install gdown && python ../../scripts/phase0/download_pretrained_encoder.py --gdown"
    )

import numpy as np
import torch
from tqdm import tqdm

from src.data.components.KITTI_dataset import KITTI
from src.models.components.vsvio import Encoder
from src.utils import custom_transform

VAL_SEQS = [s.strip() for s in os.environ.get("KITTI_VAL_SEQS", "05,07,10").split(",") if s.strip()]
SAVE_DIR = os.environ.get("KITTI_LATENT_VAL_DIR", "kitti_latent_data/val_10")


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


transform = custom_transform.Compose(
    [custom_transform.ToTensor(), custom_transform.Resize((256, 512))]
)
dataset = KITTI("kitti_data", train_seqs=VAL_SEQS, transform=transform, sequence_length=11)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

params = ObjFromDict(
    {"img_w": 512, "img_h": 256, "v_f_len": 512, "i_f_len": 256, "imu_dropout": 0.1, "seq_len": 11}
)
model = FeatureEncodingModel(params)
pretrained_w = torch.load(PRETRAINED, map_location="cpu")
model_dict = model.state_dict()
update_dict = {k: v for k, v in pretrained_w.items() if k in model_dict}
assert len(update_dict) == len(model_dict), "Some encoder weights are not loaded"
model_dict.update(update_dict)
model.load_state_dict(model_dict)
for param in model.Feature_net.parameters():
    param.requires_grad = False

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(SAVE_DIR, exist_ok=True)
model.eval()
model.to(device)
print(f"Caching {len(dataset)} windows -> {SAVE_DIR} (seqs={VAL_SEQS}, device={device})")

with torch.no_grad():
    for i, ((imgs, imus, rot, w), gts) in tqdm(enumerate(loader), total=len(loader)):
        imgs = imgs.to(device).float()
        imus = imus.to(device).float()
        feat_v, feat_i = model(imgs, imus)
        latent_vector = torch.cat((feat_v, feat_i), 2).squeeze(0)
        np.save(os.path.join(SAVE_DIR, f"{i}.npy"), latent_vector.cpu().numpy())
        np.save(os.path.join(SAVE_DIR, f"{i}_gt.npy"), gts.cpu().numpy())
        np.save(os.path.join(SAVE_DIR, f"{i}_rot.npy"), rot.cpu().numpy())
        np.save(os.path.join(SAVE_DIR, f"{i}_w.npy"), w.cpu().numpy())

print("Done.")
