"""Second learned VIO model we did not train: Visual-Selective-VIO (Yang et al., ECCV 2022), authors'
pretrained weights (vf_512_if_256_3e-05.model, the file VIFT uses for its encoder). Inference on KITTI
05 / 07 / 10 with VS-VIO's own protocol (non-overlapping 11-frame windows, LSTM state carried across
windows), two visual-selection modes: the learned policy (gumbel-softmax, seeded) and always-on vision
(selection='random', p=1). Dumps <seq>_estimated_poses.npy / <seq>_gt_poses.npy per mode for the oracle
swap, gyro rotation and accelerometer length. CPU is enough. Outputs land in /kaggle/working/vsvio/.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
OUT = WORK / "vsvio"
OUT.mkdir(parents=True, exist_ok=True)
# src.utils/__init__ imports hydra / lightning helpers, missing on the plain image
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "omegaconf", "lightning", "natsort", "rich"])


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def walk_find(pred, base=INPUT, max_depth=6):
    hits = []
    for root, dirs, files in os.walk(base, followlinks=True):
        if root[len(str(base)):].count(os.sep) >= max_depth or root.endswith(("image_2", "image_3", "velodyne")):
            dirs[:] = []
        if pred(Path(root), files):
            hits.append(Path(root))
    return hits


code = walk_find(lambda r, f: (r / "third_party/VIFT/src/models/components/vsvio.py").is_file())
assets = walk_find(lambda r, f: (r / "imus/05.mat").is_file() or "05.mat" in f)
kitti = walk_find(lambda r, f: (r / "poses/05.txt").is_file() and (r / "sequences").is_dir())
assert code and assets and kitti, (code, assets, kitti)
IPS_SRC, ASSETS, KITTI_ROOT = code[0], assets[0], kitti[0]
IMUS = ASSETS / "imus" if (ASSETS / "imus/05.mat").is_file() else ASSETS
ips = WORK / "IPS"
if not ips.is_dir():
    shutil.copytree(IPS_SRC, ips, ignore=shutil.ignore_patterns(".venv", "logs", "*.ckpt"))
vift = ips / "third_party/VIFT"
kd = vift / "data/kitti_data"
kd.mkdir(parents=True, exist_ok=True)
for name, src in (("sequences", KITTI_ROOT / "sequences"), ("poses", KITTI_ROOT / "poses"), ("imus", IMUS)):
    if not (kd / name).exists():
        (kd / name).symlink_to(src)
enc = vift / "pretrained_models/vf_512_if_256_3e-05.model"
if not enc.is_file():
    subprocess.call([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    rc = subprocess.call([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"), "--vift-root", str(vift)])
    if rc != 0:
        subprocess.call([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"), "--vift-root", str(vift), "--gdown"])
assert enc.is_file(), "VS-VIO weights unavailable"
log(f"weights {enc} ({enc.stat().st_size / 1e6:.1f} MB)")

sys.path.insert(0, str(vift))
os.chdir(vift)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from src.models.components.vsvio import DeepVIO  # noqa: E402
from src.utils.kitti_eval import data_partition  # noqa: E402


class Opt:  # VS-VIO defaults (train.py / test.py of the authors' repository)
    img_w, img_h, v_f_len, i_f_len, imu_dropout, seq_len = 512, 256, 512, 256, 0.0, 11
    rnn_hidden_size, rnn_dropout_out, rnn_dropout_between, fuse_method = 1024, 0.2, 0.2, "cat"
    data_dir = str(kd)
    # unused by DeepVIO but read by the module file
    embedding_dim, nhead, num_layers, dim_feedforward, dropout = 768, 8, 4, 512, 0.1


device = "cuda" if torch.cuda.is_available() else "cpu"
net = DeepVIO(Opt())
w = torch.load(enc, map_location="cpu")
w = w.get("state_dict", w) if isinstance(w, dict) else w
w = {k.replace("module.", ""): v for k, v in w.items()}
res = net.load_state_dict(w, strict=False)
log(f"missing {len(res.missing_keys)} {res.missing_keys[:6]} | unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:6]}")
assert not res.missing_keys, "VS-VIO weights do not cover the full model"
net.eval().to(device)
status = {"device": device, "missing": res.missing_keys, "unexpected": res.unexpected_keys}

for mode, kw in (("policy", dict(selection="gumbel-softmax")), ("fullvision", dict(selection="random", p=1.0))):
    d = OUT / mode
    d.mkdir(exist_ok=True)
    torch.manual_seed(0)
    for seq in ("05", "07", "10"):
        df = data_partition(Opt(), seq)
        poses, usage, hc = [], [], None
        t = time.time()
        for i in range(len(df)):
            imgs, imus, gt = df[i]
            with torch.no_grad():
                pose, dec, probs, hc = net(imgs.unsqueeze(0).to(device), imus.unsqueeze(0).to(device), is_first=(i == 0), hc=hc, **kw)
            poses.append(pose[0].cpu().numpy())
            if dec.numel():
                usage.append(dec[0, :, 0].float().mean().item())
        est = np.vstack(poses)
        np.save(d / f"{seq}_estimated_poses.npy", est)
        np.save(d / f"{seq}_gt_poses.npy", df.poses_rel)
        status[f"{mode}_{seq}_visual_usage"] = float(np.mean(usage)) if usage else None
        log(f"{mode} {seq}: {est.shape} poses, visual usage {status[f'{mode}_{seq}_visual_usage']}, {(time.time() - t) / 60:.1f} min")
import json  # noqa: E402

(OUT / "status.json").write_text(json.dumps(status, indent=2))
shutil.rmtree(ips, ignore_errors=True)
log("done")
