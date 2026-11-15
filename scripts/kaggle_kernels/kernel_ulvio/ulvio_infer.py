"""Audit paper, E1: a third learned VIO model we did not train: UL-VIO (Park et al., ECCV 2024), authors'
released weights (model_ulvio_v1.pth, in their repository). It reads the same VS-VIO imus/*.mat (byte-identical).
Inference on KITTI 05 / 07 / 10 and 08 with UL-VIO's own test protocol (non-overlapping 11-frame windows, no
state carried, no test-time adaptation). Dumps <seq>_estimated_poses.npy / <seq>_gt_poses.npy for the oracle
swap, gyro rotation, accelerometer length and gate v3, and prints the authors' own t_rel / r_rel as a sanity
check against the paper. 08 is in UL-VIO's training split, so its row is training-set performance.
Outputs land in /kaggle/working/ulvio_out/.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
OUT = WORK / "ulvio_out"
OUT.mkdir(parents=True, exist_ok=True)
SEQS = ("05", "07", "10", "08")
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "path"])


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


kitti = walk_find(lambda r, f: (r / "poses/05.txt").is_file() and (r / "sequences").is_dir())
assert kitti, "KITTI odometry not found"
repo = WORK / "ulvio"
if not repo.is_dir():
    assert subprocess.call(["git", "clone", "-q", "--depth", "1", "https://github.com/jp4327/ulvio.git", str(repo)]) == 0
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
for name in ("sequences", "poses"):
    if not (repo / "data" / name).exists():
        (repo / "data" / name).symlink_to(kitti[0] / name)
log(f"UL-VIO {commit}, KITTI {kitti[0]}")

sys.path.insert(0, str(repo))
os.chdir(repo)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from ULVIO import ULVIO  # noqa: E402
from params import par  # noqa: E402
from utils.kitti_eval_temporal import data_partition, kitti_eval  # noqa: E402


class Opt:  # UL-VIO test.py defaults
    data_dir, seq_len, img_w, img_h = str(repo / "data"), 11, 512, 256


device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
net = ULVIO(par)
w = torch.load(repo / "model_ulvio_v1.pth", map_location="cpu")
w = {k.replace("module.", ""): v for k, v in w.items()}
res = net.load_state_dict(w, strict=True)
net.eval().to(device)
status = {"commit": commit, "device": device, "params": sum(p.numel() for p in net.parameters())}
log(f"loaded, {status['params']} parameters, {device}")

for seq in SEQS:
    df = data_partition(Opt(), seq)
    poses, t = [], time.time()
    for i in range(len(df)):
        imgs, imus, _ = df[i]
        with torch.no_grad():
            pose, _ = net(imgs.unsqueeze(0).to(device), imus.unsqueeze(0).to(device), None)
        poses.append(pose[0].cpu().numpy())
    est = np.vstack(poses)
    np.save(OUT / f"{seq}_estimated_poses.npy", est)
    np.save(OUT / f"{seq}_gt_poses.npy", df.poses_rel)
    _, _, t_rel, r_rel, t_rmse, r_rmse, _ = kitti_eval(est, df.poses_rel)
    status[seq] = {"n": int(len(est)), "t_rel": float(t_rel), "r_rel": float(r_rel)}
    log(f"{seq}: {est.shape} poses, authors' metric t_rel {t_rel:.2f} r_rel {r_rel:.2f}, {(time.time() - t) / 60:.1f} min")

m = np.mean([[status[s]["t_rel"], status[s]["r_rel"]] for s in ("05", "07", "10")], 0)
status["mean_05_07_10"] = m.tolist()
log(f"mean 05/07/10: t_rel {m[0]:.2f} r_rel {m[1]:.2f}")
(OUT / "status.json").write_text(json.dumps(status, indent=2))
log("done")
