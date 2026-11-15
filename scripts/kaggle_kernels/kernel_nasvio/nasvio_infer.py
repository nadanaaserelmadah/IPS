"""Audit paper, E1: a fourth learned VIO model we did not train: NASVIO (Chen et al., ICASSP 2023), the two
released searched checkpoints (flops_target, latency_target; zips in their repository). It reads the same VS-VIO
imus/*.mat (byte-identical). Inference on KITTI 05 / 07 / 10 and 08, replicating the authors' test.py: 41-frame
windows that share one frame, RNN state carried across windows, no crop. The repository never builds the
poses/*.npy its test.py reads (data_prep.sh does not, and deepvio/preprocess.py stops at a pdb breakpoint), so the
GT relative poses are computed here with its own helper.cal_rel_pose (same Euler convention as VS-VIO).
Dumps <seq>_estimated_poses.npy / <seq>_gt_poses.npy per checkpoint for the oracle swap, gyro rotation,
accelerometer length and gate v3. 08 is in NASVIO's training split. Outputs land in /kaggle/working/nasvio_out/.
"""
import glob
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
OUT = WORK / "nasvio_out"
OUT.mkdir(parents=True, exist_ok=True)
SEQS = ("05", "07", "10", "08")
TEST_LEN = 41  # test.py


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
KITTI = kitti[0]
repo = WORK / "nasvio"
if not repo.is_dir():
    assert subprocess.call(["git", "clone", "-q", "--depth", "1", "https://github.com/unchenyu/NASVIO.git", str(repo)]) == 0
commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"]).decode().strip()
ckpts = {}
for target in ("flops", "latency"):
    with zipfile.ZipFile(repo / f"{target}_target.zip") as z:
        log(f"{target}_target.zip: {z.namelist()}")
        z.extractall(repo / f"ckpt_{target}")
    files = [p for p in (repo / f"ckpt_{target}").rglob("*") if p.is_file() and not p.name.startswith(".")]
    ckpts[target] = max(files, key=lambda p: p.stat().st_size)
log(f"NASVIO {commit}, KITTI {KITTI}, checkpoints {ckpts}")

sys.path.insert(0, str(repo))
os.chdir(repo)
import numpy as np  # noqa: E402
import scipy.io as sio  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
from helper import cal_rel_pose  # noqa: E402
from params import par  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"


def gt_rel(seq):
    rows = [[float(v) for v in line.split()] for line in (KITTI / "poses" / f"{seq}.txt").read_text().splitlines() if line.strip()]
    return np.array([cal_rel_pose(rows[i - 1], rows[i])[:6] for i in range(1, len(rows))])


def windows(seq):  # data_helper.get_data_info_test with imu_prev = 0
    fpaths = sorted(glob.glob(str(KITTI / "sequences" / seq / "image_2" / "*.png")))
    imus = sio.loadmat(repo / "data" / "imus" / f"{seq}.mat")["imu_data_interp"]
    out, start = [], 0
    while start + TEST_LEN < len(fpaths):
        out.append((fpaths[start:start + TEST_LEN], imus[start * 10:(start + TEST_LEN - 1) * 10 + 1]))
        start += TEST_LEN - 1
    out.append((fpaths[start:], imus[start * 10:]))
    return out


def load_img(p):
    img = TF.resize(Image.open(p), size=(par.img_h, par.img_w))
    return (TF.to_tensor(img) - 0.5).unsqueeze(0)


status = {"commit": commit, "device": device, "ckpts": {k: str(v) for k, v in ckpts.items()}}
for target, ck in ckpts.items():
    par.target = target
    import importlib  # noqa: E402
    import model  # noqa: E402
    importlib.reload(model)
    torch.manual_seed(0)
    np.random.seed(0)
    net = model.DeepVIO(par)
    w = torch.load(ck, map_location="cpu")
    w = w.get("state_dict", w) if isinstance(w, dict) and "state_dict" in w else w
    w = {k.replace("module.", ""): v for k, v in w.items()}
    net.load_state_dict(w, strict=True)
    net.eval().to(device)
    d = OUT / target
    d.mkdir(exist_ok=True)
    for seq in SEQS:
        gt = gt_rel(seq)
        prev, poses, t = None, [], time.time()
        for paths, imu in windows(seq):
            x = torch.cat([load_img(p) for p in paths], 0).unsqueeze(0).to(device)
            i_in = torch.FloatTensor(np.asarray(imu)).unsqueeze(0).to(device)
            with torch.no_grad():
                angle, trans, prev = net(x, i_in, prev=prev)
            poses.append(np.hstack([angle[0].cpu().numpy(), trans[0].cpu().numpy()]))
        est = np.vstack(poses)
        assert len(est) == len(gt), (seq, est.shape, gt.shape)
        np.save(d / f"{seq}_estimated_poses.npy", est)
        np.save(d / f"{seq}_gt_poses.npy", gt)
        status[f"{target}_{seq}_n"] = int(len(est))
        log(f"{target} {seq}: {est.shape} poses, {(time.time() - t) / 60:.1f} min")
    del net
    torch.cuda.empty_cache()

(OUT / "status.json").write_text(json.dumps(status, indent=2))
log("done")
