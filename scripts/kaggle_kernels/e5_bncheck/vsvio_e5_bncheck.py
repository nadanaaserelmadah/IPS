"""Audit paper, E5 check: are the retrained VS-VIO models worse than the released one because of BatchNorm statistics?

E5 trained on two T4s (DataParallel, batch 16 split 8 + 8), so every BatchNorm running statistic was estimated from
8-sample batches on GPU 0 with momentum 0.1. The authors' default is one GPU (batch 16). The retrained models
overestimate forward motion by 4-8 % (released: 1.5-2.5 %), and their test error jumps between fine-tuning epochs
at lr 1e-6, both of which noisy BatchNorm statistics could cause.

Test, on one GPU, with no weight changed: for each checkpoint (E5 best-on-08 of seeds 0/1/2, and the released
vf_512_if_256_3e-05.model as a control)
  asis  - evaluate as saved;
  recal - reset every BatchNorm running statistic and re-estimate it as a plain average over N_BATCHES training
          batches of 16 (clean split 00 01 02 04 06 09, the training forward of the fine-tuning stage), then evaluate.
Each evaluation is VS-VIO's own protocol (KITTI_tester data, 11-frame windows, LSTM state carried, gumbel-softmax
selection, p 0.5) on 08 / 05 / 07 / 10, repeated for K gumbel seeds to measure the evaluation's own noise.
This checks the statistics only; weights learned under 8-sample batch normalisation stay as they are.
Outputs in /kaggle/working/bn/: results.jsonl, bn_shift.json, <model>/<variant>/k<k>/<seq>_*_poses.npy.
"""
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
OUT = WORK / "bn"
OUT.mkdir(parents=True, exist_ok=True)
COMMIT = "27b06b38dcf07a015fba22038050113462387e7d"
N_BATCHES = int(os.environ.get("BN_BATCHES", "150"))
K = int(os.environ.get("BN_K", "3"))
SEQS = ["08", "05", "07", "10"]
CLEAN_TRAIN = ["00", "01", "02", "04", "06", "09"]
CKPTS = {"s0_best08": ("best08_s0.pth", 81), "s1_best08": ("best08_s1.pth", 89), "s2_best08": ("best08_s2.pth", 97),
         "released": ("vf_512_if_256_3e-05.model", 99)}  # epoch only sets the training-forward temperature


def log(msg):
    line = f"[{(time.time() - T0) / 60:6.1f} min] {msg}"
    print(line, flush=True)
    with open(OUT / "log.txt", "a") as f:
        f.write(line + "\n")


def walk_find(pred, base=INPUT, max_depth=7):
    hits = []
    for root, dirs, files in os.walk(base, followlinks=True):
        if root[len(str(base)):].count(os.sep) >= max_depth or root.endswith(("image_2", "image_3", "velodyne")):
            dirs[:] = []
        if pred(Path(root), files):
            hits.append(Path(root))
    return hits


kitti = walk_find(lambda r, f: (r / "poses/05.txt").is_file() and (r / "sequences").is_dir())
imus = walk_find(lambda r, f: "05.mat" in f and "08.mat" in f)
ck = walk_find(lambda r, f: "best08_s0.pth" in f and "vf_512_if_256_3e-05.model" in f)
assert kitti and imus and ck, (kitti, imus, ck)
CK = ck[0]
repo = WORK / "vsvio"
if not repo.is_dir():
    subprocess.check_call(["git", "clone", "-q", "https://github.com/mingyuyng/Visual-Selective-VIO.git", str(repo)])
subprocess.check_call(["git", "-C", str(repo), "checkout", "-q", COMMIT])
subprocess.call([sys.executable, "-m", "pip", "install", "-q", "path"])
data = WORK / "kitti"
data.mkdir(exist_ok=True)
for name, src in (("sequences", kitti[0] / "sequences"), ("poses", kitti[0] / "poses"), ("imus", imus[0])):
    if not (data / name).exists():
        (data / name).symlink_to(src)
os.chdir(repo)
sys.path.insert(0, str(repo))

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.argv = ["train.py", "--data_dir", str(data)]
import train as T  # noqa: E402  (parses the authors' defaults; main() only runs under __main__)
from dataset.KITTI_dataset import KITTI  # noqa: E402
from model import DeepVIO  # noqa: E402
from utils import custom_transform  # noqa: E402
from utils.kitti_eval import KITTI_tester, kitti_eval  # noqa: E402

args = T.args
args.val_seq = SEQS
args.train_seq = CLEAN_TRAIN
log(f"torch {torch.__version__}, gpus {torch.cuda.device_count()}, using cuda:0 only; N_BATCHES {N_BATCHES}, K {K}")
dev = torch.device("cuda:0")
tester = KITTI_tester(args)


def bn_layers(m):
    return [x for x in m.modules() if isinstance(x, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]


def load(fname):
    net = DeepVIO(args)
    sd = torch.load(CK / fname, map_location="cpu", weights_only=False)
    net.load_state_dict(sd)
    return net.to(dev).eval()


nets = {}
for name, (fname, _) in CKPTS.items():
    nets[(name, "asis")] = load(fname)
    nets[(name, "recal")] = load(fname)
    log(f"loaded {name} ({fname}), {len(bn_layers(nets[(name, 'asis')]))} BatchNorm layers")

# ---------------------------------------------------------------- re-estimate BatchNorm statistics (recal copies)
tf = custom_transform.Compose([custom_transform.ToTensor(), custom_transform.Resize((args.img_h, args.img_w))])
train_ds = KITTI(args.data_dir, sequence_length=args.seq_len, train_seqs=CLEAN_TRAIN, transform=tf)
g = torch.Generator().manual_seed(0)
loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4, generator=g, drop_last=True)
for (name, var), net in nets.items():
    if var == "recal":
        net.eval()
        for b in bn_layers(net):
            b.reset_running_stats()
            b.momentum = None  # cumulative average over all batches seen
            b.train()
t = time.time()
torch.manual_seed(0)
with torch.no_grad():
    for i, (imgs, imus_b, gts, rot, w) in enumerate(loader):
        if i >= N_BATCHES:
            break
        imgs, imus_b = imgs.to(dev).float(), imus_b.to(dev).float()
        for (name, var), net in nets.items():
            if var == "recal":
                temp = args.temp_init * math.exp(-args.eta * (CKPTS[name][1] - args.epochs_warmup))
                net(imgs, imus_b, is_first=True, hc=None, temp=temp, selection="gumbel-softmax", p=0.5)
        if i % 25 == 0:
            log(f"BN pass: batch {i}/{N_BATCHES}")
log(f"BN re-estimation: {N_BATCHES} batches of 16 in {(time.time() - t) / 60:.1f} min")
shift = {}
for name in CKPTS:
    a, r = bn_layers(nets[(name, "asis")]), bn_layers(nets[(name, "recal")])
    rows = []
    for la, lr in zip(a, r):
        lr.eval()
        dm = (la.running_mean - lr.running_mean).abs() / lr.running_var.sqrt().clamp_min(1e-5)
        vr = (la.running_var / lr.running_var.clamp_min(1e-8)).log().abs()
        rows.append({"features": la.num_features, "kind": type(la).__name__,
                     "mean_shift_in_sd": float(dm.mean()), "abs_log_var_ratio": float(vr.mean())})
    shift[name] = rows
    log(f"{name}: BN shift (saved vs re-estimated), mean |dmean|/sd per layer: "
        + " ".join(f"{x['mean_shift_in_sd']:.2f}" for x in rows))
(OUT / "bn_shift.json").write_text(json.dumps(shift, indent=1))

# ---------------------------------------------------------------- evaluate every net, K gumbel seeds
for net in nets.values():
    net.eval()
with open(OUT / "results.jsonl", "a") as fres:
    for k in range(K):
        for si, seq in enumerate(SEQS):
            t = time.time()
            df = tester.dataloader[si]
            dl = torch.utils.data.DataLoader(df, batch_size=None, shuffle=False, num_workers=4)
            torch.manual_seed(2000 + 10 * k + si)
            hc = {key: None for key in nets}
            poses = {key: [] for key in nets}
            decs = {key: [] for key in nets}
            with torch.no_grad():
                for i, (image_seq, imu_seq, gt_seq) in enumerate(dl):
                    x_in, i_in = image_seq.unsqueeze(0).to(dev), imu_seq.unsqueeze(0).to(dev)
                    for key, net in nets.items():  # the authors' test_one_path with num_gpu 1
                        pose, decision, probs, hc[key] = net(x_in, i_in, is_first=(i == 0), hc=hc[key],
                                                             selection="gumbel-softmax", p=0.5)
                        poses[key].append(pose[0].cpu().numpy())
                        decs[key].append(decision[0].cpu().numpy()[:, 0])
            summary = []
            for key in nets:
                pe, de = np.vstack(poses[key]), np.hstack(decs[key])
                _, _, t_rel, r_rel, _, _, usage, _ = kitti_eval(pe, de, df.poses_rel)
                d = OUT / key[0] / key[1] / f"k{k}"
                d.mkdir(parents=True, exist_ok=True)
                np.save(d / f"{seq}_estimated_poses.npy", pe)
                np.save(d / f"{seq}_gt_poses.npy", df.poses_rel)
                rec = {"model": key[0], "variant": key[1], "k": k, "seq": seq, "t_rel": float(t_rel),
                       "r_rel": float(r_rel), "usage": float(usage),
                       "fwd_scale": float(pe[:, 5].sum() / df.poses_rel[:, 5].sum())}
                fres.write(json.dumps(rec) + "\n")
                summary.append(f"{key[0]}/{key[1]} {t_rel:.2f} (x{rec['fwd_scale']:.3f})")
            fres.flush()
            log(f"k {k} seq {seq} ({(time.time() - t) / 60:.1f} min): " + " | ".join(summary))
log("done")
