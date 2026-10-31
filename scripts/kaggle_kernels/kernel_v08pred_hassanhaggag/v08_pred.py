"""KITTI final scoring: several checkpoints x {plain, flip TTA} x ensemble, on 05 / 07 / 10 (+ 08).

Checkpoints: every epoch_144.ckpt / rpmg_*.ckpt found under the inputs (the reproduction's assets
dataset and the outputs of the RPMG kernels attached as kernel sources), filtered by KF_CKPTS
(comma-separated substrings of the path, optional). All are latent_kitti_vio nets with the same
PoseTransformer(768, 768, 4, 6, 128) and Euler + t output.

The frozen encoder is run once per sequence for the plain and once for the mirrored input
(images flipped left-right, IMU mirrored: acc y, gyro x, gyro z change sign; see vift-kitti-flip),
and each net then runs on the cached latents with the same history logic as
KITTI_tester_latent.test_one_path. The mirrored prediction is mapped back (Euler about camera
y, z and translation x change sign). Variants per checkpoint: plain, flip, tta = mean of the two;
plus the same three averaged over all checkpoints ("ensemble").
The constant pitch / roll bias from held-out 08 is applied offline (scripts/kitti/bias_upside.py).

Outputs: /kaggle/working/kitti_final/<name>/<variant>/<seq>_{estimated,gt}_poses.npy, summary.json.
Local smoke test: KD_LOCAL=1 KD_INPUT=... KD_WORK=... KD_SEQS=05 KD_DEVICE=cpu KD_ENCODER=fake.model
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

T0 = time.time()
LOCAL = os.environ.get("KD_LOCAL") == "1"
INPUT = Path(os.environ.get("KD_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KD_WORK", "/kaggle/working"))
OUT = WORK / "kitti_final"
SEQS = os.environ.get("KD_SEQS", "08").split(",")  # held-out 08 for the gate v3 design (no-08 models never trained on it)
DEVICE = os.environ.get("KD_DEVICE", "cuda")


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 3600:5.2f} h] {msg}", flush=True)


def walk_find(pred, base: Path = INPUT, max_depth: int = 6):
    hits = []
    for root, dirs, files in os.walk(base, followlinks=True):
        if root[len(str(base)):].count(os.sep) >= max_depth or root.endswith(("image_2", "image_3", "velodyne")):
            dirs[:] = []
        if pred(Path(root), files):
            hits.append(Path(root))
    return hits


def sh(cmd: list[str], **kw) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, **kw)


# ---------------------------------------------------------------- locate inputs
code = walk_find(lambda r, f: (r / "third_party/VIFT/src/models/components/pose_transformer.py").is_file())
assets = walk_find(lambda r, f: (r / "imus/05.mat").is_file() or "05.mat" in f)  # a lone imus/ folder is flattened by Kaggle
kitti = walk_find(lambda r, f: (r / "poses/05.txt").is_file() and (r / "sequences").is_dir())
assert code and assets and kitti, f"inputs missing: code={code} assets={assets} kitti={kitti}"
IPS_SRC, ASSETS, KITTI = code[0], assets[0], kitti[0]
log(f"code {IPS_SRC}\nassets {ASSETS}\nkitti {KITTI}\nsequences {SEQS}")

# ---------------------------------------------------------------- workspace
ips = WORK / "IPS"
if not ips.is_dir():
    shutil.copytree(IPS_SRC, ips, ignore=shutil.ignore_patterns(".venv", "logs", "*.ckpt"))
vift = ips / "third_party/VIFT"
kd = vift / "data/kitti_data"
kd.mkdir(parents=True, exist_ok=True)
IMUS = ASSETS / "imus" if (ASSETS / "imus/05.mat").is_file() else ASSETS
for name, src in (("sequences", KITTI / "sequences"), ("poses", KITTI / "poses"), ("imus", IMUS)):
    if not (kd / name).exists():
        (kd / name).symlink_to(src)
(vift / ".project-root").touch()
os.environ["PROJECT_ROOT"] = str(vift)

if LOCAL:
    encoder = os.environ["KD_ENCODER"]
else:
    assert sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(vift / "requirements.txt")]) == 0
    if sh([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"), "--vift-root", str(vift)]) != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        assert sh([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"),
                   "--vift-root", str(vift), "--gdown"]) == 0, "encoder weights unavailable"
    encoder = str(vift / "pretrained_models/vf_512_if_256_3e-05.model")
assert Path(encoder).is_file(), encoder

sys.path.insert(0, str(vift))
os.chdir(vift)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.metrics.kitti_metrics_calculator import KITTIMetricsCalculator  # noqa: E402
from src.models.components.pose_transformer import PoseTransformer  # noqa: E402
from src.testers.kitti_latent_tester import KITTILatentTester  # noqa: E402

if not LOCAL:
    assert torch.cuda.is_available(), "GPU not enabled for this session"

import src.utils.kitti_eval as kitti_eval  # noqa: E402
from PIL import Image as PILImage  # noqa: E402
import types  # noqa: E402

# copy the small result files of attached training kernels (their full output lists ~90k latent
# files, which the Kaggle API rate-limits when downloaded from outside)
for r, _, fs in os.walk(INPUT, followlinks=True):
    r = Path(r)
    if r.name == "kitti_rpmg" and "status.json" in fs:
        dst = OUT / "sources" / r.relative_to(INPUT).parts[0]
        dst.mkdir(parents=True, exist_ok=True)
        for n in fs:
            if n.endswith((".json", ".csv", ".txt")):
                shutil.copy2(r / n, dst / n)
        for d in r.glob("poses_*"):
            shutil.copytree(d, dst / d.name, dirs_exist_ok=True)
        log(f"copied results of {r}")

FILTER = [f for f in os.environ.get("KF_CKPTS", "").split(",") if f]
ckpt_paths = sorted({Path(r) / n for r, _, fs in os.walk(INPUT, followlinks=True) for n in fs
                     if n == "rpmg_best.ckpt"})  # last.ckpt is the same top-1 link
if FILTER:
    ckpt_paths = [p for p in ckpt_paths if any(f in str(p) for f in FILTER)]
assert ckpt_paths, "no checkpoints found"


def ckpt_name(p: Path) -> str:
    if p.name == "epoch_144.ckpt":
        return "repro_e144"
    parts = p.relative_to(INPUT).parts  # e.g. notebooks/<owner>/<kernel-slug>/kitti_rpmg/rpmg_best.ckpt
    return f"{'_'.join(parts[1:3]) if parts[0] == 'notebooks' else parts[0]}_{p.stem}"  # unique per kernel source


nets = {}
assert len({ckpt_name(p) for p in ckpt_paths}) == len(ckpt_paths), "checkpoint names collide"
for p in ckpt_paths:
    net = PoseTransformer(input_dim=768, embedding_dim=768, num_layers=4, nhead=6, dim_feedforward=128, dropout=0.0)
    c = torch.load(p, map_location="cpu", weights_only=False)
    net.load_state_dict({k.removeprefix("net."): v for k, v in c["state_dict"].items() if k.startswith("net.")}, strict=True)
    nets[ckpt_name(p)] = net.eval().to(DEVICE)
    log(f"loaded {ckpt_name(p)}: {p} (epoch {c.get('epoch')})")


def make_tester():
    return KITTILatentTester(
        val_seqs=SEQS, data_dir=str(kd), seq_len=11, folder="sequences", img_w=512, img_h=256,
        wrapper_weights_path=encoder, device=DEVICE, v_f_len=512, i_f_len=256, use_history_in_eval=True,
    ).kitti_latent_tester


def encode(inner):
    """Per sequence: list of (latents [1, T, 768] on device, gt window) and the GT relative poses."""
    out = {}
    for seq, df in zip(SEQS, inner.dataloader):
        wins = []
        with torch.inference_mode():
            for image_seq, imu_seq, gt_seq in df:
                wins.append((inner.wrapper_model(image_seq.unsqueeze(0).to(DEVICE), imu_seq.unsqueeze(0).to(DEVICE)), gt_seq))
        out[seq] = (wins, df.poses_rel)
        log(f"encoded {seq}: {len(wins)} windows")
    return out


def run(net, wins):
    """KITTI_tester_latent.test_one_path with use_history_in_eval=True, on cached latents."""
    pose_list, hist = [], None
    with torch.inference_mode():
        for latents, gt_seq in wins:
            if hist is not None:
                res = torch.zeros(latents.shape[0], latents.shape[1], 6, device=latents.device)
                for idx in range(latents.shape[1]):
                    hist = torch.roll(hist, -1, 1)
                    hist[:, -1, :] = latents[:, idx, :]
                    r = net((hist, None, None), gt_seq)
                    res[:, idx, :] = r[:, -1, :]
                pose = res
            else:
                hist = latents.clone()
                pose = net((latents, None, None), gt_seq)
            pose_list.append(pose[0].detach().cpu().numpy())
    return np.vstack(pose_list).astype(np.float64)


def mirror_imu(x):
    x = np.array(x, dtype=np.float64, copy=True)
    x[:, [1, 3, 5]] *= -1  # acc y, gyro x, gyro z
    return x


def unmirror_pose(p):
    p = np.array(p, dtype=np.float64, copy=True)
    p[:, [1, 2, 3]] *= -1  # Euler about camera y, z; translation x
    return p


summary = {}


def score(est: dict, gt: dict, name: str, variant: str):
    results = {s: {"estimated_poses": est[s], "gt_poses": gt[s]} for s in SEQS}
    m = KITTIMetricsCalculator().calculate_metrics(results)
    test = [s for s in SEQS if s in ("05", "07", "10")] or SEQS
    for key in ("t_rel", "r_rel"):
        m[f"mean_{key}"] = float(np.mean([m[f"{s}_{key}"] for s in test]))
    d = OUT / name / variant
    d.mkdir(parents=True, exist_ok=True)
    for s in SEQS:
        np.save(d / f"{s}_estimated_poses.npy", est[s])
        np.save(d / f"{s}_gt_poses.npy", gt[s])
    summary[f"{name}/{variant}"] = {k: float(v) for k, v in m.items()}
    log(f"{name:28s} {variant:5s} " + " ".join(f"{s} {m[s + '_t_rel']:.2f}/{m[s + '_r_rel']:.2f}" for s in SEQS)
        + f" | mean(05,07,10) {m['mean_t_rel']:.2f} / {m['mean_r_rel']:.2f}")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))


# ---------------------------------------------------------------- encode plain and mirrored once
enc_plain = encode(make_tester())
inner = make_tester()
for dp in inner.dataloader:
    dp.imus_list = [mirror_imu(x) for x in dp.imus_list]
kitti_eval.Image = types.SimpleNamespace(open=lambda p: PILImage.open(p).transpose(PILImage.FLIP_LEFT_RIGHT))
try:
    enc_flip = encode(inner)
finally:
    kitti_eval.Image = PILImage
del inner
gt = {s: np.asarray(enc_plain[s][1][:, :6], dtype=np.float64) for s in SEQS}

# ---------------------------------------------------------------- every net, then the ensemble
allv = {"plain": [], "flip": [], "tta": []}
for name, net in nets.items():
    pl = {s: run(net, enc_plain[s][0]) for s in SEQS}
    fl = {s: unmirror_pose(run(net, enc_flip[s][0])) for s in SEQS}
    tt = {s: 0.5 * (pl[s] + fl[s]) for s in SEQS}
    for v, e in (("plain", pl), ("flip", fl), ("tta", tt)):
        e = {s: e[s][: len(gt[s])] for s in SEQS}
        score(e, gt, name, v)
        allv[v].append(e)
if len(nets) > 1:
    for v, lst in allv.items():
        score({s: np.mean([e[s] for e in lst], 0) for s in SEQS}, gt, "ensemble", v)
shutil.copytree(KITTI / "poses", OUT / "poses", dirs_exist_ok=True)
if not LOCAL:
    shutil.rmtree(ips, ignore_errors=True)
log("done")
