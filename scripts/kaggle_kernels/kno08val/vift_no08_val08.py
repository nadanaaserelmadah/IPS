"""NO-08 SPLIT WITH HELD-OUT VALIDATION (2026-10-28, review fix): train 00 01 02 04 06 09 with mirror
augmentation, select the checkpoint by val/loss on 08 (never on 05 / 07 / 10), test best and last on
05 / 07 / 10. Everything else as in the earlier kernels:

KITTI: frozen-encoder latent VIFT with the RPMG rotation loss (angle weight 40), 11-frame windows,
on the VIFT paper's training split (00, 01, 02, 04, 06, 09; the paper drops 08, its released caching
script and our first RPMG run keep it).

Our reproduction (`latent_kitti_vio_paper`, AngleWeightedPoseLoss, angle weight 10) tests at
2.11% / 0.67 vs VIFT's published 1.96 / 0.46. Table 7b showed rotation is the bottleneck:
with GT rotation the network's own translation scores 1.55%. VIFT's ablation credits RPMG
with r_rel 0.71 -> 0.46, and our run never used it. `latent_kitti_vio_weighted_tf` is the
same config with RPMGPoseLoss (tau 1/4, lambda 0.01) and angle weight 40.

Steps: (1) cache 11-frame latents with the frozen VS-VIO encoder, train 00/01/02/04/06/08/09
(stride 1) and val 05/07/10 (same format as data/latent_caching.py, written with a parallel
DataLoader); (2) train 200 epochs; (3) test best val/loss and last on 05 / 07 / 10 with
use_history_in_eval (val = test sequences, as in the VIFT code and our 2.10% run).
Outputs: /kaggle/working/kitti_rpmg/{test_metrics_<tag>.csv, metrics.csv, status.json, poses_<tag>/,
rpmg_<tag>.ckpt}; the latent caches stay in /kaggle/working/kitti_latent_data for reuse.

MIRROR AUGMENTATION (this kernel): every training window is cached twice, as is and mirrored
(images flipped left-right; IMU in the OXTS frame x fwd / y left / z up: acc y, gyro x, gyro z
negated; GT relative pose in the camera frame: Euler about y, z and translation x negated; rotation
magnitude and LDS weight unchanged). Table 7b: the gap to VIFT is rotation, mostly a per-drive yaw
bias; a net trained on both mirror images cannot keep a constant left/right bias. Flip TTA at test
time did not help because mirrored KITTI (left-hand traffic) was out of distribution; here it is
in. Everything else is the no-08 RPMG run, so the difference is the augmentation.

Local smoke test: KR_LOCAL=1 KR_INPUT=... KR_WORK=... KR_TRAIN=05 KR_VAL=05 KR_EPOCHS=1 KR_ENCODER=fake.model
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
LOCAL = os.environ.get("KR_LOCAL") == "1"
INPUT = Path(os.environ.get("KR_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("KR_WORK", "/kaggle/working"))
OUT = WORK / "kitti_rpmg"
TRAIN = os.environ.get("KR_TRAIN", "00,01,02,04,06,09").split(",")  # VIFT paper split (Sec 4.1): no 08
VAL = os.environ.get("KR_VAL", "08").split(",")  # checkpoint selection on 08 (held out), NOT on the test drives
TEST = os.environ.get("KR_TEST", "05,07,10").split(",")
EPOCHS = int(os.environ.get("KR_EPOCHS", "200"))
EXPERIMENT = os.environ.get("KR_EXPERIMENT", "latent_kitti_vio_paper")  # our 2.11% loss (AngleWeightedPoseLoss, angle weight 10): RPMG no-08 tested worse, 2.46 / 1.00
SEED = os.environ.get("KR_SEED", "")
MIRROR = os.environ.get("KR_MIRROR", "1") == "1"
DEVICE = "cpu" if LOCAL else "cuda"


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
code = walk_find(lambda r, f: (r / "third_party/VIFT/src/metrics/weighted_loss.py").is_file())
assets = walk_find(lambda r, f: (r / "imus/05.mat").is_file() or "05.mat" in f)  # a lone imus/ folder is flattened by Kaggle
kitti = walk_find(lambda r, f: (r / "poses/05.txt").is_file() and (r / "sequences").is_dir())
assert code and assets and kitti, f"inputs missing: code={code} assets={assets} kitti={kitti}"
IPS_SRC, ASSETS, KITTI_ROOT = code[0], assets[0], kitti[0]
IMUS = ASSETS / "imus" if (ASSETS / "imus/05.mat").is_file() else ASSETS
log(f"code {IPS_SRC}\nimus {IMUS}\nkitti {KITTI_ROOT}\ntrain {TRAIN} val {VAL} epochs {EPOCHS} ({EXPERIMENT})")

# ---------------------------------------------------------------- workspace
ips = WORK / "IPS"
if not ips.is_dir():
    shutil.copytree(IPS_SRC, ips, ignore=shutil.ignore_patterns(".venv", "logs", "*.ckpt"))
vift = ips / "third_party/VIFT"
kd = vift / "data/kitti_data"
kd.mkdir(parents=True, exist_ok=True)
for name, src in (("sequences", KITTI_ROOT / "sequences"), ("poses", KITTI_ROOT / "poses"), ("imus", IMUS)):
    if not (kd / name).exists():
        (kd / name).symlink_to(src)
(vift / ".project-root").touch()
os.environ["PROJECT_ROOT"] = str(vift)
enc_path = vift / "pretrained_models/vf_512_if_256_3e-05.model"
if LOCAL:
    enc_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(os.environ["KR_ENCODER"], enc_path)
else:
    assert sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(vift / "requirements.txt")]) == 0
    if sh([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"), "--vift-root", str(vift)]) != 0:
        sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
        assert sh([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"),
                   "--vift-root", str(vift), "--gdown"]) == 0, "encoder weights unavailable"
assert enc_path.is_file()

sys.path.insert(0, str(vift))
os.chdir(vift)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.components.KITTI_dataset import KITTI  # noqa: E402
from src.models.components.vsvio import Encoder  # noqa: E402
from src.utils import custom_transform  # noqa: E402

if not LOCAL:
    assert torch.cuda.is_available(), "GPU not enabled for this session"
    # RPMG (src/utils/rpmg.py, tools.py) is CUDA-only upstream code: check forward/backward before caching.
    from src.metrics.weighted_loss import RPMGPoseLoss  # noqa: E402

    _p = (torch.randn(4, 10, 6, device="cuda") * 0.01).requires_grad_()
    RPMGPoseLoss(None, angle_weight=40)(_p, torch.randn(4, 10, 6, device="cuda") * 0.01, None).backward()
    assert torch.isfinite(_p.grad).all(), "RPMG backward produced non-finite gradients"
    log("RPMG forward/backward on GPU: ok")
OUT.mkdir(parents=True, exist_ok=True)
status: dict = {"experiment": EXPERIMENT, "train": TRAIN, "val": VAL, "epochs": EPOCHS}


def save_status() -> None:
    status["session_hours"] = round((time.time() - T0) / 3600, 2)
    (OUT / "status.json").write_text(json.dumps(status, indent=2))


# ---------------------------------------------------------------- (1) latent caches
class _P:
    img_w, img_h, v_f_len, i_f_len, imu_dropout, seq_len = 512, 256, 512, 256, 0.1, 11


class FeatureEncodingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.Feature_net = Encoder(_P())

    def forward(self, imgs, imus):
        return self.Feature_net(imgs, imus)


encoder = FeatureEncodingModel()
w = torch.load(enc_path, map_location="cpu")
md = encoder.state_dict()
upd = {k: v for k, v in w.items() if k in md}
assert len(upd) == len(md), "Some weights are not loaded"
encoder.load_state_dict(upd)
encoder.eval().to(DEVICE)
transform = custom_transform.Compose([custom_transform.ToTensor(), custom_transform.Resize((256, 512))])
latent_root = WORK / "kitti_latent_data"
(vift / "data/kitti_latent_data").unlink(missing_ok=True)
(vift / "data/kitti_latent_data").symlink_to(latent_root)


def cache(seqs: list[str], name: str, mirror: bool = False) -> int:
    out = latent_root / name
    done = out / "DONE"
    if done.is_file():
        return int(done.read_text())
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    ds = KITTI(str(kd), train_seqs=seqs, transform=transform, sequence_length=11)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=False, num_workers=0 if LOCAL else 4)
    i = 0
    t = time.time()
    with torch.no_grad():
        for (imgs, imus, rot, wt), gts in loader:
            passes = [(imgs, imus, gts)]
            if mirror:
                mi, mg = imus.clone(), gts.clone()
                mi[..., [1, 3, 5]] *= -1  # acc y, gyro x, gyro z
                mg[..., [1, 2, 3]] *= -1  # Euler about camera y, z; translation x
                passes.append((torch.flip(imgs, dims=[-1]), mi, mg))
            for pi, pm, pg in passes:
                fv, fi = encoder(pi.to(DEVICE).float(), pm.to(DEVICE).float())
                lat = torch.cat((fv, fi), 2).cpu().numpy()
                for b in range(lat.shape[0]):  # same per-sample files and shapes as data/latent_caching.py
                    np.save(out / f"{i}.npy", lat[b])
                    np.save(out / f"{i}_gt.npy", pg[b:b + 1].numpy())
                    np.save(out / f"{i}_rot.npy", rot[b:b + 1].numpy())
                    np.save(out / f"{i}_w.npy", wt[b:b + 1].numpy())
                    i += 1
            if i % 2000 < 8:
                log(f"  {name}: {i}/{len(ds)} ({(time.time() - t) / 60:.1f} min)")
    assert i == len(ds) * (2 if mirror else 1), (i, len(ds))
    done.write_text(str(i))
    return i


status["mirror"] = MIRROR
status["n_train"] = cache(TRAIN, "train_10", mirror=MIRROR)
status["n_val"] = cache(VAL, "val_10")
log(f"cached train_10 {status['n_train']} / val_10 {status['n_val']}")
save_status()
del encoder
if not LOCAL:
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- (2) train
run_dir = WORK / "rpmg_run"
common = [f"experiment={EXPERIMENT}", "logger=csv", f"model.tester.val_seqs=[{','.join(VAL)}]",
          f"model.tester.device={DEVICE}", f"trainer={'cpu' if LOCAL else 'gpu'}"] + ([f"seed={SEED}"] if SEED else [])
common_test = [c if not c.startswith("model.tester.val_seqs=") else f"model.tester.val_seqs=[{','.join(TEST)}]" for c in common]
train_cmd = [sys.executable, "src/train.py", *common, "train=true", "test=false", f"trainer.max_epochs={EPOCHS}",
             "trainer.min_epochs=1", f"hydra.run.dir={run_dir}", f"data.num_workers={0 if LOCAL else 4}"]
status["train_exit_code"] = sh(train_cmd, cwd=vift)
ckpts = run_dir / "checkpoints"
best = sorted(p for p in ckpts.glob("epoch_*.ckpt"))
assert best, f"no best checkpoint in {ckpts}"
status["best_ckpt"] = best[-1].name
for m in run_dir.rglob("metrics.csv"):
    shutil.copy2(m, OUT / "metrics.csv")
save_status()


# ---------------------------------------------------------------- (3) test best (val/loss) and last
def test(ckpt: Path, tag: str) -> int:
    d = WORK / f"rpmg_test_{tag}"
    cmd = [sys.executable, "src/train.py", *common_test, "train=false", "test=true", f"hydra.run.dir={d}", f"ckpt_path='{ckpt}'"]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=vift, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f"test_log_{tag}.txt").write_text(p.stdout)
    print(p.stdout[-2500:])
    for m in d.rglob("metrics.csv"):
        shutil.copy2(m, OUT / f"test_metrics_{tag}.csv")
    (OUT / f"poses_{tag}").mkdir(exist_ok=True)
    for f in d.rglob("*_poses.npy"):
        shutil.copy2(f, OUT / f"poses_{tag}" / f.name)
    return p.returncode


for tag, ck in (("best", best[-1]), ("last", ckpts / "last.ckpt")):
    if ck.is_file():
        c = torch.load(ck, map_location="cpu", weights_only=False)
        status[f"{tag}_epoch"] = int(c["epoch"])
        c["optimizer_states"], c["lr_schedulers"] = [], []
        torch.save(c, OUT / f"rpmg_{tag}.ckpt")
        status[f"test_{tag}_exit_code"] = test(ck, tag)
        save_status()

# summary vs paper
import csv  # noqa: E402

for tag in ("best", "last"):
    f = OUT / f"test_metrics_{tag}.csv"
    if f.is_file():
        rows = list(csv.DictReader(f.open()))
        row = {k: v for r in rows for k, v in r.items() if v}
        t = [float(row[f"test/{s}_t_rel"]) for s in TEST]
        r = [float(row[f"test/{s}_r_rel"]) for s in TEST]
        status[f"{tag}_t_rel"], status[f"{tag}_r_rel"] = t, r
        status[f"{tag}_mean"] = [round(float(np.mean(t)), 3), round(float(np.mean(r)), 3)]
        log(f"{tag}: t_rel {np.round(t, 2)} r_rel {np.round(r, 2)} mean {status[f'{tag}_mean']}  (VIFT paper 1.96 / 0.46; ours before 2.11 / 0.67)")
save_status()
if not LOCAL:
    shutil.rmtree(ips, ignore_errors=True)
    shutil.rmtree(latent_root, ignore_errors=True)  # ~90k files; listing them rate-limits `kaggle kernels output`
log("done")
