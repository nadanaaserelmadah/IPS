"""KITTI: gyro-conditioned VIFT. Each 768-d latent gets the gyro rotation of its frame pair (3-d).

Table 7d: the gyro, integrated over the real ~104 ms image interval, gives r_rel 0.18 (VIFT 0.46),
and gyro rotation + our network's translation already scores 1.70% (VIFT 1.96). The remaining error
is translation (Table 7e). Here the network sees the gyro rotation of every frame pair as 3 extra
input channels (rotation vector in the camera frame, degrees; time offset and IMU->camera rotation
fitted per recording day on training sequences only: 2011_10_03 on 00/01/02, 2011_09_30 on
04/06/08/09), so it no longer has to infer rotation and can spend its capacity on translation.
At test, rotation is taken from the gyro and translation from the network (hybrid); the network's
own rotation is scored too.

Recipe otherwise = our 2.11% run: latent_kitti_vio_paper (AngleWeightedPoseLoss, angle weight 10),
seed 42, train 00/01/02/04/06/09 (VIFT paper split), val 05/07/10 (= test, as in VIFT's code),
200 epochs; PoseTransformer input_dim 771. KR_MIRROR=1 adds left-right mirror augmentation (gyro
rotation vector mirrored as (x, -y, -z)).
Inputs: IPS code, KITTI imus, hocop1/kitti-odometry, and a gyro-features dataset with
<seq>_gyro_rotvec_deg.npy and <seq>_gyro_R.npy for 00-10 (scripts/kitti/gyro_hybrid.py features).
Testing does not go through train.py (its image tester has no gyro channel): latents are encoded
with KITTI_tester_latent's windows and run with the same history logic (as vift-kitti-final).
Outputs: /kaggle/working/kitti_rpmg/{status.json, metrics.csv, poses_<tag>/, hybrid_<tag>/, rpmg_<tag>.ckpt}.

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
VAL = os.environ.get("KR_VAL", "05,07,10").split(",")
EPOCHS = int(os.environ.get("KR_EPOCHS", "200"))
EXPERIMENT = os.environ.get("KR_EXPERIMENT", "latent_kitti_vio_paper")  # our 2.11% loss (AngleWeightedPoseLoss, angle weight 10): RPMG no-08 tested worse, 2.46 / 1.00
SEED = os.environ.get("KR_SEED", "")
GYRO_DIM = 3
MIRROR = os.environ.get("KR_MIRROR", "0") == "1"
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
gyro = walk_find(lambda r, f: "05_gyro_rotvec_deg.npy" in f)
assert code and assets and kitti and gyro, f"inputs missing: code={code} assets={assets} kitti={kitti} gyro={gyro}"
GYRO = gyro[0]
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


GFEAT = {}


def gyro_feat(seq: str) -> np.ndarray:
    if seq not in GFEAT:
        GFEAT[seq] = np.load(GYRO / f"{seq}_gyro_rotvec_deg.npy").astype(np.float32)
    return GFEAT[seq]


MIRROR_G = np.array([1.0, -1.0, -1.0], dtype=np.float32)  # rotation vector under the camera-x reflection


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
    j = 0  # dataset index of the next sample in the loader (shuffle=False)
    t = time.time()
    with torch.no_grad():
        for (imgs, imus, rot, wt), gts in loader:
            gf = []
            for b in range(imgs.shape[0]):  # gyro rotation of each of the 10 frame pairs of window j
                first = Path(ds.samples[j + b]["imgs"][0])
                seq, f0 = first.parent.parent.name, int(first.stem)
                gf.append(gyro_feat(seq)[f0:f0 + 10])
            j += imgs.shape[0]
            gf = np.stack(gf)
            passes = [(imgs, imus, gts)]
            if mirror:
                mi, mg = imus.clone(), gts.clone()
                mi[..., [1, 3, 5]] *= -1  # acc y, gyro x, gyro z
                mg[..., [1, 2, 3]] *= -1  # Euler about camera y, z; translation x
                passes.append((torch.flip(imgs, dims=[-1]), mi, mg))
            for pn, (pi, pm, pg) in enumerate(passes):
                fv, fi = encoder(pi.to(DEVICE).float(), pm.to(DEVICE).float())
                lat = torch.cat((fv, fi), 2).cpu().numpy()
                lat = np.concatenate([lat, gf * (MIRROR_G if pn == 1 else 1.0)], axis=2)
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
          f"model.tester.device={DEVICE}", f"trainer={'cpu' if LOCAL else 'gpu'}", f"model.net.input_dim={768 + GYRO_DIM}"] + ([f"seed={SEED}"] if SEED else [])
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


# ---------------------------------------------------------------- (3) own test: gyro channel appended, best and last
from src.metrics.kitti_metrics_calculator import KITTIMetricsCalculator  # noqa: E402
from src.models.components.pose_transformer import PoseTransformer  # noqa: E402
from src.testers.kitti_latent_tester import KITTILatentTester  # noqa: E402
from src.utils.kitti_utils import euler_from_matrix  # noqa: E402

inner = KITTILatentTester(val_seqs=VAL, data_dir=str(kd), seq_len=11, folder="sequences", img_w=512, img_h=256,
                          wrapper_weights_path=str(enc_path), device=DEVICE, v_f_len=512, i_f_len=256,
                          use_history_in_eval=True).kitti_latent_tester
enc = {}
with torch.inference_mode():
    for seq, df in zip(VAL, inner.dataloader):
        g = torch.from_numpy(gyro_feat(seq)).to(DEVICE)
        wins, start = [], 0
        for image_seq, imu_seq, gt_seq in df:  # data_partition windows: start, start+10, ...
            lat = inner.wrapper_model(image_seq.unsqueeze(0).to(DEVICE), imu_seq.unsqueeze(0).to(DEVICE))
            n = lat.shape[1]
            wins.append((torch.cat([lat, g[start:start + n].unsqueeze(0)], 2), gt_seq))
            start += n
        enc[seq] = (wins, np.asarray(df.poses_rel[:, :6], dtype=np.float64))
        log(f"encoded {seq}: {len(wins)} windows, {start} frame pairs (gyro {len(gyro_feat(seq))})")
del inner


def run(net, wins):
    """KITTI_tester_latent.test_one_path with use_history_in_eval=True, on the cached latents."""
    pose_list, hist = [], None
    with torch.inference_mode():
        for latents, gt_seq in wins:
            if hist is not None:
                res = torch.zeros(latents.shape[0], latents.shape[1], 6, device=latents.device)
                for idx in range(latents.shape[1]):
                    hist = torch.roll(hist, -1, 1)
                    hist[:, -1, :] = latents[:, idx, :]
                    res[:, idx, :] = net((hist, None, None), gt_seq)[:, -1, :]
                pose = res
            else:
                hist = latents.clone()
                pose = net((latents, None, None), gt_seq)
            pose_list.append(pose[0].detach().cpu().numpy())
    return np.vstack(pose_list).astype(np.float64)


def score(est: dict, gt: dict) -> dict:
    m = KITTIMetricsCalculator().calculate_metrics({s: {"estimated_poses": est[s], "gt_poses": gt[s]} for s in VAL})
    m["mean_t_rel"] = float(np.mean([m[f"{s}_t_rel"] for s in VAL]))
    m["mean_r_rel"] = float(np.mean([m[f"{s}_r_rel"] for s in VAL]))
    return {k: float(v) for k, v in m.items()}


gt = {s: enc[s][1] for s in VAL}
gyro_eul = {s: np.array([euler_from_matrix(r) for r in np.load(GYRO / f"{s}_gyro_R.npy")]) for s in VAL}
for tag, ck in (("best", best[-1]), ("last", ckpts / "last.ckpt")):
    if not ck.is_file():
        continue
    c = torch.load(ck, map_location="cpu", weights_only=False)
    status[f"{tag}_epoch"] = int(c["epoch"])
    net = PoseTransformer(input_dim=768 + GYRO_DIM, embedding_dim=768, num_layers=4, nhead=6, dim_feedforward=128, dropout=0.0)
    net.load_state_dict({k.removeprefix("net."): v for k, v in c["state_dict"].items() if k.startswith("net.")}, strict=True)
    net.eval().to(DEVICE)
    est = {s: run(net, enc[s][0])[: len(gt[s])] for s in VAL}
    hyb = {s: np.hstack([gyro_eul[s][: len(gt[s])], est[s][:, 3:]]) for s in VAL}
    for kind, e in (("poses", est), ("hybrid", hyb)):
        d = OUT / f"{kind}_{tag}"
        d.mkdir(exist_ok=True)
        for s in VAL:
            np.save(d / f"{s}_estimated_poses.npy", e[s])
            np.save(d / f"{s}_gt_poses.npy", gt[s])
        m = score(e, gt)
        status[f"{tag}_{kind}_t_rel"] = [m[f"{s}_t_rel"] for s in VAL]
        status[f"{tag}_{kind}_r_rel"] = [m[f"{s}_r_rel"] for s in VAL]
        status[f"{tag}_{kind}_mean"] = [round(m["mean_t_rel"], 3), round(m["mean_r_rel"], 3)]
        log(f"{tag} {kind:6s}: " + " ".join(f"{s} {m[s + '_t_rel']:.2f}/{m[s + '_r_rel']:.2f}" for s in VAL)
            + f" | mean {m['mean_t_rel']:.2f} / {m['mean_r_rel']:.2f}  (VIFT 1.96 / 0.46; ours 2.11 / 0.67; gyro + our net 1.70 / 0.18)")
    c["optimizer_states"], c["lr_schedulers"] = [], []
    torch.save(c, OUT / f"rpmg_{tag}.ckpt")
    save_status()
save_status()
if not LOCAL:
    shutil.rmtree(ips, ignore_errors=True)
    shutil.rmtree(latent_root, ignore_errors=True)  # ~90k files; listing them rate-limits `kaggle kernels output`
log("done")
