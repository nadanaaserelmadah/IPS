"""TUM-VI fine-tune of the fixed-GT EuRoC PI-VIFT: the third dataset for the frozen observability gate v3.

Train room1-4, test room5-6 (all inside the mocap room, so GT is optical motion capture throughout).
room1/2 come from nadaelmadah/tum-vi; room3-6 are streamed from the official TUM server. TUM-VI goes
through the EuRoC loader via an EuRoC-layout shadow tree (mocap0 as state_groundtruth_estimate0; T_BS
from dso/camchain.yaml). Frames are 16-bit PNGs: the code dataset must have load_image_rgb (before it,
convert("RGB") clipped every TUM-VI frame to white).
Before training, a fatal real-data check per room: gyro vs mocap rotation, dp physics, real motion,
non-saturated frames. Then tests the EuRoC checkpoint zero-shot and the fine-tuned last.ckpt (no
selection) on room5/6 in window_last, keeping pose dumps with timestamps, plus room5/6 IMU, mocap and
calibration for the local gate run. Outputs land in /kaggle/working/tumft/.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

SEED = 123
CKPT_NAME = "pivift_s123_last.ckpt"
ENCODER = "vf_512_if_256_3e-05.model"
EPOCHS = 15
T0 = time.time()
TRAIN_BUDGET_S = 8.5 * 3600  # Kaggle stops at 12 h; leave room for downloads and two tests
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "tumft"
RUN_DIR = WORK / "tumft_run"
RAW = Path("/tmp/tumvi_raw")
ROOMS = [f"dataset-room{i}_512_16" for i in range(1, 7)]
TRAIN, TEST = ROOMS[:4], ROOMS[4:]
URL = "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/{}.tar"


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 3600:5.2f} h] {msg}", flush=True)


def walk_find(pred, base: Path = INPUT, max_depth: int = 7):
    hits = []
    for root, dirs, files in os.walk(base, followlinks=True):
        rel_depth = root[len(str(base)):].count(os.sep)
        if root.endswith(("/cam0", "/cam1")):
            dirs[:] = [d for d in dirs if d != "data"]
        if rel_depth >= max_depth:
            dirs[:] = []
        if pred(Path(root), files):
            hits.append(Path(root))
    return hits


def sh(cmd, **kw) -> int:
    log("$ " + (cmd if isinstance(cmd, str) else " ".join(cmd)))
    return subprocess.call(cmd, **kw)


def find_rooms(base: Path) -> dict:
    return {d.name: d for d in walk_find(lambda r, f: r.name in ROOMS and (r / "mav0/cam0/data.csv").is_file(), base)}


# ---------------------------------------------------------------- locate inputs
ips_hits = walk_find(lambda r, f: (r / "third_party/VIFT/src/models/components/pi_vift.py").is_file())
assert ips_hits, "IPS code dataset not attached"
IPS_SRC = ips_hits[0]
assert "def load_image_rgb" in (IPS_SRC / "third_party/VIFT/src/utils/euroc_utils.py").read_text(), \
    "code dataset predates the 16-bit PNG fix"
ckpt_hits = walk_find(lambda r, f: CKPT_NAME in f)
enc_hits = walk_find(lambda r, f: ENCODER in f)
assert ckpt_hits and enc_hits, (ckpt_hits, enc_hits)
CKPT, ENC = ckpt_hits[0] / CKPT_NAME, enc_hits[0] / ENCODER
log(f"IPS code {IPS_SRC} | ckpt {CKPT} | encoder {ENC}")

rooms = find_rooms(INPUT)
RAW.mkdir(parents=True, exist_ok=True)
for r in ROOMS:
    if r in rooms:
        continue
    for attempt in range(3):
        if sh(["bash", "-o", "pipefail", "-c", f"curl -sSfL --retry 3 '{URL.format(r)}' | tar -x -C {RAW}"]) == 0:
            break
        log(f"download of {r} failed (attempt {attempt + 1})")
    rooms.update(find_rooms(RAW))
    assert r in rooms, f"{r} not found after download"
log("rooms: " + json.dumps({k: str(v) for k, v in sorted(rooms.items())}))

# ---------------------------------------------------------------- workspace
ips = WORK / "IPS"
if not ips.is_dir():
    shutil.copytree(IPS_SRC, ips, ignore=shutil.ignore_patterns(".venv", "logs", "*.ckpt"))
vift = ips / "third_party/VIFT"
(vift / ".project-root").touch()
os.environ["PROJECT_ROOT"] = str(vift)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(vift))
(vift / "pretrained_models").mkdir(exist_ok=True)
if not (vift / "pretrained_models" / ENCODER).exists():
    (vift / "pretrained_models" / ENCODER).symlink_to(ENC)

# EuRoC-layout shadow tree: data/euroc/<room>/{dso, mav0/{cam0,cam1,imu0,mocap0,state_groundtruth_estimate0}}
euroc = vift / "data/euroc"
for r, src in rooms.items():
    dst = euroc / r
    (dst / "mav0").mkdir(parents=True, exist_ok=True)
    if not (dst / "dso").exists():
        (dst / "dso").symlink_to(src / "dso")
    for s in ("cam0", "cam1", "imu0", "mocap0"):
        if not (dst / "mav0" / s).exists():
            (dst / "mav0" / s).symlink_to(src / "mav0" / s)
    if not (dst / "mav0/state_groundtruth_estimate0").exists():
        (dst / "mav0/state_groundtruth_estimate0").symlink_to(src / "mav0/mocap0")

assert sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(vift / "requirements.txt")]) == 0

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert torch.cuda.is_available(), "GPU not enabled for this session"
assert sh([sys.executable, "tests/test_pi_vift.py"], cwd=vift) == 0, "PI-VIFT unit tests failed"
assert sh([sys.executable, "tests/test_tum_vi_gt.py"], cwd=vift) == 0, "TUM-VI GT / 16-bit loader tests failed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- fatal real-data check per room
from src.utils.euroc_utils import build_sequence_samples, euler_to_rotation_matrix, load_image_rgb  # noqa: E402
from src.utils.imu_canonical import load_T_BS  # noqa: E402
from src.utils.preintegration_torch import preintegrate_accel_first  # noqa: E402

checks = {}
for r in ROOMS:
    seq = euroc / r
    t_bs = load_T_BS(seq)
    assert not np.allclose(t_bs, np.eye(4)), f"{r}: T_BS fell back to identity"
    t_ci = np.linalg.inv(t_bs)[:3, 3]
    samples, _ = build_sequence_samples(seq)
    rot, dpe, step = [], [], []
    for s in samples[::10]:
        im = np.asarray(s["imus"])
        dts = (s["t_end"] - s["t_start"]) / 1e9 / 10
        win = torch.tensor(np.stack([im[k * 10: k * 10 + 11] for k in range(10)])[None], dtype=torch.float64)
        dR, _, dp, _ = (x[0].numpy() for x in preintegrate_accel_first(win, dt=dts / 10))
        for k, rel in enumerate(np.asarray(s["gts"])):
            r_gt = euler_to_rotation_matrix(rel[:3])
            rot.append(np.degrees(np.arccos(np.clip((np.trace(dR[k].T @ r_gt) - 1) / 2, -1, 1))))
            p = s["vel_cam"][k] * dts + 0.5 * s["g_cam"][k] * dts ** 2 + dp[k] + (np.eye(3) - r_gt) @ t_ci
            dpe.append(np.linalg.norm(p - rel[3:]))
            step.append(np.linalg.norm(rel[3:]))
    # mocap gaps: fraction of 100 ms edges whose span contains a mocap gap > 50 ms (GT interpolated there)
    mt = np.loadtxt(seq / "mav0/mocap0/data.csv", delimiter=",", comments="#", usecols=0).astype(np.int64)
    ct = np.loadtxt(seq / "mav0/cam0/data.csv", delimiter=",", comments="#", usecols=0, dtype=np.int64)[::2]
    gi = np.where(np.diff(mt) > 50_000_000)[0]
    bad = np.zeros(len(ct) - 1, bool)
    for g in gi:
        bad |= (ct[1:] > mt[g]) & (ct[:-1] < mt[g + 1])
    img = load_image_rgb(samples[len(samples) // 2]["imgs"][0])
    checks[r] = {"windows": len(samples), "rot_med_deg": float(np.median(rot)), "dp_med_mm": 1e3 * float(np.median(dpe)),
                 "step_med_m": float(np.median(step)), "gap_edge_frac": float(bad.mean()),
                 "img_sat_frac": float((img == 255).mean()), "img_std": float(img.std())}
    log(f"{r}: {json.dumps(checks[r])}")
(OUT / "checks.json").write_text(json.dumps(checks, indent=2))
for r, c in checks.items():
    assert c["step_med_m"] > 0.02 and c["rot_med_deg"] < 0.5 and c["dp_med_mm"] < 3.0, (r, c)
    assert c["img_sat_frac"] < 0.2 and c["img_std"] > 5, (r, c)

# room5/6 IMU, mocap and calibration for the local gate run
for r in TEST:
    d = OUT / "tum_meta" / r
    for s in ("imu0", "mocap0", "cam0"):
        (d / "mav0" / s).mkdir(parents=True, exist_ok=True)
        shutil.copy2(euroc / r / "mav0" / s / "data.csv", d / "mav0" / s / "data.csv")
    shutil.copytree(euroc / r / "dso", d / "dso", ignore=shutil.ignore_patterns("images", "*.png"), dirs_exist_ok=True)

SEQ_OVERRIDES = [
    "experiment=euroc_joint_pivift",
    "data.train_loader.sequences=[" + ",".join(TRAIN) + "]",
    "data.val_loader.sequences=[" + TRAIN[-1] + "]",  # monitoring only; the gate uses last.ckpt
    "model.tester.sequences=[" + ",".join(TEST) + "]",
]


def run(cmd: list[str], workers: int, deadline: float, logfile: Path) -> tuple[int, bool]:
    """Train; at the deadline send SIGINT so Lightning stops cleanly. Returns (rc, stopped)."""
    full = cmd + [f"data.num_workers={workers}", f"data.pin_memory={workers > 0}"]
    log("$ " + " ".join(full))
    with open(logfile, "w") as fh:
        p = subprocess.Popen(full, cwd=vift, stdout=fh, stderr=subprocess.STDOUT)
        try:
            return p.wait(timeout=max(60.0, deadline - time.time())), False
        except subprocess.TimeoutExpired:
            log("Training budget reached: sending SIGINT")
            p.send_signal(signal.SIGINT)
            try:
                return p.wait(timeout=900), True
            except subprocess.TimeoutExpired:
                p.kill()
                return p.wait(), True


def test(ckpt: Path, tag: str) -> int:
    out_dir = WORK / f"tumft_test_{tag}"
    dumps = WORK / "phase5_pose_dumps"
    shutil.rmtree(dumps, ignore_errors=True)
    cmd = [sys.executable, "src/train.py", *SEQ_OVERRIDES, "train=false", "test=true", "logger=csv",
           f"hydra.run.dir={out_dir}", f"ckpt_path='{ckpt}'"]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=vift, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f"test_log_{tag}.txt").write_text(p.stdout)
    print(p.stdout[-3000:])
    for m in out_dir.rglob("metrics.csv"):
        shutil.copy2(m, OUT / f"test_metrics_{tag}.csv")
    if dumps.is_dir():
        shutil.copytree(dumps, OUT / f"pose_dumps_{tag}", dirs_exist_ok=True)
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return p.returncode


status = {"seed": SEED, "init_ckpt": CKPT_NAME, "train": TRAIN, "test": TEST, "epochs": EPOCHS}
status["test_zeroshot_exit_code"] = test(CKPT, "zeroshot")
(OUT / "status.json").write_text(json.dumps(status, indent=2))

# ---------------------------------------------------------------- fine-tune
ckpt_dir = RUN_DIR / "checkpoints"
train = [
    sys.executable, "src/train.py", *SEQ_OVERRIDES,
    "train=true", "test=false", "logger=csv", f"seed={SEED}",
    f"ckpt_path='{CKPT}'", "init_ckpt_weights=true",
    f"trainer.max_epochs={EPOCHS}",
    f"hydra.run.dir={RUN_DIR}",
    "callbacks.model_checkpoint.every_n_epochs=1",
    "callbacks.model_checkpoint.save_top_k=1",
    "callbacks.model_checkpoint_periodic.every_n_epochs=1",
    "~callbacks.checkpoint_zip",
]
deadline = T0 + TRAIN_BUDGET_S
workers = min(4, os.cpu_count() or 1)
t_start = time.time()
rc, stopped = run(train, workers, deadline, OUT / "train_log.txt")
if rc != 0 and not stopped and time.time() - t_start < 1800:
    log("Early failure with DataLoader workers: retrying with num_workers=0")
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    rc, stopped = run(train, 0, deadline, OUT / "train_log.txt")
tl = (OUT / "train_log.txt").read_text(errors="ignore")
print(tl[-4000:])
status.update(train_exit_code=rc, time_stopped=stopped,
              warm_start_missing_keys="Warm-start missing keys" in tl,
              warm_start_unexpected_keys="Warm-start unexpected keys" in tl)
for p in RUN_DIR.rglob("metrics.csv"):
    shutil.copy2(p, OUT / "metrics.csv")
last = ckpt_dir / "last.ckpt"
if last.is_file():
    shutil.copy2(last, OUT / "tumft_last.ckpt")
    status["last_epochs"] = int(torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)) + 1
    status["test_last_exit_code"] = test(OUT / "tumft_last.ckpt", "last")
status["session_hours"] = round((time.time() - T0) / 3600, 2)
(OUT / "status.json").write_text(json.dumps(status, indent=2))
log(json.dumps(status))

shutil.rmtree(ips, ignore_errors=True)
shutil.rmtree(ckpt_dir, ignore_errors=True)
log("done")
