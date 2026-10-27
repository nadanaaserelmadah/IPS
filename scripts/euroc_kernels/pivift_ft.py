"""PI-VIFT on the fixed GT with the encoder fine-tuned (scene-independence test for P8-D).

P8-D: away from the training rooms the estimated translation direction is close to random
(VIFT-style V2_02 55 deg, PI-VIFT V1_01 90 deg), i.e. the frozen KITTI encoder + head looks
scene-memorizing. Here the encoder is unfrozen after epoch 1 at 0.1x the fusion LR
(model.freeze_encoder_epochs=1), with colour + IMU-bias augmentation, dropout 0.1 and weight
decay 0.05 (euroc_joint_pivift_aug). Otherwise the pivift-qfix-baseline protocol: 12 epochs,
best val/loss + best val/ate_se3 + last, tested on MH_05, V2_02, V1_01 and V1_02 in one run
(4-sequence P8 means; compare to PI-VIFT 4.43-4.98 and VIFT-style 4.02-4.11 m, floor 3.15 m).
Outputs land in /kaggle/working/pivift_ft/.

Local smoke test: FT_LOCAL=1 FT_INPUT=... FT_WORK=... FT_EXTRA="..." FT_TEST_SEQS=MH_01_easy
(skips pip / encoder download / GPU check, trains in place of the Kaggle copy).
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

EXP = "ft"
LOCAL = os.environ.get("FT_LOCAL") == "1"
T0 = time.time()
EPOCHS = int(os.environ.get("FT_EPOCHS", "12"))
TRAIN_BUDGET_S = 10.0 * 3600  # Kaggle stops at 12 h; leave room for three tests
INPUT = Path(os.environ.get("FT_INPUT", "/kaggle/input"))
WORK = Path(os.environ.get("FT_WORK", "/kaggle/working"))
OUT = WORK / f"pivift_{EXP}"
RUN_DIR = WORK / "pivift_run"
TRAIN_SEQS = ["MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "V2_01_easy"]
TEST_SEQS = os.environ.get("FT_TEST_SEQS", "MH_05_difficult,V2_02_medium,V1_01_easy,V1_02_medium").split(",")
NEEDED_SEQS = TRAIN_SEQS + ["V1_03_difficult"] + TEST_SEQS
EXTRA = os.environ.get("FT_EXTRA", "").split()  # local smoke overrides


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


def sh(cmd: list[str], **kw) -> int:
    log("$ " + " ".join(cmd))
    return subprocess.call(cmd, **kw)


# ---------------------------------------------------------------- locate inputs
ips_hits = walk_find(lambda r, f: (r / "third_party/VIFT/src/models/components/pi_vift.py").is_file())
assert ips_hits, "IPS code dataset not attached"
IPS_SRC = ips_hits[0]
assert "s * qw * qx" in (IPS_SRC / "third_party/VIFT/src/utils/euroc_utils.py").read_text(), \
    "code dataset predates the GT quaternion fix"
assert "val/ate_se3" in (IPS_SRC / "third_party/VIFT/src/models/euroc_joint_module.py").read_text(), \
    "code dataset predates val/ate_se3"
log(f"IPS code : {IPS_SRC}  (experiment {EXP})")

by_name = {}
for d in walk_find(lambda r, f: (r / "mav0/cam0/data.csv").is_file()):
    by_name.setdefault(d.name, d)
missing = [s for s in NEEDED_SEQS if s not in by_name]
assert LOCAL or not missing, f"Missing EuRoC sequences: {missing}"

EXP_OVERRIDES = ["experiment=euroc_joint_pivift_aug", "model.freeze_encoder_epochs=1", "model.encoder_lr_scale=0.1"]

# ---------------------------------------------------------------- workspace
ips = WORK / "IPS"
if LOCAL:
    ips = IPS_SRC
elif not ips.is_dir():
    shutil.copytree(IPS_SRC, ips, ignore=shutil.ignore_patterns(".venv", "logs", "*.ckpt"))
vift = ips / "third_party/VIFT"
(vift / ".project-root").touch()
os.environ["PROJECT_ROOT"] = str(vift)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, str(vift))

euroc = vift / "data/euroc"
euroc.mkdir(parents=True, exist_ok=True)
for s in NEEDED_SEQS:
    link = euroc / s
    if not link.exists() and s in by_name:
        link.symlink_to(by_name[s])

if not LOCAL:
  assert sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(vift / "requirements.txt")]) == 0
  if sh([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"), "--vift-root", str(vift)]) != 0:
    sh([sys.executable, "-m", "pip", "install", "-q", "gdown"])
    assert sh([sys.executable, str(ips / "scripts/phase0/download_pretrained_encoder.py"),
               "--vift-root", str(vift), "--gdown"]) == 0, "encoder weights unavailable"

import torch  # noqa: E402

assert LOCAL or torch.cuda.is_available(), "GPU not enabled for this session"
assert sh([sys.executable, "tests/test_pi_vift.py"], cwd=vift) == 0, "PI-VIFT unit tests failed"
# Fatal: includes the real-data check (gyro vs GT rotation < 1 deg, |g_cam| = 9.81) on MH_05.
assert sh([sys.executable, "tests/test_gt_quaternion.py"], cwd=vift,
          env={**os.environ, **({} if LOCAL else {"EUROC_REAL_ROOT": str(euroc)})}) == 0, "GT quaternion tests failed"
OUT.mkdir(parents=True, exist_ok=True)


def ckpt_epoch(path: Path) -> int:
    return int(torch.load(path, map_location="cpu", weights_only=False).get("epoch", -1)) + 1


def run(cmd: list[str], workers: int, deadline: float) -> tuple[int, bool]:
    """Train; at the deadline send SIGINT so Lightning stops cleanly. Returns (rc, stopped)."""
    full = cmd + [f"data.num_workers={workers}", f"data.pin_memory={workers > 0}"]
    log("$ " + " ".join(full))
    p = subprocess.Popen(full, cwd=vift)
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


# ---------------------------------------------------------------- train
ckpt_dir = RUN_DIR / "checkpoints"
train = [
    sys.executable, "src/train.py", *EXP_OVERRIDES,
    "train=true", "test=false", "logger=csv", *EXTRA,
    "model.tester.sequences=[" + ",".join(TEST_SEQS) + "]",  # the tester is built at train time too
    f"trainer.max_epochs={EPOCHS}",
    f"hydra.run.dir={RUN_DIR}",
    # Best val/loss at per-epoch resolution, plus last.ckpt every epoch.
    "callbacks.model_checkpoint.every_n_epochs=1",
    "callbacks.model_checkpoint.save_top_k=1",
    "callbacks.model_checkpoint_periodic.every_n_epochs=1",
    "~callbacks.checkpoint_zip",
    # Second best: val/ate_se3 (SE3 ATE of V1_03 from window_last edges).
    "+callbacks.model_checkpoint_ate._target_=lightning.pytorch.callbacks.ModelCheckpoint",
    f"+callbacks.model_checkpoint_ate.dirpath={ckpt_dir}",
    "+callbacks.model_checkpoint_ate.filename='bestate_{epoch:03d}'",
    "+callbacks.model_checkpoint_ate.monitor=val/ate_se3",
    "+callbacks.model_checkpoint_ate.mode=min",
    "+callbacks.model_checkpoint_ate.save_top_k=1",
    "+callbacks.model_checkpoint_ate.every_n_epochs=1",
    "+callbacks.model_checkpoint_ate.auto_insert_metric_name=false",
]
deadline = T0 + TRAIN_BUDGET_S
workers = min(4, os.cpu_count() or 1)
t_start = time.time()
rc, stopped = run(train, workers, deadline)
if rc != 0 and not stopped and time.time() - t_start < 1800:
    log("Early failure with DataLoader workers: retrying with num_workers=0")
    shutil.rmtree(RUN_DIR, ignore_errors=True)
    rc, stopped = run(train, 0, deadline)
log(f"Training exit code: {rc} (time-stopped: {stopped})")

status = {"experiment": EXP, "overrides": EXP_OVERRIDES, "train_exit_code": rc, "time_stopped": stopped}
ckpts: dict[str, Path] = {}
for tag, pattern in (("best", "best_*.ckpt"), ("bestate", "bestate_*.ckpt")):
    found = sorted(ckpt_dir.glob(pattern))
    assert len(found) == 1, f"{pattern}: {found}"
    shutil.copy2(found[0], OUT / f"pivift_{tag}.ckpt")
    ckpts[tag] = OUT / f"pivift_{tag}.ckpt"
    status[f"{tag}_ckpt"], status[f"{tag}_epochs"] = found[0].name, ckpt_epoch(found[0])
last = ckpt_dir / "last.ckpt"
if last.is_file():
    shutil.copy2(last, OUT / "pivift_last.ckpt")
    ckpts["last"] = OUT / "pivift_last.ckpt"
    status["last_epochs"] = ckpt_epoch(last)
for p in RUN_DIR.rglob("metrics.csv"):
    shutil.copy2(p, OUT / "metrics.csv")
log(json.dumps(status))


# ---------------------------------------------------------------- test best (val/loss), best (val ATE), last
def test(ckpt: Path, tag: str) -> int:
    out_dir = WORK / f"pivift_test_{tag}"
    cmd = [
        sys.executable, "src/train.py", *EXP_OVERRIDES,
        "train=false", "test=true", "logger=csv", *EXTRA,
        "model.tester.sequences=[" + ",".join(TEST_SEQS) + "]",
        f"hydra.run.dir={out_dir}", f"ckpt_path='{ckpt}'",
    ]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=vift, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f"test_log_{tag}.txt").write_text(p.stdout)
    print(p.stdout[-4000:])
    for m in out_dir.rglob("metrics.csv"):
        shutil.copy2(m, OUT / f"test_metrics_{tag}.csv")
    dumps = WORK / "phase5_pose_dumps"
    if dumps.is_dir():
        shutil.copytree(dumps, OUT / f"pose_dumps_{tag}", dirs_exist_ok=True)
    shutil.rmtree(out_dir / "checkpoints", ignore_errors=True)
    return p.returncode


tested: dict[int, str] = {}
for tag in ("best", "bestate", "last"):
    if tag not in ckpts:
        continue
    ep = status[f"{tag}_epochs"]
    if ep in tested:  # same epoch as an earlier tag: same weights, skip the rerun
        status[f"test_{tag}"] = f"same as {tested[ep]}"
        continue
    tested[ep] = tag
    status[f"test_{tag}_exit_code"] = test(ckpts[tag], tag)
    status["session_hours"] = round((time.time() - T0) / 3600, 2)
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
(OUT / "status.json").write_text(json.dumps(status, indent=2))  # also records skipped "same as" tags
log(json.dumps(status))

if not LOCAL:
    shutil.rmtree(ips, ignore_errors=True)
shutil.rmtree(ckpt_dir, ignore_errors=True)
log("done")
