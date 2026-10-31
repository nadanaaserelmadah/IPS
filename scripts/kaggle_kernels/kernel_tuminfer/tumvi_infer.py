"""Zero-shot PI-VIFT (fixed-GT EuRoC checkpoints, no TUM-VI training) on TUM-VI room1-4: DESIGN data for gate v4.

Gate v4 (estimating the network's direction uncertainty online instead of using KITTI's sigma_net) is designed
on held-out data only: KITTI 08 and these TUM-VI rooms, which are not test rooms (room5/6 are). Zero-shot models
are used so the network is out of domain, as on the test rooms. Same loader path, shadow tree, 16-bit PNG fix and
window_last tester as pivift-tumvi-ft. Outputs: /kaggle/working/tumft/pose_dumps_<ckpt>/.
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

SEED = 42
CKPT_NAME = "pivift_s42_last.ckpt"
ENCODER = "vf_512_if_256_3e-05.model"
EPOCHS = 15
T0 = time.time()
TRAIN_BUDGET_S = 8.5 * 3600  # Kaggle stops at 12 h; leave room for downloads and two tests
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "tumft"
RUN_DIR = WORK / "tumft_run"
RAW = Path("/tmp/tumvi_raw")
ROOMS = [f"dataset-room{i}_512_16" for i in range(1, 5)]
TRAIN, TEST = ROOMS[:4], ROOMS[4:]
DESIGN = ROOMS[:4]
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

OUT.mkdir(parents=True, exist_ok=True)
ckpts = sorted(p for h in walk_find(lambda r, f: any(x.startswith("pivift_") and x.endswith(".ckpt") for x in f))
               for p in h.glob("pivift_*.ckpt"))
log("checkpoints: " + ", ".join(p.name for p in ckpts))
assert len(ckpts) >= 3, ckpts
SEQ_OVERRIDES = [
    "experiment=euroc_joint_pivift",
    "data.train_loader.sequences=[" + ",".join(DESIGN) + "]",
    "data.val_loader.sequences=[" + DESIGN[-1] + "]",
    "model.tester.sequences=[" + ",".join(DESIGN) + "]",
]
status = {}
for ck in ckpts:
    tag = ck.stem
    out_dir = WORK / f"tuminfer_{tag}"
    dumps = WORK / "phase5_pose_dumps"
    shutil.rmtree(dumps, ignore_errors=True)
    cmd = [sys.executable, "src/train.py", *SEQ_OVERRIDES, "train=false", "test=true", "logger=csv",
           f"hydra.run.dir={out_dir}", f"ckpt_path='{ck}'"]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=vift, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f"test_log_{tag}.txt").write_text(p.stdout)
    print(p.stdout[-2000:])
    if dumps.is_dir():
        shutil.copytree(dumps, OUT / f"pose_dumps_{tag}", dirs_exist_ok=True)
    status[tag] = p.returncode
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
log(json.dumps(status))
shutil.rmtree(ips, ignore_errors=True)
log("done")
