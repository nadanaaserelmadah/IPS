"""Re-create the lost pose dumps of PI-VIFT pivift_s123_last on the four EuRoC test sequences (MH_05, V2_02, V1_01, V1_02),
needed for Fig. 4 (measure vs learn) of the Scientific Reports version (scripts/gate/plot_observability.py reads
EUROC_DUMPS/pivift_s123_last/<seq>_{estimated,gt}_poses.npy). Same loader, tester and overrides as kernel_v203_infer
(itself the same as kernel_tuminfer), one checkpoint, no geometry (the EuRoC geometry is cached locally).
Expected network SE(3) ATE (docs/RESULTS.md, PI-VIFT s123 last): 9.08 / 5.11 / 4.06 / 3.63, mean 5.47 m.
Outputs: /kaggle/working/tumft/pose_dumps_pivift_s123_last/<seq>_*.npy and test_log_pivift_s123_last.txt.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEQS = ["MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium"]
CKPT_NAMES = ["pivift_s123_last.ckpt"]
ENCODER = "vf_512_if_256_3e-05.model"
T0 = time.time()
INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
OUT = WORK / "tumft"


def log(msg: str) -> None:
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


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


# ---------------------------------------------------------------- locate inputs
ips_hits = walk_find(lambda r, f: (r / "third_party/VIFT/src/models/components/pi_vift.py").is_file())
assert ips_hits, "IPS code dataset not attached"
IPS_SRC = ips_hits[0]
SEQ_DIRS = {}
for SEQ in SEQS:
    seq_hits = walk_find(lambda r, f, s=SEQ: r.name == s and (r / "mav0/cam0/data.csv").is_file())
    assert seq_hits, f"{SEQ} not found in the attached datasets"
    SEQ_DIRS[SEQ] = seq_hits[0]
enc_hits = walk_find(lambda r, f: ENCODER in f)
assert enc_hits, "encoder not found"
ENC = enc_hits[0] / ENCODER
ckpts = []
for name in CKPT_NAMES:
    h = walk_find(lambda r, f, n=name: n in f)
    assert h, f"checkpoint {name} not found"
    ckpts.append(h[0] / name)
log(f"IPS code {IPS_SRC} | sequences {SEQ_DIRS} | encoder {ENC} | ckpts {[c.name for c in ckpts]}")

# ---------------------------------------------------------------- workspace and EuRoC-layout tree
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
for SEQ, SEQ_DIR in SEQ_DIRS.items():
    dst = vift / "data/euroc" / SEQ
    (dst / "mav0").mkdir(parents=True, exist_ok=True)
    for s in ("cam0", "cam1", "imu0", "state_groundtruth_estimate0"):
        if (SEQ_DIR / "mav0" / s).exists() and not (dst / "mav0" / s).exists():
            (dst / "mav0" / s).symlink_to(SEQ_DIR / "mav0" / s)

assert sh([sys.executable, "-m", "pip", "install", "-q", "-r", str(vift / "requirements.txt")]) == 0
import torch  # noqa: E402

assert torch.cuda.is_available(), "GPU not enabled for this session"
assert sh([sys.executable, "tests/test_pi_vift.py"], cwd=vift) == 0, "PI-VIFT unit tests failed"
OUT.mkdir(parents=True, exist_ok=True)

OVERRIDES = [
    "experiment=euroc_joint_pivift",
    f"data.train_loader.sequences=[{SEQS[0]}]",
    f"data.val_loader.sequences=[{SEQS[0]}]",
    f"model.tester.sequences=[{','.join(SEQS)}]",
]
status = {}
for ck in ckpts:
    tag = ck.stem
    dumps = WORK / "phase5_pose_dumps"
    shutil.rmtree(dumps, ignore_errors=True)
    cmd = [sys.executable, "src/train.py", *OVERRIDES, "train=false", "test=true", "logger=csv",
           f"hydra.run.dir={WORK / ('euroc4_' + tag)}", f"ckpt_path='{ck}'"]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=vift, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f"test_log_{tag}.txt").write_text(p.stdout)
    print(p.stdout[-1500:])
    if dumps.is_dir():
        shutil.copytree(dumps, OUT / f"pose_dumps_{tag}", dirs_exist_ok=True)
    status[tag] = p.returncode
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
log(json.dumps(status))

missing = [s for s in SEQS if not (OUT / f"pose_dumps_pivift_s123_last/{s}_estimated_poses.npy").is_file()]
shutil.rmtree(ips, ignore_errors=True)
log(f"done, missing dumps: {missing}")
sys.exit(1 if missing else 0)
