"""EuRoC V2_03_difficult, a sequence that no stage of the paper has touched (training MH_01-04 + V2_01, validation V1_03,
test MH_05 / V2_02 / V1_01 / V1_02): zero-shot predictions of the three released-for-the-paper PI-VIFT EuRoC checkpoints,
and the two-view geometry of the sequence, so that gate v3 / v4 can be run on it unchanged (scripts/gate/run_final.py).

Same loader path, shadow tree and window_last tester as pivift-tumvi-infer14 (kernel_tuminfer); the sequence comes from
nadaelmadah/eurocmav. Outputs: /kaggle/working/tumft/pose_dumps_<ckpt>/V2_03_difficult_*.npy and /kaggle/working/geo/.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SEQ = "V2_03_difficult"
CKPT_NAMES = ["pivift_s123_last.ckpt", "pivift_s42_last.ckpt", "pivift_s7_best.ckpt"]
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
seq_hits = walk_find(lambda r, f: r.name == SEQ and (r / "mav0/cam0/data.csv").is_file())
assert seq_hits, f"{SEQ} not found in the attached datasets"
SEQ_DIR = seq_hits[0]
enc_hits = walk_find(lambda r, f: ENCODER in f)
assert enc_hits, "encoder not found"
ENC = enc_hits[0] / ENCODER
ckpts = []
for name in CKPT_NAMES:
    h = walk_find(lambda r, f, n=name: n in f)
    assert h, f"checkpoint {name} not found"
    ckpts.append(h[0] / name)
# the private dataset amrelmadah/ips-euroc-geo-scripts holds euroc/*.py and pi_vift/shape_diag.py at its root (the repo's scripts/ folder)
geo_hits = walk_find(lambda r, f: (r / "euroc/geo_direction.py").is_file() and (r / "pi_vift/shape_diag.py").is_file())
log(f"IPS code {IPS_SRC} | sequence {SEQ_DIR} | encoder {ENC} | ckpts {[c.name for c in ckpts]} | geo scripts {geo_hits[:1]}")
assert geo_hits, "euroc/geo_direction.py not found (attach amrelmadah/ips-euroc-geo-scripts)"

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
    f"data.train_loader.sequences=[{SEQ}]",
    f"data.val_loader.sequences=[{SEQ}]",
    f"model.tester.sequences=[{SEQ}]",
]
status = {}
for ck in ckpts:
    tag = ck.stem
    dumps = WORK / "phase5_pose_dumps"
    shutil.rmtree(dumps, ignore_errors=True)
    cmd = [sys.executable, "src/train.py", *OVERRIDES, "train=false", "test=true", "logger=csv",
           f"hydra.run.dir={WORK / ('v203_' + tag)}", f"ckpt_path='{ck}'"]
    log("$ " + " ".join(cmd))
    p = subprocess.run(cmd, cwd=vift, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (OUT / f"test_log_{tag}.txt").write_text(p.stdout)
    print(p.stdout[-1500:])
    if dumps.is_dir():
        shutil.copytree(dumps, OUT / f"pose_dumps_{tag}", dirs_exist_ok=True)
    status[tag] = p.returncode
    (OUT / "status.json").write_text(json.dumps(status, indent=2))
log(json.dumps(status))

# ---------------------------------------------------------------- two-view geometry of the sequence (CPU)
first = next((OUT / f"pose_dumps_{c.stem}" for c in ckpts if (OUT / f"pose_dumps_{c.stem}/{SEQ}_timestamps.npy").is_file()), None)
assert first is not None, "no dump with timestamps: inference failed"
geo_out = WORK / "geo"
geo_ips = WORK / "geo_ips"
shutil.rmtree(geo_ips, ignore_errors=True)
(geo_ips / "third_party").mkdir(parents=True)
os.symlink(vift, geo_ips / "third_party" / "VIFT")
shutil.copytree(geo_hits[0], geo_ips / "scripts", ignore=shutil.ignore_patterns("*.zip", "__pycache__"))
args = [f"{c.stem}={OUT / ('pose_dumps_' + c.stem)}" for c in ckpts if (OUT / f"pose_dumps_{c.stem}/{SEQ}_estimated_poses.npy").is_file()]
p = subprocess.run([sys.executable, str(geo_ips / "scripts/euroc/geo_direction.py"), str(SEQ_DIR.parent), str(geo_out), f"{first}:{SEQ}", *args],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
geo_out.mkdir(exist_ok=True)
(geo_out / "geo_log.txt").write_text(p.stdout)
print(p.stdout[-4000:])
shutil.rmtree(ips, ignore_errors=True)
shutil.rmtree(geo_ips, ignore_errors=True)
log(f"done, geometry exit code {p.returncode}")
sys.exit(p.returncode)
