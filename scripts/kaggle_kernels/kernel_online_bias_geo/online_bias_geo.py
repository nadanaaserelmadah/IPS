"""Online gyro bias, stage 3: EuRoC geometry caches (gyro rotations R + two-view directions) recomputed with the frozen causal online
bias (IPS_ONLINE_BIAS_DIR) on MH_05, V2_02, V1_01, V1_02 and V2_03, plus a control: MH_05 with the original static-prefix bias (must equal the
cache already used by the paper). CPU only. Outputs: /kaggle/working/geo_online/, /kaggle/working/geo_static_ctrl/."""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
SEQS = ["MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium", "V2_03_difficult"]
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "omegaconf", "lightning", "natsort", "rich", "opencv-python-headless"], check=False)
T0 = time.time()


def find(pred, depth=8):
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if root.endswith(("/cam0/data", "/cam1/data")):
            dirs[:] = []
        if root[len(str(INPUT)):].count(os.sep) >= depth:
            dirs[:] = []
        if pred(Path(root), dirs, files):
            return Path(root)
    raise SystemExit("not found")


code = find(lambda r, d, f: r.name == "VIFT" and r.parent.name == "third_party")
scr = find(lambda r, d, f: (r / "euroc/geo_direction.py").is_file() and (r / "pi_vift/shape_diag.py").is_file())
ob = find(lambda r, d, f: "MH_05_difficult_online_bias.npz" in f)
tsd = find(lambda r, d, f: "MH_05_difficult_timestamps.npy" in f)
first = find(lambda r, d, f: (r / "mav0/cam0/data.csv").is_file())
root = first.parent
print("code", code, "| scripts", scr, "| bias", ob, "| ts", tsd, "| root", root, flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(scr, ips / "scripts", ignore=shutil.ignore_patterns("*.zip", "__pycache__"))
sys.path.insert(0, str(ips / "scripts/euroc"))
import numpy as np  # noqa: E402

import geo_direction as G  # noqa: E402

for tag, env_dir, seqs in (("geo_static_ctrl", None, ["MH_05_difficult"]), ("geo_online", str(ob), SEQS)):
    if env_dir:
        os.environ["IPS_ONLINE_BIAS_DIR"] = env_dir
    else:
        os.environ.pop("IPS_ONLINE_BIAS_DIR", None)
    out = WORK / tag
    out.mkdir(exist_ok=True)
    for seq in seqs:
        t_end = np.load(tsd / f"{seq}_timestamps.npy").astype(np.int64)
        R, d, t_st, info = G.geo_sequence(root, seq, t_end, out)
        print(f"[{(time.time() - T0) / 60:.1f} min] {tag} {seq}: {len(t_end)} edges, geometry on {np.mean(~np.isnan(d[:, 0])):.0%}", flush=True)
shutil.rmtree(ips, ignore_errors=True)
