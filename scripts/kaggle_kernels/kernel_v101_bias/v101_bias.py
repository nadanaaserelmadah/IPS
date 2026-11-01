"""EuRoC V1_01 gyro-bias diagnostic (pre-registered in RESULTS, 2026-11-01). CPU only.

1. Visual gyro bias (geo_direction.visual_gyro_bias, whole sequence, no ground truth) on MH_05, V1_02 and V2_02,
   which have a static start, compared with the static-prefix bias (validation: within 0.005 rad/s per axis).
2. The same estimate on V1_01, which has no static start (the original code used zero bias there).
3. V1_01 geometry cache recomputed with that bias (env IPS_GYRO_BIAS_JSON), everything else unchanged.
Outputs land in /kaggle/working/v101/: bias.json, validation.json, V1_01_easy_geo.npz, log.txt.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "omegaconf", "lightning", "natsort", "rich"], check=False)


def find(pred):
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if root.endswith(("/data", "/cam1")):
            dirs[:] = []
        if pred(Path(root), dirs, files):
            return Path(root)
    raise SystemExit("not found")


code = find(lambda r, d, f: r.name == "VIFT" and r.parent.name == "third_party")
euroc = find(lambda r, d, f: "MH_05_difficult" in d and "V1_01_easy" in d)
inp = find(lambda r, d, f: "dumps" in d and "scripts" in d)
print("code", code, "| euroc", euroc, "| inputs", inp, flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(inp / "scripts", ips / "scripts")
sys.path.insert(0, str(ips / "scripts/euroc"))
out = WORK / "v101"
out.mkdir(exist_ok=True)

import numpy as np  # noqa: E402

import geo_direction as G  # noqa: E402
import gyro_hybrid_euroc as gh  # noqa: E402

val, bias = {}, {}
for seq in ("MH_05_difficult", "V1_02_medium", "V2_02_medium", "V1_01_easy"):
    imu = np.loadtxt(euroc / seq / "mav0/imu0/data.csv", delimiter=",", comments="#")
    b_static, s_static = gh.static_bias(imu[:, 0].astype(np.int64), imu[:, 1:4], imu[:, 4:7])
    b_vis, info = G.visual_gyro_bias(euroc, seq)
    row = {"static": None if b_static is None else b_static.tolist(), "static_s": s_static,
           "visual": None if b_vis is None else b_vis.tolist(), **info}
    if b_static is not None and b_vis is not None:
        row["max_abs_diff"] = float(np.abs(b_static - b_vis).max())
    val[seq] = row
    if b_vis is not None:
        bias[seq] = b_vis.tolist()
    print(seq, json.dumps(row), flush=True)
(out / "validation.json").write_text(json.dumps(val, indent=2))
(out / "bias.json").write_text(json.dumps(bias, indent=2))
checked = [v["max_abs_diff"] for k, v in val.items() if "max_abs_diff" in v]
print("validation: max diff per sequence", checked, "-> pass" if checked and max(checked) <= 0.005 else "-> FLAG", flush=True)

os.environ["IPS_GYRO_BIAS_JSON"] = str(out / "bias.json")
t_end = np.load(inp / "dumps/pivift_dir_best/V1_01_easy_timestamps.npy").astype(np.int64)
cache = WORK / "cache"
cache.mkdir(exist_ok=True)
G.geo_sequence(euroc, "V1_01_easy", t_end, cache)
shutil.copy2(cache / "V1_01_easy_geo.npz", out / "V1_01_easy_geo.npz")
print("done", flush=True)
shutil.rmtree(ips, ignore_errors=True)
