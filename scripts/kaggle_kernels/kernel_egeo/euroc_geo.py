"""EuRoC: gyro rotation + geometric direction (adaptive baseline for slow motion) x network /
stereo / accelerometer length, on MH_05, V2_02, V1_01, V1_02 with the saved P8 / P8-G pose dumps.
CPU only, no training. Outputs land in /kaggle/working/geo/.
"""
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
geo = find(lambda r, d, f: "dumps" in d and "scripts" in d)
print("code", code, "| euroc", euroc, "| geo", geo, flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(geo / "scripts", ips / "scripts")
out = WORK / "geo"
dumps = geo / "dumps"
args = [f"{d.name}={d}" for d in sorted(dumps.iterdir())]
cmd = [sys.executable, str(ips / "scripts/euroc/geo_direction.py"), str(euroc), str(out),
       f"{dumps / 'pivift_dir_best'}:MH_05_difficult,V2_02_medium,V1_01_easy,V1_02_medium"] + args
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out.mkdir(exist_ok=True)
(out / "geo_log.txt").write_text(p.stdout)
print(p.stdout[-20000:])
shutil.rmtree(ips, ignore_errors=True)
sys.exit(p.returncode)
