"""(Copy of kernel_kgeo run on the test drives 05 / 07 / 10, for the audit paper scoring.) KITTI: geometric translation direction given the gyro rotation, for two gyro-timing variants
(lag 0 = image timestamps, lag 60 ms = calibrated GT lag), on training drives 04 / 06 / 09 (where the
variant is chosen) and test drives 05 / 07 / 10. CPU only. Outputs land in /kaggle/working/kgeo/.
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
        if root.endswith(("image_2", "image_3")):
            dirs[:] = []
        if pred(Path(root), dirs, files):
            return Path(root)
    raise SystemExit("not found")


code = find(lambda r, d, f: r.name == "VIFT" and r.parent.name == "third_party")
kitti = find(lambda r, d, f: "sequences" in d and "poses" in d)
geo = find(lambda r, d, f: "imgR_lag0" in d and "scripts" in d)
print("code", code, "| kitti", kitti, "| geo", geo, flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(geo / "scripts", ips / "scripts")
out = WORK / "kgeo"
out.mkdir(exist_ok=True)
log = []
for tag in ("lag60",):  # lag chosen on training drives (Table 7l)
    cmd = [sys.executable, str(ips / "scripts/kitti/geo_direction_kitti.py"), str(kitti), str(geo / f"imgR_{tag}"),
           str(out / tag), os.environ.get("KG_SEQS", "05,07,10")]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log.append(f"== {tag} (exit {p.returncode})\n{p.stdout[-6000:]}")
    print(log[-1], flush=True)
(out / "kgeo_log.txt").write_text("\n".join(log))
shutil.rmtree(ips, ignore_errors=True)
