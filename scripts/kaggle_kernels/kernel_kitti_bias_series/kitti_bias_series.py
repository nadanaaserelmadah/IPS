"""Online gyro bias, KITTI check: per-frame visual-rotation residual series on test drives 05, 07, 10 (CPU).
The frozen estimator (M=200, N_MIN=10, B_MIN=0.01 rad/s) is then run locally on these series to see whether its bias norm ever
exceeds B_MIN on KITTI (docs/ONLINE_GYRO_BIAS_DESIGN.md). Output: /kaggle/working/kbias/."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "omegaconf", "lightning", "natsort", "rich", "opencv-python-headless"], check=False)


def find(pred, depth=8):
    for root, dirs, files in os.walk(INPUT, followlinks=True):
        if root.endswith(("image_2", "image_3")):
            dirs[:] = []
        if root[len(str(INPUT)):].count(os.sep) >= depth:
            dirs[:] = []
        if pred(Path(root), dirs, files):
            return Path(root)
    raise SystemExit("not found")


code = find(lambda r, d, f: r.name == "VIFT" and r.parent.name == "third_party")
scr = find(lambda r, d, f: (r / "euroc/geo_direction.py").is_file() and (r / "pi_vift/shape_diag.py").is_file())
kin = find(lambda r, d, f: (r / "audit/bias_series_kitti.py").is_file() or (r / "bias_series_kitti.py").is_file())
rdir = find(lambda r, d, f: "05_imgR.npy" in f)
kitti = find(lambda r, d, f: "sequences" in d and "poses" in d)
print("code", code, "| scripts", scr, "| kitti inputs", kin, "| imgR", rdir, "| kitti", kitti, flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(scr, ips / "scripts", ignore=shutil.ignore_patterns("*.zip", "__pycache__"))
(ips / "scripts/kitti").mkdir(parents=True, exist_ok=True)
(ips / "scripts/audit").mkdir(parents=True, exist_ok=True)
shutil.copy(next(kin.rglob("geo_direction_kitti.py")), ips / "scripts/kitti/geo_direction_kitti.py")
shutil.copy(next(kin.rglob("bias_series_kitti.py")), ips / "scripts/audit/bias_series_kitti.py")
out = WORK / "kbias"
p = subprocess.run([sys.executable, str(ips / "scripts/audit/bias_series_kitti.py"), str(kitti), str(rdir), str(out), "05", "07", "10"],
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out.mkdir(exist_ok=True)
(out / "log.txt").write_text(p.stdout)
print(p.stdout[-3000:])
shutil.rmtree(ips, ignore_errors=True)
sys.exit(p.returncode)
