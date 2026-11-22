"""EuRoC V2_03_difficult: two-view geometry only (CPU), for the dumps produced by kernel pivift-v203-infer (attached as a kernel
source). Split out of v203_infer.py after the geometry step of its first run failed on a frame without corners
(scripts/euroc/geo_direction.py track_points now returns early on an empty point list). Output: /kaggle/working/geo/."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

SEQ = "V2_03_difficult"
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "omegaconf", "lightning", "natsort", "rich", "opencv-python-headless"], check=False)


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
seq_dir = find(lambda r, d, f: r.name == SEQ and (r / "mav0/cam0/data.csv").is_file())
scr = find(lambda r, d, f: (r / "euroc/geo_direction.py").is_file() and (r / "pi_vift/shape_diag.py").is_file())
dump_dirs = sorted({p.parent for p in INPUT.rglob(f"{SEQ}_timestamps.npy") if p.parent.name.startswith("pose_dumps_")})
assert dump_dirs, "no dumps from pivift-v203-infer"
print("code", code, "| seq", seq_dir, "| scripts", scr, "| dumps", [d.name for d in dump_dirs], flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(scr, ips / "scripts", ignore=shutil.ignore_patterns("*.zip", "__pycache__"))
out = WORK / "geo"
args = [f"{d.name.replace('pose_dumps_', '')}={d}" for d in dump_dirs]
cmd = [sys.executable, str(ips / "scripts/euroc/geo_direction.py"), str(seq_dir.parent), str(out), f"{dump_dirs[0]}:{SEQ}"] + args
p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
out.mkdir(exist_ok=True)
(out / "geo_log.txt").write_text(p.stdout)
print(p.stdout[-6000:])
shutil.rmtree(ips, ignore_errors=True)
sys.exit(p.returncode)
