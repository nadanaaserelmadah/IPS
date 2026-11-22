"""Online gyro bias, stage 1: per-edge visual-rotation residual series for all 11 EuRoC sequences (CPU).
Runs scripts/audit/bias_series.py (embedded below, base64) and also copies each sequence's imu0/data.csv and
state_groundtruth_estimate0/data.csv (for the ground-truth bias used on the design sequences). Outputs: /kaggle/working/bias/."""
import base64
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "hydra-core", "omegaconf", "lightning", "natsort", "rich", "opencv-python-headless"], check=False)
SRC_B64 = "IiIiUGVyLWVkZ2UgdmlzdWFsLXJvdGF0aW9uIHJlc2lkdWFsIHNlcmllcyBmb3IgdGhlIG9ubGluZSBneXJvLWJpYXMgZXN0aW1hdG9yIChkb2NzL09OTElORV9HWVJPX0JJQVNfREVTSUdOLm1kKS4KCkZvciBldmVyeSAxMDAgbXMgZWRnZSAoY2FtMCBmcmFtZXMgaSAtPiBpKzIsIG5vbi1vdmVybGFwcGluZykgb2YgYSBFdVJvQy1sYXlvdXQgc2VxdWVuY2UgKEV1Um9DLCBvciB0aGUgVFVNLVZJIHNoYWRvdyB0cmVlKToKNS1wb2ludCBlc3NlbnRpYWwgbWF0cml4IChSQU5TQUMsIDEgcHgpIG9uIHRyYWNrZWQgY29ybmVycyAtPiByZWNvdmVyUG9zZSByb3RhdGlvbiwgYW5kIHRoZSByYXcgKGJpYXMtZnJlZSkgZ3lybyByb3RhdGlvbiBvdmVyCnRoZSBzYW1lIGludGVydmFsLiBUaGUgc2VyaWVzIGRvZXMgbm90IGRlcGVuZCBvbiBhbnkgYmlhcywgc28gdGhlIGVzdGltYXRvciBjYW4gYmUgdHVuZWQgYW5kIHJ1biBvZmZsaW5lIG9uIGl0IHdpdGhvdXQKcmUtcmVhZGluZyBpbWFnZXM7IHRoZSBlc3RpbWF0b3IgaXRzZWxmIG9ubHkgZXZlciB1c2VzIGVkZ2VzIHVwIHRvIHRoZSBjdXJyZW50IG9uZSAoc2VlIGJpYXNfb25saW5lLnB5KS4KU2F2ZWQgcGVyIHNlcXVlbmNlIChucHopOiB0MCwgdDEgKG5zKSwgZHQgKHMpLCB0aF9neXJvIChyYXcgZ3lybyByb3RhdGlvbiB2ZWN0b3IsIGJvZHkgZnJhbWUpLCByZXMwIC8gcmVzMQoodGhfZ3lybyAtIHRoX3ZpcyBmb3IgdGhlIHR3byByZWNvdmVyUG9zZSBjb252ZW50aW9ucyksIHBhcjAgLyBwYXIxIChtZWRpYW4gZGUtcm90YXRlZCBwYXJhbGxheCwgcHgpLCBuX2lubCwgbl9wdHMuCiAgcHl0aG9uIHNjcmlwdHMvYXVkaXQvYmlhc19zZXJpZXMucHkgUk9PVCBPVVRfRElSIFNFUSBbU0VRIC4uLl0gICAgICAob25lIHByb2Nlc3MgcGVyIHNlcXVlbmNlLCBydW4gaW4gcGFyYWxsZWwpCiIiIgppbXBvcnQgb3MKaW1wb3J0IHN5cwpmcm9tIG11bHRpcHJvY2Vzc2luZyBpbXBvcnQgUG9vbApmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCmltcG9ydCBudW1weSBhcyBucAoKSEVSRSA9IFBhdGgoX19maWxlX18pLnJlc29sdmUoKS5wYXJlbnQKc3lzLnBhdGguaW5zZXJ0KDAsIHN0cihIRVJFLnBhcmVudCAvICJldXJvYyIpKQppbXBvcnQgZ2VvX2RpcmVjdGlvbiBhcyBHICAjIG5vcWE6IEU0MDIKaW1wb3J0IGd5cm9faHlicmlkX2V1cm9jIGFzIGdoICAjIG5vcWE6IEU0MDIKCk1JTl9JTkwgPSAzMAoKCmRlZiBzZXJpZXMoYXJncyk6CiAgICByb290LCBvdXQsIHNlcSA9IGFyZ3MKICAgIGltcG9ydCBjdjIKICAgIGZyb20gc2NpcHkuc3BhdGlhbC50cmFuc2Zvcm0gaW1wb3J0IFJvdGF0aW9uCgogICAgZHN0ID0gUGF0aChvdXQpIC8gZiJ7c2VxfV9iaWFzX3Nlcmllcy5ucHoiCiAgICBpZiBkc3QuaXNfZmlsZSgpOgogICAgICAgIHJldHVybiBzZXEsICJjYWNoZWQiCiAgICBtYXYgPSBQYXRoKHJvb3QpIC8gc2VxIC8gIm1hdjAiCiAgICBpbXUgPSBucC5sb2FkdHh0KG1hdiAvICJpbXUwL2RhdGEuY3N2IiwgZGVsaW1pdGVyPSIsIiwgY29tbWVudHM9IiMiKQogICAgdF9ucywgZ3lybyA9IGltdVs6LCAwXS5hc3R5cGUobnAuaW50NjQpLCBpbXVbOiwgMTo0XQogICAgUl9icyA9IGdoLmxvYWRfVF9CUyhQYXRoKHJvb3QpIC8gc2VxKVs6MywgOjNdCiAgICBLLCBkaXN0LCBjYW1fdHMsIGZpbGVzID0gRy5yZWFkX2NhbShtYXYgLyAiY2FtMCIpCiAgICBmX3B4ID0gS1swLCAwXQogICAgY29scyA9IHtrOiBbXSBmb3IgayBpbiAoInQwIiwgInQxIiwgImR0IiwgInRoX2d5cm8iLCAicmVzMCIsICJyZXMxIiwgInBhcjAiLCAicGFyMSIsICJuX2lubCIsICJuX3B0cyIpfQogICAgaSA9IDAKICAgIHdoaWxlIGkgKyAyIDwgbGVuKGNhbV90cyk6CiAgICAgICAgdDAsIHQxID0gaW50KGNhbV90c1tpXSksIGludChjYW1fdHNbaSArIDJdKQogICAgICAgIGlhX2ksIGliX2kgPSBpLCBpICsgMgogICAgICAgIGkgKz0gMgogICAgICAgIGlmIG5vdCAodF9uc1swXSA8IHQwIGFuZCB0MSA8IHRfbnNbLTFdKToKICAgICAgICAgICAgY29udGludWUKICAgICAgICBpYSA9IGN2Mi5pbXJlYWQoc3RyKGZpbGVzW2lhX2ldKSwgY3YyLklNUkVBRF9HUkFZU0NBTEUpCiAgICAgICAgaWIgPSBjdjIuaW1yZWFkKHN0cihmaWxlc1tpYl9pXSksIGN2Mi5JTVJFQURfR1JBWVNDQUxFKQogICAgICAgIHBhLCBwYiA9IEcudHJhY2soaWEsIGliKQogICAgICAgIGlmIHBhIGlzIE5vbmU6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgeGEsIHhiID0gRy5iZWFyaW5ncyhwYSwgSywgZGlzdCksIEcuYmVhcmluZ3MocGIsIEssIGRpc3QpCiAgICAgICAgb2sgPSBucC5pc2Zpbml0ZSh4YSkuYWxsKDEpICYgbnAuaXNmaW5pdGUoeGIpLmFsbCgxKQogICAgICAgIHhhLCB4YiA9IHhhW29rXSwgeGJbb2tdCiAgICAgICAgaWYgbGVuKHhhKSA8IE1JTl9JTkw6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgRSwgbWFzayA9IGN2Mi5maW5kRXNzZW50aWFsTWF0KHhhWzosIDoyXSwgeGJbOiwgOjJdLCBmb2NhbD0xLjAsIHBwPSgwLjAsIDAuMCksIG1ldGhvZD1jdjIuUkFOU0FDLCBwcm9iPTAuOTk5LAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB0aHJlc2hvbGQ9MS4wIC8gZl9weCkKICAgICAgICBpZiBFIGlzIE5vbmUgb3IgRS5zaGFwZSAhPSAoMywgMyk6CiAgICAgICAgICAgIGNvbnRpbnVlCiAgICAgICAgbl9pbiwgUnIsIF8sIF8gPSBjdjIucmVjb3ZlclBvc2UoRSwgeGFbOiwgOjJdLCB4Yls6LCA6Ml0sIGZvY2FsPTEuMCwgcHA9KDAuMCwgMC4wKSwgbWFzaz1tYXNrKQogICAgICAgIGlmIG5faW4gPCBNSU5fSU5MIG9yIG5faW4gPCAwLjUgKiBsZW4oeGEpOgogICAgICAgICAgICBjb250aW51ZQogICAgICAgIGlubCA9IG1hc2sucmF2ZWwoKSA+IDAKICAgICAgICB0aF9neXJvID0gUm90YXRpb24uZnJvbV9tYXRyaXgoZ2guaW50ZWdyYXRlKHRfbnMsIGd5cm8sIHQwLCB0MSkpLmFzX3JvdHZlYygpCiAgICAgICAgcm93ID0geyJ0MCI6IHQwLCAidDEiOiB0MSwgImR0IjogKHQxIC0gdDApICogMWUtOSwgInRoX2d5cm8iOiB0aF9neXJvLCAibl9pbmwiOiBpbnQobl9pbiksICJuX3B0cyI6IGxlbih4YSl9CiAgICAgICAgZm9yIGMsIFJjIGluICgoMCwgUnIpLCAoMSwgUnIuVCkpOgogICAgICAgICAgICB0aF92aXMgPSBSb3RhdGlvbi5mcm9tX21hdHJpeChSX2JzIEAgUmMgQCBSX2JzLlQpLmFzX3JvdHZlYygpCiAgICAgICAgICAgIHJvd1tmInJlc3tjfSJdID0gdGhfZ3lybyAtIHRoX3ZpcwogICAgICAgICAgICB5ID0geGFbaW5sXSBAIFJjLlQKICAgICAgICAgICAgcm93W2YicGFye2N9Il0gPSBmbG9hdChucC5tZWRpYW4obnAubGluYWxnLm5vcm0oeGJbaW5sLCA6Ml0gLSB5WzosIDoyXSAvIHlbOiwgMjozXSwgYXhpcz0xKSkgKiBmX3B4KQogICAgICAgIGZvciBrLCB2IGluIHJvdy5pdGVtcygpOgogICAgICAgICAgICBjb2xzW2tdLmFwcGVuZCh2KQogICAgUGF0aChvdXQpLm1rZGlyKHBhcmVudHM9VHJ1ZSwgZXhpc3Rfb2s9VHJ1ZSkKICAgIG5wLnNhdmV6KGRzdCwgKip7azogbnAuYXJyYXkodikgZm9yIGssIHYgaW4gY29scy5pdGVtcygpfSkKICAgIHJldHVybiBzZXEsIGYie2xlbihjb2xzWyd0MCddKX0gZWRnZXMiCgoKaWYgX19uYW1lX18gPT0gIl9fbWFpbl9fIjoKICAgIHJvb3QsIG91dCwgc2VxcyA9IHN5cy5hcmd2WzFdLCBzeXMuYXJndlsyXSwgc3lzLmFyZ3ZbMzpdCiAgICBuID0gbWluKGxlbihzZXFzKSwgaW50KG9zLmVudmlyb24uZ2V0KCJOUFJPQyIsIG9zLmNwdV9jb3VudCgpIG9yIDEpKSkKICAgIHdpdGggUG9vbChuKSBhcyBwOgogICAgICAgIGZvciBzZXEsIG1zZyBpbiBwLmltYXBfdW5vcmRlcmVkKHNlcmllcywgWyhyb290LCBvdXQsIHMpIGZvciBzIGluIHNlcXNdKToKICAgICAgICAgICAgcHJpbnQoc2VxLCBtc2csIGZsdXNoPVRydWUpCg=="


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
first = find(lambda r, d, f: (r / "mav0/cam0/data.csv").is_file())
root = first.parent
seqs = sorted(p.name for p in root.iterdir() if (p / "mav0/cam0/data.csv").is_file())
print("code", code, "| scripts", scr, "| root", root, "| seqs", seqs, flush=True)
ips = WORK / "IPS"
shutil.rmtree(ips, ignore_errors=True)
(ips / "third_party").mkdir(parents=True)
os.symlink(code, ips / "third_party" / "VIFT")
shutil.copytree(scr, ips / "scripts", ignore=shutil.ignore_patterns("*.zip", "__pycache__"))
(ips / "scripts/audit").mkdir(parents=True, exist_ok=True)
(ips / "scripts/audit/bias_series.py").write_bytes(base64.b64decode(SRC_B64))
out = WORK / "bias"
out.mkdir(exist_ok=True)
for s in seqs:
    for sub in ("imu0", "state_groundtruth_estimate0"):
        f = root / s / "mav0" / sub / "data.csv"
        if f.is_file():
            (out / s).mkdir(exist_ok=True)
            shutil.copy(f, out / s / f"{sub}.csv")
env = dict(os.environ, NPROC="4")
p = subprocess.run([sys.executable, str(ips / "scripts/audit/bias_series.py"), str(root), str(out), *seqs], env=env,
                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
(out / "log.txt").write_text(p.stdout)
print(p.stdout[-3000:])
shutil.rmtree(ips, ignore_errors=True)
sys.exit(p.returncode)
