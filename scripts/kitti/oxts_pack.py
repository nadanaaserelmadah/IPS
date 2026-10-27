"""Pack raw OXTS (100 Hz) + sync image timestamps per odometry sequence into one npz.
  python scripts/kitti/oxts_pack.py RAW_DIR OUT_DIR [SEQS]
"""
import sys, numpy as np
from pathlib import Path
from datetime import datetime

SEQ = {"00": ("2011_10_03_drive_0027", 0, 4540), "01": ("2011_10_03_drive_0042", 0, 1100), "02": ("2011_10_03_drive_0034", 0, 4660),
       "04": ("2011_09_30_drive_0016", 0, 270), "05": ("2011_09_30_drive_0018", 0, 2760), "06": ("2011_09_30_drive_0020", 0, 1100),
       "07": ("2011_09_30_drive_0027", 0, 1100), "08": ("2011_09_30_drive_0028", 1100, 5170), "09": ("2011_09_30_drive_0033", 0, 1590),
       "10": ("2011_09_30_drive_0034", 0, 1200)}


def ts(p):
    out = []
    for line in Path(p).read_text().split("\n"):
        line = line.strip()
        if not line:
            continue
        d, frac = line.split(".")
        t = datetime.strptime(d, "%Y-%m-%d %H:%M:%S")
        out.append(t.hour * 3600 + t.minute * 60 + t.second + float("0." + frac))
    return np.array(out)


root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
for s, (drive, a, b) in SEQ.items():
    if len(sys.argv) > 3 and s not in sys.argv[3].split(","):
        continue
    d = root / drive
    t_ox = ts(d / "extract/oxts/timestamps.txt")
    files = sorted((d / "extract/oxts/data").glob("*.txt"))
    assert len(files) == len(t_ox), (s, len(files), len(t_ox))
    ox = np.array([np.fromstring(f.read_text(), sep=" ") for f in files])
    t_img = ts(d / "sync/image_02/timestamps.txt")[a:b + 1]
    np.savez(out / f"{s}.npz", t_ox=t_ox, ox=ox, t_img=t_img)
    dt = np.diff(t_ox)
    print(s, drive, "oxts", ox.shape, f"dt median {np.median(dt)*1e3:.2f} ms, gaps>15ms {int((dt>0.015).sum())}, max {dt.max()*1e3:.0f} ms | frames", len(t_img),
          f"img dt median {np.median(np.diff(t_img))*1e3:.1f} ms | oxts covers img: {t_ox[0] <= t_img[0]} {t_ox[-1] >= t_img[-1]}")
