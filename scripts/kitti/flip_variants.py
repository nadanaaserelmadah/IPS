"""KITTI: score flip test-time augmentation variants from the vift-kitti-flip kernel output (no training).

08 (held out of the checkpoint's training) is the selection set; 05 / 07 / 10 the test set.
  python scripts/kitti/flip_variants.py KITTI_FLIP_DIR   (holds plain/ and flip/)
"""
import sys, contextlib, io, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_rotation_test as g
D = sys.argv[1]
seqs = ["08", "05", "07", "10"]
L = lambda v, s, k: np.load(f"{D}/{v}/{s}_{k}_poses.npy").astype(float)
def sc(e, gt):
    with contextlib.redirect_stdout(io.StringIO()):
        return g.score(e, gt)
rows = {}
for s in seqs:
    p, f, gt = L("plain", s, "estimated"), L("flip", s, "estimated"), L("plain", s, "gt")
    V = {"plain": p, "tta all": 0.5 * (p + f)}
    for name, cols in [("tta rotation only", [0, 1, 2]), ("tta yaw only", [1]), ("tta roll+yaw", [1, 2]), ("tta translation only", [3, 4, 5])]:
        e = p.copy(); e[:, cols] = 0.5 * (p[:, cols] + f[:, cols]); V[name] = e
    for w in (0.25,):
        V[f"tta all w_flip={w}"] = (1 - w) * p + w * f
    for k, e in V.items():
        rows.setdefault(k, []).append(sc(e, gt))
print("| variant | 08 (calib) | 05 | 07 | 10 | mean 05/07/10 |\n|---|---|---|---|---|---|")
for k, r in rows.items():
    r = np.array(r)
    print(f"| {k} | " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in r) + f" | **{r[1:,0].mean():.2f}** / {r[1:,1].mean():.2f} |")
