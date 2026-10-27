"""KITTI: re-score after removing a constant per-axis rotation bias from the network (no training).

Bias per sequence = mean(est - GT) Euler angles per frame. "LOO" takes it from the other test
sequences; "oracle-own" from the sequence itself. With CALIB_DIR (dumps of held-out calibration
sequences, e.g. 08 for a checkpoint trained without it), "calib" takes it from those instead.
  python scripts/kitti/bias_upside.py DUMP_DIR [CALIB_DIR [CALIB_SEQS]]
"""
import sys, contextlib, io, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_rotation_test as g
D = sys.argv[1]
seqs = ["05", "07", "10"]
E = {s: np.load(f"{D}/{s}_estimated_poses.npy").astype(float) for s in seqs}
G = {s: np.load(f"{D}/{s}_gt_poses.npy").astype(float) for s in seqs}
bias = {s: (E[s][:, :3] - G[s][:, :3]).mean(0) for s in seqs}
calib = None
if len(sys.argv) > 2:
    cs = (sys.argv[3] if len(sys.argv) > 3 else "08").split(",")
    ce = np.concatenate([np.load(f"{sys.argv[2]}/{c}_estimated_poses.npy").astype(float)[:, :3] for c in cs])
    cg = np.concatenate([np.load(f"{sys.argv[2]}/{c}_gt_poses.npy").astype(float)[:, :3] for c in cs])
    calib = (ce - cg).mean(0)
    print(f"calibration bias from {cs} (deg/frame, x y z): {np.round(np.degrees(calib), 4)}")
for s in seqs:
    print(f"{s} own bias (deg/frame, x y z): {np.round(np.degrees(bias[s]), 4)}")
def sc(e, gt):
    with contextlib.redirect_stdout(io.StringIO()):
        return g.score(e, gt)
for name, axes in [("none", []), ("pitch+roll (0,2)", [0, 2]), ("all 3", [0, 1, 2]), ("roll only (2)", [2])]:
    for mode in ["LOO", "oracle-own"] + (["calib"] if calib is not None else []):
        r = []
        for s in seqs:
            b = {"LOO": lambda: np.mean([bias[o] for o in seqs if o != s], 0), "oracle-own": lambda: bias[s], "calib": lambda: calib}[mode]()
            e = E[s].copy()
            for a in axes: e[:, a] -= b[a]
            r.append(sc(e, G[s]))
        r = np.array(r)
        print(f"{name:18s} {mode:10s} " + " ".join(f"{x:.2f}/{y:.2f}" for x, y in r) + f" | mean {r[:,0].mean():.2f} / {r[:,1].mean():.2f}")
        if not axes: break
