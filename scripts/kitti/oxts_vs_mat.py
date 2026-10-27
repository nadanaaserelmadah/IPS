"""How was VS-VIO imus/*.mat made? Compare with raw OXTS interpolated at image times (+offset)."""
import sys, numpy as np, scipy.io as sio
from pathlib import Path
NPZ, MAT = Path(sys.argv[1]), Path(sys.argv[2])
COLS = {"ax..wz (11-13,17-19)": [11, 12, 13, 17, 18, 19], "af..wu (14-16,20-22)": [14, 15, 16, 20, 21, 22]}
for s in ["04", "05", "06", "07", "08", "09", "10"]:
    z = np.load(NPZ / f"{s}.npz"); t_ox, ox, t_img = z["t_ox"], z["ox"], z["t_img"]
    m = sio.loadmat(MAT / f"{s}.mat")["imu_data_interp"]
    n = len(t_img) - 1
    gaps = np.where(np.diff(t_ox) > 0.015)[0]
    ingap = [(t_ox[g], t_ox[g + 1]) for g in gaps if t_ox[g + 1] > t_img[0] and t_ox[g] < t_img[-1]]
    gap_frames = [(int(np.searchsorted(t_img, a)), round(b - a, 2)) for a, b in ingap]
    res = {}
    for cname, cols in COLS.items():
        for mode in ("per-frame", "0.01s"):
            best = None
            for off in np.arange(-0.10, 0.101, 0.005):
                if mode == "per-frame":
                    tq = (t_img[:-1, None] + np.arange(10)[None] * np.diff(t_img)[:, None] / 10).ravel()
                    tq = np.r_[tq, t_img[-1]] + off
                else:
                    tq = t_img[0] + np.arange(10 * n + 1) * 0.01 + off
                c = np.stack([np.interp(tq, t_ox, ox[:, k]) for k in cols], 1)
                L = min(len(c), len(m))
                e = np.sqrt(np.mean((c[:L] - m[:L]) ** 2, 0))
                if best is None or e[3:].mean() < best[1][3:].mean():
                    best = (off, e)
            res[(cname, mode)] = best
    k = min(res, key=lambda q: res[q][1][3:].mean())
    off, e = res[k]
    print(f"{s}: mat {m.shape} vs 10*{n}+1={10*n+1} | best {k[0]}, {k[1]}, offset {off*1e3:+.0f} ms, RMSE acc {np.round(e[:3],4)} gyro {np.round(e[3:],5)} | gaps in seq (frame, s): {gap_frames}")
    for q, (o, ee) in res.items():
        if q != k: print(f"      {q[0]:22s} {q[1]:9s} off {o*1e3:+4.0f} ms gyro RMSE {ee[3:].mean():.5f}")
