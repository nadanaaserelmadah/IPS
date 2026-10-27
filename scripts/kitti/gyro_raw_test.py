"""Gyro-rotation test redone with real timestamps: raw 100 Hz OXTS integrated over each image interval."""
import sys, contextlib, io, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_rotation_test as g

NPZ, DUMP = Path(sys.argv[1]), Path(sys.argv[2])
TRAIN, TEST = ["04", "06", "08", "09"], ["05", "07", "10"]
_, P = {}, {}
for s in TRAIN + TEST:
    P[s], _ = g.read_pose_from_text(str(DUMP / "poses" / f"{s}.txt"))


def integ(s, lag, src="raw", bias=np.zeros(3)):
    z = np.load(NPZ / f"{s}.npz"); t_ox, ox, t_img = z["t_ox"], z["ox"], z["t_img"]
    w = ox[:, 17:20] - bias
    out = []
    for i in range(len(t_img) - 1):
        a, b = t_img[i] + lag, t_img[i + 1] + lag
        if src == "raw":  # integrate the piecewise-linear rate over [a, b] with substeps at OXTS samples
            inner = t_ox[(t_ox > a) & (t_ox < b)]
            tk = np.r_[a, inner, b]
        else:  # VS-VIO style: 10 equal substeps of the image interval
            tk = np.linspace(a, b, 11)
        wk = np.stack([np.interp(tk, t_ox, w[:, c]) for c in range(3)], 1)
        R = np.eye(3)
        for k in range(len(tk) - 1):
            dt = 0.01 if src == "mat-dt0.01" else tk[k + 1] - tk[k]  # mat-dt0.01: our first test (fixed 10 ms substeps)
            R = R @ g.so3_exp(0.5 * (wk[k] + wk[k + 1]) * dt)
        out.append(R)
    return out


def fit(lag, src):
    ai, ac = [], []
    for s in TRAIN:
        Ri, Rc = integ(s, lag, src), g.gt_rel_R(P[s])
        n = min(len(Ri), len(Rc))
        ai += [g.so3_log(r) for r in Ri[:n]]; ac += [g.so3_log(r) for r in Rc[:n]]
    ai, ac = np.array(ai), np.array(ac)
    R = g.kabsch(ai, ac)
    return np.median(np.linalg.norm((R @ ai.T).T - ac, axis=1)), R


def sc(e, gt):
    with contextlib.redirect_stdout(io.StringIO()):
        return g.score(e, gt)


for src in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["raw"]):
    errs = {lag: fit(lag, src)[0] for lag in np.arange(-0.08, 0.081, 0.01)}
    l0 = min(errs, key=errs.get)
    errs2 = {lag: fit(lag, src)[0] for lag in np.arange(l0 - 0.01, l0 + 0.0101, 0.0025)}
    lag = min(errs2, key=errs2.get)
    _, R = fit(lag, src)
    print(f"[{src}] fitted lag {lag*1e3:+.1f} ms (train median per-frame err {np.degrees(errs2[lag]):.4f} deg); lag curve: " + " ".join(f"{k*1e3:+.0f}:{np.degrees(v):.4f}" for k, v in errs.items()))
    rows = {}
    for s in TEST:
        est = np.load(DUMP / f"{s}_estimated_poses.npy").astype(float); gt = np.load(DUMP / f"{s}_gt_poses.npy").astype(float)
        Rg = [R @ r @ R.T for r in integ(s, lag, src)][: len(gt)]
        gy6 = np.array([g.euler_from_matrix(r) for r in Rg])
        med = np.median(np.degrees([np.linalg.norm(g.so3_log(g.eulerAnglesToRotationMatrix(x[:3]).T @ r)) for x, r in zip(gt, Rg)]))
        for name, e6 in (("network", est), ("gyro R + GT t", np.hstack([gy6, gt[:, 3:]])), ("gyro R + net t", np.hstack([gy6, est[:, 3:]]))):
            rows.setdefault(name, []).append(sc(e6, gt))
        print(f"  {s}: gyro per-frame rot err median {med:.4f} deg")
    print("| variant | 05 | 07 | 10 | mean t_rel | mean r_rel |\n|---|---|---|---|---|---|")
    for k, v in rows.items():
        v = np.array(v); print(f"| {k} | " + " | ".join(f"{a:.2f}/{b:.2f}" for a, b in v) + f" | **{v[:,0].mean():.2f}** | {v[:,1].mean():.2f} |")
