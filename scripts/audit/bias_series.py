"""Per-edge visual-rotation residual series for the online gyro-bias estimator (docs/ONLINE_GYRO_BIAS_DESIGN.md).

For every 100 ms edge (cam0 frames i -> i+2, non-overlapping) of a EuRoC-layout sequence (EuRoC, or the TUM-VI shadow tree):
5-point essential matrix (RANSAC, 1 px) on tracked corners -> recoverPose rotation, and the raw (bias-free) gyro rotation over
the same interval. The series does not depend on any bias, so the estimator can be tuned and run offline on it without
re-reading images; the estimator itself only ever uses edges up to the current one (see bias_online.py).
Saved per sequence (npz): t0, t1 (ns), dt (s), th_gyro (raw gyro rotation vector, body frame), res0 / res1
(th_gyro - th_vis for the two recoverPose conventions), par0 / par1 (median de-rotated parallax, px), n_inl, n_pts.
  python scripts/audit/bias_series.py ROOT OUT_DIR SEQ [SEQ ...]      (one process per sequence, run in parallel)
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "euroc"))
import geo_direction as G  # noqa: E402
import gyro_hybrid_euroc as gh  # noqa: E402

MIN_INL = 30


def series(args):
    root, out, seq = args
    import cv2
    from scipy.spatial.transform import Rotation

    dst = Path(out) / f"{seq}_bias_series.npz"
    if dst.is_file():
        return seq, "cached"
    mav = Path(root) / seq / "mav0"
    imu = np.loadtxt(mav / "imu0/data.csv", delimiter=",", comments="#")
    t_ns, gyro = imu[:, 0].astype(np.int64), imu[:, 1:4]
    R_bs = gh.load_T_BS(Path(root) / seq)[:3, :3]
    K, dist, cam_ts, files = G.read_cam(mav / "cam0")
    f_px = K[0, 0]
    cols = {k: [] for k in ("t0", "t1", "dt", "th_gyro", "res0", "res1", "par0", "par1", "n_inl", "n_pts")}
    i = 0
    while i + 2 < len(cam_ts):
        t0, t1 = int(cam_ts[i]), int(cam_ts[i + 2])
        ia_i, ib_i = i, i + 2
        i += 2
        if not (t_ns[0] < t0 and t1 < t_ns[-1]):
            continue
        ia = cv2.imread(str(files[ia_i]), cv2.IMREAD_GRAYSCALE)
        ib = cv2.imread(str(files[ib_i]), cv2.IMREAD_GRAYSCALE)
        pa, pb = G.track(ia, ib)
        if pa is None:
            continue
        xa, xb = G.bearings(pa, K, dist), G.bearings(pb, K, dist)
        ok = np.isfinite(xa).all(1) & np.isfinite(xb).all(1)
        xa, xb = xa[ok], xb[ok]
        if len(xa) < MIN_INL:
            continue
        E, mask = cv2.findEssentialMat(xa[:, :2], xb[:, :2], focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999,
                                       threshold=1.0 / f_px)
        if E is None or E.shape != (3, 3):
            continue
        n_in, Rr, _, _ = cv2.recoverPose(E, xa[:, :2], xb[:, :2], focal=1.0, pp=(0.0, 0.0), mask=mask)
        if n_in < MIN_INL or n_in < 0.5 * len(xa):
            continue
        inl = mask.ravel() > 0
        th_gyro = Rotation.from_matrix(gh.integrate(t_ns, gyro, t0, t1)).as_rotvec()
        row = {"t0": t0, "t1": t1, "dt": (t1 - t0) * 1e-9, "th_gyro": th_gyro, "n_inl": int(n_in), "n_pts": len(xa)}
        for c, Rc in ((0, Rr), (1, Rr.T)):
            th_vis = Rotation.from_matrix(R_bs @ Rc @ R_bs.T).as_rotvec()
            row[f"res{c}"] = th_gyro - th_vis
            y = xa[inl] @ Rc.T
            row[f"par{c}"] = float(np.median(np.linalg.norm(xb[inl, :2] - y[:, :2] / y[:, 2:3], axis=1)) * f_px)
        for k, v in row.items():
            cols[k].append(v)
    Path(out).mkdir(parents=True, exist_ok=True)
    np.savez(dst, **{k: np.array(v) for k, v in cols.items()})
    return seq, f"{len(cols['t0'])} edges"


if __name__ == "__main__":
    root, out, seqs = sys.argv[1], sys.argv[2], sys.argv[3:]
    n = min(len(seqs), int(os.environ.get("NPROC", os.cpu_count() or 1)))
    with Pool(n) as p:
        for seq, msg in p.imap_unordered(series, [(root, out, s) for s in seqs]):
            print(seq, msg, flush=True)
