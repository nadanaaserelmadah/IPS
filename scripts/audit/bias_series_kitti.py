"""KITTI version of bias_series.py: per frame pair k -> k+1 (10 Hz, 0.1 s) visual rotation vs the gyro rotation of the same pair
(camera frame, imgR_lag60 = calibrated gyro integrated with the 60 ms lag and no bias removed). Same fixed convention as EuRoC:
th_vis = rotvec(R_recoverPose^T), parallax with R_recoverPose. Saved: <seq>_bias_series.npz with the keys bias_online.load reads
(t1 as the frame index in units of 0.1 s * 1e9 ns, dt, res1, par0, n_inl).
  python scripts/audit/bias_series_kitti.py KITTI_ROOT IMGR_DIR OUT_DIR SEQ [SEQ ...]"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "euroc"))
sys.path.insert(0, str(HERE.parent / "kitti"))
import geo_direction as G  # noqa: E402
import geo_direction_kitti as GK  # noqa: E402

MIN_INL = 30


def run(root, rdir, out, seqs):
    import cv2
    from scipy.spatial.transform import Rotation

    out.mkdir(parents=True, exist_ok=True)
    dist = np.zeros(4)
    for s in seqs:
        K = GK.read_K(root / "sequences" / s / "calib.txt")
        imgs = sorted((root / "sequences" / s / "image_2").glob("*.png"))
        R = np.load(rdir / f"{s}_imgR.npy")
        n = min(len(imgs) - 1, len(R))
        f = K[0, 0]
        cols = {k: [] for k in ("t1", "dt", "res1", "par0", "n_inl")}
        ia = cv2.imread(str(imgs[0]), cv2.IMREAD_GRAYSCALE)
        for k in range(n):
            ib = cv2.imread(str(imgs[k + 1]), cv2.IMREAD_GRAYSCALE)
            pa, pb = G.track(ia, ib, 800)
            ia = ib
            if pa is None:
                continue
            xa, xb = G.bearings(pa, K, dist), G.bearings(pb, K, dist)
            E, mask = cv2.findEssentialMat(xa[:, :2], xb[:, :2], focal=1.0, pp=(0.0, 0.0), method=cv2.RANSAC, prob=0.999, threshold=1.0 / f)
            if E is None or E.shape != (3, 3):
                continue
            n_in, Rr, _, _ = cv2.recoverPose(E, xa[:, :2], xb[:, :2], focal=1.0, pp=(0.0, 0.0), mask=mask)
            if n_in < MIN_INL or n_in < 0.5 * len(xa):
                continue
            inl = mask.ravel() > 0
            th_gyro = Rotation.from_matrix(R[k]).as_rotvec()
            th_vis = Rotation.from_matrix(Rr.T).as_rotvec()
            y = xa[inl] @ Rr.T
            cols["t1"].append(int((k + 1) * 1e8))
            cols["dt"].append(0.1)
            cols["res1"].append(th_gyro - th_vis)
            cols["par0"].append(float(np.median(np.linalg.norm(xb[inl, :2] - y[:, :2] / y[:, 2:3], axis=1)) * f))
            cols["n_inl"].append(int(n_in))
        np.savez(out / f"{s}_bias_series.npz", **{k: np.array(v) for k, v in cols.items()})
        print(s, n, "pairs,", len(cols["t1"]), "usable", flush=True)


if __name__ == "__main__":
    run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:])
