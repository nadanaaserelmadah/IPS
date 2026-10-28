"""KITTI: per-frame translation direction from two-view geometry, given the gyro rotation (no network).

Table 7h left the network's direction as the last learned part of the KITTI pose: GT direction x
accelerometer length scores 0.45% against 1.27% with the network's direction. Here the direction of
frame k -> k+1 comes from the rectified left colour camera (image_2, the camera VIFT uses):
LK-tracked corners, the image-time gyro rotation (lag 0, rotated into the camera frame with the
training-fitted R_cam_imu), and the 2-point RANSAC / SVD / cheirality solver of
scripts/euroc/geo_direction.py. Frames with < 2 px de-rotated parallax are re-measured over
k-1 -> k+1 (same rule as EuRoC); frames that still fail are NaN (the caller falls back).
  python scripts/kitti/geo_direction_kitti.py KITTI_ROOT IMG_R_DIR OUT_DIR [SEQS]
  python scripts/kitti/geo_direction_kitti.py imgR NPZ_DIR GYRO_FEATS_DIR OUT_DIR [LAG_S]
KITTI_ROOT holds sequences/<s>/image_2 and calib.txt; IMG_R_DIR holds <s>_imgR.npy (image-interval
gyro rotations, camera frame, from `make_img_rotations` below).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "euroc"))
import geo_direction as gd  # noqa: E402


def read_K(calib: Path):
    for line in calib.read_text().splitlines():
        if line.startswith("P2:"):
            P = np.array([float(x) for x in line.split()[1:]]).reshape(3, 4)
            return P[:, :3]
    raise ValueError(calib)


def make_img_rotations(npz_dir: Path, gyro_dir: Path, out: Path, seqs=("04", "05", "06", "07", "09", "10"), lag=0.0):
    """Gyro rotation over each image interval, shifted by `lag` (0 = image timestamps; the calibrated
    GT lag is the alternative; the choice is made on training drives 04 / 06 / 09)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gyro_hybrid as h
    import imu_scale_kitti as isk

    out.mkdir(parents=True, exist_ok=True)
    for s in seqs:
        R_ci = np.array(json.loads((gyro_dir / f"calib_{isk.DAY[s]}.json").read_text())["R_cam_imu"])
        Rs = np.array([R_ci @ r @ R_ci.T for r in h.integrate(npz_dir / f"{s}.npz", lag)])
        np.save(out / f"{s}_imgR.npy", Rs)
        print(s, Rs.shape)


def run(root: Path, rdir: Path, out: Path, seqs):
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    dist = np.zeros(4)
    for s in seqs:
        K = read_K(root / "sequences" / s / "calib.txt")
        imgs = sorted((root / "sequences" / s / "image_2").glob("*.png"))
        R = np.load(rdir / f"{s}_imgR.npy")
        n = min(len(imgs) - 1, len(R))
        rng = np.random.default_rng(0)
        d_all, infos = np.full((n, 3), np.nan), []
        gray = lambda i: cv2.imread(str(imgs[i]), cv2.IMREAD_GRAYSCALE)  # noqa: E731
        ia = gray(0)
        for k in range(n):
            ib = gray(k + 1)
            pa, pb = gd.track(ia, ib, 800)
            d, info = (None, {"n": 0}) if pa is None else gd.direction(gd.bearings(pa, K, dist), gd.bearings(pb, K, dist), R[k], K[0, 0], rng)
            if (d is None or info.get("parallax_px", 0.0) < gd.ADAPT_PX) and k > 0:
                ip = gray(k - 1)
                q, qb = gd.track(ip, ib, 800)
                if q is not None:
                    d2, info2 = gd.direction(gd.bearings(q, K, dist), gd.bearings(qb, K, dist), R[k - 1] @ R[k], K[0, 0], rng)
                    if d2 is not None and info2["parallax_px"] >= gd.ADAPT_PX:
                        d, info = R[k - 1].T @ d2, dict(info2, baseline_frames=2)
            if d is not None:
                d_all[k] = d
            infos.append(info)
            ia = ib
        np.savez(out / f"{s}_geo_dir.npz", d=d_all, info=json.dumps(infos))
        print(f"{s}: {n} frames, geometric direction on {np.mean(~np.isnan(d_all[:, 0])):.0%}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "imgR":
        make_img_rotations(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]), lag=float(sys.argv[5]) if len(sys.argv) > 5 else 0.0)
    else:
        run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]),
            sys.argv[4].split(",") if len(sys.argv) > 4 else ["05", "07", "10"])
