"""KITTI: replace the network's per-frame rotation with gyro preintegration and re-score (no training).

Inputs:
  --dump DIR   <seq>_estimated_poses.npy / <seq>_gt_poses.npy (6-DoF Euler + t, frame i -> i+1,
               camera frame) from the kitti_dump kernel, plus DIR/poses/<seq>.txt (KITTI GT)
  --imus DIR   KITTI imus/<seq>.mat ('imu_data_interp', 10 samples per frame interval, 100 Hz)
The IMU-to-camera rotation, the IMU-to-pose time offset (the KITTI imus lag the GT poses by
~5-6 samples) and optionally a constant gyro bias are fitted on training sequences only (default 04, 06, 08, 09: same 2011_09_30 recording day as 05 / 07 / 10),
then applied unchanged to the test sequences.

Variants scored with the repo's KITTI metric (t_rel %, r_rel deg/100 m):
  network          : predictions as tested
  gyro R + net t   : the proposed swap
  gyro R + GT t    : rotation-only quality of the gyro
  GT R + net t     : translation-only quality of the network
  python scripts/kitti/gyro_rotation_test.py --dump DIR --imus DIR [--train 04,06,08,09] [--test 05,07,10]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

VIFT = Path(__file__).resolve().parents[2] / "third_party/VIFT"
sys.path.insert(0, str(VIFT))
from src.metrics.kitti_metrics_calculator import KITTIMetricsCalculator  # noqa: E402
from src.utils.kitti_utils import eulerAnglesToRotationMatrix, euler_from_matrix, read_pose_from_text  # noqa: E402

DT = 0.01  # 10 interpolated IMU samples per 0.1 s frame interval


def so3_exp(w):
    a = np.linalg.norm(w)
    if a < 1e-12:
        return np.eye(3)
    k = w / a
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def so3_log(R):
    a = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v * (0.5 if a < 1e-8 else a / (2 * np.sin(a)))


def gyro_rel(imu: np.ndarray, n_pairs: int, bias: np.ndarray, gyro_cols) -> list[np.ndarray]:
    """IMU-frame relative rotation for frame pairs 0..n_pairs-1 (midpoint integration)."""
    w = imu[:, gyro_cols] - bias
    out = []
    for i in range(n_pairs):
        R = np.eye(3)
        for k in range(10 * i, 10 * i + 10):
            R = R @ so3_exp(0.5 * (w[k] + w[k + 1]) * DT)
        out.append(R)
    return out


def shift(imu: np.ndarray, lag: float) -> np.ndarray:
    """Sample k of the result is input sample k + lag (linear interpolation, edges held)."""
    idx = np.arange(len(imu), dtype=np.float64) + lag
    return np.stack([np.interp(idx, np.arange(len(imu)), imu[:, c]) for c in range(imu.shape[1])], 1)


def gt_rel_R(poses_abs: np.ndarray) -> list[np.ndarray]:
    return [poses_abs[i][:3, :3].T @ poses_abs[i + 1][:3, :3] for i in range(len(poses_abs) - 1)]


def kabsch(a_imu: np.ndarray, a_cam: np.ndarray) -> np.ndarray:
    """R with a_cam ~ R a_imu (rotation vectors, rows)."""
    H = a_imu.T @ a_cam
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def load(seq, args):
    imu = sio.loadmat(args.imus / f"{seq}.mat")["imu_data_interp"].astype(np.float64)
    poses_abs, _ = read_pose_from_text(str(args.dump / "poses" / f"{seq}.txt"))
    return imu, poses_abs


def score(est6, gt6):
    m = KITTIMetricsCalculator().calculate_metrics({"s": {"estimated_poses": est6, "gt_poses": gt6}})
    return m["s_t_rel"], m["s_r_rel"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--imus", type=Path, required=True)
    p.add_argument("--train", default="04,06,08,09")
    p.add_argument("--test", default="05,07,10")
    args = p.parse_args()

    # which columns are the gyro: acc has |mean| ~ g, gyro is small
    imu0, _ = load(args.train.split(",")[0], args)
    norms = np.abs(imu0.mean(0))
    gyro_cols = [3, 4, 5] if norms[:3].max() > 5 else [0, 1, 2]
    print(f"column |mean|: {np.round(norms, 3)} -> gyro cols {gyro_cols}")

    # time offset: per-frame gyro-vs-GT error (after a Kabsch fit) over candidate lags, on train
    raw = {s: load(s, args) for s in args.train.split(",")}

    def lag_err(lag):
        ai, ac = [], []
        for imu, P in raw.values():
            n = min(len(P) - 1, (len(imu) - 1) // 10) - 2
            ai += [so3_log(r) for r in gyro_rel(shift(imu, lag), n, np.zeros(3), gyro_cols)][5:-5]
            ac += [so3_log(r) for r in gt_rel_R(P)[:n]][5:-5]
        ai, ac = np.array(ai), np.array(ac)
        return np.median(np.linalg.norm((kabsch(ai, ac) @ ai.T).T - ac, axis=1))

    coarse = {lag: lag_err(lag) for lag in range(-2, 13)}
    best = min(coarse, key=coarse.get)
    fine = {lag: lag_err(lag) for lag in np.arange(best - 1, best + 1.01, 0.25)}
    lag = float(min(fine, key=fine.get))
    print("lag (samples) -> median err (deg): " + " ".join(f"{k:g}:{np.degrees(v):.4f}" for k, v in {**coarse, **fine}.items()))
    print(f"fitted IMU lag: {lag:+.2f} samples ({lag * DT * 1000:+.0f} ms)")

    # fit R_cam_imu (and a constant bias) on training sequences
    A_imu, A_cam, train = [], [], {}
    for s in args.train.split(","):
        imu, P = raw[s]
        imu = shift(imu, lag)
        n = min(len(P) - 1, (len(imu) - 1) // 10)
        Ri, Rc = gyro_rel(imu, n, np.zeros(3), gyro_cols), gt_rel_R(P)[:n]
        train[s] = (imu, P, n, Rc)
        A_imu += [so3_log(r) for r in Ri]
        A_cam += [so3_log(r) for r in Rc]
    A_imu, A_cam = np.array(A_imu), np.array(A_cam)
    R_ci = kabsch(A_imu, A_cam)
    print("R_cam_imu (fit on train):\n", np.round(R_ci, 4))
    # constant bias in the IMU frame from residual rates on train
    resid = (R_ci.T @ A_cam.T).T - A_imu  # per-interval rotation-vector residual (imu frame), ~ -bias*0.1
    bias = -resid.mean(0) / (10 * DT)
    print(f"fitted gyro bias (rad/s): {np.round(bias, 5)}  |b|*0.1 s = {np.degrees(np.linalg.norm(bias))*0.1:.4f} deg/frame")
    for s, (imu, P, n, Rc) in train.items():
        for name, b in (("no bias", np.zeros(3)), ("bias", bias)):
            Rg = [R_ci @ r @ R_ci.T for r in gyro_rel(imu, n, b, gyro_cols)]
            err = np.degrees([np.linalg.norm(so3_log(a.T @ c)) for a, c in zip(Rg, Rc)])
            print(f"  train {s} gyro vs GT per frame ({name}): median {np.median(err):.4f} mean {err.mean():.4f} deg")

    print("\n| seq | variant | t_rel % | r_rel deg/100 m | rot err/frame median (deg) |")
    print("|---|---|---|---|---|")
    tot = {}
    for s in args.test.split(","):
        est = np.load(args.dump / f"{s}_estimated_poses.npy").astype(np.float64)
        gt = np.load(args.dump / f"{s}_gt_poses.npy").astype(np.float64)
        imu, P = load(s, args)
        imu = shift(imu, lag)
        n = len(est)
        assert len(gt) == n and (len(imu) - 1) // 10 >= n, (s, len(est), len(gt), len(imu))
        Rgt = [eulerAnglesToRotationMatrix(g[:3]) for g in gt]
        Rnet = [eulerAnglesToRotationMatrix(e[:3]) for e in est]
        for bname, b in (("", np.zeros(3)), (" (bias-corr.)", bias)):
            Rgy = [R_ci @ r @ R_ci.T for r in gyro_rel(imu, n, b, gyro_cols)]
            gy6 = np.array([euler_from_matrix(r) for r in Rgy])
            variants = {"network": (est, Rnet), f"gyro R + net t{bname}": (np.hstack([gy6, est[:, 3:]]), Rgy),
                        f"gyro R + GT t{bname}": (np.hstack([gy6, gt[:, 3:]]), Rgy)}
            if not bname:
                variants["GT R + net t"] = (np.hstack([gt[:, :3], est[:, 3:]]), Rgt)
            else:
                variants.pop("network")
            for name, (e6, Rs) in variants.items():
                tr, rr = score(e6, gt)
                rot = np.median(np.degrees([np.linalg.norm(so3_log(a.T @ c)) for a, c in zip(Rs, Rgt)]))
                tot.setdefault(name, []).append((tr, rr))
                print(f"| {s} | {name} | {tr:.2f} | {rr:.2f} | {rot:.4f} |")
    print("\n| variant | mean t_rel % | mean r_rel deg/100 m |\n|---|---|---|")
    for name, v in tot.items():
        v = np.array(v)
        print(f"| {name} | **{v[:, 0].mean():.2f}** | {v[:, 1].mean():.2f} |")
    print("| VIFT paper | 1.96 | 0.46 |")


if __name__ == "__main__":
    main()
