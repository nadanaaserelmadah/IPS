"""KITTI: fuse network and gyro rotations offline (no training) and re-score.

Per frame, the gyro rotation (lag / extrinsic fitted on train as in gyro_rotation_test.py) has
low noise but a slowly varying bias; the network rotation has no drift-prone bias but more
per-frame noise. Two simple fusions, both causal:
  blend a     : rotation vector  a·net + (1−a)·gyro
  bias W      : gyro minus a bias estimated as the mean (gyro − net) over the previous W frames
Translation stays the network's. Scored with the repo's KITTI metric.
  python scripts/kitti/gyro_fusion_test.py --dump DIR --imus DIR
"""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path

import numpy as np

import gyro_rotation_test as g


def fit_calibration(args, gyro_cols, train):
    raw = {s: g.load(s, args) for s in train}

    def fit(lag):
        ai, ac = [], []
        for imu, P in raw.values():
            n = min(len(P) - 1, (len(imu) - 1) // 10) - 2
            ai += [g.so3_log(r) for r in g.gyro_rel(g.shift(imu, lag), n, np.zeros(3), gyro_cols)][5:-5]
            ac += [g.so3_log(r) for r in g.gt_rel_R(P)[:n]][5:-5]
        ai, ac = np.array(ai), np.array(ac)
        R = g.kabsch(ai, ac)
        return np.median(np.linalg.norm((R @ ai.T).T - ac, axis=1)), R

    lags = np.arange(3, 8.01, 0.25)
    errs = [fit(l)[0] for l in lags]
    lag = float(lags[int(np.argmin(errs))])
    return lag, fit(lag)[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--imus", type=Path, required=True)
    p.add_argument("--train", default="04,06,08,09")
    p.add_argument("--test", default="05,07,10")
    args = p.parse_args()
    gyro_cols = [3, 4, 5]
    lag, R_ci = fit_calibration(args, gyro_cols, args.train.split(","))
    print(f"lag {lag:+.2f} samples, R_cam_imu fitted on {args.train}")

    rows = {}
    for s in args.test.split(","):
        est = np.load(args.dump / f"{s}_estimated_poses.npy").astype(np.float64)
        gt = np.load(args.dump / f"{s}_gt_poses.npy").astype(np.float64)
        imu, _ = g.load(s, args)
        n = len(est)
        Rgy = [R_ci @ r @ R_ci.T for r in g.gyro_rel(g.shift(imu, lag), n, np.zeros(3), gyro_cols)]
        a_gy = np.array([g.so3_log(r) for r in Rgy])
        a_net = np.array([g.so3_log(g.eulerAnglesToRotationMatrix(e[:3])) for e in est])
        a_gt = np.array([g.so3_log(g.eulerAnglesToRotationMatrix(x[:3])) for x in gt])
        e_net, e_gy = a_net - a_gt, a_gy - a_gt
        print(f"{s}: per-frame rot-vector error std (deg) net {np.degrees(e_net.std(0)).round(4)} gyro {np.degrees(e_gy.std(0)).round(4)}"
              f" | mean net {np.degrees(e_net.mean(0)).round(4)} gyro {np.degrees(e_gy.mean(0)).round(4)}"
              f" | corr(net err, gyro err) per axis {[round(np.corrcoef(e_net[:, i], e_gy[:, i])[0, 1], 2) for i in range(3)]}")
        variants = {"network": a_net, "gyro": a_gy}
        for a in (0.25, 0.5, 0.75):
            variants[f"blend a={a}"] = a * a_net + (1 - a) * a_gy
        d = a_gy - a_net
        for W in (25, 100, 400, 1600):
            c = np.cumsum(np.vstack([np.zeros(3), d]), 0)
            k = np.arange(n)
            lo = np.maximum(0, k - W)
            b = np.where((k - lo)[:, None] > 0, (c[k] - c[lo]) / np.maximum(k - lo, 1)[:, None], 0.0)  # mean of d over [k-W, k)
            variants[f"gyro − bias(W={W})"] = a_gy - b
        for name, av in variants.items():
            e6 = np.hstack([np.array([g.euler_from_matrix(g.so3_exp(v)) for v in av]), est[:, 3:]])
            with contextlib.redirect_stdout(io.StringIO()):
                tr, rr = g.score(e6, gt)
            rows.setdefault(name, []).append((tr, rr))
    print(f"\n| rotation source (translation = network) | {' | '.join(args.test.split(','))} t_rel | mean t_rel % | mean r_rel |")
    print("|---|" + "---|" * (len(args.test.split(",")) + 2))
    for name, v in rows.items():
        v = np.array(v)
        print(f"| {name} | {' | '.join(f'{x:.2f}' for x in v[:, 0])} | **{v[:, 0].mean():.2f}** | {v[:, 1].mean():.2f} |")
    print("| VIFT paper | 2.02 | 1.75 | 2.11 | 1.96 | 0.46 |")


if __name__ == "__main__":
    main()
