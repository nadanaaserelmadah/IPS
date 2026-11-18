"""Audit paper: why does training on KITTI 08 matter so much for the 05/07/10 test sequences?

Training VS-VIO with 08 lowers its median test t_rel from 4.29 to 2.85 % (same code, seed and hardware,
results_cache/2026-11-18). This script compares 08 with the rest of the training set and with the test
sequences, from the KITTI odometry ground-truth poses and the released imus/*.mat:
  - size: frames, length, share of the training windows;
  - recording day (KITTI odometry -> raw drive mapping from the KITTI devkit): the test sequences are all from
    2011_09_30, so is 08, and so is only part of the rest of the training set;
  - motion: speed, yaw rate, turning share, stops, vertical motion, specific force from the inertial files;
  - coverage: how well the training set, with and without 08, covers the (speed, yaw rate) distribution of
    the test frames (Jensen-Shannon divergence and share of test frames in bins the training set never visits).
  python scripts/audit/seq08_stats.py POSES_DIR IMUS_DIR [OUT_TXT]
"""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

DT = 0.1037  # s, measured mean image interval on these drives (RESULTS.md; 103.7-104.1 ms)
DAY = {"00": "2011_10_03", "01": "2011_10_03", "02": "2011_10_03", "03": "2011_09_26", "04": "2011_09_30",
       "05": "2011_09_30", "06": "2011_09_30", "07": "2011_09_30", "08": "2011_09_30", "09": "2011_09_30",
       "10": "2011_09_30"}  # KITTI devkit odometry -> raw mapping (drives 0027/0042/0034/0016/0018/0020/0027/0028/0033/0034)
TRAIN_NO08 = ["00", "01", "02", "04", "06", "09"]
TEST = ["05", "07", "10"]


def load_poses(path):
    P = np.loadtxt(path).reshape(-1, 3, 4)
    return P[:, :, :3], P[:, :, 3]


def motion(seq, poses_dir):
    R, t = load_poses(poses_dir / f"{seq}.txt")
    step = np.diff(t, axis=0)
    speed = np.linalg.norm(step, axis=1) / DT
    # relative rotation per frame in the camera frame; yaw = rotation about the camera y axis (KITTI: y down)
    rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
    yaw = np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 2, 2])) / DT
    ang = np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1))) / DT
    vert = np.abs(step[:, 1]) / DT  # camera y is down: vertical speed in the world frame of the first camera
    return {"speed": speed, "yaw": yaw, "ang": ang, "vert": vert, "frames": len(t),
            "length_km": np.linalg.norm(step, axis=1).sum() / 1000,
            "heading_total_deg": np.abs(np.degrees(np.arctan2(rel[:, 0, 2], rel[:, 2, 2]))).sum()}


def imu(seq, imus_dir):
    d = sio.loadmat(imus_dir / f"{seq}.mat")["imu_data_interp"]  # columns: accel xyz (m/s^2), gyro xyz (rad/s)
    acc, gyr = d[:, :3], d[:, 3:]
    return {"acc_norm_std": float(np.linalg.norm(acc, axis=1).std()),
            "gyr_norm_mean_dps": float(np.degrees(np.linalg.norm(gyr, axis=1)).mean())}


def hist2d(m_list, edges):
    s = np.concatenate([m["speed"] for m in m_list])
    y = np.concatenate([m["yaw"] for m in m_list])
    h, _, _ = np.histogram2d(s, np.clip(y, edges[1][0], edges[1][-1] - 1e-9), bins=edges)
    return h / h.sum()


def js(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.sum(np.where(a > 0, a * np.log(np.where(a > 0, a, 1) / np.where(b > 0, b, 1)), 0))
    return float((kl(p, m) + kl(q, m)) / 2)


def main():
    poses_dir, imus_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out = []
    say = lambda s="": (print(s), out.append(s))
    seqs = TRAIN_NO08 + ["08"] + TEST
    M = {s: motion(s, poses_dir) for s in seqs}
    I = {s: imu(s, imus_dir) for s in seqs}

    say("## Per sequence (KITTI odometry ground truth; inertial statistics from imus/*.mat)\n")
    say("| seq | role | day | frames | km | speed m/s: mean (p90) | stopped % | |yaw rate| deg/s: mean (p90) | turning % (>5 deg/s) | vertical m/s mean | accel-norm SD | gyro-norm mean deg/s |")
    say("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in seqs:
        m, i = M[s], I[s]
        role = "test" if s in TEST else ("train (08)" if s == "08" else "train")
        say(f"| {s} | {role} | {DAY[s]} | {m['frames']} | {m['length_km']:.2f} | {m['speed'].mean():.1f} ({np.percentile(m['speed'], 90):.1f}) | "
            f"{100 * np.mean(m['speed'] < 0.5):.1f} | {np.abs(m['yaw']).mean():.2f} ({np.percentile(np.abs(m['yaw']), 90):.1f}) | "
            f"{100 * np.mean(np.abs(m['yaw']) > 5):.1f} | {m['vert'].mean():.2f} | {i['acc_norm_std']:.2f} | {i['gyr_norm_mean_dps']:.2f} |")

    say("\n## Share of the training frames by recording day\n")
    say("| training set | frames | from 2011_09_30 (the test day) | share | of which 08 |")
    say("|---|---|---|---|---|")
    for name, tr in (("without 08", TRAIN_NO08), ("with 08 (authors')", TRAIN_NO08 + ["08"])):
        n = sum(M[s]["frames"] for s in tr)
        nd = sum(M[s]["frames"] for s in tr if DAY[s] == "2011_09_30")
        say(f"| {name} | {n} | {nd} | {100 * nd / n:.0f} % | {100 * M['08']['frames'] / nd:.0f} % |" if "08" in tr else
            f"| {name} | {n} | {nd} | {100 * nd / n:.0f} % | – |")

    edges = (np.linspace(0, 25, 26), np.linspace(-30, 30, 31))  # 1 m/s x 2 deg/s bins
    P_test = hist2d([M[s] for s in TEST], edges)
    say("\n## Coverage of the test frames' (speed, yaw rate) distribution\n")
    say("| training set | Jensen-Shannon divergence to test | test frames in bins with no training frame |")
    say("|---|---|---|")
    for name, tr in (("without 08", TRAIN_NO08), ("with 08", TRAIN_NO08 + ["08"]), ("08 alone", ["08"]),
                     ("2011_09_30 training drives without 08 (04, 06, 09)", ["04", "06", "09"])):
        P = hist2d([M[s] for s in tr], edges)
        say(f"| {name} | {js(P_test, P):.3f} | {100 * P_test[P == 0].sum():.2f} % |")
    for s in TEST:
        Ps = hist2d([M[s]], edges)
        a, b = js(Ps, hist2d([M[x] for x in TRAIN_NO08], edges)), js(Ps, hist2d([M[x] for x in TRAIN_NO08 + ["08"]], edges))
        say(f"| test {s} alone: without 08 / with 08 | {a:.3f} / {b:.3f} | |")

    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
