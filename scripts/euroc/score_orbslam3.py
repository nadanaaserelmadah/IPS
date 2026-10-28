"""Score our own ORB-SLAM3 runs (kernel nadaelmadah/orbslam3-euroc) with our EuRoC protocol.

Protocol, as for every method in Table P8-H: positions sampled at our 10 Hz edge timestamps (t_end),
SE(3)-aligned ATE (gh.ate). ORB-SLAM3 saves body (IMU) poses for its inertial modes, so it is scored
against the ground-truth body position (state_groundtruth p_RS_R) at the same times. Frames before
ORB-SLAM3 initialises (or inside tracking gaps > 0.2 s) are dropped; the coverage is reported, and our
network-free pipeline (gyro + geometric direction x accelerometer length, |g| check) is re-scored on the
same subset against the same body ground truth for a like-for-like row.
  python scripts/euroc/score_orbslam3.py ORB_DIR EUROC_ROOT GEO_CACHE_DIR DUMPS_DIR
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid_euroc as gh  # noqa: E402
import imu_scale  # noqa: E402

SEQS = ("MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium")
RUNS = ("monoi_nolc", "monoi_lc", "stereoi_nolc", "stereoi_lc")


def interp(t_src, x_src, t_q, max_gap_ns=200_000_000):
    ok = (t_q >= t_src[0]) & (t_q <= t_src[-1])
    i = np.clip(np.searchsorted(t_src, t_q), 1, len(t_src) - 1)
    gap = t_src[i] - t_src[i - 1]
    ok &= gap <= max_gap_ns
    out = np.stack([np.interp(t_q, t_src, x_src[:, j]) for j in range(3)], 1)
    return out, ok


def ours_network_free(root, seq, cache, t_end):
    z = np.load(cache / f"{seq}_geo.npz")
    R, d = z["R"], z["d"]
    ok = ~np.isnan(d[:, 0])
    d_full, last = [], np.array([0.0, 0.0, 1.0])
    for k in range(len(t_end)):
        last = d[k] if ok[k] else last
        d_full.append(last)
    d_full = np.array(d_full)
    l_imu, _ = imu_scale.imu_lengths(root, seq, t_end, d_full, g_tol=0.5)
    L, last_l = [], 0.0
    for x in l_imu:
        last_l = x if not np.isnan(x) else last_l
        L.append(last_l)
    return gh.compose(list(R), d_full * np.array(L)[:, None])  # camera positions, N+1 (origin first)


def main():
    orb, root, cache, dumps = (Path(a) for a in sys.argv[1:5])
    rows = {}
    for seq in SEQS:
        t_end = np.load(dumps / "pivift_dir_best" / f"{seq}_timestamps.npy").astype(np.int64)
        gt = np.loadtxt(root / seq / "mav0/state_groundtruth_estimate0/data.csv", delimiter=",", comments="#")
        g_body, _ = interp(gt[:, 0].astype(np.int64), gt[:, 1:4], t_end)
        ours = ours_network_free(root, seq, cache, t_end)[1:]  # position at each t_end
        for run in RUNS:
            f = orb / f"{run}_{seq}_frames.txt"
            tr = np.loadtxt(f)
            t_o, p_o = tr[:, 0].astype(np.int64), tr[:, 1:4]
            order = np.argsort(t_o)
            p_q, ok = interp(t_o[order], p_o[order], t_end)
            cov = float(ok.mean())
            ate_orb = gh.ate(p_q[ok], g_body[ok])[0]
            ate_ours = gh.ate(ours[ok], g_body[ok])[0]
            rows.setdefault(run, []).append((seq, cov, ate_orb, ate_ours))
    print("| ORB-SLAM3 run | " + " | ".join(s.split("_")[0] + "_" + s.split("_")[1] for s in SEQS) + " | mean |")
    print("|---|" + "---|" * (len(SEQS) + 1))
    for run, v in rows.items():
        cells = [f"{a:.3f} ({c:.0%})" for _, c, a, _ in v]
        print(f"| {run} | " + " | ".join(cells) + f" | **{np.mean([a for _, _, a, _ in v]):.3f}** |")
    v = rows["monoi_nolc"]
    print(f"| ours, network-free (same subset as monoi_nolc, body GT) | " + " | ".join(f"{o:.2f}" for *_, o in v)
          + f" | **{np.mean([o for *_, o in v]):.2f}** |")


if __name__ == "__main__":
    main()
