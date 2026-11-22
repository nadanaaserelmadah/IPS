"""Design data only: condition-number bound for 20-edge EuRoC windows (rule fixed in docs/COND_THRESHOLD_EUROC_DESIGN.md).

For the EuRoC TRAINING sequences, solve the accelerometer-length alignment with the ground-truth direction of every edge
(no gravity check, no condition bound), and tabulate, for each candidate bound, the share of edges kept and the median
relative length error of the kept edges. Test sequences are never read.
  python scripts/audit/cond_threshold_euroc.py EUROC_ROOT
"""
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/euroc"))
import gyro_hybrid_euroc as gh  # noqa: E402
import imu_scale  # noqa: E402

SEQS = ("MH_01_easy", "MH_02_easy", "MH_03_medium", "MH_04_difficult", "V2_01_easy")
BOUNDS = (1e3, 1e4, 1e5, 1e6, 1e7)
MIN_STEP_M = 0.02


def gt_edges(root: Path, seq: str, gt_dir: str = "state_groundtruth_estimate0"):
    """t_end (ns), ground-truth camera direction (camera frame at the edge start) and length per edge.
    gt_dir: EuRoC's state_groundtruth_estimate0, or mocap0 for TUM-VI."""
    mav = root / seq / "mav0"
    cam = np.loadtxt(mav / "cam0/data.csv", delimiter=",", comments="#", usecols=0).astype(np.int64)
    t_end = cam[20::2]  # as in the paper's 10 Hz edge chain
    gt = np.loadtxt(mav / gt_dir / "data.csv", delimiter=",", comments="#")
    tg, pg, qg = gt[:, 0], gt[:, 1:4], gt[:, 4:8]  # t, p_RS_R, q_RS (w x y z)
    rot = Rotation.from_quat(qg[:, [1, 2, 3, 0]])
    T_bs = gh.load_T_BS(root / seq)
    R_bs, p_bs = T_bs[:3, :3], T_bs[:3, 3]
    ok = (t_end - imu_scale.EDGE_NS >= tg[0]) & (t_end <= tg[-1])
    t1 = t_end.astype(np.float64)
    t0 = t1 - imu_scale.EDGE_NS
    sl = Slerp(tg.astype(np.float64), rot)

    def cam_pose(t):
        tt = np.clip(t, tg[0], tg[-1])
        R_wb = sl(tt).as_matrix()
        p = np.stack([np.interp(tt, tg, pg[:, i]) for i in range(3)], 1)
        return R_wb @ R_bs, p + np.einsum("nij,j->ni", R_wb, p_bs)

    R0, p0 = cam_pose(t0)
    _, p1 = cam_pose(t1)
    rel = np.einsum("nji,nj->ni", R0, p1 - p0)  # R0^T (p1 - p0)
    L = np.linalg.norm(rel, axis=1)
    d = rel / np.maximum(L, 1e-9)[:, None]
    return t_end, d, L, ok


def main():
    root = Path(sys.argv[1])
    conds, errs, grav = [], [], []
    for seq in SEQS:
        t_end, d, L, ok = gt_edges(root, seq)
        cond = np.full(len(t_end), np.nan)
        lens, g = imu_scale.imu_lengths(root, seq, t_end, d, g_tol=None, cond_out=cond)
        e = np.abs(lens / np.maximum(L, 1e-9) - 1)
        use = ok & ~np.isnan(lens) & (L >= MIN_STEP_M)
        conds.append(cond[use]); errs.append(e[use])
        print(f"{seq}: edges {len(t_end)}, solved {int((~np.isnan(lens)).sum())}, used for stats {int(use.sum())}, "
              f"cond median {np.nanmedian(cond[use]):.1e}, median length error {np.median(e[use]) * 100:.1f}%, |g|-9.81 median {np.median(np.abs(g - 9.81)):.2f}")
    c, e = np.concatenate(conds), np.concatenate(errs)
    print(f"\nall training edges used: {len(c)}; median error with no bound {np.median(e) * 100:.1f}%")
    print("\n| bound | edges kept | median rel. length error of kept edges | median error of rejected edges |\n|---|---|---|---|")
    rows = []
    for b in BOUNDS:
        k = c < b
        mk = np.median(e[k]) * 100 if k.any() else float("nan")
        mr = np.median(e[~k]) * 100 if (~k).any() else float("nan")
        rows.append((b, k.mean(), mk))
        print(f"| {b:.0e} | {100 * k.mean():.0f}% | {mk:.1f}% | {mr:.1f}% |")
    ok = [r for r in rows if r[1] >= 0.9 and r[2] <= 5.0]
    if ok:
        pick = ok[0][0]
    else:
        cand = [r for r in rows if r[1] >= 0.9]
        pick = min(cand, key=lambda r: (r[2], -r[0]))[0] if cand else None
    print(f"\nrule (docs/COND_THRESHOLD_EUROC_DESIGN.md) selects: {pick:.0e}" if pick else "\nno candidate keeps 90% of the edges")


if __name__ == "__main__":
    main()
