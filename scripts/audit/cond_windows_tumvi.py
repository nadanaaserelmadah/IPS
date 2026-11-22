"""TUM-VI room5/6 (test rooms): condition numbers of the 20-edge windows and the gravity check, with the ground-truth
(mocap) direction of every edge. Diagnostic only: it says how many windows a condition-number bound would reject, it does not
run the gate. Edges are the 10 Hz chain taken from the camera timestamps (cam0[20::2]), not each run's own dump timestamps.
  python scripts/audit/cond_windows_tumvi.py TUM_ROOT
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cond_threshold_euroc as ct  # noqa: E402
import imu_scale  # noqa: E402

ROOMS = ("dataset-room5_512_16", "dataset-room6_512_16")
BOUNDS = (1e3, 1e4, 1e5, 1e6, 1e7)


def main():
    root = Path(sys.argv[1])
    conds, errs = [], []
    for r in ROOMS:
        t_end, d, L, ok = ct.gt_edges(root, r, gt_dir="mocap0")
        cond = np.full(len(t_end), np.nan)
        lens, g = imu_scale.imu_lengths(root, r, t_end, d, g_tol=None, cond_out=cond)
        e = np.abs(lens / np.maximum(L, 1e-9) - 1)
        use = ok & ~np.isnan(lens) & (L >= ct.MIN_STEP_M)
        print(f"{r}: edges {len(t_end)}, solved {int((~np.isnan(lens)).sum())}, used {int(use.sum())}, cond median {np.nanmedian(cond[use]):.1e}, "
              f"max {np.nanmax(cond[use]):.1e}, median length error {np.median(e[use]) * 100:.1f}%, windows with |g|-9.81 > 0.5: {int((np.abs(g - 9.81) > 0.5).sum())} of {len(g)}")
        conds.append(cond[use]); errs.append(e[use])
    c, e = np.concatenate(conds), np.concatenate(errs)
    print("\n| bound | edges kept | median rel. length error, kept | median error, rejected |\n|---|---|---|---|")
    for b in BOUNDS:
        k = c < b
        print(f"| {b:.0e} | {100 * k.mean():.1f}% | {np.median(e[k]) * 100 if k.any() else float('nan'):.1f}% | "
              f"{np.median(e[~k]) * 100 if (~k).any() else float('nan'):.1f}% |")


if __name__ == "__main__":
    main()
