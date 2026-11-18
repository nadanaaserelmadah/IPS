"""Audit paper: forward-scale error of retrained VS-VIO models by true speed, on KITTI 05/07/10.

The clean-protocol models (trained without 08) overestimate forward motion by about 6 %; models trained with 08
by about 2 %. Binning the frames by their true speed shows where: a model pulled towards the mean speed of its
training set (10.6 m/s without 08, test 7.3 m/s) overestimates slow motion and underestimates fast motion.
  python scripts/audit/scale_by_speed.py NAME=DUMP_DIR [NAME=DUMP_DIR ...]
DUMP_DIR holds <seq>_estimated_poses.npy and <seq>_gt_poses.npy (VS-VIO pose layout: rotation, then translation).
"""
import sys
from pathlib import Path

import numpy as np

DT = 0.1037  # s, mean image interval on these drives
BINS = [0, 3, 6, 9, 12, 30]  # m/s


def main():
    print("| model | " + " | ".join(f"{BINS[i]}-{BINS[i + 1]} m/s" for i in range(len(BINS) - 1)) + " | all |")
    print("|---|" + "---|" * len(BINS))
    counts = None
    for arg in sys.argv[1:]:
        name, d = arg.split("=", 1)
        d = Path(d)
        gt = np.concatenate([np.load(d / f"{s}_gt_poses.npy") for s in ("05", "07", "10")]).astype(float)
        est = np.concatenate([np.load(d / f"{s}_estimated_poses.npy")[:len(np.load(d / f"{s}_gt_poses.npy"))]
                              for s in ("05", "07", "10")]).astype(float)
        g, e = np.linalg.norm(gt[:, 3:], axis=1), np.linalg.norm(est[:, 3:], axis=1)
        v = g / DT
        idx = [(v >= BINS[i]) & (v < BINS[i + 1]) for i in range(len(BINS) - 1)]
        counts = [int(m.sum()) for m in idx]
        print(f"| {name} | " + " | ".join(f"{e[m].sum() / g[m].sum():.3f}" for m in idx) + f" | {e.sum() / g.sum():.3f} |")
    print("| frames | " + " | ".join(str(c) for c in counts) + f" | {sum(counts)} |")
    print("\nCells: summed estimated step length / summed true step length (1.000 = correct scale).")


if __name__ == "__main__":
    main()
