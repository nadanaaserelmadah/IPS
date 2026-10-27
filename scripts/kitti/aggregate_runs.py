"""KITTI: mean ± std over runs, per recipe, with network and gyro rotation, plus the ensemble of each recipe.

Input: a JSON file mapping recipe name -> list of dump directories (<seq>_estimated_poses.npy /
<seq>_gt_poses.npy for 05, 07, 10), and the gyro calibration directory from gyro_hybrid.py.
Output: a markdown table for the paper (mean ± sample std of the per-run mean t_rel / r_rel), and
for each recipe with ≥ 2 runs the score of the averaged prediction (ensemble).
  python scripts/kitti/aggregate_runs.py RUNS.json GYRO_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h  # noqa: E402

SEQS = ("05", "07", "10")


def load(d: Path):
    gt = {s: np.load(d / f"{s}_gt_poses.npy").astype(np.float64) for s in SEQS}
    est = {s: np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt[s])] for s in SEQS}
    return est, gt


def scores(est, gt, gyro_dir):
    net = np.array([h.score(est[s], gt[s]) for s in SEQS])
    gyr = np.array([h.score(h.hybrid(est[s], np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)), gt[s]) for s in SEQS])
    return net.mean(0), gyr.mean(0)  # (t_rel, r_rel) means over 05 / 07 / 10


def fmt(x):
    x = np.asarray(x)
    return f"{x[:, 0].mean():.2f} ± {x[:, 0].std(ddof=1):.2f}" if len(x) > 1 else f"{x[0, 0]:.2f}"


def main():
    runs = json.loads(Path(sys.argv[1]).read_text())
    gyro_dir = Path(sys.argv[2])
    print("| Recipe | runs | network t_rel % | gyro rotation t_rel % | ensemble network | ensemble gyro |")
    print("|---|---|---|---|---|---|")
    for recipe, dirs in runs.items():
        loaded = [load(Path(d)) for d in dirs]
        per = [scores(e, g, gyro_dir) for e, g in loaded]
        net, gyr = [p[0] for p in per], [p[1] for p in per]
        ens_n = ens_g = "—"
        if len(loaded) > 1:
            gt = loaded[0][1]
            mean = {s: np.mean([e[s] for e, _ in loaded], 0) for s in SEQS}
            en, eg = scores(mean, gt, gyro_dir)
            ens_n, ens_g = f"{en[0]:.2f}", f"{eg[0]:.2f}"
        print(f"| {recipe} | {len(dirs)} | {fmt(net)} | {fmt(gyr)} | {ens_n} | {ens_g} |")
        for d, (n, g) in zip(dirs, per):
            print(f"|   {Path(d).parent.parent.name or d} | | {n[0]:.2f} / {n[1]:.2f} | {g[0]:.2f} / {g[1]:.2f} | | |")
    print("| VIFT paper | | 1.96 | | | |")


if __name__ == "__main__":
    main()
