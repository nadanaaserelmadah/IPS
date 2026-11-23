"""Audit / thesis link: how often does an odometry error change a "has this place been visited?" decision?

Multi-robot exploration avoids sending a robot to a place another one has already covered. That needs each robot's
position, so odometry drift turns into two kinds of wrong map entries. On a grid of square cells (side c metres, in
the ground plane) built from a robot's trajectory, comparing the map built from ESTIMATED poses with the one from the
TRUE poses gives, per sequence:
  redundancy risk  = visited cells missing from the estimated map / visited cells   (another robot would revisit them)
  coverage loss    = cells in the estimated map that were never visited / cells in the estimated map
                     (another robot would skip ground that nobody covered)
  cell mismatch    = share of frames whose estimated cell differs from the true cell
Assumption, stated in every output: every trajectory starts at a known, shared origin (identity), i.e. a common frame.
Place recognition, loop closure, map merging and communication are NOT modelled, so this measures only what odometry
drift contributes.

Stages, as scored everywhere in the audit (scripts/audit/score_released.py): network output, gyroscope rotation + the
network's translation, + accelerometer length, and the frozen gate v3 (C5b).
  python scripts/audit/revisit_decisions.py NAME=DUMP_DIR[,DUMP_DIR...] [...] -- GEO NPZ GYRO RAW_CALIB [OUT_PNG]
Several comma-separated dump directories under one NAME are seeds of the same model: mean and SD over them.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/kitti"))
sys.path.insert(0, str(ROOT / "scripts/gate"))
import gyro_hybrid as h  # noqa: E402
import gate_v3_select as V  # noqa: E402
import run_final  # noqa: E402
from src.utils.kitti_utils import path_accu  # noqa: E402

SEQS = ("05", "07", "10")
CELLS = (5.0, 10.0, 20.0)  # metres
STAGES = ("network", "gyro + net t", "+ accel length", "gate v3")
SIGMA_NET, VARIANT = 0.829, "C5b"  # the frozen gate, as in score_released.py


def positions(rel6):
    """Ground-plane (x, z) positions of the camera from relative 6-DoF poses (KITTI camera frame: x right, z forward)."""
    T = path_accu(np.asarray(rel6, dtype=np.float64))
    return np.array([[m[0, 3], m[2, 3]] for m in T])


def stage_poses(est, gt, s, geo, npz_dir, gyro_dir, raw_calib):
    res, _, _ = V.candidates(s, est, gt, geo / f"{s}_geo_dir.npz", npz_dir, gyro_dir, raw_calib, SIGMA_NET)
    gR, gap = np.load(gyro_dir / f"{s}_gyro_R.npy"), h.load_gap(gyro_dir, s)
    return {"network": est, "gyro + net t": h.hybrid(est, gR, gap),
            "+ accel length": h.hybrid(res["hand-chosen (net dir + accel len)"], gR, gap),
            "gate v3": h.hybrid(res[run_final.NAMES[VARIANT]], gR, gap)}


def cells(xz, c):
    return {(int(np.floor(x / c)), int(np.floor(z / c))) for x, z in xz}


def metrics(p_est, p_gt, c):
    n = min(len(p_est), len(p_gt))
    p_est, p_gt = p_est[:n], p_gt[:n]
    ce, cg = cells(p_est, c), cells(p_gt, c)
    redundancy = len(cg - ce) / len(cg)
    coverage_loss = len(ce - cg) / len(ce)
    mismatch = float(np.mean(np.any(np.floor(p_est / c) != np.floor(p_gt / c), axis=1)))
    return redundancy, coverage_loss, mismatch


def main():
    a = sys.argv[1:]
    i = a.index("--")
    models = [(x.split("=", 1)[0], [Path(q) for q in x.split("=", 1)[1].split(",")]) for x in a[:i]]
    geo, npz_dir, gyro_dir, raw_calib = (Path(x) for x in a[i + 1:i + 5])
    out_png = Path(a[i + 5]) if len(a) > i + 5 else None
    missing = [s for s in SEQS if not (gyro_dir / f"{s}_gyro_maxgap_s.npy").is_file()]
    assert not missing and h.USE_GAP, f"gyro gap files missing for {missing}"

    R = {}  # (model, stage, c) -> per dump directory: array (drives, 3 metrics)
    traj = {}
    for name, dirs in models:
        for d in dirs:
            per = {}
            for s in SEQS:
                gt = np.load(d / f"{s}_gt_poses.npy").astype(np.float64)
                est = np.load(d / f"{s}_estimated_poses.npy").astype(np.float64)[: len(gt)]
                p_gt = positions(gt)
                for stage, e6 in stage_poses(est, gt, s, geo, npz_dir, gyro_dir, raw_calib).items():
                    p = positions(e6)
                    if d == dirs[0]:
                        traj[(name, s, stage)] = p
                    for c in CELLS:
                        per.setdefault((stage, c), []).append(metrics(p, p_gt, c))
                traj[(name, s, "gt")] = p_gt
            for (stage, c), v in per.items():
                R.setdefault((name, stage, c), []).append(np.array(v) * 100)

    def cell(name, stage, c, k):
        v = np.array([x.mean(0)[k] for x in R[(name, stage, c)]])  # per dump directory: mean over the three drives
        return f"{v.mean():.1f} ± {v.std(ddof=1):.1f}" if len(v) > 1 else f"{v[0]:.1f}"

    print("## Wrong 'visited?' map entries caused by odometry drift (KITTI 05/07/10, mean over the three drives)\n")
    print("Assumes a known common start for every trajectory; no loop closure or map merging is modelled. "
          "Several seeds: mean ± SD over seeds.\n")
    for c in CELLS:
        print(f"### Cell side {c:g} m\n")
        print("| model | stage | redundancy risk % | coverage loss % | frames in the wrong cell % |\n|---|---|---|---|---|")
        for name, _ in models:
            for stage in STAGES:
                print(f"| {name} | {stage} | {cell(name, stage, c, 0)} | {cell(name, stage, c, 1)} | {cell(name, stage, c, 2)} |")
        print()
    print("### Per drive, cell side 10 m, mean over seeds (redundancy risk % / coverage loss % / wrong-cell %)\n")
    print("| model | stage | 05 | 07 | 10 |\n|---|---|---|---|---|")
    for name, _ in models:
        for stage in STAGES:
            v = np.mean(R[(name, stage, 10.0)], axis=0)
            print(f"| {name} | {stage} | " + " | ".join(f"{x[0]:.0f} / {x[1]:.0f} / {x[2]:.0f}" for x in v) + " |")

    if out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        name = models[0][0]
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.9), facecolor="#fcfcfb")
        for ax, s in zip(axes, SEQS):
            ax.set_facecolor("#fcfcfb")
            g = traj[(name, s, "gt")]
            ax.plot(g[:, 0], g[:, 1], color="#0b0b0b", lw=2.2, label="true")
            ax.plot(*traj[(name, s, "network")].T, color="#eb6834", lw=1.4, label="network")
            ax.plot(*traj[(name, s, "gate v3")].T, color="#2a78d6", lw=1.4, label="after gyro + accel + gate")
            from matplotlib.ticker import MultipleLocator
            ax.xaxis.set_major_locator(MultipleLocator(100))
            ax.yaxis.set_major_locator(MultipleLocator(100))
            ax.grid(color="#e1e0d9", lw=0.6)
            ax.set_aspect("equal", adjustable="datalim")
            ax.tick_params(labelsize=6, colors="#898781")
            ax.set_title(f"KITTI {s}", fontsize=10, loc="left")
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes[0].legend(fontsize=7, frameon=False, loc="best")
        fig.text(0.01, 0.01, f"Model: {name}. Ground-plane positions in metres; every trajectory starts at the origin.", fontsize=7, color="#898781")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        fig.savefig(out_png, dpi=200, facecolor="#fcfcfb")
        print(f"\nfigure: {out_png}")


if __name__ == "__main__":
    main()
