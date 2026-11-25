"""Scientific Reports version of the KITTI heading-error figure (vector PDF): same data as
scripts/kitti/plot_figures.py (second figure), only the styling differs.
  python scripts/scireports/fig_heading_error.py DUMP_DIR GYRO_DIR OUT_DIR [ignored extra args]
Writes OUT_DIR/kitti_heading_error.pdf. Network heading error (dashed blue) and gyro-rotation heading error (dotted
orange) vs distance travelled on KITTI 05 / 07 / 10; the zero line is the ground-truth heading.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/kitti"))
sys.path.insert(0, str(HERE))
import sr_style as sr  # noqa: E402

sr.apply()
import matplotlib.pyplot as plt  # noqa: E402

import gyro_hybrid as h  # noqa: E402
import plot_figures as pf  # noqa: E402  (accumulate; its main() is not run)


def main():
    dump, gyro, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    out.mkdir(parents=True, exist_ok=True)
    seqs = ("05", "07", "10")
    data = {}
    for s in seqs:
        est = np.load(dump / f"{s}_estimated_poses.npy").astype(np.float64)
        gt = np.load(dump / f"{s}_gt_poses.npy").astype(np.float64)
        hyb = h.hybrid(est, np.load(gyro / f"{s}_gyro_R.npy"), h.load_gap(gyro, s))
        data[s] = {k: pf.accumulate(v) for k, v in (("gt", gt), ("net", est), ("hyb", hyb))}

    fig, axes = plt.subplots(1, 3, figsize=(sr.WIDTH_IN, 1.9), sharey=True)
    fig.subplots_adjust(left=0.095, right=0.995, bottom=0.235, top=0.90, wspace=0.12)
    for i, (ax, s) in enumerate(zip(axes, seqs)):
        d = data[s]
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(d["gt"][:, :3, 3], axis=0), axis=1))]
        ax.axhline(0, color=sr.BLACK, lw=1.0, zorder=1)  # ground-truth heading
        for key, col, ls, lab in (("net", sr.BLUE, sr.DASH, "Network"), ("hyb", sr.ORANGE, sr.DOT, "Gyro rotation")):
            E = np.einsum("nji,njk->nik", d["gt"][:, :3, :3], d[key][:, :3, :3])  # R_gt^T R_est
            err = np.degrees(np.arctan2(E[:, 0, 2], E[:, 0, 0]))  # rotation about camera y (yaw)
            ax.plot(dist / 1000, err, color=col, lw=sr.LW, ls=ls, label=lab, zorder=2)
        ax.margins(x=0.02, y=0.08)
        ax.set_xlabel("Distance travelled (km)")
        if i == 0:
            ax.set_ylabel("Heading error (°)")
        if i == 2:
            ax.legend(loc="lower left", handlelength=2.4, borderaxespad=0.2)
        sr.clean(ax)
        ax.text(0.02, 0.97, f"Drive {s}", transform=ax.transAxes, ha="left", va="top")
        sr.panel_letter(ax, "abc"[i], -34 if i == 0 else -8)
    fig.savefig(out / "kitti_heading_error.pdf", facecolor="white")
    print("wrote", out / "kitti_heading_error.pdf")


if __name__ == "__main__":
    main()
