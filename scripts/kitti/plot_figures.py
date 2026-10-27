"""Paper figures for the KITTI gyro result (static PNGs).

Fig. A: top-down trajectories on 05 / 07 / 10 (x-z plane): GT, network (reproduction), network
translation + gyro rotation, and (with NPZ_DIR RAW_CALIB_DIR) also network direction x accelerometer
step length + gyro rotation (imu_scale_kitti.py, W=500). Fig. B: absolute heading (yaw) error vs distance.
  python scripts/kitti/plot_figures.py DUMP_DIR GYRO_DIR OUT_DIR [NPZ_DIR RAW_CALIB_DIR]
Colors: reference categorical slots 1-3 (blue #2a78d6, orange #eb6834, aqua #1baf7a); GT in neutral ink, solid;
estimates also differ by line style, so identity is never color alone.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid as h  # noqa: E402
import gyro_rotation_test as g  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def accumulate(rel6: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    out = [T.copy()]
    for p in rel6:
        M = np.eye(4)
        M[:3, :3] = g.eulerAnglesToRotationMatrix(p[:3])
        M[:3, 3] = p[3:]
        T = T @ M
        out.append(T.copy())
    return np.array(out)


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)


def main():
    dump, gyro, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    imu = (Path(sys.argv[4]), Path(sys.argv[5])) if len(sys.argv) > 5 else None
    out.mkdir(parents=True, exist_ok=True)
    seqs = ("05", "07", "10")
    data = {}
    for s in seqs:
        est = np.load(dump / f"{s}_estimated_poses.npy").astype(np.float64)
        gt = np.load(dump / f"{s}_gt_poses.npy").astype(np.float64)
        hyb = h.hybrid(est, np.load(gyro / f"{s}_gyro_R.npy"), h.load_gap(gyro, s))
        data[s] = {k: accumulate(v) for k, v in (("gt", gt), ("net", est), ("hyb", hyb))}
        data[s]["scores"] = (h.score(est, gt), h.score(hyb, gt))
        if imu:
            import imu_scale_kitti as isk

            e6, gap = isk.run_lengths(imu[0], gyro, imu[1], s, est[: len(gt)], 500)
            acc = h.hybrid(e6, np.load(gyro / f"{s}_gyro_R.npy"), gap)
            data[s]["acc"] = accumulate(acc)
            data[s]["acc_score"] = h.score(acc, gt)

    fig, axes = plt.subplots(1, 3, figsize=(12, 5.0), facecolor=SURFACE)
    for ax, s in zip(axes, seqs):
        d = data[s]
        (nt, nr), (ht, hr) = d["scores"]
        ax.plot(d["gt"][:, 0, 3], d["gt"][:, 2, 3], color=INK, lw=2, label="Ground truth")
        ax.plot(d["net"][:, 0, 3], d["net"][:, 2, 3], color=BLUE, lw=2, ls=(0, (5, 2)), label="Network (reproduction)")
        ax.plot(d["hyb"][:, 0, 3], d["hyb"][:, 2, 3], color=ORANGE, lw=2, ls=(0, (1, 1.5)), label="Gyro rotation + network translation")
        if "acc" in d:
            ax.plot(d["acc"][:, 0, 3], d["acc"][:, 2, 3], color=AQUA, lw=2, ls=(0, (6, 2, 1, 2)), label="Gyro rotation + accelerometer step length")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(f"Sequence {s}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("x (m)", color=MUTED, fontsize=9)
        if s == "05":
            ax.set_ylabel("z (m)", color=MUTED, fontsize=9)
        style(ax)
        extra = f"\n+ accelerometer length {d['acc_score'][0]:.2f}% / {d['acc_score'][1]:.2f}" if "acc" in d else ""
        ax.text(0.01, -0.17, f"network {nt:.2f}% / {nr:.2f}\ngyro rotation {ht:.2f}% / {hr:.2f}{extra}", transform=ax.transAxes,
                fontsize=8, color=MUTED, va="top", linespacing=1.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4 if imu else 3, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("KITTI test sequences: t_rel % / r_rel °/100 m (reproduction checkpoint, with-08 split)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out / "kitti_gyro_trajectories.png", dpi=200, facecolor=SURFACE)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), facecolor=SURFACE, sharey=True)
    for ax, s in zip(axes, seqs):
        d = data[s]
        dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(d["gt"][:, :3, 3], axis=0), axis=1))]
        for key, col, ls, lab in (("net", BLUE, (0, (5, 2)), "Network"), ("hyb", ORANGE, (0, (1, 1.5)), "Gyro rotation")):
            E = np.einsum("nji,njk->nik", d["gt"][:, :3, :3], d[key][:, :3, :3])  # R_gt^T R_est
            err = np.degrees(np.arctan2(E[:, 0, 2], E[:, 0, 0]))  # rotation about camera y (yaw)
            ax.plot(dist / 1000, err, color=col, lw=2, ls=ls, label=lab)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(f"Sequence {s}", color=INK, fontsize=11, loc="left")
        ax.set_xlabel("distance travelled (km)", color=MUTED, fontsize=9)
        if s == "05":
            ax.set_ylabel("heading error (°)", color=MUTED, fontsize=9)
            ax.legend(fontsize=8, frameon=False, labelcolor=INK)
        style(ax)
    fig.tight_layout()
    fig.savefig(out / "kitti_gyro_heading_error.png", dpi=200, facecolor=SURFACE)
    print("wrote", out / "kitti_gyro_trajectories.png", out / "kitti_gyro_heading_error.png")


if __name__ == "__main__":
    main()
