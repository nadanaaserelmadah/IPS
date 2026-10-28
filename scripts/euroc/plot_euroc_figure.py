"""Paper figure: EuRoC top-down trajectories (MH_05, V2_02, V1_01, V1_02), SE(3)-aligned to ground truth.

Lines: ground truth; a learned model (VIFT-style s123 last where its dump exists, else PI-VIFT s123 last);
the network-free same-sensor pipeline (gyro rotation + geometric direction x accelerometer length,
with the |g| plausibility check);
and the stereo pipeline (extra camera). Uses the per-edge caches written by geo_direction.py.
  python scripts/euroc/plot_euroc_figure.py EUROC_ROOT GEO_CACHE_DIR DUMPS_DIR OUT_PNG
Colors: reference categorical slots 1-3 (blue, orange, aqua); GT in neutral ink; line styles differ too.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gyro_hybrid_euroc as gh  # noqa: E402
import imu_scale  # noqa: E402
from src.metrics.euroc_metrics_calculator import align_trajectory_lengths, align_trajectory_se3  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQS = ("MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium")


def aligned(xyz, gt):
    e, g = align_trajectory_lengths(xyz, gt)
    return align_trajectory_se3(e, g), g


def main():
    root, cache, dumps, out = (Path(a) for a in sys.argv[1:5])
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.4), facecolor=SURFACE)
    for ax, seq in zip(axes, SEQS):
        ts = dumps / "pivift_dir_best"
        t_end = np.load(ts / f"{seq}_timestamps.npy").astype(np.int64)
        gxyz = np.load(ts / f"{seq}_gt_xyz.npy").reshape(-1, 3).astype(np.float64)
        z = np.load(cache / f"{seq}_geo.npz")
        R, d, t_st = z["R"], z["d"], z["t_st"]
        ok = ~np.isnan(d[:, 0])
        d_full, last = [], np.array([0.0, 0.0, 1.0])
        for k in range(len(t_end)):
            last = d[k] if ok[k] else last
            d_full.append(last)
        d_full = np.array(d_full)
        l_imu, _ = imu_scale.imu_lengths(root, seq, t_end, d_full, g_tol=0.5)  # gravity plausibility check
        L, last_l = [], 0.0
        for x in l_imu:
            last_l = x if not np.isnan(x) else last_l
            L.append(last_l)
        mono = gh.compose(list(R), d_full * np.array(L)[:, None])
        okst, lastt, tp = ~np.isnan(t_st[:, 0]), np.zeros(3), []
        for k in range(len(t_end)):  # same fallback as geo_direction.evaluate (Table P8-H)
            if okst[k]:
                lastt = t_st[k]
            tp.append(t_st[k] if okst[k] else (d[k] * np.linalg.norm(lastt) if ok[k] else lastt))
        stereo = gh.compose(list(R), np.array(tp))
        name = "vift_s123_last" if (dumps / "vift_s123_last" / f"{seq}_estimated_poses.npy").is_file() else "pivift_s123_last"
        est = np.load(dumps / name / f"{seq}_estimated_poses.npy").astype(np.float64)
        net = gh.compose([gh.euler_to_rotation_matrix(p[:3]) for p in est], est[:, 3:])
        g = None
        lines = []
        for xyz, col, ls, lab in ((net, BLUE, (0, (5, 2)), "Learned (VIFT-style; PI-VIFT on V1_xx)"),
                                  (mono, AQUA, (0, (6, 2, 1, 2)), "Gyro + geometric direction + accelerometer (no network)"),
                                  (stereo, ORANGE, (0, (1, 1.5)), "Gyro + stereo (no network; extra camera)")):
            a, g = aligned(xyz, gxyz)
            ate = float(np.sqrt(np.mean(np.sum((a - g) ** 2, 1))))
            lines.append((a, col, ls, f"{lab}", ate))
        ax.plot(g[:, 0], g[:, 1], color=INK, lw=2, label="Ground truth")
        for a, col, ls, lab, ate in lines:
            ax.plot(a[:, 0], a[:, 1], color=col, lw=1.8, ls=ls, label=lab)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(seq.split("_")[0] + "_" + seq.split("_")[1], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("x (m)", color=MUTED, fontsize=9)
        if seq == SEQS[0]:
            ax.set_ylabel("y (m)", color=MUTED, fontsize=9)
        ax.set_facecolor(SURFACE)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(color=GRID, linewidth=0.6)
        txt = "\n".join(f"{l.split(' (')[0]}: {ate:.2f} m" for _, _, _, l, ate in lines)
        ax.text(0.01, -0.17, txt, transform=ax.transAxes, fontsize=7.5, color=MUTED, va="top", linespacing=1.4)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle("EuRoC test sequences, SE(3)-aligned top-down trajectories (ATE per method below each panel)",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.15, 1, 1))
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
