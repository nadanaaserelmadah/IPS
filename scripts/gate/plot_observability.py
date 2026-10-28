"""Paper figure: when is a pose component better measured than learned? Per-edge evidence on KITTI and EuRoC.

(a) Translation direction: network error minus two-view geometric error (deg), binned by the edge's de-rotated
    parallax (px). Above zero, the measurement is better. KITTI (VIFT, July checkpoint, 05 / 07 / 10) and EuRoC
    (PI-VIFT s123, MH_05 / V2_02 / V1_01 / V1_02).
(b) Step length on KITTI: network length error minus accelerometer length error (|L/L_gt - 1|, %), binned by the
    accelerometer window's condition number. Above zero, the accelerometer is better.
Lines are bin medians, bands the interquartile range. Colours: reference categorical slots 1-2 (blue, orange);
the two datasets also differ by marker shape.
  python scripts/gate/plot_observability.py KITTI_DUMP KITTI_GEO NPZ_DIR GYRO_DIR RAW_CALIB EUROC_DUMPS EUROC_GEO OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/kitti"))
sys.path.insert(0, str(ROOT / "scripts/gate"))
import gyro_hybrid as h  # noqa: E402,F401
import imu_scale_kitti as isk  # noqa: E402
import observability_gate as G  # noqa: E402

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def binned(x, y, edges):
    c, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b) & np.isfinite(y)
        if m.sum() >= 20:
            c.append(np.sqrt(a * b))
            med.append(np.median(y[m]))
            q1.append(np.percentile(y[m], 25))
            q3.append(np.percentile(y[m], 75))
    return map(np.array, (c, med, q1, q3))


def kitti_direction(dump, geo):
    xs, ys = [], []
    for s in ("05", "07", "10"):
        gt = np.load(dump / f"{s}_gt_poses.npy")
        n = len(gt)
        te = np.load(dump / f"{s}_estimated_poses.npy")[:n, 3:]
        z = np.load(geo / f"{s}_geo_dir.npz")
        dg, inf = z["d"][:n], json.loads(str(z["info"]))
        par = G.parallax(inf, n)
        mv = (np.linalg.norm(gt[:, 3:], axis=1) > 0.1) & ~np.isnan(dg[:, 0]) & np.isfinite(par)
        xs.append(par[mv])
        ys.append(G.angle_deg(te[mv], gt[mv, 3:]) - G.angle_deg(dg[mv], gt[mv, 3:]))
    return np.concatenate(xs), np.concatenate(ys)


def euroc_direction(dumps, geo, run="pivift_s123_last"):
    xs, ys = [], []
    for seq in ("MH_05_difficult", "V2_02_medium", "V1_01_easy", "V1_02_medium"):
        f = dumps / run / f"{seq}_estimated_poses.npy"
        if not f.is_file():
            continue
        est = np.load(f)
        gt = np.load(dumps / run / f"{seq}_gt_poses.npy")
        z = np.load(geo / f"{seq}_geo.npz")
        dg, inf = z["d"], json.loads(str(z["info"]))
        par = G.parallax(inf, len(gt))
        mv = (np.linalg.norm(gt[:, 3:], axis=1) > 0.01) & ~np.isnan(dg[:, 0]) & np.isfinite(par)
        xs.append(par[mv])
        ys.append(G.angle_deg(est[mv, 3:], gt[mv, 3:]) - G.angle_deg(dg[mv], gt[mv, 3:]))
    return np.concatenate(xs), np.concatenate(ys)


def kitti_length(dump, npz_dir, gyro_dir, raw_calib):
    xs, ys = [], []
    for s in ("05", "07", "10"):
        gt = np.load(dump / f"{s}_gt_poses.npy")
        n = len(gt)
        te = np.load(dump / f"{s}_estimated_poses.npy")[:n, 3:]
        Ln, Lg = np.linalg.norm(te, axis=1), np.linalg.norm(gt[:, 3:], axis=1)
        calib = json.loads((gyro_dir / f"calib_{isk.DAY[s]}.json").read_text())
        p_c, _ = isk.cam_in_imu(raw_calib, isk.DAY[s])
        cond = np.full(n, np.nan)
        old = isk.COND_MAX
        isk.COND_MAX = np.inf  # keep every window's solution, to plot it against its conditioning
        La, _, _ = isk.seq_lengths(npz_dir / f"{s}.npz", calib, p_c, te / Ln[:, None], 500, gap=h.load_gap(gyro_dir, s), cond_out=cond)
        isk.COND_MAX = old
        mv = (Lg > 0.1) & np.isfinite(La) & np.isfinite(cond)
        xs.append(cond[mv])
        ys.append(100 * (np.abs(Ln[mv] / Lg[mv] - 1) - np.abs(La[mv] / Lg[mv] - 1)))
    return np.concatenate(xs), np.concatenate(ys)


PLAIN = FuncFormatter(lambda v, _: f"{v:g}")


def style(ax):
    ax.xaxis.set_major_formatter(PLAIN)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.axhline(0, color=MUTED, lw=0.9)


def main():
    kd, kg, npz, gyro, raw, ed, eg, out = (Path(a) for a in sys.argv[1:9])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURFACE)
    edges = np.geomspace(0.5, 60, 16)
    for (x, y), col, mk, lab in ((kitti_direction(kd, kg), BLUE, "o", "KITTI (VIFT)"),
                                 (euroc_direction(ed, eg), ORANGE, "s", "EuRoC (PI-VIFT)")):
        c, m, q1, q3 = binned(x, y, edges)
        a1.fill_between(c, q1, q3, color=col, alpha=0.18, lw=0)
        a1.plot(c, m, color=col, lw=2, marker=mk, ms=5, label=lab)
    a1.set_xscale("log")
    a1.set_yscale("symlog", linthresh=1.0)  # KITTI differences are below 1 deg, EuRoC's are tens of degrees
    a1.set_yticks([-2, -1, 0, 1, 10, 100])
    a1.yaxis.set_major_formatter(PLAIN)
    a1.set_xlabel("de-rotated parallax of the edge (px)", color=MUTED, fontsize=9)
    a1.set_ylabel("network minus geometric direction error (°)", color=MUTED, fontsize=9)
    a1.set_title("(a) Translation direction", color=INK, fontsize=11, loc="left")
    style(a1)
    a1.legend(fontsize=8, frameon=False, labelcolor=INK, loc="center left")
    a1.set_xticks([1, 2, 5, 10, 20, 50])
    a1.text(0.98, 0.42, "above 0: measuring is better\nbelow 0: the network is better", transform=a1.transAxes, ha="right", fontsize=8, color=MUTED)
    x, y = kitti_length(kd, npz, gyro, raw)
    c, m, q1, q3 = binned(x, y, np.geomspace(max(x.min(), 1.0), x.max() * 1.01, 12))
    a2.fill_between(c, q1, q3, color=BLUE, alpha=0.18, lw=0)
    a2.plot(c, m, color=BLUE, lw=2, marker="o", ms=5, label="KITTI (VIFT)")
    a2.axvline(isk.COND_MAX, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    a2.set_xscale("log")
    a2.text(isk.COND_MAX, a2.get_ylim()[1], "gate threshold ", fontsize=8, color=MUTED, va="top", ha="right")
    a2.set_xlabel("accelerometer window condition number", color=MUTED, fontsize=9)
    a2.set_ylabel("network minus accelerometer length error (%)", color=MUTED, fontsize=9)
    a2.set_title("(b) Step length", color=INK, fontsize=11, loc="left")
    style(a2)
    a2.text(0.98, 0.04, "above 0: measuring is better", transform=a2.transAxes, ha="right", fontsize=8, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
