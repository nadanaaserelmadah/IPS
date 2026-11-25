"""Scientific Reports version of the "measure vs learn" figure (vector PDF): same data, bins, medians and
interquartile bands as scripts/gate/plot_observability.py, only the styling differs.
  python scripts/scireports/fig_measure_vs_learn.py KITTI_DUMP KITTI_GEO NPZ_DIR GYRO_DIR RAW_CALIB EUROC_DUMPS EUROC_GEO OUT_PDF
(a) network minus two-view geometric direction error vs de-rotated parallax (KITTI solid, circles; EuRoC dashed, squares);
(b) KITTI network minus accelerometer length error vs accelerometer-window condition number.
Lines are bin medians, grey bands the interquartile range.
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts/gate"))
sys.path.insert(0, str(HERE))
import sr_style as sr  # noqa: E402

sr.apply()
import matplotlib.pyplot as plt  # noqa: E402

import plot_observability as po  # noqa: E402  (data code; its main() is not run)

isk = po.isk


def main():
    kd, kg, npz, gyro, raw, ed, eg, out = (Path(a) for a in sys.argv[1:9])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(sr.WIDTH_IN, 2.3))
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.20, top=0.88, wspace=0.42)
    edges = np.geomspace(0.5, 60, 16)

    for (x, y), col, mk, ls, lab in ((po.kitti_direction(kd, kg), sr.BLUE, "o", "-", "KITTI (base)"),
                                     (po.euroc_direction(ed, eg), sr.ORANGE, "s", sr.DASH, "EuRoC (PI-VIFT)")):
        c, m, q1, q3 = po.binned(x, y, edges)
        a1.fill_between(c, q1, q3, **sr.BAND)
        a1.plot(c, m, color=col, lw=sr.LW, ls=ls, marker=mk, ms=3.5, label=lab)
    a1.axhline(0, color=sr.BLACK, lw=1.0, zorder=1)
    a1.set_xscale("log")
    a1.set_yscale("symlog", linthresh=1.0)  # KITTI differences are below 1 deg, EuRoC's are tens of degrees
    a1.set_yticks([-2, -1, 0, 1, 10, 100])
    a1.set_xticks([1, 2, 5, 10, 20, 50])
    sr.plain_axis(a1.xaxis)
    sr.plain_axis(a1.yaxis)
    a1.set_xlabel("De-rotated parallax of the frame pair (px)")
    a1.set_ylabel("Network minus geometric\ndirection error (°)")
    a1.set_title("Translation direction", loc="left", pad=4)
    sr.clean(a1)
    a1.legend(loc="upper left", bbox_to_anchor=(0.01, 0.80), handlelength=2.4, borderaxespad=0.0)
    a1.text(0.98, 0.42, "Above 0: measuring is better\nBelow 0: the network is better", transform=a1.transAxes,
            ha="right", va="bottom")

    x, y = po.kitti_length(kd, npz, gyro, raw)
    c, m, q1, q3 = po.binned(x, y, np.geomspace(max(x.min(), 1.0), x.max() * 1.01, 12))
    a2.fill_between(c, q1, q3, **sr.BAND)
    a2.axhline(0, color=sr.BLACK, lw=1.0, zorder=1)
    a2.plot(c, m, color=sr.BLUE, lw=sr.LW, marker="o", ms=3.5, label="KITTI (base)")
    a2.axvline(isk.COND_MAX, color=sr.BLACK, lw=1.0, ls=(0, (4, 3)))
    a2.set_xscale("log")
    a2.margins(y=0.10)
    sr.log_ticks(a2, extra=(isk.COND_MAX,))
    sr.plain_axis(a2.yaxis)
    a2.text(isk.COND_MAX, a2.get_ylim()[1], "Gate threshold ", va="top", ha="right")
    a2.set_xlabel("Accelerometer window condition number")
    a2.set_ylabel("Network minus accelerometer\nlength error (%)")
    a2.set_title("Step length", loc="left", pad=4)
    sr.clean(a2)
    a2.text(isk.COND_MAX * 0.8, 0.04, "Above 0: measuring is better", transform=a2.get_xaxis_transform(),
            ha="right", va="bottom")

    sr.panel_letter(a1, "a", -36)
    sr.panel_letter(a2, "b", -44)
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
