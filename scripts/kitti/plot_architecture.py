"""Paper figure: architecture of the gyro-anchored hybrid on top of VIFT (static PNG + SVG).

Gray boxes are VIFT's (frozen encoder, causal pose transformer); teal boxes are ours (timestamp
correction, gyro + accelerometer preintegration, hybrid pose). Dashed arrow: the gyro-conditioned variant, where the
gyro rotation is also fed to the network.
  python scripts/kitti/plot_architecture.py OUT_DIR
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, MUTED = "#0b0b0b", "#52514e"
GRAY = dict(fill="#F1EFE8", edge="#5F5E5A", title="#2C2C2A", sub="#5F5E5A")
TEAL = dict(fill="#E1F5EE", edge="#0F6E56", title="#04342C", sub="#0F6E56")

W, H = 680, 540
BOXES = {  # key: (x, y_top, w, h, title, subtitle, palette)
    "img": (60, 40, 240, 56, "Camera images", "10 Hz, left camera", GRAY),
    "imu": (380, 40, 240, 56, "Raw IMU (OXTS)", "gyro + accel, true sample times", GRAY),
    "enc": (60, 136, 240, 56, "Frozen encoder", "VS-VIO visual + inertial, 768-d", GRAY),
    "ts": (380, 136, 240, 56, "Timestamp correction", "dt ≈ 104 ms, lag +60 ms", TEAL),
    "tf": (60, 232, 240, 56, "Causal pose transformer", "VIFT; outputs direction of t", GRAY),
    "gyr": (380, 232, 240, 56, "Gyro + accel. preintegration", "outputs R and step length |t|", TEAL),
    "hyb": (200, 328, 280, 56, "Hybrid pose", "R gyro, |t| accel., direction net", TEAL),
    "out": (200, 424, 280, 56, "Trajectory, KITTI 05 / 07 / 10", "1.27 / 0.18 vs VIFT 1.96 / 0.46", GRAY),
}


def y(v):  # top-down layout coordinates -> matplotlib (origin bottom-left)
    return H - v


def box(ax, x, top, w, h, title, sub, p):
    ax.add_patch(FancyBboxPatch((x, y(top + h)), w, h, boxstyle="round,pad=0,rounding_size=8",
                                fc=p["fill"], ec=p["edge"], lw=0.8))
    ax.text(x + w / 2, y(top + 19), title, ha="center", va="center", fontsize=10.5, weight="bold", color=p["title"])
    ax.text(x + w / 2, y(top + 39), sub, ha="center", va="center", fontsize=9, color=p["sub"])


def arrow(ax, pts, dashed=False):
    xs = [p[0] for p in pts]
    ys = [y(p[1]) for p in pts]
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=MUTED, lw=1.0, ls="--" if dashed else "-")
    ax.add_patch(FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>", mutation_scale=10,
                                 color=MUTED, lw=1.0, ls="--" if dashed else "-", shrinkA=0, shrinkB=0))


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    for b in BOXES.values():
        box(ax, *b)
    for x in (180, 500):
        for top in (96, 192):
            arrow(ax, [(x, top), (x, top + 40)])
    arrow(ax, [(380, 260), (300, 260)], dashed=True)
    arrow(ax, [(180, 288), (180, 356), (200, 356)])
    arrow(ax, [(500, 288), (500, 356), (480, 356)])
    arrow(ax, [(340, 384), (340, 424)])
    for x, p, label in ((60, GRAY, "From VIFT"), (190, TEAL, "Ours")):
        ax.add_patch(FancyBboxPatch((x, y(514)), 14, 14, boxstyle="round,pad=0,rounding_size=3",
                                    fc=p["fill"], ec=p["edge"], lw=0.8))
        ax.text(x + 22, y(507), label, va="center", fontsize=9, color=INK)
    ax.plot([290, 324], [y(507)] * 2, color=MUTED, lw=1.0, ls="--")
    ax.text(332, y(507), "gyro rotation also fed to the network (variant)", va="center", fontsize=9, color=INK)
    for ext in ("png", "svg"):
        fig.savefig(out / f"kitti_gyro_hybrid_architecture.{ext}", dpi=220, facecolor="white")
    print(out / "kitti_gyro_hybrid_architecture.png")


if __name__ == "__main__":
    main()
