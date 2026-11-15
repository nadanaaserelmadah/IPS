"""Audit paper, E6: is the gap between published learned-VIO methods larger than one method's seed spread?

Reads the gate tables printed by scripts/audit/score_released.py (OXTS rows, KITTI 05 / 07 / 10):
  - released checkpoints (E1-E3 output): one run per method;
  - VIFT clean protocol (E4 output): base_s* (released IMU files) and fix_s* (lag-corrected), same code,
    only the seed and the IMU files differ.
Prints the spread table, where each released checkpoint falls in the seed distribution, and the expected best
of k seeds (what a paper reports when it keeps the best seed by test score). Draws a strip plot.
  python scripts/audit/e6_spread.py RELEASED_TXT SEEDS_TXT OUT_PNG
"""
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

STAGES = ["network", "gyro rot + net t", "+ accel length", "gate v3"]
NAMES = {"vsvio/policy": "VS-VIO policy", "vsvio/fullvision": "VS-VIO full vision", "ulvio/ulvio_out": "UL-VIO",
         "nasvio_out/flops": "NASVIO flops", "nasvio_out/latency": "NASVIO latency"}
VIFT_PAPER_NET = 1.96  # VIFT (arXiv 2409.08769) Table, mean t_rel of 05/07/10, its own protocol
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_BASE, C_FIX, C_REL = "#2a78d6", "#1baf7a", "#eb6834"  # validated: scripts/validate_palette.js, all pairs


def gate_rows(path):
    """{model: [network, gyro, accel, gate]} from the 'KITTI test, variant C5b' table, OXTS rows."""
    text = Path(path).read_text().split("### KITTI test")[1].split("\n## ")[0]
    rows = {}
    for line in text.splitlines():
        m = re.match(r"\| (\S+) \| OXTS \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| \*\*([\d.]+)\*\* \|", line)
        if m:
            rows[m[1]] = [float(m[i]) for i in range(2, 6)]
    return rows


def best_of_k(x, k):
    """Expected minimum of k runs drawn without replacement from x."""
    return float(np.mean([min(c) for c in combinations(x, k)]))


def main():
    released_txt, seeds_txt, out_png = sys.argv[1:4]
    rel = {NAMES[k]: v for k, v in gate_rows(released_txt).items() if k in NAMES}
    seeds = gate_rows(seeds_txt)
    base = {k.split("/")[0]: v for k, v in seeds.items() if k.startswith("base_")}
    fix = {k.split("/")[0]: v for k, v in seeds.items() if k.startswith("fix_")}
    assert len(rel) == 5 and len(base) == 6 and len(fix) == 6, (rel.keys(), base.keys(), fix.keys())
    R, B, F = (np.array(list(d.values())) for d in (rel, base, fix))
    A = np.vstack([B, F])

    print("## E6: spread across released methods vs across seeds of one method (t_rel %, 05/07/10, OXTS)\n")
    print("| stage | 5 released checkpoints: min–max (range) | VIFT, 6 seeds, released IMU: min–max (range), SD |"
          " VIFT, 12 runs (both IMU sets): min–max (range), SD |")
    print("|---|---|---|---|")
    for i, s in enumerate(STAGES):
        f = lambda x: f"{x[:, i].min():.2f}–{x[:, i].max():.2f} ({np.ptp(x[:, i]):.2f})"
        print(f"| {s} | {f(R)} | {f(B)}, {B[:, i].std(ddof=1):.2f} | {f(A)}, {A[:, i].std(ddof=1):.2f} |")

    print("\n## Where each released checkpoint falls among the 12 VIFT runs (share of VIFT runs that are worse)\n")
    print("| released | network | VIFT runs worse | gate v3 | VIFT runs worse |\n|---|---|---|---|---|")
    for name, v in rel.items():
        print(f"| {name} | {v[0]:.2f} | {np.mean(A[:, 0] > v[0]):.0%} | {v[3]:.2f} | {np.mean(A[:, 3] > v[3]):.0%} |")

    print("\n## Keeping the best of k seeds by test score (VIFT, 6 seeds, released IMU): expected reported number\n")
    print("| k | network | gate v3 |\n|---|---|---|")
    for k in range(1, 7):
        print(f"| {k} | {best_of_k(B[:, 0], k):.2f} | {best_of_k(B[:, 3], k):.2f} |")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), sharey=True, facecolor=SURFACE)
    groups = [("Released\nmodels", R, C_REL, "D"), ("VIFT seeds\nreleased IMU", B, C_BASE, "o"),
              ("VIFT seeds\nlag-fixed IMU", F, C_FIX, "s")]
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, [(0, "Network output"), (3, "After gyro + accel + gate v3")]):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for gx, (label, X, c, mk) in enumerate(groups):
            y = X[:, col]
            jit = rng.uniform(-0.12, 0.12, len(y)) if X is not R else np.zeros(len(y))
            ax.plot([gx - 0.3, gx + 0.3], [y.mean()] * 2, color=AXIS, lw=1.5, zorder=1, solid_capstyle="round")
            ax.scatter(gx + jit, y, s=46, c=c, marker=mk, edgecolors=SURFACE, linewidths=1.5, zorder=3)
            if X is R:  # direct labels, pushed apart so none collide; a leader line when a label moved
                order = np.argsort(y)
                ly = y[order].copy()
                for j in range(1, len(ly)):
                    ly[j] = max(ly[j], ly[j - 1] + 0.11)
                for j, oi in enumerate(order):
                    ax.annotate(list(rel)[oi], (gx, y[oi]), xytext=(gx - 0.16, ly[j]), textcoords="data", ha="right",
                                va="center", fontsize=7, color=INK2,
                                arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.6, shrinkA=0, shrinkB=4)
                                if abs(ly[j] - y[oi]) > 0.02 else None)
        if col == 0:
            ax.axhline(VIFT_PAPER_NET, color=MUTED, lw=1, ls=(0, (4, 3)))
            ax.annotate("VIFT paper, reported (its own protocol)", (2.4, VIFT_PAPER_NET), xytext=(0, -10),
                        textcoords="offset points", ha="right", fontsize=7, color=MUTED)
        ax.set_xticks(range(3), [g[0] for g in groups], color=INK2, fontsize=8)
        ax.set_xlim(-1.2, 2.5)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[0].set_ylabel("t_rel % (KITTI 05/07/10, OXTS)")
    fig.text(0.01, 0.01, "Bars: group mean. Each VIFT point is one seed (42, 2, 3, 4, 5, 6), same code and data.",
             fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    fig.savefig(Path(out_png).with_suffix(".pdf"), facecolor=SURFACE)
    print(f"\nfigure: {out_png}")


if __name__ == "__main__":
    main()
