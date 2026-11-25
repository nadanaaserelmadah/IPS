"""Shared Scientific Reports figure style (vector PDF, Arial 7 pt, 1.0 pt minimum line width, no boxing).

Imported by fig_heading_error.py and fig_measure_vs_learn.py. Styling only: no numbers are computed here.
"""
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, NullFormatter

matplotlib.use("Agg")

FONT_PT = 7
WIDTH_IN = 5.1  # included at \textwidth = 5.17 in, so 7 pt stays 7 pt
BLUE, ORANGE, BLACK, GREY = "#1f4e79", "#b5651d", "#000000", "#7f7f7f"
BAND = dict(color=GREY, alpha=0.25, lw=0)  # median / IQR band: light grey, never colour
LW = 1.2  # data lines; axes, ticks and reference lines use 1.0
DASH = (0, (5, 2))
DOT = (0, (1, 1.5))
ARIAL = "/System/Library/Fonts/Supplemental/Arial.ttf"


def apply():
    if Path(ARIAL).is_file():
        font_manager.fontManager.addfont(ARIAL)
    matplotlib.rcParams.update({
        "font.family": ["Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": FONT_PT, "axes.labelsize": FONT_PT, "axes.titlesize": FONT_PT, "axes.titleweight": "normal",
        "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.grid": False, "legend.frameon": False,
        "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
        "xtick.direction": "out", "ytick.direction": "out",
        "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "text.color": BLACK,
        "xtick.color": BLACK, "ytick.color": BLACK,
        "lines.linewidth": LW, "lines.markeredgewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300,
    })


def clean(ax):
    """No boxing, no grid, no panel background."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.0)
    ax.grid(False)
    ax.set_facecolor("none")


def panel_letter(ax, letter, dx_pt, dy_pt=3):
    """Bold lowercase letter, top-left, outside the axes (offset in points from the axes' top-left corner)."""
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction", xytext=(dx_pt, dy_pt), textcoords="offset points",
                ha="left", va="bottom", fontsize=FONT_PT, fontweight="bold", annotation_clip=False)


def num(v):
    """Plain number: comma thousands separators only from 5 digits, true minus sign."""
    s = f"{int(round(v)):,}" if abs(v) >= 1e4 else f"{v:g}"
    return s.replace("-", "−")


PLAIN = FuncFormatter(lambda v, _: num(v))


def plain_axis(axis):
    axis.set_major_formatter(PLAIN)
    axis.set_minor_formatter(NullFormatter())


def log_ticks(ax, extra=()):
    """Tidy decade ticks on a log x axis within the current limits (1 per decade for wide ranges)."""
    lo, hi = ax.get_xlim()
    dec = np.log10(hi / lo)
    mult = (1,) if dec >= 3 else (1, 3) if dec >= 1.5 else (1, 2, 5)
    ticks = [m * 10.0 ** k for k in range(int(np.floor(np.log10(lo))) - 1, int(np.ceil(np.log10(hi))) + 2) for m in mult
             if lo * 0.999 <= m * 10.0 ** k <= hi * 1.001]
    ticks = sorted(set(ticks) | {float(t) for t in extra})
    ax.set_xticks(ticks)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    plain_axis(ax.xaxis)
