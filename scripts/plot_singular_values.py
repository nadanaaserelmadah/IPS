"""Draw what the largest and smallest singular values are: a unit circle of inputs becomes an ellipse (docs/figures/singular_values.svg and .png).

For a 2x2 matrix A, every input vector of length 1 is sent to A @ v. The set of outputs is an ellipse; its long half-axis
is the largest singular value and its short half-axis is the smallest. The condition number is the ratio of the two.
All numbers come from numpy here.

  python scripts/plot_singular_values.py
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1500, 860
INK = "#1C2430"
PX = 70  # pixels per unit
parts = []


def fmt(s):
    s = re.sub(r" {2,}", lambda m: " " * len(m.group()), s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label(x, y, s, size=17, color=INK, anchor="start", bold=False):
    wt = ' font-weight="700"' if bold else ""
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{wt}>{fmt(s)}</text>')


def arrow(x1, y1, x2, y2, color, width=4):
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" marker-end="url(#ah_{color[1:]})"/>')


def curve(cx, cy, pts, color, width=3, fill="none"):
    p = " ".join(f"{cx + x * PX:.1f},{cy - y * PX:.1f}" for x, y in pts)
    parts.append(f'<polygon points="{p}" fill="{fill}" fill-opacity="0.12" stroke="{color}" stroke-width="{width}"/>')


def box(x, y, w, h, fill, stroke):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


RED, BLUE, GREY = "#B42318", "#3B6BA5", "#8A8880"
t = np.linspace(0, 2 * np.pi, 361)
circle = np.stack([np.cos(t), np.sin(t)], 1)

label(40, 40, "Largest and smallest singular value: a circle of inputs becomes an ellipse", 24, "#555", bold=True)
label(40, 70, "Take every input vector of length 1 (the circle). The matrix turns each one into an output vector. The outputs fill an ellipse.", 17, INK)

rows = [
    ("Matrix A = [[1, 1], [1, −1]]", np.array([[1, 1], [1, -1]], float), 230, "#E1F1E4", "#3B7A4A"),
    ("Matrix A = [[1, 1], [1, 1.05]]", np.array([[1, 1], [1, 1.05]], float), 590, "#FBE8D3", "#B45309"),
]
for title, A, cy, bg, edge in rows:
    U, S, Vt = np.linalg.svd(A)
    box(30, cy - 150, W - 60, 300, bg, edge)
    label(60, cy - 118, title, 20, INK, bold=True)
    cx_in, cx_out = 300, 800
    # input circle with the two special directions
    curve(cx_in, cy + 20, circle, GREY, 3, "#8A8880")
    label(cx_in, cy + 138, "inputs: all vectors of length 1", 16, "#555", "middle")
    for k, col in ((0, RED), (1, BLUE)):
        v = Vt[k]
        arrow(cx_in, cy + 20, cx_in + v[0] * PX, cy + 20 - v[1] * PX, col)
    for k, col in ((0, RED), (1, BLUE)):
        v = Vt[k]
        label(cx_in + v[0] * (PX + 14), cy + 20 - v[1] * (PX + 14) + 5, f"direction {k + 1}", 15, col, "start" if v[0] >= 0 else "end", bold=True)
    # arrow between
    parts.append(f'<line x1="{cx_in + 170}" y1="{cy + 20}" x2="{cx_out - 170}" y2="{cy + 20}" stroke="#444" stroke-width="3" marker-end="url(#ah_444444)"/>')
    label((cx_in + cx_out) / 2, cy - 2, "multiply by A", 18, INK, "middle", bold=True)
    # output ellipse
    out = circle @ A.T
    curve(cx_out, cy + 20, out, edge, 3, edge)
    label(cx_out, cy + 138, "outputs: an ellipse", 16, "#555", "middle")
    for k, col in ((0, RED), (1, BLUE)):
        u = U[:, k] * S[k]
        arrow(cx_out, cy + 20, cx_out + u[0] * PX, cy + 20 - u[1] * PX, col)
    # result text
    tx = 1040
    label(tx, cy - 40, f"largest singular value  =  {S[0]:.3g}", 19, RED, bold=True)
    label(tx, cy - 14, "(longest output: direction 1)", 15, "#555")
    label(tx, cy + 28, f"smallest singular value  =  {S[1]:.3g}", 19, BLUE, bold=True)
    label(tx, cy + 54, "(shortest output: direction 2)", 15, "#555")
    label(tx, cy + 102, f"condition number  =  {S[0]:.3g} ÷ {S[1]:.3g}  =  {S[0] / S[1]:.0f}", 19, INK, bold=True)

label(40, 800, "Round ellipse: every direction is treated alike, the equations can tell the unknowns apart.", 17, "#3B7A4A", bold=True)
label(40, 826, "Almost a line: direction 2 is squashed to nearly nothing, so changing the unknowns along it hardly changes the equations. That is the part we cannot determine.", 17, "#B45309", bold=True)

defs = "".join(f'<marker id="ah_{c[1:]}" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M1 1L9 5L1 9z" fill="{c}"/></marker>'
               for c in (RED, BLUE, "#444444"))
svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}"><defs>{defs}</defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "singular_values.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'singular_values.png'}", (OUT / "singular_values.svg").as_uri()],
               check=True, capture_output=True)
print(OUT / "singular_values.png")
