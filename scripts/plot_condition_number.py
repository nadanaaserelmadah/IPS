"""Draw what a condition number means with two 2x2 systems (docs/figures/condition_number.svg and .png).

Each equation is a line in the (x, y) plane and the solution is where the two lines cross.
  well conditioned : x + y = 2 and x - y = 0       (lines at right angles, cond = 1)
  badly conditioned: x + y = 2 and x + 1.05 y = 2.05 (almost parallel lines, cond ~ 84)
The right-hand side of the second equation is then changed by 0.2 and the solution is recomputed.
The numbers are computed here, not typed in.

  python scripts/plot_condition_number.py
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1940, 1180
INK = "#1C2430"
parts = []


def fmt(s):
    s = re.sub(r" {2,}", lambda m: " " * len(m.group()), s)
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def label(x, y, s, size=17, color=INK, anchor="start", bold=False):
    wt = ' font-weight="700"' if bold else ""
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{wt}>{fmt(s)}</text>')


def line(x1, y1, x2, y2, color="#888", width=1.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{d}/>')


def box(x, y, w, h, fill, stroke, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>')


def solve(A, b):
    return np.linalg.solve(np.array(A, float), np.array(b, float))


class Plot:
    """A square plot area with data limits; maps (x, y) data to pixels."""

    def __init__(self, px, py, size, xlim, ylim):
        self.px, self.py, self.size, self.xlim, self.ylim = px, py, size, xlim, ylim

    def pt(self, x, y):
        fx = (x - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        fy = (y - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        return self.px + fx * self.size, self.py + self.size - fy * self.size

    def frame(self):
        box(self.px, self.py, self.size, self.size, "#FFFFFF", "#C9C4B6")
        for v in range(int(np.ceil(self.xlim[0])), int(np.floor(self.xlim[1])) + 1):
            x0, y0 = self.pt(v, self.ylim[0])
            line(x0, y0, x0, self.py, "#ECE9E0", 1)
            label(x0, y0 + 20, str(v), 14, "#777", "middle")
        for v in range(int(np.ceil(self.ylim[0])), int(np.floor(self.ylim[1])) + 1):
            x0, y0 = self.pt(self.xlim[0], v)
            line(x0, y0, self.px + self.size, y0, "#ECE9E0", 1)
            label(x0 - 8, y0 + 5, str(v), 14, "#777", "end")

    def eq_line(self, a, b, c, color, width=3, dash=""):
        """Draw a x + b y = c across the plot."""
        pts = []
        for xv in self.xlim:
            if abs(b) > 1e-12:
                pts.append((xv, (c - a * xv) / b))
        for yv in self.ylim:
            if abs(a) > 1e-12:
                pts.append(((c - b * yv) / a, yv))
        pts = [p for p in pts if self.xlim[0] - 1e-9 <= p[0] <= self.xlim[1] + 1e-9 and self.ylim[0] - 1e-9 <= p[1] <= self.ylim[1] + 1e-9]
        pts = sorted(set((round(p[0], 9), round(p[1], 9)) for p in pts))
        (xa, ya), (xb, yb) = self.pt(*pts[0]), self.pt(*pts[-1])
        line(xa, ya, xb, yb, color, width, dash)

    def dot(self, x, y, color, hollow=False, r=9):
        cx, cy = self.pt(x, y)
        fill = "#FFFFFF" if hollow else color
        parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{r}" fill="{fill}" stroke="{color}" stroke-width="3.5"/>')


# systems
A_good = [[1, 1], [1, -1]]
A_bad = [[1, 1], [1, 1.05]]
b_good, b_bad = [2, 0], [2, 2.05]
db = np.array([0.0, 0.2])
cond_good, cond_bad = np.linalg.cond(A_good), np.linalg.cond(A_bad)
x0g, x1g = solve(A_good, b_good), solve(A_good, np.array(b_good) + db)
x0b, x1b = solve(A_bad, b_bad), solve(A_bad, np.array(b_bad) + db)
move_g, move_b = float(np.linalg.norm(x1g - x0g)), float(np.linalg.norm(x1b - x0b))

label(40, 40, "What a condition number means: two lines and where they cross", 24, "#555", bold=True)
label(40, 70, "Each equation is a line. The solution is the crossing point. Now change the data of the second equation by 0.2 and see how far the crossing moves.", 17, INK)

SZ = 520
# left plot
L = Plot(120, 130, SZ, (-0.5, 2.5), (-0.5, 2.5))
label(120 + SZ / 2, 112, f"Well conditioned: cond = {cond_good:.0f}", 21, "#0F766E", "middle", bold=True)
L.frame()
L.eq_line(1, 1, 2, "#3B6BA5")
L.eq_line(1, -1, 0, "#0F766E")
L.eq_line(1, -1, 0.2, "#0F766E", 3, "9 6")
L.dot(*x0g, "#1C2430")
L.dot(*x1g, "#B45309", hollow=True)
a1 = L.pt(*x0g)
label(a1[0] + 18, a1[1] - 14, f"({x0g[0]:.1f}, {x0g[1]:.1f})", 16, INK, bold=True)
b1 = L.pt(*x1g)
label(b1[0] + 18, b1[1] + 26, f"({x1g[0]:.1f}, {x1g[1]:.1f})", 16, "#B45309", bold=True)
label(L.px, L.py + SZ + 48, "x + y = 2", 16, "#3B6BA5", bold=True)
label(L.px + 150, L.py + SZ + 48, "x − y = 0   (dashed: x − y = 0.2)", 16, "#0F766E", bold=True)

# right plot
R = Plot(1000, 130, SZ, (-4, 3), (-1, 6))
label(1000 + SZ / 2, 112, f"Badly conditioned: cond ≈ {cond_bad:.0f}", 21, "#B45309", "middle", bold=True)
R.frame()
R.eq_line(1, 1, 2, "#3B6BA5")
R.eq_line(1, 1.05, 2.05, "#B45309")
R.eq_line(1, 1.05, 2.25, "#B45309", 3, "9 6")
R.dot(*x0b, "#1C2430")
R.dot(*x1b, "#B45309", hollow=True)
a2, b2 = R.pt(*x0b), R.pt(*x1b)
parts.append(f'<line x1="{a2[0]:.1f}" y1="{a2[1]:.1f}" x2="{b2[0]:.1f}" y2="{b2[1]:.1f}" stroke="#B42318" stroke-width="2.5" stroke-dasharray="3 5"/>')
label(a2[0] - 20, a2[1] + 38, f"({x0b[0]:.0f}, {x0b[1]:.0f})", 16, INK, "end", bold=True)
label(b2[0] + 16, b2[1] - 10, f"({x1b[0]:.0f}, {x1b[1]:.0f})", 16, "#B45309", bold=True)
label(R.px, R.py + SZ + 48, "x + y = 2", 16, "#3B6BA5", bold=True)
label(R.px + 150, R.py + SZ + 48, "x + 1.05 y = 2.05   (dashed: = 2.25)", 16, "#B45309", bold=True)

# captions under the plots
box(120, 745, SZ, 120, "#E1F1E4", "#3B7A4A")
label(120 + SZ / 2, 779, "Lines cross at a wide angle", 18, INK, "middle", bold=True)
label(120 + SZ / 2, 807, f"data moved 0.20  →  answer moved {move_g:.2f}", 18, INK, "middle")
label(120 + SZ / 2, 835, "A small error stays a small error: the answer can be trusted", 16, "#33414F", "middle")
box(1000, 745, SZ, 120, "#FBE8D3", "#B45309")
label(1000 + SZ / 2, 779, "Lines are almost parallel", 18, INK, "middle", bold=True)
label(1000 + SZ / 2, 807, f"data moved 0.20  →  answer moved {move_b:.1f}", 18, INK, "middle")
label(1000 + SZ / 2, 835, f"The error is multiplied by about {move_b / 0.2:.0f}: the answer cannot be trusted", 16, "#33414F", "middle")

# bottom: connection to the length problem
label(40, 917, "In our length problem", 22, "#555", bold=True)
box(40, 935, 900, 170, "#E1F1E4", "#3B7A4A")
label(490, 969, "The car accelerates and turns", 19, INK, "middle", bold=True)
label(490, 999, "the window's equations differ from each other (wide angle)", 17, "#33414F", "middle")
label(490, 1027, "the length is separated from v₀ and gravity", 17, "#33414F", "middle")
label(490, 1065, "small condition number: use the accelerometer length", 17, "#0F766E", "middle", bold=True)
box(1000, 935, 900, 170, "#FBE8D3", "#B45309")
label(1450, 969, "The car keeps a constant speed on a straight line", 19, INK, "middle", bold=True)
label(1450, 999, "the equations are almost the same one (almost parallel)", 17, "#33414F", "middle")
label(1450, 1027, "length and initial speed cannot be told apart", 17, "#33414F", "middle")
label(1450, 1065, "condition number ≥ 10⁴: keep the network length", 17, "#B45309", "middle", bold=True)
label(40, 1145, "Condition number = largest ÷ smallest singular value of the matrix (np.linalg.cond). The 2×2 numbers are computed in scripts/plot_condition_number.py.", 15, "#555")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "condition_number.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'condition_number.png'}", (OUT / "condition_number.svg").as_uri()],
               check=True, capture_output=True)
print(f"cond good {cond_good:.2f} bad {cond_bad:.1f}; moves {move_g:.3f} vs {move_b:.3f}")
print(OUT / "condition_number.png")
