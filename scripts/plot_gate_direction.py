"""Draw the direction rule of the observability gate (docs/figures/gate_direction.svg and .png).

Rule (scripts/gate/gate_v3_select.py fuse, scripts/gate/observability_gate.py direction_gate):
  sigma_g^2 = sigma_geo^2 + (A / parallax)^2              A = 6.755 deg*px
  consistent  : angle(d_net, d_geo) <= 3 * sqrt(sigma_g^2 + sigma_net^2)   -> inverse-variance fusion
  inconsistent: geometry available -> d_geo ; no geometry -> d_net
Fusion weight of the network: w_net = sigma_g^2 / (sigma_g^2 + sigma_net^2).
sigma_net = 0.829 deg (v3, KITTI 08); v4 replaces it by an online estimate (a distrusted network, sigma_hat = 10 deg, is shown too).
sigma_geo = 0.3 deg is ILLUSTRATIVE (the solver gives it per edge); everything else is the code's constant.

  python scripts/plot_gate_direction.py
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1500, 1000
INK = "#1C2430"
A_PX, K, SIG_GEO, SIG_NET, SIG_BAD = 6.755, 3.0, 0.3, 0.829, 10.0
parts = []


def fmt(s):
    s = re.sub(r" {2,}", lambda m: " " * len(m.group()), s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"_\{([^}]*)\}", r'<tspan baseline-shift="sub" font-size="70%">\1</tspan>', s)
    return re.sub(r"\^\{([^}]*)\}", r'<tspan baseline-shift="super" font-size="70%">\1</tspan>', s)


def label(x, y, s, size=17, color=INK, anchor="start", bold=False, rot=None):
    wt = ' font-weight="700"' if bold else ""
    tr = f' transform="rotate({rot} {x} {y})"' if rot is not None else ""
    parts.append(f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{wt}{tr}>{fmt(s)}</text>')


def box(x, y, w, h, fill, stroke, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>')


class Axes:
    def __init__(self, x, y, w, h, xlim, ylim):
        self.x, self.y, self.w, self.h, self.xlim, self.ylim = x, y, w, h, xlim, ylim

    def p(self, vx, vy):
        return (self.x + (vx - self.xlim[0]) / (self.xlim[1] - self.xlim[0]) * self.w,
                self.y + self.h - (vy - self.ylim[0]) / (self.ylim[1] - self.ylim[0]) * self.h)

    def frame(self, xticks, yticks, xlabel, ylabel, ytick_fmt="{:g}"):
        box(self.x, self.y, self.w, self.h, "#FFFFFF", "#C9C4B6")
        for v in xticks:
            px, py = self.p(v, self.ylim[0])
            parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{px:.1f}" y2="{self.y}" stroke="#ECE9E0" stroke-width="1"/>')
            label(px, py + 22, f"{v:g}", 14, "#777", "middle")
        for v in yticks:
            px, py = self.p(self.xlim[0], v)
            parts.append(f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{self.x + self.w}" y2="{py:.1f}" stroke="#ECE9E0" stroke-width="1"/>')
            label(px - 8, py + 5, ytick_fmt.format(v), 14, "#777", "end")
        label(self.x + self.w / 2, self.y + self.h + 50, xlabel, 16, INK, "middle", bold=True)
        label(self.x - 52, self.y + self.h / 2, ylabel, 16, INK, "middle", bold=True, rot=-90)

    def curve(self, xs, ys, color, width=3.5, dash=""):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        pts = " ".join(f"{self.p(a, b)[0]:.1f},{self.p(a, b)[1]:.1f}" for a, b in zip(xs, ys))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}"{d}/>')


def sigma_g(p, sg=SIG_GEO):
    return np.sqrt(sg ** 2 + (A_PX / p) ** 2)


label(40, 40, "The direction rule of the gate: fuse when the two directions agree, otherwise use the geometry", 24, "#555", bold=True)
label(40, 68, "Per edge: θ = angle between the network direction and the geometric direction. σ_geo = 0.3° is illustrative; A = 6.755°·px, K = 3 and σ_net = 0.829° are the code's constants.", 16, INK)

ps = np.linspace(1.5, 30, 200)
# Panel A: threshold vs parallax
AX = Axes(110, 140, 560, 380, (0, 30), (0, 20))
label(AX.x + AX.w / 2, 124, "A.  Which rule applies (σ_net = 0.829°)", 19, "#0F766E", "middle", bold=True)
AX.frame([0, 5, 10, 15, 20, 25, 30], [0, 5, 10, 15, 20], "parallax p (pixels, after removing the rotation)", "angle θ between d_net and d_geo (degrees)")
T = K * np.sqrt(sigma_g(ps) ** 2 + SIG_NET ** 2)
# shaded fuse region
poly = [AX.p(ps[0], 0)] + [AX.p(a, min(b, 20)) for a, b in zip(ps, T)] + [AX.p(ps[-1], 0)]
parts.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in poly) + '" fill="#0F766E" fill-opacity="0.15"/>')
poly2 = [AX.p(ps[0], 20)] + [AX.p(a, min(b, 20)) for a, b in zip(ps, T)] + [AX.p(ps[-1], 20)]
parts.append('<polygon points="' + " ".join(f"{a:.1f},{b:.1f}" for a, b in poly2) + '" fill="#B45309" fill-opacity="0.12"/>')
AX.curve(ps, np.minimum(T, 20), "#1C2430", 3.5)
label(AX.p(23, 1.0)[0], AX.p(23, 1.0)[1], "FUSE (inverse-variance)", 17, "#0F766E", "middle", bold=True)
label(AX.p(15, 15.5)[0], AX.p(15, 15.5)[1], "USE THE GEOMETRY", 17, "#B45309", "middle", bold=True)
label(AX.p(15, 3.2)[0] + 20, AX.p(15, 3.2)[1] - 30, "threshold 3·√(σ_g² + σ_net²)", 15, INK, "start")
for th, col, txt in ((1.5, "#0F766E", "θ = 1.5° → fuse"), (8, "#B45309", "θ = 8° → geometry")):
    cx, cy = AX.p(8, th)
    parts.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="8" fill="{col}" stroke="#fff" stroke-width="2"/>')
    label(cx + 14, cy + 5, txt, 15, col, "start", bold=True)
label(AX.p(1, 18.3)[0] + 8, AX.p(1, 18.3)[1], "small parallax: the geometry is unreliable,", 14, "#555")
label(AX.p(1, 17.2)[0] + 8, AX.p(1, 17.2)[1], "so the threshold is wide", 14, "#555")

# Panel B: network weight in fusion
BX = Axes(850, 140, 560, 380, (0, 30), (0, 100))
label(BX.x + BX.w / 2, 124, "B.  How much the network counts in the fusion", 19, "#0F766E", "middle", bold=True)
BX.frame([0, 5, 10, 15, 20, 25, 30], [0, 25, 50, 75, 100], "parallax p (pixels)", "network weight in the fused direction (%)")
for sn, col, dash, txt in ((SIG_NET, "#0F766E", "", "trusted network, σ_net = 0.83° (v3)"), (SIG_BAD, "#B42318", "9 6", "distrusted network, σ_net = 10° (v4)")):
    wn = sigma_g(ps) ** 2 / (sigma_g(ps) ** 2 + sn ** 2) * 100
    BX.curve(ps, wn, col, 3.5, dash)
label(BX.p(14, 62)[0], BX.p(14, 62)[1], "σ_net = 0.83°", 16, "#0F766E", "start", bold=True)
label(BX.p(14, 62)[0], BX.p(14, 62)[1] + 20, "88% at p = 3 px → 17% at p = 30 px", 14, "#0F766E", "start")
label(BX.p(8, 12)[0], BX.p(8, 12)[1], "distrusted network (10°): under 2% for p ≥ 5 px", 15, "#B42318", "start", bold=True)
label(BX.p(9, 94)[0], BX.p(9, 94)[1], "more parallax → the geometry is trusted more", 14, "#555")

# Panel C: decision order
label(40, 640, "C.  The decision, edge by edge", 22, "#555", bold=True)
cards = [
    ("1. No geometry on this edge", "(too few matches)", "→ use the network direction", "#ECEBE6", "#8A8880"),
    ("2. θ ≤ 3·√(σ_g² + σ_net²)", "the two agree", "→ fuse: d ∝ d_net/σ_net² + d_geo/σ_g²", "#E1F1E4", "#3B7A4A"),
    ("3. θ > 3·√(σ_g² + σ_net²)", "they disagree", "→ use the geometry (the network is the suspect)", "#FBE8D3", "#B45309"),
]
for i, (t1, t2, t3, f, s) in enumerate(cards):
    x = 40 + i * 480
    box(x, 665, 450, 130, f, s)
    label(x + 225, 702, t1, 18, INK, "middle", bold=True)
    label(x + 225, 730, t2, 16, "#555", "middle")
    label(x + 225, 765, t3, 16, INK, "middle", bold=True)
box(40, 815, 1420, 150, "#FFFFFF", "#C9C4B6")
label(60, 848, "Why the geometry wins when they disagree: it has an explicit error model per edge (σ_geo from the solver plus A/p from the parallax), while the network has one global σ_net.", 16, INK)
label(60, 876, "The network can be off by tens of degrees in a new domain (30–90° on EuRoC), the geometry by 1–2°. On KITTI the network is the more accurate one (0.3–0.7° against 0.8–1.0°),", 16, INK)
label(60, 904, "so this is a design choice, not a law. v4 estimates σ_net online from the last 100 confident edges; above 10° the network is distrusted (panel B, dashed), and edges", 16, INK)
label(60, 932, "without geometry hold the last gated direction.", 16, INK)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "gate_direction.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'gate_direction.png'}", (OUT / "gate_direction.svg").as_uri()],
               check=True, capture_output=True)
w0, w1 = [sigma_g(np.array([p])) ** 2 / (sigma_g(np.array([p])) ** 2 + SIG_NET ** 2) * 100 for p in (3.0, 30.0)]
print(f"network weight at p=3: {w0[0]:.0f}%  at p=30: {w1[0]:.0f}%  thr(8px)={K*np.sqrt(sigma_g(8)**2+SIG_NET**2):.2f} deg")
print(OUT / "gate_direction.png")
