"""Draw how the correct gyro integration of one image pair works (docs/figures/gyro_integration.svg and .png).

Everything drawn follows scripts/kitti/gyro_hybrid.py integrate() and gyro_rotation_test.py so3_exp():
window [t_img[k] + lag, t_img[k+1] + lag], raw samples inside it plus the two interpolated end values,
trapezoid mean of neighbouring angular velocities times the real time gap, exponential map, ordered product,
then the camera frame and the 50 ms gap rule. The numbers in the examples are illustrative.

  python scripts/plot_gyro_integration.py
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1940, 1150
INK = "#1C2430"
FILLS = {"in": ("#E6EEF8", "#3B6BA5"), "frozen": ("#ECEBE6", "#8A8880"), "add": ("#FBE8D3", "#B45309"),
         "phys": ("#DDF0EC", "#0F766E"), "out": ("#E1F1E4", "#3B7A4A"), "tok": ("#FFFFFF", "#555555")}
parts = []


def fmt(s):
    s = re.sub(r" {2,}", lambda m: " " * len(m.group()), s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\^\{([^}]*)\}", r'<tspan baseline-shift="super" font-size="70%">\1</tspan>', s)
    s = re.sub(r"_\{([^}]*)\}", r'<tspan baseline-shift="sub" font-size="70%">\1</tspan>', s)
    return s


def visible(s):
    return len(re.sub(r"[\^_{}]", "", s))


def box(x, y, w, h, lines, kind, size=18, bold_first=True):
    fill, stroke = FILLS[kind]
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    lh = size * 1.5
    y0 = y + (h - len(lines) * lh) / 2 + lh * 0.72
    for i, ln in enumerate(lines):
        if visible(ln) * size * 0.54 > w - 20:
            print(f"warning: line may not fit ({w}px): {ln!r}")
        wt = "700" if (bold_first and i == 0) else "400"
        parts.append(f'<text x="{x + w / 2}" y="{y0 + i * lh:.1f}" text-anchor="middle" font-size="{size}" font-weight="{wt}">{fmt(ln)}</text>')


def arrow(pts, both=False, color="#444"):
    p = " ".join(f"{x},{y}" for x, y in pts)
    start = ' marker-start="url(#ah)"' if both else ""
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#ah)"{start}/>')


def line(x1, y1, x2, y2, color="#888", width=1.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>')


def label(x, y, s, size=17, color=INK, anchor="start", bold=False):
    wt = ' font-weight="700"' if bold else ""
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{wt}>{fmt(s)}</text>')


# 1. window and samples (1 ms = 8 px)
label(40, 38, "1. The window and the raw samples inside it (drawn to scale, 1 ms = 8 px)", 20, "#555", bold=True)
x0 = 260
a, b = x0 + 30, x0 + 30 + 104 * 8


def wy(x):
    return 135 - 28 * math.sin((x - 180) / 190)


pts = " ".join(f"{x},{wy(x):.1f}" for x in range(180, 1301, 5))
parts.append(f'<polyline points="{pts}" fill="none" stroke="#9AA3AD" stroke-width="2"/>')
label(40, 140, "ω (one axis)", 18, INK, bold=True)
label(40, 226, "raw samples", 18, INK, bold=True)
line(180, 215, 1300, 215, "#555", 1.5)
for j in range(-1, 14):
    xx = x0 + 80 * j
    line(xx, 207, xx, 223, "#555", 1.5)
for j in range(1, 11):
    parts.append(f'<circle cx="{x0 + 80 * j}" cy="{wy(x0 + 80 * j):.1f}" r="6.5" fill="#0F766E"/>')
for j in (0, 11):
    parts.append(f'<circle cx="{x0 + 80 * j}" cy="{wy(x0 + 80 * j):.1f}" r="4.5" fill="#9AA3AD"/>')
ya = wy(x0) + (wy(x0 + 80) - wy(x0)) * 30 / 80
xb0 = x0 + 800
yb = wy(xb0) + (wy(xb0 + 80) - wy(xb0)) * (b - xb0) / 80
line(x0, wy(x0), x0 + 80, wy(x0 + 80), "#B45309", 1.5, "4 3")
line(xb0, wy(xb0), xb0 + 80, wy(xb0 + 80), "#B45309", 1.5, "4 3")
for xx, yy in ((a, ya), (b, yb)):
    parts.append(f'<circle cx="{xx}" cy="{yy:.1f}" r="7.5" fill="#FFFFFF" stroke="#B45309" stroke-width="3"/>')
line(a, 90, a, 250, "#999", 1.5, "4 4")
line(b, 90, b, 250, "#999", 1.5, "4 4")
label(a, 80, "t_{a} = τ_{k} + lag", 17, INK, "middle", bold=True)
label(b, 80, "t_{b} = τ_{k+1} + lag", 17, INK, "middle", bold=True)
for (xs, xe, txt, xc) in ((a + 4, x0 + 76, "first: 6.3 ms", (a + x0 + 80) / 2), (x0 + 480 + 4, x0 + 560 - 4, "typical: 10 ms", x0 + 520), (xb0 + 4, b - 4, "last: 7.8 ms", (xb0 + b) / 2)):
    arrow([(xs, 262), (xe, 262)], both=True)
    label(xc, 288, txt, 15, "#555", "middle")
label(40, 330, "The window ends fall between samples, so the two end values (hollow circles) are interpolated linearly from the neighbours.", 17)
label(40, 354, "Every Δt is the real gap between neighbouring samples (the first and last are shorter), not a fixed 10 ms.", 17)

# 2. one step
label(40, 400, "2. One step between two neighbouring samples", 20, "#555", bold=True)
px0, pxe, pbase, ws = 110, 490, 640, 90
parts.append(f'<polygon points="{px0},{pbase} {px0},{pbase - ws} {pxe},{pbase - 2 * ws} {pxe},{pbase}" fill="#0F766E" fill-opacity="0.22" stroke="#0F766E" stroke-width="2.5"/>')
parts.append(f'<rect x="{px0}" y="{pbase - ws}" width="{pxe - px0}" height="{ws}" fill="none" stroke="#8A8880" stroke-width="2" stroke-dasharray="7 5"/>')
line(90, pbase, 520, pbase, "#555", 1.5)
line(px0, pbase + 6, px0, 450, "#555", 1.5)
label(px0 - 12, pbase - ws + 6, "ω_{i}", 17, INK, "end", bold=True)
label(pxe + 14, pbase - 2 * ws + 6, "ω_{i+1}", 17, INK, "start", bold=True)
label((px0 + pxe) / 2, pbase + 26, "Δt_{i}", 17, INK, "middle")
label(60, 700, "trapezoid: ½ (ω_{i} + ω_{i+1}) · Δt_{i} = 0.0150 rad", 17, "#0F766E", bold=True)
label(60, 724, "Euler (dashed): ω_{i} · Δt_{i} = 0.0100 rad", 17, "#555")
label(60, 748, "example: ω from 1.0 to 2.0 rad/s in 10 ms (illustrative)", 15, "#555")
box(600, 430, 350, 120, ["Mean angular velocity", "ω_{avg} = ½ (ω_{i} + ω_{i+1})", "for each of the 3 axes"], "add")
box(990, 430, 350, 120, ["Rotation vector", "φ = ω_{avg} · Δt_{i}", "axis ω_{avg}/|ω_{avg}|, angle θ = |φ|"], "add")
box(1380, 430, 480, 120, ["Exponential map (Rodrigues)", "ΔR_{i} = I + sin θ · K + (1 − cos θ) · K²", "K: skew-symmetric matrix of the axis"], "add")
arrow([(950, 490), (990, 490)])
arrow([(1340, 490), (1380, 490)])
box(600, 600, 1260, 100, ["Each step is a small rotation (about 0.3° at 0.5 rad/s over 10 ms)", "the subtlety is how the steps are combined: see step 3"], "in", size=17, bold_first=False)

# 3. compose
label(40, 790, "3. Compose the steps in time order (the product is not commutative)", 20, "#555", bold=True)
names = ["ΔR_{1}", "ΔR_{2}", "ΔR_{3}", "…", "ΔR_{n}"]
for i, nm in enumerate(names):
    bx = 60 + i * 215
    box(bx, 815, 150, 58, [nm], "phys", size=20, bold_first=False)
    if i < len(names) - 1:
        label(bx + 182, 852, "·", 28, INK, "middle", bold=True)
label(60 + 4 * 215 + 150 + 40, 852, "=", 28, INK, "middle", bold=True)
box(1150, 815, 220, 58, ["R_{imu}"], "out", size=20, bold_first=False)
label(1420, 838, "rotations about different axes do not commute:", 16, "#555")
label(1420, 860, "the order in time cannot be changed", 16, "#555")

# 4. camera frame and gap rule
label(40, 925, "4. To the camera frame, and when the gyroscope is not used", 20, "#555", bold=True)
box(60, 945, 880, 100, ["Camera frame", "R_{cam} = R_{ci} · R_{imu} · R_{ci}^{T}", "R_{ci} fitted by Kabsch on the training drives (KITTI: bias b_{g} = 0)"], "add")
box(980, 945, 880, 100, ["Gap rule", "largest spacing between raw samples in the window > 50 ms", "the edge keeps the network rotation (KITTI OXTS outages)"], "add")
box(60, 1060, 1800, 62, ["Why the true Δt matters (illustrative, ω_{z} = 0.5 rad/s):  104 ms gives 0.0520 rad = 2.98°,  10 × 10 ms gives 0.0500 rad = 2.865°:  0.115° lost per frame (about 4%),  3.6° lost over a 90° turn"], "out", size=17, bold_first=False)
label(40, 1142, "Code: scripts/kitti/gyro_hybrid.py integrate(), gyro_rotation_test.py so3_exp(), kabsch()", 15, "#555")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M1 1L9 5L1 9z" fill="#444"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "gyro_integration.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'gyro_integration.png'}", (OUT / "gyro_integration.svg").as_uri()],
               check=True, capture_output=True)
print(OUT / "gyro_integration.png")
