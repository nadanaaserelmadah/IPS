"""Draw what the KITTI lag is and how it is fitted (docs/figures/lag_calibration.svg and .png).

Facts drawn here come from docs/RESULTS.md and scripts/kitti/gyro_hybrid.py: the integration window of each image pair is
[t_img[k] + lag, t_img[k+1] + lag]; the lag is the grid-searched shift that makes the integrated gyro rotation match the
ground-truth rotation of the pair (training drives only); the documented effect is on the two-view direction error.

  python scripts/plot_lag_calibration.py
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1940, 1140
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


def box(x, y, w, h, lines, kind, size=18, bold_first=True, dashed=False):
    fill, stroke = FILLS[kind]
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
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


# 1. one image pair on three clocks (1 ms = 8 px)
label(40, 38, "1. One image pair on three clocks (drawn to scale, 1 ms = 8 px)", 20, "#555", bold=True)
x0, px, lag = 260, 8, 60
t1 = x0 + 104 * px
s0, s1 = x0 + lag * px, t1 + lag * px
for yy, name in ((110, "images"), (185, "ground truth"), (250, "IMU / OXTS")):
    label(40, yy + 6, name, 18, INK, bold=True)
    if name.startswith("IMU"):
        label(40, yy + 28, "raw, own timestamps", 14, "#555")
parts.append(f'<rect x="{x0 - 3}" y="97" width="6" height="26" fill="{INK}"/><rect x="{t1 - 3}" y="97" width="6" height="26" fill="{INK}"/>')
label(x0 - 8, 88, "τ_{k}", 18)
label(t1 - 14, 88, "τ_{k+1}", 18)
arrow([(x0 + 10, 140), (t1 - 10, 140)], both=True)
label((x0 + t1) / 2, 134, "≈ 104 ms (image timestamps)", 16, INK, "middle")
box(x0, 165, t1 - x0, 40, ["rotation of the pair in the ground truth: what the gyro has to reproduce"], "phys", size=16, bold_first=False)
line(x0 - 90, 250, s1 + 110, 250, "#555", 1.5)
for j in range(-1, 18):
    line(x0 + 80 * j, 240, x0 + 80 * j, 260, "#555", 1.5)
label(s1 + 125, 256, "100 Hz samples", 16, "#555")
box(x0, 280, t1 - x0, 40, ["lag = 0: window [τ_{k}, τ_{k+1}]"], "frozen", size=16, bold_first=False, dashed=True)
box(s0, 335, t1 - x0, 40, ["lag = +60 ms: window [τ_{k} + lag, τ_{k+1} + lag]"], "out", size=16, bold_first=False)
line(x0, 205, x0, 400, "#999", 1.5, "4 4")
line(t1, 205, t1, 320, "#999", 1.5, "4 4")
line(s0, 320, s0, 400, "#999", 1.5, "4 4")
arrow([(x0 + 4, 395), (s0 - 4, 395)], both=True)
label((x0 + s0) / 2, 420, "lag", 18, INK, "middle", bold=True)
label(s0 + 20, 424, "constant for a recording day: +60 ms (2011-09-30), +62.5 ms (2011-10-03)", 16, "#555")
label(40, 462, "The raw samples are never changed: each keeps its own timestamp. The lag only moves which samples are integrated for each image pair.", 17, "#B45309", bold=True)
label(40, 488, "The rotation that matches the pair shows up 60 ms later in the IMU stream. Whose clock is off (image, ground truth or IMU) is not determined.", 17, "#B45309", bold=True)

# 2. how the lag is fitted
label(40, 540, "2. How the lag is fitted (training drives 04, 06, 08, 09 only)", 20, "#555", bold=True)
xs = [40, 420, 800, 1180, 1560]
box(xs[0], 558, 340, 120, ["Candidate lags", "−80 … +80 ms, step 10 ms,", "then ±10 ms, step 2.5 ms"], "add")
box(xs[1], 558, 340, 120, ["Integrate the gyro", "over [τ_{k} + lag, τ_{k+1} + lag]", "with the true time steps"], "add")
box(xs[2], 558, 340, 120, ["Align to the GT rotation", "Kabsch alignment of the", "IMU-to-camera rotation"], "add")
box(xs[3], 558, 340, 120, ["Median rotation residual", "one number for each", "candidate lag"], "add")
box(xs[4], 558, 340, 120, ["Keep the lowest", "+60 ms (OXTS), +65 ms (SuMa)", "+62.5 ms on 2011-10-03"], "out")
for i in range(4):
    arrow([(xs[i] + 340, 618), (xs[i + 1], 618)])
label(40, 708, "the residual-versus-lag curve itself was not saved in the notes, only the chosen values", 16, "#555")

# 3. documented effect
label(40, 740, "3. Documented effect on the two-view direction error (KITTI, median, degrees)", 20, "#555", bold=True)
base, scale = 1005, 85
bars = [(200, 1.43, "#8A8880"), (310, 0.63, "#0F766E"), (520, 2.21, "#8A8880"), (630, 0.92, "#0F766E")]
line(150, base, 760, base, "#555", 1.5)
for bx, v, col in bars:
    h = v * scale
    parts.append(f'<rect x="{bx}" y="{base - h:.1f}" width="90" height="{h:.1f}" rx="4" fill="{col}"/>')
    label(bx + 45, base - h - 10, f"{v:.2f}°", 18, INK, "middle", bold=True)
label(255, base + 28, "train 04 / 06 / 09", 17, INK, "middle")
label(575, base + 28, "test 05 / 07 / 10", 17, INK, "middle")
parts.append(f'<rect x="150" y="{base + 50}" width="20" height="20" rx="3" fill="#8A8880"/>')
label(178, base + 66, "lag 0 (image timestamps)", 16)
parts.append(f'<rect x="450" y="{base + 50}" width="20" height="20" rx="3" fill="#0F766E"/>')
label(478, base + 66, "lag +60 ms (chosen on the training drives)", 16)
box(850, 770, 1010, 250, ["How to read it", "a positive lag moves the IMU window later", "the same offset was needed by the ground truth (+60 ms) and by the lidar poses (+65 ms),",
                          "which suggests the image timestamps, but this is not proven", "origin unresolved: ground-truth timing or lidar-sweep timing"], "add", size=18)
label(40, 1118, "Numbers: docs/RESULTS.md (Table 7l and the calibration notes). Code: scripts/kitti/gyro_hybrid.py, window start = t_img + lag.", 16, "#555")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
       '<path d="M1 1L9 5L1 9z" fill="#444"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "lag_calibration.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'lag_calibration.png'}", (OUT / "lag_calibration.svg").as_uri()],
               check=True, capture_output=True)
print(OUT / "lag_calibration.png")
