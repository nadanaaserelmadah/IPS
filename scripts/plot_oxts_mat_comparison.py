"""Draw how the VS-VIO imus/*.mat files were identified from the raw OXTS stream (docs/figures/oxts_vs_mat.svg and .png).

The method is the one in scripts/kitti/oxts_vs_mat.py: rebuild the .mat by interpolating the raw 100 Hz OXTS at candidate
query times (two hypotheses, offsets -100..+100 ms) and keep the candidate with the lowest gyroscope RMSE.

  python scripts/plot_oxts_mat_comparison.py
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1940, 1080
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


def arrow(pts, dashed=False, color="#444"):
    p = " ".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    parts.append(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="2.2" marker-end="url(#ah)"{dash}/>')


def line(x1, y1, x2, y2, color="#888", width=1.5, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"{d}/>')


def label(x, y, s, size=17, color=INK, anchor="start", bold=False):
    wt = ' font-weight="700"' if bold else ""
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{wt}>{fmt(s)}</text>')


# 1. inputs
label(40, 38, "1. Inputs", 20, "#555", bold=True)
box(40, 55, 430, 110, ["Raw OXTS, 100 Hz", "oxts/data/*.txt: ax ay az wx wy wz ...", "t_ox: one timestamp per sample"], "in")
box(510, 55, 430, 110, ["Image timestamps", "sync/image_02/timestamps.txt", "t_img: about 104 ms apart"], "in")
box(980, 55, 430, 110, ["VS-VIO  imus/*.mat", "imu_data_interp", "10 n + 1 rows × 6 columns"], "frozen")
label(40, 190, "packed into one npz per drive by oxts_pack.py", 16, "#555")

# 2. query-time hypotheses for one image interval, drawn to scale (1 ms = 12 px)
label(40, 245, "2. Two hypotheses for the query times inside one image interval (drawn to scale, 1 ms = 11 px)", 20, "#555", bold=True)
x0, px = 260, 11
tau1 = x0 + 104 * px
for yy, name in ((315, "raw OXTS"), (385, "image times"), (480, "per-frame"), (570, "0.01 s")):
    label(40, yy + 6, name, 18, INK, bold=True)
line(x0 - 40, 315, tau1 + 120, 315, "#555", 1.5)
for j in range(12):
    line(x0 + 10 * px * j, 305, x0 + 10 * px * j, 325, "#555", 1.5)
label(tau1 + 120, 321, "100 Hz samples, 10 ms apart", 16, "#555")
line(x0, 402, x0, 600, "#999", 1.5, "4 4")
line(tau1, 402, tau1, 600, "#999", 1.5, "4 4")
line(x0, 385, x0, 385, INK, 3)
parts.append(f'<rect x="{x0 - 3}" y="372" width="6" height="26" fill="{INK}"/><rect x="{tau1 - 3}" y="372" width="6" height="26" fill="{INK}"/>')
label(x0 - 8, 360, "τ_{k}", 18, INK)
label(tau1 - 12, 360, "τ_{k+1}", 18, INK)
arrow([(x0 + 8, 420), (tau1 - 8, 420)])
arrow([(tau1 - 8, 420), (x0 + 8, 420)])
label((x0 + tau1) / 2, 412, "true interval ≈ 104 ms (image timestamps)", 16, INK, "middle")
for j in range(11):  # A: ten equal parts of the true interval
    parts.append(f'<circle cx="{x0 + 10.4 * px * j:.1f}" cy="480" r="7" fill="#0F766E"/>')
label(tau1 + 90, 486, "10 equal parts of the interval", 16, "#0F766E", bold=True)
label(tau1 + 90, 508, "(≈ 10.4 ms): matches the .mat", 16, "#0F766E", bold=True)
for j in range(11):  # B: fixed 10 ms steps
    parts.append(f'<circle cx="{x0 + 10 * px * j:.1f}" cy="570" r="7" fill="#B45309"/>')
line(x0 + 100 * px, 570, tau1, 570, "#B42318", 4)
label(tau1 + 90, 576, "fixed 10 ms steps: ends", 16, "#B45309", bold=True)
label(tau1 + 90, 598, "≈ 4% short of the next image", 16, "#B45309", bold=True)

# 3. comparison
label(40, 655, "3. Comparison (scripts/kitti/oxts_vs_mat.py)", 20, "#555", bold=True)
box(40, 675, 400, 110, ["For each hypothesis", "and each time offset", "−100 … +100 ms, step 5 ms"], "add")
box(490, 675, 440, 110, ["Interpolate the raw OXTS", "np.interp(tq + offset, t_ox, ox[:, c])", "six columns: ax ay az wx wy wz"], "add")
box(980, 675, 400, 110, ["RMSE against the .mat", "per column (3 accel., 3 gyro)", "over the whole drive"], "add")
box(1430, 675, 430, 110, ["Keep the lowest", "gyroscope RMSE", "(over hypothesis and offset)"], "add")
for xa, xb in ((440, 490), (930, 980), (1380, 1430)):
    arrow([(xa, 730), (xb, 730)])

# result and consequence
box(40, 840, 900, 150, ["Consequence", "the code read those parts as 10 ms steps,", "so every integrated rotation was ≈ 4% too small"], "add")
box(980, 840, 880, 150, ["Result on drives 04–10", "best: per-frame, offset 0 ms, gyro RMSE ≤ 3.5 × 10^{-4} rad/s", "so the .mat is the raw OXTS on 10 equal parts of each real image interval"], "out")
arrow([(1650, 785), (1650, 840)])
arrow([(980, 915), (940, 915)])
label(40, 1030, "The 104 ms itself comes from np.diff(t_img) in oxts_pack.py, not from this comparison. The comparison shows how the .mat was built.", 16, "#555")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       '<path d="M1 1L9 5L1 9z" fill="#444"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "oxts_vs_mat.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'oxts_vs_mat.png'}", (OUT / "oxts_vs_mat.svg").as_uri()],
               check=True, capture_output=True)
print(OUT / "oxts_vs_mat.png")
