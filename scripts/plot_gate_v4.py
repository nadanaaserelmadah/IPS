"""Draw why gate v3 failed on TUM-VI and what v4 changes, with arrows (docs/figures/gate_v3_v4.svg and .png).

One toy trajectory of 12 edges. Every arrow is the direction of motion of one edge (up = true heading).
The network is wrong by about 40 degrees (as on TUM-VI); two-view geometry is right where it exists, but 3 edges have none.
  v3: edge with geometry -> geometry (the two disagree, so the geometry wins); edge without geometry -> the network direction.
  v4: edge without geometry -> hold the last gated direction once the network is judged unreliable (sigma_hat > 10 deg).
The numbers below are illustrative (a toy), the rule is the code's (scripts/gate/gate_v3_select.py, gate_v4.py).

  python scripts/plot_gate_v4.py
"""
import math
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1500, 1000
INK = "#1C2430"
GREEN, ORANGE, TEAL, RED, BLUE, GREY = "#3B7A4A", "#E0703E", "#0F766E", "#C0392B", "#3B6BA5", "#8A8880"
parts = []


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=20, color=INK, anchor="middle", bold=False):
    wt = ' font-weight="700"' if bold else ""
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" direction="rtl" font-size="{size}" fill="{color}"{wt}>{esc(s)}</text>')


def box(x, y, w, h, fill, stroke, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" stroke="{stroke}" stroke-width="2"{d}/>')


def arrow(cx, cy, ang, color, length=64, width=6):
    """Arrow centred on (cx, cy) pointing `ang` degrees clockwise from up."""
    a = math.radians(ang)
    dx, dy = math.sin(a) * length / 2, -math.cos(a) * length / 2
    x1, y1, x2, y2 = cx - dx, cy - dy, cx + dx, cy + dy
    parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')
    h = 15
    for s in (-1, 1):
        b = a + math.pi + s * 0.5
        parts.append(f'<line x1="{x2:.1f}" y1="{y2:.1f}" x2="{x2 + math.sin(b) * h:.1f}" y2="{y2 - math.cos(b) * h:.1f}" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>')


truth = [0, 0, 5, 10, 10, 15, 15, 10, 5, 0, 0, -5]
net_off = [38, -42, 51, 35, -47, 44, -39, 52, -45, 36, 41, -40]
geo_noise = [1, -2, 1, 0, 2, -1, 1, -2, 1, 0, 2, -1]
missing = {4, 5, 9}  # edges with no two-view geometry
net = [t + o for t, o in zip(truth, net_off)]
geo = [t + n for t, n in zip(truth, geo_noise)]

v3, v4 = [], []
for i in range(12):
    v3.append(net[i] if i in missing else geo[i])
    v4.append(v4[-1] if (i in missing and v4) else geo[i])
err3 = [abs(a - b) for a, b in zip(v3, truth)]
err4 = [abs(a - b) for a, b in zip(v4, truth)]

text(750, 52, "نفس الحالة في v3 وv4: الشبكة غلطانة، وبعض الحواف مفيهاش هندسة", 32, INK, bold=True)
text(750, 88, "كل سهم هو اتجاه الحركة في حافة واحدة (السهم لفوق = الاتجاه الصح). الحواف الـ 12 جنب بعض من الشمال لليمين.", 19, "#555")

rows = [
    ("الاتجاه الحقيقي", GREEN, truth, "#E1F1E4", GREEN),
    ("الشبكة (غلطانة حوالي 40°)", ORANGE, net, "#FBE8D3", ORANGE),
    ("الهندسة (موجودة إلا في 3 حواف)", TEAL, geo, "#DDF0EC", TEAL),
    ("v3: القرار النهائي", BLUE, v3, "#E6EEF8", BLUE),
    ("v4: القرار النهائي", BLUE, v4, "#E6EEF8", BLUE),
]
ys = [190, 320, 450, 600, 740]
xs = [380 + i * 95 for i in range(12)]
for (name, col, arr, bg, edge), y in zip(rows, ys):
    box(40, y - 55, 1430, 110, bg, edge)
    text(190, y + 8, name, 22, INK, bold=True)
    for i, x in enumerate(xs):
        if name.startswith("الهندسة") and i in missing:
            box(x - 36, y - 38, 72, 76, "#FFFFFF", GREY, "6 5")
            text(x, y + 8, "لا يوجد", 17, GREY)
            continue
        c = col
        if name.startswith("v3") or name.startswith("v4"):
            e = (err3 if name.startswith("v3") else err4)[i]
            c = RED if e > 10 else BLUE
        arrow(x, y, arr[i], c)
        if name.startswith("v3") or name.startswith("v4"):
            e = (err3 if name.startswith("v3") else err4)[i]
            if e > 10:
                text(x, y + 50, f"خطأ {e:.0f}°", 16, RED, bold=True)

# the three edges without geometry, called out
for i in sorted(missing):
    parts.append(f'<rect x="{xs[i] - 44}" y="{ys[2] - 62}" width="88" height="{ys[4] - ys[2] + 124}" rx="14" fill="none" stroke="{RED}" stroke-width="3" stroke-dasharray="9 6"/>')
text(750, 831, "الحواف المحاطة بخط أحمر مقطّع هي اللي مفيهاش هندسة: هنا الفرق بين v3 وv4", 21, RED, bold=True)

# the trust meter
box(40, 840, 1430, 135, "#FFFFFF", "#C9C4B6")
text(190, 885, "ثقتنا في الشبكة (σ̂)", 22, INK, bold=True)
x0, x1 = 380, 1420
parts.append(f'<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="{GREEN}"/><stop offset="0.35" stop-color="#E5B73C"/><stop offset="1" stop-color="{RED}"/></linearGradient></defs>')
parts.append(f'<rect x="{x0}" y="880" width="{x1 - x0}" height="26" rx="13" fill="url(#g)" opacity="0.85"/>')
sx = lambda d: x0 + (x1 - x0) * d / 60
parts.append(f'<line x1="{sx(10)}" y1="868" x2="{sx(10)}" y2="920" stroke="{INK}" stroke-width="4"/>')
text(sx(10), 940, "العتبة 10°", 19, INK, bold=True)
parts.append(f'<circle cx="{sx(40)}" cy="893" r="17" fill="#FFFFFF" stroke="{INK}" stroke-width="5"/>')
text(sx(40), 940, "الشبكة بعيدة حوالي 40° عن الهندسة", 19, RED, bold=True)
text(sx(0), 960, "0°", 17, "#555")
text(sx(60), 960, "60°", 17, "#555")
text(sx(25), 960, "σ̂ = وسيط الزاوية بين الشبكة والهندسة على الحواف اللي الهندسة فيها موثوقة", 17, "#555")

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "gate_v3_v4.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'gate_v3_v4.png'}", (OUT / "gate_v3_v4.svg").as_uri()],
               check=True, capture_output=True)
print(f"mean error v3 {sum(err3) / 12:.1f} deg, v4 {sum(err4) / 12:.1f} deg")
print(OUT / "gate_v3_v4.png")
