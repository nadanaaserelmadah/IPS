"""Draw the PI-VIFT architecture (docs/figures/pivift_architecture.svg and .png).

Everything drawn here is read from third_party/VIFT/src/models/components/pi_vift.py and the PIViftLoss config:
two extra tokens (preintegration, state), an 18-number residual head, the physics step with the carried state.

  python scripts/plot_pivift_architecture.py
"""
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures"
W, H = 1940, 1110
INK = "#1C2430"
FILLS = {  # fill, stroke
    "in": ("#E6EEF8", "#3B6BA5"), "frozen": ("#ECEBE6", "#8A8880"), "add": ("#FBE8D3", "#B45309"),
    "phys": ("#DDF0EC", "#0F766E"), "out": ("#E1F1E4", "#3B7A4A"), "tok": ("#FFFFFF", "#555555"),
}
parts = []


def fmt(s):
    s = re.sub(r" {2,}", lambda m: "\u00a0" * len(m.group()), s)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\^\{([^}]*)\}", r'<tspan baseline-shift="super" font-size="70%">\1</tspan>', s)
    s = re.sub(r"_\{([^}]*)\}", r'<tspan baseline-shift="sub" font-size="70%">\1</tspan>', s)
    s = re.sub(r"\^(\w)", r'<tspan baseline-shift="super" font-size="70%">\1</tspan>', s)
    s = re.sub(r"_(\w)", r'<tspan baseline-shift="sub" font-size="70%">\1</tspan>', s)
    return s


def visible(s):
    return len(re.sub(r"[\^_{}]", "", s))


def box(x, y, w, h, lines, kind, size=17, bold_first=True, left=False, dashed=False):
    fill, stroke = FILLS[kind]
    dash = ' stroke-dasharray="7 5"' if dashed else ""
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="2"{dash}/>')
    lh = size * 1.5
    y0 = y + (h - len(lines) * lh) / 2 + lh * 0.72
    for i, ln in enumerate(lines):
        if visible(ln) * size * 0.54 > w - 20:
            print(f"warning: line may not fit ({w}px): {ln!r}")
        weight = "700" if (bold_first and i == 0) else "400"
        anchor, tx = ("start", x + 24) if left else ("middle", x + w / 2)
        parts.append(f'<text x="{tx}" y="{y0 + i * lh:.1f}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}">{fmt(ln)}</text>')


def arrow(pts, dashed=False):
    p = " ".join(f"{x},{y}" for x, y in pts)
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    parts.append(f'<polyline points="{p}" fill="none" stroke="#444" stroke-width="2.2" marker-end="url(#ah)"{dash}/>')


def label(x, y, s, size=15, color="#555", anchor="start", bold=False):
    wt = ' font-weight="700"' if bold else ""
    parts.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" fill="{color}"{wt}>{fmt(s)}</text>')


# inputs
box(30, 50, 190, 90, ["Frame pair", "I_{k−1}, I_k", "256 × 512"], "in")
box(30, 190, 190, 90, ["IMU window", "11 samples × 6", "(acc, gyro)"], "in")
box(30, 560, 190, 130, ["Carried state", "v, g, b_g, b_a", "(12 numbers, from", "the previous window)"], "add")
# encoders, preintegration, state MLP
box(270, 50, 270, 90, ["Visual encoder", "FlowNet-style CNN,", "9 conv layers (frozen)"], "frozen")
box(270, 190, 270, 90, ["Inertial encoder", "3 × 1-D conv,", "linear projection (frozen)"], "frozen")
box(270, 330, 270, 160, ["Preintegration", "IMU window, dt = 0.01 s", "bias-corrected (b_g, b_a)", "ΔR (6-D), Δv, Δp, dt", "= 13 numbers"], "add")
box(270, 565, 270, 120, ["State MLP", "(v, g, b_g, b_a) = 12", "12 → 64 → 128"], "add")
# tokens
box(590, 70, 200, 50, ["visual · 512"], "tok", bold_first=False)
box(590, 210, 200, 50, ["inertial · 256"], "tok", bold_first=False)
box(590, 372, 200, 76, ["preintegration · 128", "(MLP 13 → 64 → 128)"], "tok", bold_first=False)
box(590, 587, 200, 76, ["state · 128", "(same for all 10 edges)"], "tok", bold_first=False)
# concat, transformer, head, residuals
box(850, 50, 130, 670, ["Token", "per edge", "512 + 256", "+ 128 + 128", "= 1024"], "tok", size=18)
box(1040, 250, 290, 260, ["Causal transformer", "window of 10 edges", "linear 1024 → 768", "+ sinusoidal positions", "4 layers · 6 heads", "causal mask"], "frozen", size=18)
box(1390, 310, 150, 140, ["MLP head", "768 → 768 → 18"], "add", size=17)
box(1580, 255, 320, 250, ["18 residual corrections", "δθ   rotation (3)", "δp   position (3)", "δv   velocity (3)", "δg   gravity (3)", "δb_g   gyro bias (3)", "δb_a   accel bias (3)"], "add", size=17)
# physics and output
box(270, 770, 1230, 250, [
    "Physics step for each edge k (all vectors in camera frame k)",
    "R_{out} = ΔR_k · exp(δθ)          Δp_{out} = v_k·dt + ½·g_k·dt² + Δp_k + δp",
    "v_{k+1} = R_{out}^T (v_k + g_k·dt + Δv_k + δv)          g_{k+1} = norm(R_{out}^T (g_k + 0.05·δg)) · 9.81",
    "b_g ← b_g + 10^{-3}·tanh(δb_g)          b_a ← b_a + 10^{-2}·tanh(δb_a)",
    "ΔR_k, Δv_k, Δp_k: preintegration of the IMU window with the current biases",
    "k = 0 starts from the carried state (first window: v from a small head, g from the mean IMU acceleration, biases 0)",
], "phys", size=18, left=True)
box(1560, 770, 340, 250, ["Pose per edge", "[θ_k, t_k]", "the gate uses t_k", "(direction and length)", "θ_k is discarded", "(rotation from the gyroscope)"], "out", size=17)

# arrows
arrow([(220, 95), (270, 95)])
arrow([(220, 235), (270, 235)])
arrow([(100, 280), (100, 395), (270, 395)])
arrow([(140, 560), (140, 445), (270, 445)])
label(150, 525, "b_g, b_a")
arrow([(220, 625), (270, 625)])
for yy in (95, 235, 410, 625):
    arrow([(540, yy), (590, yy)])
    arrow([(790, yy), (850, yy)])
arrow([(980, 380), (1040, 380)])
arrow([(1330, 380), (1390, 380)])
arrow([(1540, 380), (1580, 380)])
arrow([(1740, 505), (1740, 640), (1400, 640), (1400, 770)])
arrow([(1500, 895), (1560, 895)])
arrow([(270, 910), (125, 910), (125, 690)])
label(140, 745, "carry to next window")
arrow([(520, 490), (520, 527), (565, 527), (565, 770)], dashed=True)
label(575, 748, "ΔR_k, Δv_k, Δp_k")

# legend and note
lx = 270
for kind, text in (("in", "input"), ("frozen", "VIFT component"), ("add", "added in PI-VIFT"), ("phys", "physics step"), ("out", "output")):
    fill, stroke = FILLS[kind]
    parts.append(f'<rect x="{lx}" y="1040" width="22" height="22" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    label(lx + 32, 1057, text, size=17, color=INK)
    lx += 250
label(270, 1095, "Training loss: multi-horizon pose loss + 0.1 × velocity loss + 0.05 × gravity-direction loss (both against ground truth)", size=16)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Arial, Helvetica, sans-serif" fill="{INK}">'
       '<defs><marker id="ah" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
       '<path d="M1 1L9 5L1 9z" fill="#444"/></marker></defs>'
       f'<rect width="{W}" height="{H}" fill="#FFFFFF"/>' + "".join(parts) + "</svg>")
(OUT / "pivift_architecture.svg").write_text(svg)
subprocess.run([CHROME, "--headless", "--disable-gpu", "--hide-scrollbars", "--force-device-scale-factor=2",
                f"--window-size={W},{H}", f"--screenshot={OUT / 'pivift_architecture.png'}", (OUT / "pivift_architecture.svg").as_uri()],
               check=True, capture_output=True)
print(OUT / "pivift_architecture.png")
