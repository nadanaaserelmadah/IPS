// English PowerPoint for the current paper (paper/submission/main.tex, 8 pages) -> docs/presentations/paper_en.pptx
//   NODE_PATH=SOME_DIR/node_modules node scripts/build_paper_deck_en.js
// Same slides and numbers as build_paper_deck_ar.js. Layout coordinates are written as in the Arabic (right-to-left) deck and
// mirrored by the helpers, so card i of a row lands at the left. Slide 15 is outside the paper and is marked as such.
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = process.env.PPTX_SKILL_DIR || "/Users/finetch/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/79318aaa-643a-485b-90dd-4a26d2773cce/b0b5a169-e2bd-4421-8d36-b221b96b46a7/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const OUT = path.join(__dirname, "../docs/presentations/paper_en.pptx");
const THEME = {
  name: "IPS Observability",
  headFontFace: "Arial",
  bodyFontFace: "Arial",
  colors: { dk1: "17252A", lt1: "FFFFFF", dk2: "0E3B43", lt2: "EEF3F2", accent1: "0E7C86", accent2: "E8A33D", accent3: "C8553D",
            accent4: "5B8C5A", accent5: "7A8B99", accent6: "3F5E8A", hlink: "0E7C86", folHlink: "7A8B99" },
};
const HEX = { teal: "0E7C86", amber: "E8A33D", red: "C8553D", green: "5B8C5A", gray: "9AA8AD", ink: "17252A", dark: "0E3B43" };

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.theme = { headFontFace: "Arial", bodyFontFace: "Arial" };
pres.title = "Measure What Is Observable, Learn the Rest";
pres.author = "IPS";
const C = pres.SchemeColor;
const W = 13.33, M = 0.6;
const mx = (x, w) => W - x - w; // mirror a box written in right-to-left coordinates

pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  slideNumber: { x: 12.2, y: 6.95, w: 0.7, h: 0.3, fontSize: 11, color: C.accent5, fontFace: "Arial", align: "right" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 1.0, fontSize: 28, bold: true, color: C.text2, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 1.9, w: W - 2 * M, h: 1.7, fontSize: 38, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({
  title: "CLOSING", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.5, w: W - 2 * M, h: 1.0, fontSize: 40, bold: true, color: C.background1, align: "left", valign: "middle", margin: 0 }, text: "" } }],
});
["Introduction", "Problem", "Diagnosis", "Method", "Results", "Conclusion"].forEach((t) => pres.addSection({ title: t }));

// ---------- helpers (all take right-to-left coordinates and mirror them) ----------
const base = { isTextBox: true, fontFace: "Arial", margin: 0, valign: "top", align: "left" };
function T(s, text, x, y, w, h, o = {}) { s.addText(text, Object.assign({ x: mx(x, w), y, w, h, fontSize: 16, color: C.text1 }, base, o)); }
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.ShapeType.roundRect, Object.assign({ x: mx(x, w), y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2, width: 0 } }, o));
}
function circle(s, x, y, d, fill, label, size = 20, color = C.background1) {
  s.addShape(pres.ShapeType.ellipse, { x: mx(x, d), y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(label, { x: mx(x, d), y, w: d, h: d, align: "center", valign: "middle", fontSize: size, bold: true, color, fontFace: "Arial", margin: 0, isTextBox: true });
}
function arrow(s, x, y, w, color = HEX.ink) { // rightward arrow
  s.addShape(pres.ShapeType.line, { x: mx(x, w), y, w, h: 0, line: { color, width: 2, endArrowType: "triangle" } });
}
function box(s, x, y, w, h, o) { s.addShape(pres.ShapeType.roundRect, Object.assign({ x: mx(x, w), y, w, h }, o)); }
function newSlide(title, sec, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec });
  s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function barOpts(extra) {
  return Object.assign({
    showTitle: false, showLegend: false, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", valAxisLabelFormatCode: "0.0", dataLabelFontSize: 14, dataLabelColor: HEX.ink,
    dataLabelFontFace: "+mn-lt", catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", catAxisLabelFontSize: 13, valAxisLabelFontSize: 12,
    catAxisLabelColor: HEX.ink, valAxisLabelColor: "5A6B70", valGridLine: { color: "D5DDDB", size: 0.5 }, catGridLine: { style: "none" }, barGapWidthPct: 45,
  }, extra);
}
function chart(s, type, data, o) { s.addChart(type, data, Object.assign({}, o, { x: mx(o.x, o.w) })); }

// ---------- 1. title ----------
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "Introduction" });
  s.addText("Measure What Is Observable, Learn the Rest", { placeholder: "title" });
  T(s, "Observability-gated fusion for learned visual–inertial odometry", M, 3.65, W - 2 * M, 0.6, { fontSize: 24, color: C.accent2, bold: true });
  T(s, "Nada El Madah · Magdy Zakaria · Samir Elmougy · Shahenda Sarhan · Mohamed Handosa", M, 4.6, W - 2 * M, 0.4, { fontSize: 14, color: C.background2 });
  circle(s, 11.55, 0.55, 0.95, C.accent1, "R", 30); circle(s, 10.55, 0.55, 0.95, C.accent2, "ℓ", 30, C.text1); circle(s, 9.55, 0.55, 0.95, C.accent3, "d", 30);
  T(s, "rotation · length · direction", 8.0, 1.62, 4.5, 0.4, { fontSize: 14, color: C.background2 });
  s.addNotes("The sensors already measure much of the motion, so we do not ask the network to learn it.");
}

// ---------- 2. the three components ----------
{
  const s = newSlide("Motion has three components, and each has a sensor", "Problem",
    "Visual–inertial odometry (VIO) estimates camera motion from images and an inertial sensor (IMU). The gyroscope measures rotation, the accelerometer can measure step length, and two camera views give the direction of travel. Learned VIO asks one neural network to predict all three at once.");
  const cw = 3.85, gap = 0.29, y = 1.65;
  const items = [["R", "Rotation", "Gyroscope", "Measures angular velocity directly", C.accent3], ["ℓ", "Step length", "Accelerometer", "Estimated from the acceleration over a time window", C.accent2], ["d", "Direction of travel", "Two camera views", "Two-view geometry, once rotation is known", C.accent1]];
  items.forEach((it, i) => {
    const x = W - M - cw - i * (cw + gap);
    card(s, x, y, cw, 3.3);
    circle(s, x + cw - 1.1, y + 0.3, 0.8, it[4], it[0], 28, it[4] === C.accent2 ? C.text1 : C.background1);
    T(s, it[1], x + 0.25, y + 1.35, cw - 0.5, 0.5, { fontSize: 24, bold: true, color: C.text2 });
    T(s, it[2], x + 0.25, y + 1.95, cw - 0.5, 0.45, { fontSize: 18, bold: true, color: C.accent1 });
    T(s, it[3], x + 0.25, y + 2.45, cw - 0.5, 0.7, { fontSize: 15 });
  });
  card(s, M, 5.3, W - 2 * M, 1.2, { fill: { color: C.text2 }, line: { color: C.text2, width: 0 } });
  T(s, "Learned VIO asks one neural network to predict all three at once. This is where it fails.", M + 0.4, 5.3, W - 2 * M - 0.8, 1.2, { fontSize: 22, bold: true, color: C.background1, valign: "middle" });
}

// ---------- 3. the question ----------
{
  const s = newSlide("Which part of the pose is wrong?", "Diagnosis",
    "Instead of asking how to improve the whole network, we ask which component is wrong. We replace one component by ground truth and re-score, without retraining. On KITTI, true rotation with the network's own translation gives 1.55%, better than the 1.96% reported for VIFT.");
  const bw = 3.6, y = 1.8;
  const steps = [["1", "Take a trained network", "VIFT on KITTI, a base-style network on EuRoC"], ["2", "Swap one component for ground truth", "Rotation, direction or length"], ["3", "Re-score", "No retraining"]];
  steps.forEach((st, i) => {
    const x = W - M - bw - i * (bw + 0.62);
    card(s, x, y, bw, 2.2);
    circle(s, x + bw - 0.85, y + 0.25, 0.6, C.accent1, st[0], 20);
    T(s, st[1], x + 0.2, y + 1.0, bw - 0.4, 0.5, { fontSize: 18, bold: true, color: C.text2 });
    T(s, st[2], x + 0.2, y + 1.55, bw - 0.4, 0.5, { fontSize: 14 });
    if (i < 2) arrow(s, x - 0.55, y + 1.1, 0.48);
  });
  card(s, M, 4.5, W - 2 * M, 2.0, { fill: { color: C.text2 }, line: { color: C.text2, width: 0 } });
  T(s, "1.55%", W - M - 3.2, 4.65, 2.8, 1.1, { fontSize: 60, bold: true, color: C.accent2, align: "center" });
  T(s, "KITTI: true rotation + network translation", W - M - 3.4, 5.75, 3.2, 0.6, { fontSize: 13, color: C.background2, align: "center" });
  T(s, "Better than the 1.96% that VIFT reports: rotation is the main problem on KITTI. On EuRoC the problem is elsewhere.", M + 0.4, 4.7, 7.5, 1.6, { fontSize: 20, color: C.background1, valign: "middle" });
}

// ---------- 4. KITTI diagnosis chart ----------
{
  const s = newSlide("On KITTI, swapping the rotation for the gyroscope changes everything", "Diagnosis",
    "t_rel in percent, mean of drives 05, 07 and 10. The gyroscope integrated with nominal 10 ms steps gives 4.62% because the true image interval is about 104 ms. Integrated over the true interval it gives 1.70% with r_rel 0.18, better than the published VIFT on both metrics.");
  const labels = ["Network as is", "True rotation + network translation", "Gyro, wrong nominal steps", "Gyro over the true interval", "VIFT, published"];
  chart(s, pres.charts.BAR, [{ name: "t_rel %", labels, values: [2.11, 1.55, 4.62, 1.70, 1.96] }], barOpts({ x: 4.3, y: 1.6, w: 8.4, h: 5.2, barDir: "bar", chartColors: [HEX.gray, HEX.green, HEX.red, HEX.teal, HEX.amber], valAxisMaxVal: 5.5, valAxisMinVal: 0, catAxisOrientation: "maxMin" }));
  card(s, M, 1.65, 3.45, 4.9);
  T(s, "What to read", M + 0.2, 1.85, 3.05, 0.5, { fontSize: 20, bold: true, color: C.text2 });
  T(s, [{ text: "Rotation error r_rel: ", options: { bold: true } }, { text: "0.67 to 0.18 as soon as the gyroscope is used.", options: { breakLine: true } },
        { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "Note: ", options: { bold: true } }, { text: "the gyroscope agrees with OXTS partly by construction, so we also measured against SuMa: 0.32." }],
    M + 0.2, 2.5, 3.05, 3.8, { fontSize: 15 });
}

// ---------- 5. timing finding ----------
{
  const s = newSlide("Timing finding: the true image interval is about 104 ms, not 100", "Diagnosis",
    "The inertial files distributed with released models hold 11 samples per image pair, read as ten 10 ms steps. They are the raw 100 Hz stream interpolated over the true image interval (103.7–104.1 ms), so every rotation is about 4% too small and the gyroscope looks biased when it is not. The +60 ms offset is fitted on training drives 04, 06, 08 and 09 only; its physical origin is unresolved.");
  const cw = 3.85, gap = 0.29, y = 1.65;
  const st = [["≈ 104 ms", "True interval between two images: 103.7 to 104.1 ms on drives 04 to 10", C.accent1], ["4%", "Rotation lost in every frame when 10 ms steps are assumed", C.accent3], ["+60 ms", "Constant offset between image timestamps and ground truth, fitted on training drives only", C.accent2]];
  st.forEach((it, i) => {
    const x = W - M - cw - i * (cw + gap);
    card(s, x, y, cw, 3.1);
    T(s, it[0], x + 0.2, y + 0.3, cw - 0.4, 1.0, { fontSize: 44, bold: true, color: it[2] === C.accent2 ? C.accent3 : it[2], align: "center" });
    T(s, it[1], x + 0.3, y + 1.5, cw - 0.6, 1.4, { fontSize: 15 });
  });
  card(s, M, 5.15, W - 2 * M, 1.4, { fill: { color: C.text2 }, line: { color: C.text2, width: 0 } });
  T(s, "After the fix: from 4.62% to 1.70%", W - M - 0.4 - 5.8, 5.15, 5.8, 1.4, { fontSize: 26, bold: true, color: C.accent2, valign: "middle" });
  T(s, "The networks never see the step, so only methods that integrate these files inherit the error.", M + 0.2, 5.15, 5.4, 1.4, { fontSize: 16, color: C.background1, valign: "middle" });
}

// ---------- 6. EuRoC / TUM-VI ----------
{
  const s = newSlide("On EuRoC and TUM-VI the problem is the direction of motion", "Diagnosis",
    "The zero-motion floor is the error of assuming the device never moved (3.15 m on average). The learned networks are worse than that. Swapping the rotation for ground truth does not help, while swapping the translation brings the error below the floor. The network's direction error grows with distance from the training rooms until it approaches a random direction.");
  const cw = 5.95, ch = 2.3, gx = 0.25, gy = 0.25, x0 = W - M - cw, y0 = 1.6;
  const st = [["3.15 m", "Zero-motion floor: the error of assuming no motion", C.accent5], ["4.02–4.98 m", "Learned networks: worse than the floor", C.accent3],
              ["0.6–2.1 m", "True translation + network rotation: below the floor", C.accent4], ["31–90°", "Network direction error on EuRoC, close to a random direction; 40–87° on TUM-VI", C.accent1]];
  st.forEach((it, i) => {
    const x = x0 - (i % 2) * (cw + gx), y = y0 + Math.floor(i / 2) * (ch + gy);
    card(s, x, y, cw, ch);
    T(s, it[0], x + 0.3, y + 0.25, cw - 0.6, 0.9, { fontSize: 38, bold: true, color: it[2] });
    T(s, it[1], x + 0.3, y + 1.3, cw - 0.6, 0.9, { fontSize: 16 });
  });
  T(s, "True rotation does not help there; the direction is the part that breaks.", M, 6.5, W - 2 * M, 0.4, { fontSize: 16, bold: true, color: C.text2 });
}

// ---------- 7. design rule ----------
{
  const s = newSlide("The rule: measure where a component is observable, learn the rest", "Method",
    "Observability varies along a trajectory: the step length is unobservable at a stop or at constant velocity, and the direction at tiny parallax. So the choice cannot be made once per sequence; it is made per edge and per component.");
  const rows = [["Rotation", "Gyroscope", "Observable almost always, given a known bias", C.accent3], ["Step length", "Accelerometer", "Unobservable at a stop or at constant velocity", C.accent2], ["Direction", "Two-view geometry", "Unobservable at tiny parallax", C.accent1]];
  rows.forEach((r, i) => {
    const y = 1.65 + i * 1.5;
    card(s, M, y, W - 2 * M, 1.25);
    circle(s, W - M - 0.95, y + 0.22, 0.8, r[3], String(i + 1), 24, r[3] === C.accent2 ? C.text1 : C.background1);
    T(s, r[0], 8.1, y + 0.25, 3.6, 0.75, { fontSize: 24, bold: true, color: C.text2, valign: "middle" });
    T(s, r[1], 5.0, y + 0.25, 3.0, 0.75, { fontSize: 18, bold: true, color: C.accent1, valign: "middle" });
    T(s, r[2], M, y + 0.25, 4.3, 0.75, { fontSize: 15, valign: "middle" });
  });
  T(s, "So the decision is made per edge, and per component.", M, 6.2, W - 2 * M, 0.5, { fontSize: 20, bold: true, color: C.text2 });
}

// ---------- 8. pipeline ----------
{
  const s = newSlide("System architecture: two branches and a gate", "Method",
    "Camera and IMU feed two branches. The learned branch is the frozen network; only its translation direction and length are used. The measurement branch computes the gyro rotation over the true image interval, the two-view direction, and the step length from a windowed accelerometer alignment, each with a quality signal. The gate chooses per edge and per component.");
  card(s, 11.0, 2.7, 1.73, 2.0, { fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 } });
  T(s, "Camera\n+\nIMU", 11.0, 2.7, 1.73, 2.0, { align: "center", valign: "middle", fontSize: 18, bold: true, color: C.background1 });
  box(s, 4.9, 1.5, 5.5, 2.5, { rectRadius: 0.1, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.5, dashType: "dash" } });
  T(s, "Measurement branch (no learning)", 4.95, 1.55, 5.4, 0.35, { fontSize: 14, bold: true, color: C.accent1 });
  ["Rotation: gyroscope", "Direction: two-view geometry", "Length: accelerometer"].forEach((t, i) => {
    card(s, 5.1, 1.95 + i * 0.68, 5.1, 0.58, { fill: { color: C.background2 }, line: { color: C.accent1, width: 1 } });
    T(s, t, 5.25, 1.95 + i * 0.68, 4.8, 0.58, { fontSize: 15, valign: "middle", bold: true });
  });
  box(s, 4.9, 4.4, 5.5, 1.6, { rectRadius: 0.1, fill: { color: C.background1 }, line: { color: C.accent5, width: 1.5, dashType: "dash" } });
  T(s, "Learned branch", 4.95, 4.45, 5.4, 0.35, { fontSize: 14, bold: true, color: C.accent5 });
  card(s, 5.1, 4.9, 5.1, 0.9, { fill: { color: C.background2 }, line: { color: C.accent5, width: 1 } });
  T(s, "Frozen network: translation direction and length only", 5.25, 4.9, 4.8, 0.9, { fontSize: 15, valign: "middle", bold: true });
  arrow(s, 10.45, 2.75, 0.5, HEX.ink); arrow(s, 10.45, 4.55, 0.5, HEX.ink);
  card(s, 2.2, 2.7, 2.2, 2.0, { fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
  T(s, "Observability gate\nper edge,\nper component", 2.2, 2.7, 2.2, 2.0, { align: "center", valign: "middle", fontSize: 17, bold: true, color: C.text1 });
  arrow(s, 4.45, 2.9, 0.4, HEX.ink); arrow(s, 4.45, 4.55, 0.4, HEX.ink);
  card(s, M, 2.7, 1.2, 2.0, { fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 } });
  T(s, "Pose", M, 2.7, 1.2, 2.0, { align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1 });
  arrow(s, 1.85, 3.7, 0.3, HEX.ink);
  T(s, "Rotation always comes from the gyroscope; the network is used only where a component is not observable.", M, 6.3, W - 2 * M, 0.5, { fontSize: 16, bold: true, color: C.text2 });
}

// ---------- 9. gate details ----------
{
  const s = newSlide("The gate decides from sensor signals alone", "Method",
    "For every edge the gate chooses between measurement and network from signals computed on the sensors: parallax after removing rotation, the uncertainty of the geometric solution, the condition number of the alignment system, and gravity consistency. v3 is pre-registered with constants fixed on every dataset; v4 is a post-hoc revision that estimates its trust in the network online.");
  const sig = ["Parallax after removing rotation", "Uncertainty of the geometric solution", "Condition number of the length system: below 10⁴ for 500-edge windows", "Gravity consistency: norm within ±0.5 m/s²"];
  T(s, "Decision signals", 7.2, 1.55, 5.5, 0.5, { fontSize: 20, bold: true, color: C.text2 });
  sig.forEach((r, i) => {
    const y = 2.15 + i * 0.95;
    card(s, 7.2, y, 5.53, 0.8, { fill: { color: C.background2 } });
    circle(s, 12.0, y + 0.12, 0.56, C.accent1, String(i + 1), 16);
    T(s, r, 7.35, y, 4.5, 0.8, { fontSize: 15, valign: "middle" });
  });
  T(s, "One constant, K = 3 (a 3-sigma rule), on every dataset.", 7.2, 6.0, 5.53, 0.5, { fontSize: 14, color: C.text2, bold: true });
  card(s, M, 1.55, 6.3, 2.45, { fill: { color: C.background2 } });
  T(s, "v3: pre-registered", M + 0.25, 1.7, 5.8, 0.45, { fontSize: 20, bold: true, color: C.accent1 });
  T(s, "σ_net = 0.83° is fixed, tuned on held-out KITTI drive 08 only, then frozen. One run on the test data.", M + 0.25, 2.25, 5.8, 1.6, { fontSize: 16 });
  card(s, M, 4.25, 6.3, 2.25, { fill: { color: C.background2 } });
  T(s, "v4: post-hoc", M + 0.25, 4.4, 5.8, 0.45, { fontSize: 20, bold: true, color: C.accent3 });
  T(s, "σ̂ is estimated online from the agreement between the network and confident geometry; the threshold τ = 10° was chosen on design data only.", M + 0.25, 4.95, 5.8, 1.45, { fontSize: 16 });
}

// ---------- 10. KITTI results ----------
{
  const s = newSlide("KITTI: from 3.04% to 1.56% without test-set selection", "Results",
    "Clean split: the paper-text split, with the checkpoint selected on held-out drive 08. Mean over three seeds. The gyroscope removes most of the error, the accelerometer length most of the rest, and the gate reaches 1.56% against 1.96% for VIFT. With drive 08 as in the released code: 1.21%. r_rel 0.18 is measured against OXTS, which shares the unit with the gyroscope; against SuMa it is 0.32.");
  const labels = ["Network", "+ gyro rotation", "+ accelerometer length", "Gate v3 = v4", "VIFT, published"];
  chart(s, pres.charts.BAR, [{ name: "t_rel %", labels, values: [3.04, 2.10, 1.62, 1.56, 1.96] }], barOpts({ x: 4.3, y: 1.55, w: 8.4, h: 5.2, barDir: "col", chartColors: [HEX.gray, HEX.gray, HEX.gray, HEX.teal, HEX.amber], valAxisMaxVal: 3.5, valAxisMinVal: 0 }));
  card(s, M, 1.65, 3.45, 4.9);
  T(s, [{ text: "Rotation error: from 1.04 to 0.18", options: { bold: true, fontSize: 20, color: HEX.teal, breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "With drive 08, as in VIFT's released code: network 2.15%, gate 1.21%.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "Against SuMa: 1.65%, rotation error 0.32.", options: {} }],
    M + 0.2, 1.9, 3.05, 4.4, { fontSize: 15 });
}

// ---------- 11. cross model ----------
{
  const s = newSlide("Models we did not train: the same improvement", "Results",
    "The same chain was run once with the gate frozen on the published weights of three families: VS-VIO, UL-VIO and NASVIO (two variants). The error drops by about 45–51%. These models were trained with drive 08, so the numbers are on the with-08 split, against OXTS.");
  const labels = ["VS-VIO", "UL-VIO", "NASVIO flops", "NASVIO latency"];
  chart(s, pres.charts.BAR, [{ name: "Network", labels, values: [2.35, 3.52, 2.58, 2.50] }, { name: "Gate", labels, values: [1.27, 1.74, 1.42, 1.38] }],
    barOpts({ x: 3.9, y: 1.55, w: 8.8, h: 5.2, barDir: "col", chartColors: [HEX.gray, HEX.teal], showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: HEX.ink, valAxisMaxVal: 4, valAxisMinVal: 0 }));
  card(s, M, 1.65, 3.0, 4.9);
  T(s, "45–51%", M + 0.15, 2.0, 2.7, 0.9, { fontSize: 38, bold: true, color: HEX.teal, align: "center" });
  T(s, "lower t_rel with the gate, on published weights we did not train. One run, gate frozen.", M + 0.2, 3.1, 2.6, 3.0, { fontSize: 15 });
}

// ---------- 12. EuRoC / TUM-VI results ----------
{
  const s = newSlide("EuRoC and TUM-VI: a large gain, but not state of the art", "Results",
    "On EuRoC the learned networks reach 4.02–4.98 m, measuring every component without a network gives 1.02 m, and the gate 0.94 or 0.99 m. But ORB-SLAM3, run by us under the same protocol, reaches 0.042 m, about 22 times more accurate, so the result is a diagnosis of where learned VIO fails rather than a state of the art. On TUM-VI v3 failed (0.69 m against a limit of 0.57 m) and v4 fixed it (0.47 m).");
  function row(y, title, vals) {
    T(s, title, M, y - 0.05, W - 2 * M, 0.45, { fontSize: 18, bold: true, color: C.text2 });
    const cw = 2.85, gap = 0.24;
    vals.forEach((v, i) => {
      const x = W - M - cw - i * (cw + gap);
      card(s, x, y + 0.5, cw, 1.9, { fill: { color: C.background2 } });
      T(s, v[0], x + 0.15, y + 0.65, cw - 0.3, 0.8, { fontSize: 28, bold: true, color: v[2], align: "center" });
      T(s, v[1], x + 0.2, y + 1.5, cw - 0.4, 0.8, { fontSize: 14, align: "center" });
    });
  }
  row(1.45, "EuRoC: mean ATE in metres over four sequences", [["4.02–4.98", "Learned networks", HEX.red], ["1.02", "Measured, no network", HEX.gray], ["0.94 / 0.99", "Gate, two models", HEX.teal], ["0.042", "ORB-SLAM3, our run", HEX.green]]);
  row(4.15, "TUM-VI: mean ATE in metres", [["5.0–8.8", "Learned networks", HEX.red], ["0.47", "Best hand-designed", HEX.gray], ["0.69", "v3 failed; limit 0.57", HEX.red], ["0.47", "v4, post-hoc", HEX.teal]]);
}

// ---------- 13. failures and v4 ----------
{
  const s = newSlide("Failures are reported: four versions of the gate", "Results",
    "v1 and v2 failed on KITTI. v3 was designed on held-out drive 08 only and met the criterion on KITTI and EuRoC but failed on TUM-VI. v4 is a post-hoc revision that estimates its trust in the network online, and it fixed TUM-VI. We ran v4 unchanged on V2_03, the only EuRoC sequence never used at any stage: it estimates the network's error well, but v4 is no better than v3. Any tau from 3 to 20 degrees gives identical results.");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dark }, align: "center", fontSize: 14, fontFace: "Arial" } });
  const D = (t, c) => ({ text: t, options: { align: "center", fontSize: 14, fontFace: "Arial", color: c || HEX.ink } });
  s.addTable([
    [H("Version"), H("Status"), H("KITTI %"), H("EuRoC m"), H("TUM-VI m")],
    [D("v1"), D("Pre-registered"), D("fail 1.96", HEX.red), D("1.01 / 0.98"), D("not run")],
    [D("v2"), D("Post-hoc"), D("fail 1.99", HEX.red), D("1.00 / 0.96"), D("not run")],
    [D("v3"), D("One test run"), D("1.21"), D("0.94 / 0.99"), D("fail 0.69", HEX.red)],
    [D("v4"), D("Post-hoc"), D("1.21"), D("0.95 / 0.98"), D("0.47")],
  ], { x: M, y: 1.6, w: 7.4, colW: [1.1, 1.9, 1.5, 1.6, 1.3], rowH: 0.62, border: { type: "solid", color: "D5DDDB", pt: 0.75 }, fill: { color: "FFFFFF" }, valign: "middle" });
  card(s, M, 1.6, W - M - 8.3, 4.95); // right-hand card: final x = 8.3
  T(s, "Test on V2_03, never used before", M + 0.2, 1.8, W - M - 8.7, 0.5, { fontSize: 18, bold: true, color: C.text2 });
  T(s, [{ text: "σ̂ estimates the network's error without ground truth: 59–70° against 61–72° actual.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "But v4 is no better than v3: 2.1 m against 2.0 m (network 3.4 m).", options: { bold: true, breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "Any τ from 3° to 20° gives identical results.", options: {} }],
    M + 0.2, 2.4, W - M - 8.7, 3.9, { fontSize: 15 });
  T(s, "v4's gain is shown on TUM-VI only.", W - M - 7.4, 5.2, 7.4, 0.5, { fontSize: 18, bold: true, color: HEX.red });
}

// ---------- 14. limitations ----------
{
  const s = newSlide("Limits: what the paper does not claim", "Conclusion",
    "We state the limits plainly: KITTI's ground truth shares the OXTS unit with the gyroscope, v4 is post-hoc, the result is not a state of the art on EuRoC, the gyro bias needs a static start, we did not check whether DB-VIO is affected by the timing error because its code and time step are not published, and the network differs between KITTI and the other datasets.");
  const lim = ["KITTI's ground truth and the gyroscope come from the same OXTS unit; against SuMa the rotation error is 0.32", "v4 is post-hoc and no better than v3 on V2_03",
               "Not a state of the art on EuRoC: ORB-SLAM3 is about 22 times more accurate", "The gyro bias needs a static start, and V1_01 has none",
               "We did not check whether DB-VIO has the timing error: its code is unpublished and its time step is not stated", "The network is VIFT on KITTI and PI-VIFT on EuRoC and TUM-VI"];
  const cw = 5.95, ch = 1.45, gx = 0.25;
  lim.forEach((t, i) => {
    const x = W - M - cw - (i % 2) * (cw + gx), y = 1.6 + Math.floor(i / 2) * (ch + 0.2);
    card(s, x, y, cw, ch);
    circle(s, x + cw - 0.75, y + 0.38, 0.55, C.accent3, String(i + 1), 16);
    T(s, t, x + 0.2, y + 0.1, cw - 1.1, ch - 0.2, { fontSize: 15, valign: "middle" });
  });
}

// ---------- 15. outside the paper: online bias ----------
{
  const s = newSlide("Next step: gyro bias estimated while moving", "Conclusion",
    "This slide is outside the paper. The supervisor asked for an online bias estimate with no static start. We implemented a causal estimator (median of the last 200 usable edges) with constants tuned on training sequences only. On V1_01 the error falls from 2.07 to 0.73 m, and on V2_03 it improves in all six cells. But where a static start exists the static mean is more accurate, and on TUM-VI the estimator needs a floor because the true bias is near zero. The hybrid rule was chosen after seeing the results, so it is post-hoc.");
  box(s, M, 1.45, 4.5, 0.45, { rectRadius: 0.06, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
  T(s, "Outside the paper; partly post-hoc", M, 1.45, 4.5, 0.45, { fontSize: 14, bold: true, color: C.text1, align: "center", valign: "middle" });
  const labels = ["MH_05", "V2_02", "V1_02", "V1_01"];
  chart(s, pres.charts.BAR, [{ name: "Original bias (static start or zero)", labels, values: [0.83, 0.38, 0.34, 2.07] }, { name: "Online bias", labels, values: [0.98, 0.41, 0.54, 0.73] }],
    barOpts({ x: 5.6, y: 1.95, w: 7.1, h: 4.6, barDir: "col", chartColors: [HEX.gray, HEX.teal], showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: "+mn-lt", legendColor: HEX.ink, valAxisMaxVal: 2.6, valAxisMinVal: 0, dataLabelFontSize: 12 }));
  card(s, M, 2.1, 4.8, 4.45);
  T(s, [{ text: "ATE in metres with gate v3.", options: { breakLine: true, bold: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "Helps where there is no static start: V1_01 from 2.07 to 0.73, and V2_03 in all six cells.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "Slightly worse with a static start: V1_02 rises by 0.20. Needs a floor on TUM-VI.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "Hybrid rule (static if present, else online): mean 0.57 instead of 0.905.", options: { bold: true } }],
    M + 0.2, 2.25, 4.4, 4.2, { fontSize: 14 });
}

// ---------- 16. conclusion ----------
{
  const s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "Conclusion" });
  s.addText("Summary", { placeholder: "title" });
  const pts = [["Find the wrong component by swapping in ground truth", "Rotation on KITTI, direction on EuRoC and TUM-VI."], ["Measure what the sensors measure, learn the rest", "From 1.96% to 1.56% on KITTI, and about 45–51% lower error on models we did not train."], ["Every failure is reported", "v1 and v2 failed on KITTI, v3 on TUM-VI; v4 is post-hoc and no better than v3 on V2_03."]];
  pts.forEach((p, i) => {
    const y = 1.9 + i * 1.45;
    circle(s, W - M - 0.85, y + 0.1, 0.75, C.accent2, String(i + 1), 24, C.text1);
    T(s, p[0], M, y, W - 2 * M - 1.2, 0.55, { fontSize: 24, bold: true, color: C.background1 });
    T(s, p[1], M, y + 0.6, W - 2 * M - 1.2, 0.55, { fontSize: 16, color: C.background2 });
  });
  T(s, "Thank you", M, 6.3, W - 2 * M, 0.7, { fontSize: 30, bold: true, color: C.accent2 });
  s.addNotes("Three messages: diagnose by swapping, measure versus learn, and report every failure.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
