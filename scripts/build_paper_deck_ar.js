// Arabic PowerPoint for the current paper (paper/submission/main.tex, 8 pages) -> docs/presentations/paper_ar.pptx
//   NODE_PATH=SOME_DIR/node_modules node scripts/build_paper_deck_ar.js
// (npm install pptxgenjs --prefix SOME_DIR). Numbers are copied from the paper's tables; slide 15 is marked as outside the paper.
const path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = process.env.PPTX_SKILL_DIR || "/Users/finetch/Library/Application Support/Claude/local-agent-mode-sessions/skills-plugin/79318aaa-643a-485b-90dd-4a26d2773cce/b0b5a169-e2bd-4421-8d36-b221b96b46a7/skills/pptx";
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));

const OUT = path.join(__dirname, "../docs/presentations/paper_ar.pptx");
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
pres.title = "قِس ما يمكن رصده، وتعلَّم الباقي";
pres.author = "IPS";
const C = pres.SchemeColor;
const W = 13.33, M = 0.6;

pres.defineSlideMaster({
  title: "CONTENT", background: { color: C.background1 },
  slideNumber: { x: 0.4, y: 6.95, w: 0.7, h: 0.3, fontSize: 11, color: C.accent5, fontFace: "Arial" },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.35, w: W - 2 * M, h: 1.0, fontSize: 34, bold: true, color: C.text2, align: "right", valign: "middle", margin: 0 }, text: "" } }],
});
pres.defineSlideMaster({
  title: "DARK", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 1.9, w: W - 2 * M, h: 1.7, fontSize: 46, bold: true, color: C.background1, align: "right", valign: "middle", margin: 0 }, text: "" } }],
});

["المقدمة", "المشكلة", "التشخيص", "الطريقة", "النتائج", "الخاتمة"].forEach((t) => pres.addSection({ title: t }));

pres.defineSlideMaster({
  title: "CLOSING", background: { color: C.text2 },
  objects: [{ placeholder: { options: { name: "title", type: "title", x: M, y: 0.5, w: W - 2 * M, h: 1.0, fontSize: 40, bold: true, color: C.background1, align: "right", valign: "middle", margin: 0 }, text: "" } }],
});

// ---------- helpers ----------
const RLM = "\u200F";
// Latin/number runs followed by punctuation get a right-to-left mark so PowerPoint keeps the punctuation on the correct side
function bd(t) {
  if (typeof t !== "string") return t;
  let r = t.replace(/([A-Za-z0-9][A-Za-z0-9_\-–.°%×\/]*)(?=[:،؛؟!)\]]|\.(?=\s|$))/g, "$1" + RLM);
  if (/^[A-Za-z]/.test(r)) r = RLM + r; // pure numbers and ranges keep their left-to-right order
  return r;
}
function bdAll(x) { return Array.isArray(x) ? x.map((r) => Object.assign({}, r, { text: bd(r.text) })) : bd(x); }
const rtl = { rtlMode: true, align: "right", isTextBox: true, fontFace: "Arial", margin: 0, valign: "top" };
function T(s, text, x, y, w, h, o = {}) { s.addText(bdAll(text), Object.assign({ x, y, w, h, fontSize: 16, color: C.text1 }, rtl, o)); }
function card(s, x, y, w, h, o = {}) {
  s.addShape(pres.ShapeType.roundRect, Object.assign({ x, y, w, h, rectRadius: 0.08, fill: { color: C.background2 }, line: { color: C.background2, width: 0 } }, o));
}
function circle(s, x, y, d, fill, label, size = 20, color = C.background1) {
  s.addShape(pres.ShapeType.ellipse, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill, width: 0 } });
  s.addText(label, { x, y, w: d, h: d, align: "center", valign: "middle", fontSize: size, bold: true, color, fontFace: "Arial", margin: 0, isTextBox: true });
}
function arrow(s, x, y, w, color = HEX.ink) { // leftward arrow (reading direction is right-to-left)
  s.addShape(pres.ShapeType.line, { x, y, w, h: 0, flipH: true, line: { color, width: 2, endArrowType: "triangle" } });
}
function newSlide(title, sec, notes) {
  const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: sec });
  s.addText(bd(title), { placeholder: "title", rtlMode: true, align: "right" });
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

// ---------- 1. title ----------
{
  const s = pres.addSlide({ masterName: "DARK", sectionTitle: "المقدمة" });
  s.addText("قِس ما يمكن رصده، وتعلَّم الباقي", { placeholder: "title", rtlMode: true, align: "right" });
  T(s, "دمج مضبوط بالرصدية لـ VIO المتعلَّم", M, 3.65, W - 2 * M, 0.6, { fontSize: 26, color: C.accent2, bold: true });
  T(s, "Measure What Is Observable, Learn the Rest: Observability-Gated Fusion for Learned Visual–Inertial Odometry", M, 4.4, W - 2 * M, 0.8,
    { fontSize: 15, color: C.background2, rtlMode: false, align: "left" });
  T(s, "Nada El Madah · Magdy Zakaria · Samir Elmougy · Shahenda Sarhan · Mohamed Handosa", M, 5.4, W - 2 * M, 0.4, { fontSize: 14, color: C.background2, rtlMode: false, align: "left" });
  circle(s, 11.55, 0.55, 0.95, C.accent1, "R", 30); circle(s, 10.55, 0.55, 0.95, C.accent2, "ℓ", 30, C.text1); circle(s, 9.55, 0.55, 0.95, C.accent3, "d", 30);
  T(s, "دوران · طول · اتجاه", 8.0, 1.62, 4.5, 0.4, { fontSize: 14, color: C.background2 });
  s.addNotes("العنوان: قِس ما يمكن رصده، وتعلَّم الباقي. الفكرة: الحساسات تقيس جزءًا كبيرًا من الحركة أصلًا، فلا نطلب من الشبكة أن تتعلمه.");
}

// ---------- 2. the three components ----------
{
  const s = newSlide("الحركة ثلاثة مكوّنات، وكل مكوّن له حساس", "المشكلة",
    "VIO تقدّر حركة الكاميرا من الصور وحساس القصور الذاتي IMU (الجيروسكوب ومقياس التسارع). الجيروسكوب يقيس الدوران، ومقياس التسارع يستطيع قياس طول الخطوة، ولقطتان من الكاميرا تعطيان اتجاه الحركة. لكن VIO المتعلَّم يطلب من شبكة واحدة أن تتنبأ بالثلاثة دفعة واحدة.");
  const cw = 3.85, gap = 0.29, y = 1.65;
  const items = [["R", "الدوران", "الجيروسكوب", "يقيس السرعة الزاوية مباشرة", C.accent3], ["ℓ", "طول الخطوة", "مقياس التسارع", "يقدّره من تسارع النافذة الزمنية", C.accent2], ["d", "اتجاه الحركة", "لقطتان من الكاميرا", "هندسة الرؤية بعد معرفة الدوران", C.accent1]];
  items.forEach((it, i) => {
    const x = W - M - cw - i * (cw + gap);
    card(s, x, y, cw, 3.3);
    circle(s, x + cw - 1.1, y + 0.3, 0.8, it[4], it[0], 28, it[4] === C.accent2 ? C.text1 : C.background1);
    T(s, it[1], x + 0.25, y + 1.35, cw - 0.5, 0.5, { fontSize: 24, bold: true, color: C.text2 });
    T(s, it[2], x + 0.25, y + 1.95, cw - 0.5, 0.45, { fontSize: 18, bold: true, color: C.accent1 });
    T(s, it[3], x + 0.25, y + 2.45, cw - 0.5, 0.7, { fontSize: 15 });
  });
  card(s, M, 5.3, W - 2 * M, 1.2, { fill: { color: C.text2 }, line: { color: C.text2, width: 0 } });
  T(s, "لكنّ VIO المتعلَّم يطلب من شبكة عصبية واحدة أن تتنبأ بالثلاثة معًا. وهنا يفشل.", M + 0.4, 5.3, W - 2 * M - 0.8, 1.2, { fontSize: 22, bold: true, color: C.background1, valign: "middle" });
}

// ---------- 3. the question ----------
{
  const s = newSlide("السؤال: أي جزء من الوضعية خاطئ؟", "التشخيص",
    "بدل أن نسأل كيف نحسّن الشبكة كلها، نسأل أي مكوّن خاطئ. نستبدل مكوّنًا واحدًا بالحقيقة الأرضية ونعيد التقييم بلا إعادة تدريب. على KITTI: دوران حقيقي مع إزاحة الشبكة يعطي 1.55%، وهو أفضل من 1.96% المنشورة لنموذج VIFT.");
  const bw = 3.6, y = 1.8;
  const steps = [["١", "خذ شبكة جاهزة", "نموذج VIFT على KITTI، وشبكة على EuRoC"], ["٢", "استبدل مكوّنًا واحدًا بالحقيقة", "الدوران أو الاتجاه أو الطول"], ["٣", "أعد القياس", "بلا إعادة تدريب"]];
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
  T(s, "على KITTI: دوران حقيقي مع إزاحة الشبكة", W - M - 3.4, 5.75, 3.2, 0.5, { fontSize: 14, color: C.background2, align: "center" });
  T(s, "أفضل من 1.96% التي ينشرها VIFT: الدوران هو المشكلة الرئيسية على KITTI. أما على EuRoC فالمشكلة في مكان آخر.", M + 0.4, 4.7, 7.5, 1.6, { fontSize: 20, color: C.background1, valign: "middle" });
}

// ---------- 4. KITTI diagnosis chart ----------
{
  const s = newSlide("على KITTI: تبديل الدوران بالجيروسكوب يغيّر كل شيء", "التشخيص",
    "t_rel النسبي بالمئة، متوسط الدرايفات 05 و07 و10. الجيروسكوب بخطوة 10 ms الاسمية يعطي 4.62% لأن الفاصل الحقيقي نحو 104 ms. عند المكاملة على الفاصل الحقيقي يعطي 1.70% مع r_rel 0.18، أقل من VIFT المنشورة على المقياسين.");
  const labels = ["الشبكة كما هي", "دوران حقيقي + إزاحة الشبكة", "جيروسكوب بخطوات افتراضية خاطئة", "جيروسكوب بالفاصل الحقيقي", "VIFT المنشورة"];
  const vals = [2.11, 1.55, 4.62, 1.70, 1.96];
  s.addChart(pres.charts.BAR, [{ name: "t_rel %", labels, values: vals }], barOpts({ x: 4.3, y: 1.55, w: 8.4, h: 5.2, barDir: "bar", chartColors: [HEX.gray, HEX.green, HEX.red, HEX.teal, HEX.amber], valAxisMaxVal: 5.5, valAxisMinVal: 0, catAxisOrientation: "maxMin" }));
  card(s, M, 1.65, 3.45, 4.9);
  T(s, "ماذا نقرأ؟", M + 0.2, 1.85, 3.05, 0.5, { fontSize: 20, bold: true, color: C.text2 });
  T(s, [{ text: "خطأ الدوران: ", options: { bold: true } }, { text: "من 0.67 إلى 0.18 بمجرد استعمال الجيروسكوب.", options: { breakLine: true } },
        { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "ملاحظة: ", options: { bold: true } }, { text: "اتفاق الجيروسكوب مع OXTS جزئيًا بالبناء، فقِسناه أيضًا مقابل SuMa فكان 0.32." }],
    M + 0.2, 2.5, 3.05, 3.8, { fontSize: 15 });
}

// ---------- 5. timing finding ----------
{
  const s = newSlide("اكتشاف التوقيت: الفاصل الحقيقي نحو 104 مللي ثانية لا 100", "التشخيص",
    "الملفات العطالية الموزّعة مع النماذج المنشورة تحمل 11 عينة لكل زوج صور، وتُقرأ كعشر خطوات كل منها 10 ms. لكن العينات هي التيار الخام 100 Hz المنقَّح على فاصل الصور الحقيقي (103.7–104.1 ms). فكل دوران ناقص نحو 4%، ويبدو الجيروسكوب منحازًا وهو ليس كذلك. الإزاحة +60 ms مضبوطة على درايفات التدريب 04 و06 و08 و09 فقط، وأصلها الفيزيائي غير مفسَّر.");
  const cw = 3.85, gap = 0.29, y = 1.65;
  const st = [["≈ 104 ms", "الفاصل الحقيقي بين صورتين، من 103.7 إلى 104.1 مللي ثانية على الدرايفات 04 إلى 10", C.accent1], ["4%", "نقص الدوران في كل إطار حين نفترض خطوات 10 مللي ثانية", C.accent3], ["+60 ms", "إزاحة زمنية ثابتة بين طوابع الصور والحقيقة الأرضية، مضبوطة على التدريب فقط", C.accent2]];
  st.forEach((it, i) => {
    const x = W - M - cw - i * (cw + gap);
    card(s, x, y, cw, 3.1);
    T(s, it[0], x + 0.2, y + 0.3, cw - 0.4, 1.0, { fontSize: 44, bold: true, color: it[2] === C.accent2 ? C.accent3 : it[2], align: "center" });
    T(s, it[1], x + 0.3, y + 1.5, cw - 0.6, 1.4, { fontSize: 15 });
  });
  card(s, M, 5.15, W - 2 * M, 1.4, { fill: { color: C.text2 }, line: { color: C.text2, width: 0 } });
  T(s, "بعد التصحيح: من 4.62% إلى 1.70%", M + 0.4, 5.15, 6.2, 1.4, { fontSize: 26, bold: true, color: C.accent2, valign: "middle" });
  T(s, "الشبكات نفسها لا ترى الخطوة، فيتأثر فقط من يكامل هذه الملفات.", 7.0, 5.15, 5.7, 1.4, { fontSize: 16, color: C.background1, valign: "middle" });
}

// ---------- 6. EuRoC / TUM-VI ----------
{
  const s = newSlide("في EuRoC وTUM-VI تكمن المشكلة في اتجاه الحركة", "التشخيص",
    "حدّ السكون هو خطأ افتراض أن الجهاز لم يتحرك (3.15 م متوسطًا). الشبكات المتعلَّمة أسوأ منه. استبدال الدوران بالحقيقة لا يفيد، بينما استبدال الإزاحة بالحقيقة يخفض الخطأ إلى ما دون حد السكون. وخطأ اتجاه الشبكة يكبر كلما ابتعدنا عن غرف التدريب، حتى يقترب من اتجاه عشوائي.");
  const cw = 5.95, ch = 2.3, gx = 0.25, gy = 0.25, x0 = W - M - cw, y0 = 1.6;
  const st = [["3.15 م", "حدّ السكون: خطأ افتراض أن الجهاز لم يتحرك", C.accent5], ["4.02–4.98 م", "الشبكات المتعلَّمة: أسوأ من حدّ السكون", C.accent3],
              ["0.6–2.1 م", "إزاحة حقيقية + دوران الشبكة: أقل من حدّ السكون", C.accent4], ["31–90°", "خطأ اتجاه الشبكة على EuRoC يقترب من اتجاه عشوائي، ويبلغ على TUM-VI نحو 40 إلى 87 درجة", C.accent1]];
  st.forEach((it, i) => {
    const x = x0 - (i % 2) * (cw + gx), y = y0 + Math.floor(i / 2) * (ch + gy);
    card(s, x, y, cw, ch);
    T(s, it[0], x + 0.3, y + 0.25, cw - 0.6, 0.9, { fontSize: 38, bold: true, color: it[2] });
    T(s, it[1], x + 0.3, y + 1.3, cw - 0.6, 0.9, { fontSize: 16 });
  });
  T(s, "الدوران الحقيقي لا يفيد هناك، والاتجاه هو الجزء الذي ينهار.", M, 6.5, W - 2 * M, 0.4, { fontSize: 16, bold: true, color: C.text2 });
}

// ---------- 7. design rule ----------
{
  const s = newSlide("القاعدة: قِس حيث يُرصد المكوّن، وتعلَّم الباقي", "الطريقة",
    "قابلية الرصد تتغير على طول المسار: طول الخطوة غير مرصود عند التوقف أو السرعة الثابتة، والاتجاه غير مرصود عند حركة (parallax) ضئيلة. لذلك لا يصلح قرار واحد للتسلسل كله، بل قرار لكل حافة ولكل مكوّن.");
  const rows = [["الدوران", "الجيروسكوب", "يُرصد دائمًا تقريبًا (مع انحياز معروف)", C.accent3], ["طول الخطوة", "مقياس التسارع", "لا يُرصد عند التوقف أو السرعة الثابتة", C.accent2], ["الاتجاه", "هندسة لقطتين", "لا يُرصد عند حركة parallax ضئيلة", C.accent1]];
  rows.forEach((r, i) => {
    const y = 1.65 + i * 1.5;
    card(s, M, y, W - 2 * M, 1.25);
    circle(s, W - M - 0.95, y + 0.22, 0.8, r[3], String(i + 1), 24, r[3] === C.accent2 ? C.text1 : C.background1);
    T(s, r[0], 8.1, y + 0.25, 3.6, 0.75, { fontSize: 24, bold: true, color: C.text2, valign: "middle" });
    T(s, r[1], 5.3, y + 0.25, 2.6, 0.75, { fontSize: 18, bold: true, color: C.accent1, valign: "middle" });
    T(s, r[2], M + 0.3, y + 0.25, 4.7, 0.75, { fontSize: 15, valign: "middle" });
  });
  T(s, "لذلك يكون القرار لكل حافة، ولكل مكوّن على حدة.", M, 6.2, W - 2 * M, 0.5, { fontSize: 20, bold: true, color: C.text2 });
}

// ---------- 8. pipeline ----------
{
  const s = newSlide("بنية النظام: فرعان وبوابة", "الطريقة",
    "الكاميرا وIMU تغذّيان فرعين. الفرع المتعلَّم هو الشبكة المجمّدة، ونستعمل منها اتجاه الإزاحة وطولها فقط. فرع القياس يحسب الدوران من الجيروسكوب على الفاصل الحقيقي، والاتجاه من هندسة لقطتين، والطول من محاذاة مقياس التسارع، ومع كل منها إشارة جودة. البوابة تختار لكل حافة ولكل مكوّن.");
  const bh = 0.85;
  // input (rightmost)
  card(s, 11.0, 2.7, 1.73, 2.0, { fill: { color: C.accent6 }, line: { color: C.accent6, width: 0 } });
  T(s, "الكاميرا\n+\nIMU", 11.0, 2.7, 1.73, 2.0, { align: "center", valign: "middle", fontSize: 18, bold: true, color: C.background1 });
  // measurement lane
  s.addShape(pres.ShapeType.roundRect, { x: 4.9, y: 1.5, w: 5.5, h: 2.5, rectRadius: 0.1, fill: { color: C.background1 }, line: { color: C.accent1, width: 1.5, dashType: "dash" } });
  T(s, "فرع القياس (بلا تعلّم)", 4.95, 1.55, 5.4, 0.35, { fontSize: 14, bold: true, color: C.accent1 });
  ["الدوران: جيروسكوب", "الاتجاه: هندسة لقطتين", "الطول: مقياس التسارع"].forEach((t, i) => {
    card(s, 5.1, 1.95 + i * 0.68, 5.1, 0.58, { fill: { color: C.background2 }, line: { color: C.accent1, width: 1 } });
    T(s, t, 5.2, 1.95 + i * 0.68, 4.9, 0.58, { fontSize: 15, valign: "middle", bold: true });
  });
  // learned lane
  s.addShape(pres.ShapeType.roundRect, { x: 4.9, y: 4.4, w: 5.5, h: 1.6, rectRadius: 0.1, fill: { color: C.background1 }, line: { color: C.accent5, width: 1.5, dashType: "dash" } });
  T(s, "الفرع المتعلَّم", 4.95, 4.45, 5.4, 0.35, { fontSize: 14, bold: true, color: C.accent5 });
  card(s, 5.1, 4.9, 5.1, 0.9, { fill: { color: C.background2 }, line: { color: C.accent5, width: 1 } });
  T(s, "الشبكة المجمّدة: اتجاه الإزاحة وطولها فقط", 5.2, 4.9, 4.9, 0.9, { fontSize: 15, valign: "middle", bold: true });
  // arrows from input
  arrow(s, 10.45, 2.75, 0.5, HEX.ink); arrow(s, 10.45, 4.55, 0.5, HEX.ink);
  // gate
  card(s, 2.2, 2.7, 2.2, 2.0, { fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
  T(s, "بوابة الرصدية\nلكل حافة\nولكل مكوّن", 2.2, 2.7, 2.2, 2.0, { align: "center", valign: "middle", fontSize: 17, bold: true, color: C.text1 });
  arrow(s, 4.45, 2.9, 0.4, HEX.ink); arrow(s, 4.45, 4.55, 0.4, HEX.ink);
  // output
  card(s, M, 2.7, 1.2, 2.0, { fill: { color: C.accent4 }, line: { color: C.accent4, width: 0 } });
  T(s, "الوضعية", M, 2.7, 1.2, 2.0, { align: "center", valign: "middle", fontSize: 16, bold: true, color: C.background1 });
  arrow(s, 1.85, 3.7, 0.3, HEX.ink);
  T(s, "الدوران يأتي دائمًا من الجيروسكوب؛ تُستعمل الشبكة فقط حين لا يُرصد المكوّن.", M, 6.3, W - 2 * M, 0.5, { fontSize: 16, bold: true, color: C.text2 });
}

// ---------- 9. gate details ----------
{
  const s = newSlide("البوابة: قرار من إشارات الحساسات وحدها", "الطريقة",
    "البوابة تقرر لكل حافة بين القياس والشبكة اعتمادًا على إشارات تُحسب من الحساسات: حركة الصورة بعد إزالة الدوران، وعدم يقين الحل الهندسي، ورقم الحالة لنظام المحاذاة، واتساق الجاذبية. النسخة v3 مسجَّلة مسبقًا وثوابتها ثابتة على كل المجموعات، وv4 معدَّلة لاحقًا (post-hoc) تقدّر ثقتها بالشبكة متصلًا.");
  const sig = [["حركة الصورة parallax بعد إزالة الدوران"], ["عدم يقين الحل الهندسي"], ["رقم الحالة لنظام الطول: أقل من 10⁴ لنوافذ 500 حافة"], ["اتساق الجاذبية: قيمتها ضمن ±0.5 م/ث²"]];
  T(s, "إشارات القرار", 7.2, 1.55, 5.5, 0.5, { fontSize: 20, bold: true, color: C.text2 });
  sig.forEach((r, i) => {
    const y = 2.15 + i * 0.95;
    card(s, 7.2, y, 5.53, 0.8, { fill: { color: C.background2 } });
    circle(s, 12.0, y + 0.12, 0.56, C.accent1, String(i + 1), 16);
    T(s, r[0], 7.35, y, 4.5, 0.8, { fontSize: 15, valign: "middle" });
  });
  T(s, "ثابت واحد، K = 3 أي قاعدة 3 سيغما، في كل المجموعات.", 7.2, 6.0, 5.53, 0.5, { fontSize: 14, color: C.text2, bold: true });
  card(s, M, 1.55, 6.3, 2.45, { fill: { color: C.background2 } });
  T(s, "النسخة v3: مسجَّلة مسبقًا", M + 0.25, 1.7, 5.8, 0.45, { fontSize: 20, bold: true, color: C.accent1 });
  T(s, "القيمة σ_net = 0.83° ثابتة، ضُبطت على KITTI 08 فقط ثم جُمّدت. تشغيلة واحدة على الاختبار.", M + 0.25, 2.25, 5.8, 1.6, { fontSize: 16 });
  card(s, M, 4.25, 6.3, 2.25, { fill: { color: C.background2 } });
  T(s, "النسخة v4: لاحقة لرؤية النتائج", M + 0.25, 4.4, 5.8, 0.45, { fontSize: 20, bold: true, color: C.accent3 });
  T(s, "تُقدَّر σ̂ متصلًا من اتفاق الشبكة مع الهندسة الواثقة، والعتبة τ = 10° اختيرت على بيانات التصميم فقط.", M + 0.25, 4.95, 5.8, 1.45, { fontSize: 16 });
}

// ---------- 10. KITTI results ----------
{
  const s = newSlide("على KITTI: من 3.04% إلى 1.56% دون اختيار على الاختبار", "النتائج",
    "الانقسام النظيف: النص الورقي، ونقطة الحفظ مختارة على الدرايف 08 المحجوز. المتوسط ± الانحراف على ثلاث بذور. الجيروسكوب يزيل معظم الخطأ، وطول مقياس التسارع معظم الباقي، والبوابة تصل إلى 1.56% مقابل 1.96% لـ VIFT. مع الدرايف 08 كما في الكود المنشور: 1.21%. r_rel 0.18 مقاس مقابل OXTS الذي يشارك وحدة الجيروسكوب، ومقابل SuMa يساوي 0.32.");
  const labels = ["الشبكة", "+ جيروسكوب", "+ طول التسارع", "البوابة v3 = v4", "VIFT المنشورة"];
  s.addChart(pres.charts.BAR, [{ name: "t_rel %", labels, values: [3.04, 2.10, 1.62, 1.56, 1.96] }], barOpts({ x: 4.3, y: 1.55, w: 8.4, h: 5.2, barDir: "col", chartColors: [HEX.gray, HEX.gray, HEX.gray, HEX.teal, HEX.amber], valAxisMaxVal: 3.5, valAxisMinVal: 0 }));
  card(s, M, 1.65, 3.45, 4.9);
  T(s, [{ text: "خطأ الدوران: من 1.04 إلى 0.18", options: { bold: true, fontSize: 22, color: HEX.teal, breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "مع الدرايف 08، كما في كود VIFT المنشور: الشبكة 2.15% ثم البوابة 1.21%.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "مقابل SuMa: 1.65%، وخطأ الدوران 0.32.", options: {} }],
    M + 0.2, 1.9, 3.05, 4.4, { fontSize: 15 });
}

// ---------- 11. cross model ----------
{
  const s = newSlide("نماذج لم ندرّبها: التحسّن نفسه", "النتائج",
    "السلسلة نفسها شُغّلت مرة واحدة والبوابة مجمّدة على الأوزان المنشورة لثلاث عائلات من النماذج: VS-VIO وUL-VIO وNASVIO بنسختيها. الخطأ ينخفض بنحو 45–51%. هذه النماذج دُرّبت مع الدرايف 08، فالأرقام على انقسام with-08 ومقابل OXTS.");
  const labels = ["VS-VIO", "UL-VIO", "NASVIO flops", "NASVIO latency"];
  s.addChart(pres.charts.BAR, [{ name: "الشبكة", labels, values: [2.35, 3.52, 2.58, 2.50] }, { name: "البوابة", labels, values: [1.27, 1.74, 1.42, 1.38] }],
    barOpts({ x: 3.9, y: 1.55, w: 8.8, h: 5.2, barDir: "col", chartColors: [HEX.gray, HEX.teal], showLegend: true, legendPos: "t", legendFontSize: 14, legendFontFace: "+mn-lt", legendColor: HEX.ink, valAxisMaxVal: 4, valAxisMinVal: 0 }));
  card(s, M, 1.65, 3.0, 4.9);
  T(s, "45–51%", M + 0.15, 2.0, 2.7, 0.9, { fontSize: 38, bold: true, color: HEX.teal, align: "center" });
  T(s, "انخفاض الخطأ t_rel بالبوابة على أوزان منشورة لم ندرّبها. تشغيلة واحدة، وبوابة مجمّدة.", M + 0.2, 3.1, 2.6, 3.0, { fontSize: 15 });
}

// ---------- 12. EuRoC / TUM-VI results ----------
{
  const s = newSlide("على EuRoC وTUM-VI: تحسّن كبير، لكن ليس رقمًا قياسيًا", "النتائج",
    "على EuRoC الشبكات المتعلَّمة 4.02–4.98 م، وقياس كل المكوّنات بلا شبكة 1.02 م، والبوابة 0.94 أو 0.99 م. لكن ORB-SLAM3 الذي شغّلناه بالبروتوكول نفسه يبلغ 0.042 م، أي أدق بنحو 22 ضعفًا، فالنتيجة تشخيص لمكان فشل VIO المتعلَّم لا حالة فنية متقدمة. على TUM-VI فشلت v3 (0.69 م مقابل حدّ 0.57 م)، وعالجتها v4 (0.47 م).");
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
  row(1.45, "على EuRoC: متوسط ATE بالمتر لأربعة تسلسلات", [["4.02–4.98", "شبكات متعلَّمة", HEX.red], ["1.02", "قياس بلا شبكة", HEX.gray], ["0.94 / 0.99", "البوابة، نموذجان", HEX.teal], ["0.042", "ORB-SLAM3 بتشغيلنا", HEX.green]]);
  row(4.15, "على TUM-VI: متوسط ATE بالمتر", [["5.0–8.8", "شبكات متعلَّمة", HEX.red], ["0.47", "أفضل يدوي", HEX.gray], ["0.69", "فشلت v3، والحد 0.57", HEX.red], ["0.47", "النسخة v4 اللاحقة", HEX.teal]]);
}

// ---------- 13. failures and v4 ----------
{
  const s = newSlide("الإخفاقات معلنة: أربع نسخ من البوابة", "النتائج",
    "النسختان v1 وv2 فشلتا على KITTI، وv3 صُمّمت على الدرايف 08 فقط وحققت المعيار على KITTI وEuRoC لكنها فشلت على TUM-VI. v4 معدَّلة لاحقًا وتقدّر الثقة بالشبكة متصلًا، فعالجت TUM-VI. اختبرنا v4 كما هي على V2_03، وهو التسلسل الوحيد في EuRoC الذي لم يدخل في أي مرحلة: خطأ الشبكة يقدَّر بدقة، لكن v4 ليست أفضل من v3. وأي τ بين 3° و20° يعطي النتائج نفسها.");
  const H = (t) => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: HEX.dark }, align: "center", fontSize: 14, fontFace: "Arial", rtlMode: true } });
  const D = (t, c) => ({ text: t, options: { align: "center", fontSize: 14, fontFace: "Arial", rtlMode: true, color: c || HEX.ink } });
  s.addTable([
    [H("TUM-VI بالمتر"), H("EuRoC بالمتر"), H("KITTI بالمئة"), H("الحالة"), H("النسخة")],
    [D("لم تُشغَّل"), D("1.01 / 0.98"), D("فشل 1.96", HEX.red), D("مسجَّلة مسبقًا"), D("v1")],
    [D("لم تُشغَّل"), D("1.00 / 0.96"), D("فشل 1.99", HEX.red), D("لاحقة"), D("v2")],
    [D("فشل 0.69", HEX.red), D("0.94 / 0.99"), D("1.21"), D("تشغيلة اختبار واحدة"), D("v3")],
    [D("0.47"), D("0.95 / 0.98"), D("1.21"), D("لاحقة لرؤية النتائج"), D("v4")],
  ], { x: M, y: 1.6, w: 7.4, colW: [1.4, 1.5, 1.5, 2.2, 0.8], rowH: 0.62, border: { type: "solid", color: "D5DDDB", pt: 0.75 }, fill: { color: "FFFFFF" }, valign: "middle", rtlMode: true });
  card(s, 8.3, 1.6, W - M - 8.3, 4.95);
  T(s, "اختبار V2_03 الذي لم يُستعمل قط", 8.5, 1.8, W - M - 8.7, 0.5, { fontSize: 18, bold: true, color: C.text2 });
  T(s, [{ text: "تقدير σ̂ لخطأ الشبكة بلا حقيقة أرضية يبلغ 59 إلى 70 درجة، والخطأ الفعلي 61 إلى 72.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "لكن v4 ليست أفضل من v3، فخطؤها 2.1 م مقابل 2.0 م، وخطأ الشبكة 3.4 م.", options: { bold: true, breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "وأي τ بين 3 و20 درجة يعطي النتائج نفسها.", options: {} }],
    8.5, 2.4, W - M - 8.7, 3.9, { fontSize: 15 });
  T(s, "مكسب النسخة v4 مثبَت على TUM-VI وحدها.", M, 5.2, 7.4, 0.5, { fontSize: 18, bold: true, color: HEX.red });
}

// ---------- 14. limitations ----------
{
  const s = newSlide("الحدود: ما لا تدّعيه الورقة", "الخاتمة",
    "نعرض الحدود بصراحة: الحقيقة الأرضية في KITTI تشارك وحدة OXTS مع الجيروسكوب، وv4 لاحقة، والنتيجة ليست حالة فنية متقدمة على EuRoC، وانحياز الجيروسكوب يحتاج بداية ساكنة، ولم نتحقق من تأثر DB-VIO بخطأ التوقيت لأنه لم ينشر كوده ولا الخطوة الزمنية، والشبكتان المستعملتان مختلفتان بين KITTI وبقية المجموعات.");
  const lim = ["الحقيقة الأرضية في KITTI والجيروسكوب من وحدة OXTS نفسها، ومقابل SuMa يبلغ خطأ الدوران 0.32", "النسخة v4 لاحقة، ولا تتفوق على v3 في V2_03",
               "ليست رقمًا قياسيًا على EuRoC، لأن ORB-SLAM3 أدق بنحو 22 ضعفًا", "انحياز الجيروسكوب يحتاج بداية ساكنة، وتسلسل V1_01 بلا بداية ساكنة",
               "لم نتحقق من تأثر DB-VIO بخطأ التوقيت، لأن الكود غير منشور والخطوة الزمنية غير مذكورة", "الشبكة على KITTI هي VIFT، وعلى EuRoC وTUM-VI هي PI-VIFT"];
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
  const s = newSlide("خطوة تالية: انحياز الجيروسكوب أثناء الحركة", "الخاتمة",
    "هذه الشريحة خارج الورقة. طلب المشرف تقدير الانحياز متصلًا بلا بداية ساكنة. نفّذناه: مقدّر سببي (وسيط آخر 200 حافة صالحة) ثوابته مضبوطة على تسلسلات التدريب فقط. على V1_01 ينزل الخطأ من 2.07 إلى 0.73 م، وعلى V2_03 يتحسن في كل الخلايا الست. لكن مع وجود بداية ساكنة يكون المتوسط الساكن أدق، وعلى TUM-VI يحتاج حدًّا أدنى لأن انحيازه الحقيقي قريب من الصفر. القاعدة الهجينة اخترناها بعد رؤية النتائج، فهي post-hoc.");
  s.addShape(pres.ShapeType.roundRect, { x: M, y: 1.45, w: 4.5, h: 0.45, rectRadius: 0.06, fill: { color: C.accent2 }, line: { color: C.accent2, width: 0 } });
  T(s, "خارج الورقة، وبعضه لاحق للنتائج", M, 1.45, 4.5, 0.45, { fontSize: 14, bold: true, color: C.text1, align: "center", valign: "middle" });
  const labels = ["MH_05", "V2_02", "V1_02", "V1_01"];
  s.addChart(pres.charts.BAR, [{ name: "الانحياز الأصلي (ساكن أو صفر)", labels, values: [0.83, 0.38, 0.34, 2.07] }, { name: "انحياز متصل", labels, values: [0.98, 0.41, 0.54, 0.73] }],
    barOpts({ x: 5.6, y: 1.95, w: 7.1, h: 4.6, barDir: "col", chartColors: [HEX.gray, HEX.teal], showLegend: true, legendPos: "t", legendFontSize: 13, legendFontFace: "+mn-lt", legendColor: HEX.ink, valAxisMaxVal: 2.6, valAxisMinVal: 0, dataLabelFontSize: 12 }));
  card(s, M, 2.1, 4.8, 4.45);
  T(s, [{ text: "الخطأ ATE بالمتر مع البوابة v3.", options: { breakLine: true, bold: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "يحسّن ما لا بداية ساكنة له: تسلسل V1_01 من 2.07 إلى 0.73، وتسلسل V2_03 في الخلايا الست كلها.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "يضرّ قليلًا مع بداية ساكنة، فيزيد خطأ V1_02 بمقدار 0.20، ويحتاج حدًّا أدنى على TUM-VI.", options: { breakLine: true } }, { text: " ", options: { breakLine: true, fontSize: 8 } },
        { text: "القاعدة الهجينة، أي ساكن إن وُجد وإلا متصل، تعطي متوسطًا 0.57 بدل 0.905.", options: { bold: true } }],
    M + 0.2, 2.25, 4.4, 4.2, { fontSize: 14 });
}

// ---------- 16. conclusion ----------
{
  const s = pres.addSlide({ masterName: "CLOSING", sectionTitle: "الخاتمة" });
  s.addText("الخلاصة", { placeholder: "title", rtlMode: true, align: "right" });
  const pts = [["حدّد المكوّن الخاطئ بالاستبدال بالحقيقة", "الدوران في KITTI، والاتجاه في EuRoC وTUM-VI."], ["قِس ما تقيسه الحساسات، وتعلَّم الباقي", "من 1.96% إلى 1.56% على KITTI، وانخفاض بنحو 45 إلى 51% على نماذج لم ندرّبها."], ["كل الإخفاقات معلنة", "أخفقت v1 وv2 على KITTI، وv3 على TUM-VI، والنسخة v4 لاحقة ولا تتفوق على v3 في V2_03."]];
  pts.forEach((p, i) => {
    const y = 1.9 + i * 1.45;
    circle(s, W - M - 0.85, y + 0.1, 0.75, C.accent2, String(i + 1), 24, C.text1);
    T(s, p[0], M, y, W - 2 * M - 1.2, 0.55, { fontSize: 24, bold: true, color: C.background1 });
    T(s, p[1], M, y + 0.6, W - 2 * M - 1.2, 0.55, { fontSize: 16, color: C.background2 });
  });
  T(s, "شكرًا", M, 6.3, W - 2 * M, 0.7, { fontSize: 30, bold: true, color: C.accent2 });
  s.addNotes("ثلاث رسائل: التشخيص بالاستبدال، قاعدة القياس مقابل التعلّم، والإعلان عن كل الإخفاقات.");
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote", OUT);
})();
