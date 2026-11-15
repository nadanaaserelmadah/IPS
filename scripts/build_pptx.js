// Builds an editable .pptx from the slide layout that build_pptx.py extracted from Chrome.
//   node scripts/build_pptx.js LAYOUT_JSON OUT_PPTX DECK_DIR
// Needs pptxgenjs (npm install pptxgenjs; point NODE_PATH at its node_modules if it is not next to this script).
const fs = require("fs");
const path = require("path");
const PptxGenJS = require("pptxgenjs");

const [layoutPath, outPath, deckDir] = process.argv.slice(2);
const L = JSON.parse(fs.readFileSync(layoutPath, "utf8"));
const deck = JSON.parse(fs.readFileSync(path.join(deckDir, "deck.json"), "utf8"));

const PX = 144; // 1920 px canvas = 13.333 in
const inch = (v) => v / PX;
const pt = (v) => v / 2;
const half = (v) => Math.round(v * 2) / 2;

const pres = new PptxGenJS();
pres.layout = "LAYOUT_WIDE";
pres.title = deck.title;
pres.author = "Nada El Madah";
pres.subject = "Thesis defense";

const sectionAt = {};
Object.values(deck.sections || {}).forEach((s) => { sectionAt[s.start] = s.description; });
let current = null;

L.slides.forEach((sl) => {
  if (sectionAt[sl.id]) { current = sectionAt[sl.id]; pres.addSection({ title: current }); }
  const s = pres.addSlide(current ? { sectionTitle: current } : undefined);
  s.background = { color: sl.bg };
  sl.items.forEach((it, n) => {
    const b = it.b;
    if (it.t === "box") {
      const w = inch(b.w), h = inch(b.h);
      const opts = {
        x: inch(b.x), y: inch(b.y), w, h,
        fill: it.fill ? { color: it.fill.hex } : { color: "FFFFFF", transparency: 100 },
        line: it.line ? { color: it.line.c.hex, width: Math.max(0.5, pt(it.line.w)) } : { type: "none" },
        objectName: `${sl.id}-box-${n}`,
      };
      if (it.rad > 0) { opts.rectRadius = Math.min(inch(it.rad), Math.min(w, h) / 2); }
      s.addShape(it.rad > 0 ? pres.ShapeType.roundRect : pres.ShapeType.rect, opts);
    } else if (it.t === "text") {
      const slack = Math.min(b.w * 0.02 + 4, 40);
      let x = b.x, w = b.w + slack;
      if (it.align === "right") x -= slack; else if (it.align === "center") x -= slack / 2;
      x = Math.max(0, x); w = Math.min(w, 1920 - x);
      s.addText(it.text, {
        x: inch(x), y: inch(b.y), w: inch(w), h: inch(b.h),
        fontFace: it.font, fontSize: half(pt(it.fs)), bold: it.bold, italic: it.italic, color: it.color,
        align: it.align, valign: "top", margin: 0, fit: "none", wrap: true, isTextBox: true, rtlMode: it.rtl,
        lineSpacingMultiple: Math.min(2, Math.max(0.9, it.lh / (it.fs * 1.15))),
        charSpacing: it.ls ? pt(it.ls) : undefined,
        objectName: `${sl.id}-text-${n}`,
      });
    } else if (it.t === "img") {
      const sc = Math.min(b.w / it.nw, b.h / it.nh), w = it.nw * sc, h = it.nh * sc;
      s.addImage({
        path: path.resolve(deckDir, it.src), x: inch(b.x + (b.w - w) / 2), y: inch(b.y + (b.h - h) / 2), w: inch(w), h: inch(h),
        altText: it.alt, objectName: `${sl.id}-image-${n}`,
      });
    } else if (it.t === "arrow") {
      s.addShape(it.dir === "left" ? pres.ShapeType.leftArrow : pres.ShapeType.rightArrow, {
        x: inch(b.x), y: inch(b.y), w: inch(b.w), h: inch(b.h), fill: { color: it.fill }, line: { type: "none" },
        objectName: `${sl.id}-arrow-${n}`,
      });
    } else if (it.t === "table") {
      const rows = it.rows.map((r) => r.cells.map((c) => ({
        text: c.text,
        options: {
          fontFace: c.font, fontSize: half(pt(c.fs)), bold: c.bold, color: c.color, align: c.align, valign: "top", rtlMode: c.rtl,
          margin: [pt(c.pad[0]), pt(c.pad[1]), pt(c.pad[2]), pt(c.pad[3])],
          border: [{ type: "none" }, { type: "none" },
            c.bb ? { type: "solid", pt: Math.max(0.5, pt(c.bb.w)), color: c.bb.c } : { type: "none" }, { type: "none" }],
        },
      })));
      s.addTable(rows, { x: inch(b.x), y: inch(b.y), w: inch(b.w), colW: it.colW.map(inch), rowH: it.rows.map((r) => inch(r.h)), objectName: `${sl.id}-table-${n}` });
    }
  });
  if (sl.notes) s.addNotes(sl.notes);
});

pres.writeFile({ fileName: outPath }).then((f) => console.log("wrote", f, L.slides.length, "slides"));
