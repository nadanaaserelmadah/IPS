"""Build editable PowerPoint files from the decks in docs/presentations/{ar,en}  ->  deck.pptx

The slides are laid out by Chrome (with Arial and Georgia, which PowerPoint has too, so line breaks match), the position of every
box, text, picture, table and arrow is read from the page, and scripts/build_pptx.js writes them as native PowerPoint objects.
Speaker notes and sections are kept.

  python scripts/build_pptx.py [ar|en ...]

Needs Chrome, Node and pptxgenjs:  npm install pptxgenjs --prefix SOME_DIR
then run with  PPTXGENJS_NODE_MODULES=SOME_DIR/node_modules  if pptxgenjs is not already resolvable from Node.
"""
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / "docs" / "presentations"
EXTRACT_JS = r"""
<script>
window.addEventListener('load', function () {
  function col(c) {
    var m = c.match(/rgba?\(([^)]+)\)/); if (!m) return null;
    var p = m[1].split(',').map(Number), a = p.length > 3 ? p[3] : 1;
    return {hex: p.slice(0, 3).map(function (v) { return ('0' + Math.round(v).toString(16)).slice(-2); }).join('').toUpperCase(), a: a};
  }
  function family(cs) { return cs.fontFamily.split(',')[0].replace(/["']/g, '').trim(); }
  function align(cs) {
    var a = cs.textAlign, rtl = cs.direction === 'rtl';
    if (a === 'start') a = rtl ? 'right' : 'left'; if (a === 'end') a = rtl ? 'left' : 'right';
    return a === 'justify' ? 'left' : a;
  }
  var out = [];
  document.querySelectorAll('section').forEach(function (sec) {
    var sr = sec.getBoundingClientRect(), items = [], maxBottom = 0;
    function R(el) { var r = el.getBoundingClientRect(); maxBottom = Math.max(maxBottom, r.bottom - sr.top);
      return {x: r.left - sr.left, y: r.top - sr.top, w: r.width, h: r.height}; }
    function painted(el, cs) {
      var bg = col(cs.backgroundColor), bw = parseFloat(cs.borderTopWidth) || 0, hasLine = bw > 0 && cs.borderTopStyle !== 'none';
      if ((bg && bg.a > 0) || hasLine)
        items.push({t: 'box', b: R(el), fill: (bg && bg.a > 0) ? bg : null, line: hasLine ? {w: bw, c: col(cs.borderTopColor)} : null,
                    rad: parseFloat(cs.borderTopLeftRadius) || 0});
    }
    function text(el, cs) {
      var t = el.textContent.replace(/\s+/g, ' ').trim(); if (!t) return;
      if (cs.textTransform === 'uppercase') t = t.toUpperCase();
      var fs = parseFloat(cs.fontSize), lh = cs.lineHeight === 'normal' ? fs * 1.2 : parseFloat(cs.lineHeight);
      items.push({t: 'text', b: R(el), text: t, fs: fs, lh: lh, bold: parseInt(cs.fontWeight) >= 600, italic: cs.fontStyle === 'italic',
                  color: col(cs.color).hex, font: family(cs), align: align(cs), rtl: cs.direction === 'rtl', ls: parseFloat(cs.letterSpacing) || 0});
    }
    function table(el) {
      var rows = [], colW = [];
      el.querySelectorAll('tr').forEach(function (tr, i) {
        var cells = [];
        tr.querySelectorAll('th,td').forEach(function (c) {
          var cs = getComputedStyle(c), bb = parseFloat(cs.borderBottomWidth) || 0;
          if (i === 0) colW.push(c.getBoundingClientRect().width);
          cells.push({text: c.textContent.replace(/\s+/g, ' ').trim(), fs: parseFloat(cs.fontSize), bold: parseInt(cs.fontWeight) >= 600,
                      color: col(cs.color).hex, font: family(cs), align: align(cs), rtl: cs.direction === 'rtl',
                      pad: [parseFloat(cs.paddingTop), parseFloat(cs.paddingRight), parseFloat(cs.paddingBottom), parseFloat(cs.paddingLeft)],
                      bb: bb > 0 ? {w: bb, c: col(cs.borderBottomColor).hex} : null});
        });
        rows.push({h: tr.getBoundingClientRect().height, cells: cells});
      });
      items.push({t: 'table', b: R(el), colW: colW, rows: rows});
    }
    function walk(el) {
      var tag = el.tagName.toLowerCase(); if (tag === 'aside') return;
      var cs = getComputedStyle(el); if (cs.display === 'none') return;
      if (tag === 'table') { table(el); return; }
      if (tag === 'img') { painted(el, cs);
        items.push({t: 'img', b: R(el), src: el.getAttribute('src'), nw: el.naturalWidth, nh: el.naturalHeight, alt: el.alt}); return; }
      if (tag === 'svg') { var pg = el.querySelector('polygon');
        items.push({t: 'arrow', b: R(el), dir: el.getAttribute('data-arrow'), fill: (pg.getAttribute('fill') || '#000000').replace('#', '')}); return; }
      painted(el, cs);
      if (/^(h1|h2|h3|p)$/.test(tag)) { text(el, cs); return; }
      for (var i = 0; i < el.children.length; i++) walk(el.children[i]);
    }
    for (var i = 0; i < sec.children.length; i++) walk(sec.children[i]);
    var notes = sec.querySelector('aside');
    out.push({id: sec.id, bg: col(getComputedStyle(sec).backgroundColor).hex, items: items, bottom: Math.round(maxBottom), notes: notes ? notes.textContent.trim() : ''});
  });
  var pre = document.createElement('pre'); pre.id = 'layout'; pre.textContent = JSON.stringify({slides: out}); document.body.appendChild(pre);
});
</script>
"""


def build(lang):
    d = ROOT / lang
    page = (d / "deck.html").read_text()
    page = re.sub(r'<link rel="stylesheet"[^>]*>', "", page)
    for old, new in (("font-family:'Cairo', Arial, sans-serif", "font-family:Arial, sans-serif"),
                     ("font-family:'IBM Plex Sans', Arial, sans-serif", "font-family:Arial, sans-serif"),
                     ("font-family:'Source Serif 4', Georgia, serif", "font-family:Georgia, serif")):
        page = page.replace(old, new)
    page = page.replace("</body>", EXTRACT_JS + "</body>")
    tmp = d / "_pptx_layout.html"
    tmp.write_text(page)
    try:
        dom = subprocess.run([CHROME, "--headless", "--disable-gpu", "--virtual-time-budget=20000", "--dump-dom", tmp.as_uri()],
                             check=True, capture_output=True, text=True).stdout
    finally:
        tmp.unlink()
    layout = json.loads(html.unescape(re.search(r'<pre id="layout">(.*?)</pre>', dom, re.S).group(1)))
    for s in layout["slides"]:
        if s["bottom"] > 1080:
            print(f"[{lang}] slide {s['id']}: content reaches y={s['bottom']} (canvas 1080); it will run past the slide in PowerPoint")
    out = d / "deck.pptx"
    with tempfile.TemporaryDirectory() as tdir:
        lp = Path(tdir) / "layout.json"
        lp.write_text(json.dumps(layout))
        env = dict(os.environ)
        if os.environ.get("PPTXGENJS_NODE_MODULES"):
            env["NODE_PATH"] = os.environ["PPTXGENJS_NODE_MODULES"]
        subprocess.run(["node", str(HERE / "build_pptx.js"), str(lp), str(out), str(d)], check=True, env=env)
    subprocess.run(["zip", "-dq", str(out), "*/"], capture_output=True)  # drop folder entries; some cloud services reject them
    print(f"[{lang}] {out}")


if __name__ == "__main__":
    for lang in sys.argv[1:] or ["ar", "en"]:
        build(lang)
