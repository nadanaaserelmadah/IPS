"""Build the thesis-defense decks in docs/presentations/{ar,en} into a standalone HTML, a PDF and a speaker-notes file.

Each deck is a deck.json (slide order, fonts) plus one HTML file per slide in slides/ (1920x1080 canvas, inline styles).
Figures are read from docs/figures.

  python scripts/build_presentations.py              # both decks: deck.html, deck.pdf, notes.md
  python scripts/build_presentations.py en           # one deck
  python scripts/build_presentations.py --check      # also report slides whose content runs past the 1920x1080 canvas
"""
import html
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME  # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "docs" / "presentations"
CSS = """
@page { size: 1920px 1080px; margin: 0; }
html, body { margin: 0; padding: 0; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
section { box-sizing: border-box; width: 1920px; height: 1080px; overflow: hidden; position: relative; break-after: page; page-break-after: always; }
section:last-of-type { break-after: auto; page-break-after: auto; }
h1, h2, h3, p, ul, ol { margin: 0; }
img { display: block; }
aside { display: none; }
table { border-collapse: collapse; table-layout: fixed; width: 100%; }
th, td { padding: 0.35em 0.6em; border-bottom: 1px solid #D4D0C4; vertical-align: top; font-weight: inherit; }
th { font-weight: 600; border-bottom: 2px solid #1C2430; }
"""
CHECK_JS = """
<script>
window.addEventListener('load', function () { document.fonts.ready.then(function () {
  var rep = [];
  document.querySelectorAll('section').forEach(function (s, i) {
    var sb = s.getBoundingClientRect(), bottom = 0, right = 0;
    s.querySelectorAll('*').forEach(function (e) {
      var r = e.getBoundingClientRect();
      if (r.width === 0 && r.height === 0) return;
      bottom = Math.max(bottom, r.bottom - sb.top); right = Math.max(right, r.right - sb.left);
    });
    rep.push({n: i + 1, id: s.id, bottom: Math.round(bottom), right: Math.round(right)});
  });
  var fonts = Array.from(document.fonts).filter(function (f) { return f.status === 'loaded'; }).map(function (f) { return f.family; });
  var pre = document.createElement('pre'); pre.id = 'rep';
  pre.textContent = JSON.stringify({fonts: fonts, rep: rep}); document.body.appendChild(pre);
}); });
</script>
"""


def arrow_svg(m):
    kind, style = m.group(1), m.group(2)
    w = int(re.search(r"width:(\d+)px", style).group(1))
    h = int(re.search(r"height:(\d+)px", style).group(1))
    col = re.search(r"background:(#[0-9A-Fa-f]{3,8})", style).group(1)
    pts = [(0, .28 * h), (.6 * w, .28 * h), (.6 * w, 0), (w, .5 * h), (.6 * w, h), (.6 * w, .72 * h), (0, .72 * h)]
    if kind == "arrow-left":
        pts = [(w - x, y) for x, y in pts]
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    d = "left" if kind == "arrow-left" else "right"
    return f'<svg data-arrow="{d}" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="flex:none;align-self:center"><polygon points="{p}" fill="{col}"/></svg>'


def plain(s):
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def build(lang, check=False):
    d = ROOT / lang
    deck = json.loads((d / "deck.json").read_text())
    sections, notes = [], []
    for n, sid in enumerate(deck["order"], 1):
        t = (d / "slides" / f"{sid}.html").read_text()
        t = t.replace("../../../figures/", "../../figures/")
        t = re.sub(r'<x-shape kind="(arrow-left|arrow-right)" style="([^"]*)"></x-shape>', arrow_svg, t)
        am = re.search(r"<aside>(.*?)</aside>", t, re.S)
        tm = re.search(r"<h[12][^>]*>(.*?)</h[12]>", t, re.S)
        notes.append((n, sid, plain(tm.group(1)) if tm else sid, plain(am.group(1)) if am else ""))
        sections.append(t.strip())
    css = CSS + ("h1, h2, h3, p, th, td { direction: rtl; }\n[dir=ltr] { unicode-bidi: isolate; }\n" if lang == "ar" else "")  # text only: the flex/grid/table layout stays as designed
    links = "".join(f'<link rel="stylesheet" href="{f["href"]}">' for f in deck["faces"].values() if "href" in f)
    page = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><title>{html.escape(deck["title"])}</title>'
            f'{links}<style>{css}</style></head><body>{"".join(sections)}{CHECK_JS if check else ""}</body></html>')
    out_html, out_pdf = d / "deck.html", d / "deck.pdf"
    out_html.write_text(page)
    (d / "notes.md").write_text(f"# {deck['title']}: speaker notes\n\n" + "\n\n".join(
        f"## Slide {n}: {title} (`{sid}`)\n\n{note or '(no notes)'}" for n, sid, title, note in notes) + "\n")
    if check:
        dom = subprocess.run([CHROME, "--headless", "--disable-gpu", "--virtual-time-budget=20000", "--dump-dom",
                              out_html.as_uri()], check=True, capture_output=True, text=True).stdout
        rep = json.loads(html.unescape(re.search(r'<pre id="rep">(.*?)</pre>', dom, re.S).group(1)))
        print(f"[{lang}] fonts loaded: {', '.join(sorted(set(rep['fonts']))) or 'none (fallback fonts in use)'}")
        for r in rep["rep"]:
            if r["bottom"] > 1000 or r["right"] > 1920:
                print(f"[{lang}] slide {r['n']} ({r['id']}): content reaches y={r['bottom']}, x={r['right']} (canvas 1080 x 1920)")
        out_html.write_text(page.replace(CHECK_JS, ""))
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=20000",
                    f"--print-to-pdf={out_pdf}", out_html.as_uri()], check=True, capture_output=True)
    pages = len(re.findall(rb"/Type\s*/Page[^s]", out_pdf.read_bytes()))
    print(f"[{lang}] {len(deck['order'])} slides -> {out_pdf} ({pages} pages)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    for lang in args or ["ar", "en"]:
        build(lang, check="--check" in sys.argv)


if __name__ == "__main__":
    main()
