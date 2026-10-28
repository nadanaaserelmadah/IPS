"""Build the Arabic teaching book docs/book/*.md into docs/IPS_BOOK_AR.pdf (RTL, figures embedded, linked TOC).

Markdown -> HTML (Python-Markdown: tables, sane_lists, md_in_html, attr_list, toc), then Chrome headless prints
the PDF. Chapter files are concatenated in name order; STYLE_GUIDE.md is skipped.
  python scripts/build_book_pdf.py [--html]   (--html also keeps docs/book/book.html for inspection)
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
BOOK = ROOT / "docs" / "book"
OUT = ROOT / "docs" / "IPS_BOOK_AR.pdf"
CSS = ROOT / "scripts" / "book.css"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
LIST = re.compile(r"^\s*([-*]|\d+\.) ")


def commonmark_lists(md: str) -> str:
    """Python-Markdown needs a blank line before a list and 4-space nesting; CommonMark does not."""
    out, fence = [], False
    for line in md.split("\n"):
        if line.lstrip().startswith("```"):
            fence = not fence
        if not fence:
            line = re.sub(r"^ {2,3}(?=[-*] |\d+\. )", "    ", line)
            prev = out[-1] if out else ""
            if LIST.match(line) and prev.strip() and not LIST.match(prev) and not prev.startswith(" ") \
                    and not prev.lstrip().startswith("<"):
                out.append("")
        out.append(line)
    return "\n".join(out)


def embed_images(html: str) -> str:
    def repl(m):
        alt, src = m.group(1), m.group(2)
        path = (BOOK / src).resolve()
        data = base64.b64encode(path.read_bytes()).decode()
        return f'<figure><img alt="{alt}" src="data:image/png;base64,{data}"><figcaption>{alt}</figcaption></figure>'
    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.png)" ?/?></p>', repl, html)
    return html


def toc(html: str) -> str:
    items = []
    for m in re.finditer(r'<h([12]) id="([^"]+)">(.*?)</h\1>', html):
        level, hid, text = m.groups()
        if 'class="notoc"' in text:
            continue
        items.append(f'<li class="l{level}"><a href="#{hid}">{re.sub("<[^>]+>", "", text)}</a></li>')
    return '<nav class="toc"><h1>المحتويات</h1><ul>' + "\n".join(items) + "</ul></nav>"


def build() -> str:
    files = sorted(p for p in BOOK.glob("[0-9][0-9]_*.md"))
    md_all = "\n\n".join(commonmark_lists(p.read_text()) for p in files)
    conv = markdown.Markdown(extensions=["tables", "sane_lists", "md_in_html", "attr_list", "toc"],
                             extension_configs={"toc": {"slugify": lambda v, s: "s-" + str(abs(hash(v)) % 10**8)}})
    body = embed_images(conv.convert(md_all))
    front, sep, rest = body.partition("<!--TOC-->")
    body = front + toc(rest) + rest if sep else toc(body) + body
    return (f'<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>IPS book</title>'
            f"<style>{CSS.read_text()}</style></head><body>{body}</body></html>")


def main():
    html = build()
    page = BOOK / "book.html"
    page.write_text(html)
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={OUT}", page.as_uri()], check=True, capture_output=True)
    if "--html" not in sys.argv:
        page.unlink()
    print(OUT)


if __name__ == "__main__":
    main()
