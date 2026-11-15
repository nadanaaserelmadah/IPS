"""Build the short Arabic research summary docs/summary/SUMMARY_AR.md into docs/summary/SUMMARY_AR.pdf.

Same pipeline and stylesheet as the book (Python-Markdown -> HTML -> Chrome headless), without cover or TOC.
  python scripts/build_summary_pdf.py [OTHER.md]   (default: docs/summary/SUMMARY_AR.md)
"""
import base64
import re
import subprocess
import sys
from pathlib import Path

import markdown

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_book_pdf import CHROME, CSS, commonmark_lists  # noqa: E402

SRC = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "summary" / "SUMMARY_AR.md"
OUT = SRC.with_suffix(".pdf")
EXTRA_CSS = """
body { font-size: 11pt; line-height: 1.75; }
h1 { break-before: auto; font-size: 20pt; text-align: center; border-bottom: none; margin-bottom: 4px; }
h1 + p, h1 + p + p { text-align: center; }
h2 { margin-top: 18px; }
figure img { max-height: 95mm; }
"""


def main():
    md = commonmark_lists(SRC.read_text())
    html = markdown.markdown(md, extensions=["tables", "sane_lists", "md_in_html", "attr_list"])

    def embed(m):
        data = base64.b64encode((SRC.parent / m.group(2)).read_bytes()).decode()
        return f'<figure><img alt="{m.group(1)}" src="data:image/png;base64,{data}"><figcaption>{m.group(1)}</figcaption></figure>'

    html = re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.png)" ?/?></p>', embed, html)
    page = SRC.with_suffix(".html")
    page.write_text(f'<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>IPS summary</title>'
                    f"<style>{CSS.read_text()}{EXTRA_CSS}</style></head><body>{html}</body></html>")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=15000",
                    f"--print-to-pdf={OUT}", page.as_uri()], check=True, capture_output=True)
    page.unlink()
    print(OUT)


if __name__ == "__main__":
    main()
