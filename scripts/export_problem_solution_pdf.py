"""Export docs/PROBLEM_SOLUTION_PAPER_DRAFT.md to PDF (Arabic part RTL, figures embedded).

Markdown -> HTML with the `markdown` package, then Chrome headless prints the PDF.
  python scripts/export_problem_solution_pdf.py
"""
import base64
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "PROBLEM_SOLUTION_PAPER_DRAFT.md"
OUT = SRC.with_suffix(".pdf")
CSS = ROOT / "scripts" / "paper_draft.css"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


LIST = re.compile(r"^\s*([-*]|\d+\.) ")


def commonmark_lists(md: str) -> str:
    """Python-Markdown needs a blank line before a list and 4-space nesting; CommonMark does not."""
    out = []
    for line in md.split("\n"):
        line = re.sub(r"^ {2,3}(?=[-*] |\d+\. )", "    ", line)
        prev = out[-1] if out else ""
        if LIST.match(line) and prev.strip() and not LIST.match(prev) and not prev.startswith(" "):
            out.append("")
        out.append(line)
    return "\n".join(out)


def to_html(md: str) -> str:
    md = re.sub(r"\A---\n.*?\n---\n", "", md, flags=re.S)  # YAML front matter
    md = commonmark_lists(md)
    md = md.replace("\\newpage", '<div class="newpage"></div>')
    parts = re.split(r"^::: \{dir=rtl\}\n(.*?)^:::\n", md, flags=re.S | re.M)
    conv = lambda s: markdown.markdown(s, extensions=["tables", "sane_lists"])  # noqa: E731
    html = []
    for i, p in enumerate(parts):  # odd indices are the RTL blocks
        html.append(f'<div dir="rtl">{conv(p)}</div>' if i % 2 else conv(p))
    body = "\n".join(html)

    def embed(m):
        src = SRC.parent / m.group(2)
        data = base64.b64encode(src.read_bytes()).decode()
        return f'<figure><img alt="{m.group(1)}" src="data:image/png;base64,{data}"><figcaption>{m.group(1)}</figcaption></figure>'

    body = re.sub(r'<p><img alt="([^"]*)" src="([^"]+\.png)" ?/?></p>', embed, body)
    return f'<!doctype html><html><head><meta charset="utf-8"><style>{CSS.read_text()}</style></head><body>{body}</body></html>'


def main():
    html = to_html(SRC.read_text())
    with tempfile.TemporaryDirectory() as d:
        page = Path(d) / "draft.html"
        page.write_text(html)
        subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                        f"--print-to-pdf={OUT}", page.as_uri()], check=True, capture_output=True)
    print(OUT)


if __name__ == "__main__":
    sys.exit(main())
