"""Export the Markdown paper draft to formatted Word and PDF files."""

from __future__ import annotations

import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    LongTable,
    PageTemplate,
    Paragraph,
    PageBreak,
    Spacer,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "PAPER_DRAFT.md"
DOCX_OUTPUT = ROOT / "docs" / "PAPER_DRAFT.docx"
PDF_OUTPUT = ROOT / "docs" / "PAPER_DRAFT.pdf"


def clean_inline(text: str) -> str:
    """Remove Markdown inline notation while preserving its visible text."""
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\*([^*]+)\*", r"\1", text)
    return text.replace("\\|", "|").strip()


def parse_blocks(markdown: str) -> list[tuple[str, object]]:
    """Parse the limited Markdown constructs used by the paper."""
    lines = markdown.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            blocks.append(("paragraph", " ".join(part.strip() for part in paragraph)))
            paragraph.clear()

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped == "---":
            flush_paragraph()
            blocks.append(("rule", ""))
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading:
            flush_paragraph()
            blocks.append((f"h{len(heading.group(1))}", clean_inline(heading.group(2))))
            index += 1
            continue
        if stripped.startswith("|") and index + 1 < len(lines):
            separator = lines[index + 1].strip()
            if separator.startswith("|") and re.fullmatch(r"[\|\-:\s]+", separator):
                flush_paragraph()
                rows: list[list[str]] = []
                while index < len(lines) and lines[index].strip().startswith("|"):
                    current = lines[index].strip()
                    if not re.fullmatch(r"[\|\-:\s]+", current):
                        rows.append(
                            [clean_inline(cell) for cell in current.strip("|").split("|")]
                        )
                    index += 1
                blocks.append(("table", rows))
                continue
        list_match = re.match(r"^(\d+\.\s+|-\s+)(.+)$", stripped)
        if list_match:
            flush_paragraph()
            kind = "number" if list_match.group(1)[0].isdigit() else "bullet"
            blocks.append((kind, list_match.group(2)))
            index += 1
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    return blocks


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    tc_pr.append(shading)


def add_docx_inline(paragraph, text: str) -> None:
    """Add text with basic Markdown bold/italic formatting."""
    cursor = 0
    pattern = re.compile(r"(\*\*.+?\*\*|\*.+?\*|`.+?`)")
    for match in pattern.finditer(text):
        if match.start() > cursor:
            paragraph.add_run(text[cursor : match.start()])
        token = match.group(0)
        run = paragraph.add_run(token.strip("*`"))
        run.bold = token.startswith("**")
        run.italic = token.startswith("*") and not token.startswith("**")
        if token.startswith("`"):
            run.font.name = "Consolas"
        cursor = match.end()
    if cursor < len(text):
        paragraph.add_run(text[cursor:])


def export_docx(blocks: list[tuple[str, object]]) -> None:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)

    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.08

    for style_name, size in (("Title", 17), ("Heading 1", 14), ("Heading 2", 12)):
        style = document.styles[style_name]
        style.font.name = "Times New Roman"
        style.font.size = Pt(size)
        style.font.bold = True

    title_seen = False
    for kind, value in blocks:
        if kind == "h1" and not title_seen:
            paragraph = document.add_paragraph(style="Title")
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            paragraph.add_run(str(value))
            title_seen = True
        elif kind.startswith("h"):
            level = min(int(kind[1:]), 2)
            document.add_heading(str(value), level=level)
        elif kind == "paragraph":
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            add_docx_inline(paragraph, str(value))
        elif kind in {"number", "bullet"}:
            style = "List Number" if kind == "number" else "List Bullet"
            paragraph = document.add_paragraph(style=style)
            add_docx_inline(paragraph, str(value))
        elif kind == "table":
            rows = value
            if not rows:
                continue
            table = document.add_table(rows=len(rows), cols=len(rows[0]))
            table.style = "Table Grid"
            table.autofit = True
            for row_index, row in enumerate(rows):
                for column_index, text in enumerate(row):
                    cell = table.cell(row_index, column_index)
                    cell.text = text
                    for paragraph in cell.paragraphs:
                        paragraph.paragraph_format.space_after = Pt(1)
                        for run in paragraph.runs:
                            run.font.name = "Times New Roman"
                            run.font.size = Pt(8)
                            run.bold = row_index == 0
                    if row_index == 0:
                        set_cell_shading(cell, "D9EAF7")
        elif kind == "rule":
            document.add_paragraph()

    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("IPS Paper Draft")
    run.font.name = "Times New Roman"
    run.font.size = Pt(8)

    document.save(DOCX_OUTPUT)


def register_pdf_font() -> str:
    candidates = [
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            pdfmetrics.registerFont(TTFont("PaperFont", str(candidate)))
            return "PaperFont"
    return "Helvetica"


def pdf_escape(text: str) -> str:
    text = clean_inline(text)
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def export_pdf(blocks: list[tuple[str, object]]) -> None:
    font = register_pdf_font()
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "PaperBody",
        parent=styles["BodyText"],
        fontName=font,
        fontSize=9,
        leading=11,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
    title = ParagraphStyle(
        "PaperTitle",
        parent=body,
        fontSize=16,
        leading=19,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    h1 = ParagraphStyle(
        "PaperH1",
        parent=body,
        fontSize=13,
        leading=15,
        spaceBefore=8,
        spaceAfter=5,
    )
    h2 = ParagraphStyle(
        "PaperH2",
        parent=body,
        fontSize=11,
        leading=13,
        spaceBefore=6,
        spaceAfter=4,
    )
    list_style = ParagraphStyle(
        "PaperList", parent=body, leftIndent=12, firstLineIndent=-7, alignment=TA_LEFT
    )
    table_style = ParagraphStyle(
        "PaperTable", parent=body, fontSize=6.6, leading=8, alignment=TA_LEFT
    )

    def draw_page(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font, 8)
        canvas.drawString(18 * mm, 10 * mm, "IPS Paper Draft")
        canvas.drawRightString(192 * mm, 10 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(PDF_OUTPUT),
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        title="Staged Domain Adaptation of a Causal-Transformer VIO Backbone",
        author="IPS Project",
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="paper")
    doc.addPageTemplates([PageTemplate(id="main", frames=frame, onPage=draw_page)])

    story = []
    title_seen = False
    number = 1
    for kind, value in blocks:
        if kind == "h1" and not title_seen:
            story.append(Paragraph(pdf_escape(str(value)), title))
            title_seen = True
        elif kind == "h1":
            story.append(Paragraph(pdf_escape(str(value)), h1))
        elif kind.startswith("h"):
            story.append(Paragraph(pdf_escape(str(value)), h2))
        elif kind == "paragraph":
            story.append(Paragraph(pdf_escape(str(value)), body))
        elif kind in {"number", "bullet"}:
            marker = f"{number}." if kind == "number" else "•"
            story.append(Paragraph(f"{marker} {pdf_escape(str(value))}", list_style))
            number = number + 1 if kind == "number" else number
        elif kind == "table":
            rows = value
            if not rows:
                continue
            data = [
                [Paragraph(pdf_escape(cell), table_style) for cell in row] for row in rows
            ]
            available = doc.width
            column_widths = [available / len(rows[0])] * len(rows[0])
            table = LongTable(data, colWidths=column_widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#777777")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 2.5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 2.5),
                        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 5)])
        elif kind == "rule":
            story.append(Spacer(1, 4))

    doc.build(story)


def main() -> None:
    blocks = parse_blocks(SOURCE.read_text(encoding="utf-8"))
    export_docx(blocks)
    export_pdf(blocks)
    print(f"Created: {DOCX_OUTPUT}")
    print(f"Created: {PDF_OUTPUT}")


if __name__ == "__main__":
    main()
