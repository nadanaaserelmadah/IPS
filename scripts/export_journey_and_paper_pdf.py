#!/usr/bin/env python3
"""Export JOURNEY_AND_PAPER bilingual PDFs (updated through Phase 5/6).

Usage (from IPS root):
  py -3 scripts/export_journey_and_paper_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from arabic_reshaper import reshape
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets" / "canva"
OUT_DIR = ROOT / "docs"

PIPELINE_PNG = ASSETS / "01_ips_pipeline.png"
ARCH_PNG = ASSETS / "02_ips_architecture.png"
P6_PNG = ASSETS / "03_phase6_flow.png"

ACCENT = colors.HexColor("#0f766e")
MUTED = colors.HexColor("#475569")
LIGHT = colors.HexColor("#f1f5f9")
WARN = colors.HexColor("#b45309")


def register_fonts() -> tuple[str, str]:
    candidates = [
        (Path(r"C:\Windows\Fonts\arial.ttf"), Path(r"C:\Windows\Fonts\arialbd.ttf")),
        (Path(r"C:\Windows\Fonts\tahoma.ttf"), Path(r"C:\Windows\Fonts\tahomabd.ttf")),
        (Path(r"C:\Windows\Fonts\segoeui.ttf"), Path(r"C:\Windows\Fonts\segoeuib.ttf")),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("IPSBody", str(regular)))
            pdfmetrics.registerFont(TTFont("IPSBold", str(bold)))
            return "IPSBody", "IPSBold"
    raise FileNotFoundError("No suitable TTF font found under C:\\Windows\\Fonts")


def ar(text: str) -> str:
    return get_display(reshape(text))


def build_styles(body: str, bold: str, *, rtl: bool) -> dict:
    align = TA_RIGHT if rtl else TA_LEFT
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontName=bold,
            fontSize=17,
            leading=22,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=styles["Normal"],
            fontName=body,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=12,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=styles["Heading1"],
            fontName=bold,
            fontSize=13,
            leading=18,
            alignment=align,
            textColor=ACCENT,
            spaceBefore=12,
            spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=styles["Heading2"],
            fontName=bold,
            fontSize=11,
            leading=15,
            alignment=align,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontName=body,
            fontSize=9.5,
            leading=14,
            alignment=TA_JUSTIFY if not rtl else TA_RIGHT,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=styles["Normal"],
            fontName=body,
            fontSize=9.5,
            leading=13,
            alignment=align,
            leftIndent=10 if not rtl else 0,
            rightIndent=10 if rtl else 0,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=styles["Normal"],
            fontName=body,
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceBefore=3,
            spaceAfter=8,
        ),
        "note": ParagraphStyle(
            "note",
            parent=styles["Normal"],
            fontName=body,
            fontSize=8.5,
            leading=12,
            alignment=align,
            textColor=WARN,
            backColor=colors.HexColor("#fffbeb"),
            borderPadding=5,
            spaceBefore=4,
            spaceAfter=8,
        ),
        "footer": ParagraphStyle(
            "footer",
            parent=styles["Normal"],
            fontName=body,
            fontSize=8,
            alignment=TA_CENTER,
            textColor=MUTED,
        ),
        "th": ParagraphStyle(
            "th",
            parent=styles["Normal"],
            fontName=bold,
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "td",
            parent=styles["Normal"],
            fontName=body,
            fontSize=7.5,
            leading=10,
            alignment=TA_CENTER,
        ),
    }


def P(text: str, style, *, rtl: bool):
    return Paragraph(ar(text) if rtl else text, style)


def add_image(story, path: Path, caption: str, styles, *, rtl: bool, width_mm: float = 170):
    if not path.is_file():
        story.append(P(f"[Missing figure: {path.name}]", styles["note"], rtl=rtl))
        return
    img = Image(str(path), width=width_mm * mm, height=(width_mm * 9 / 16) * mm)
    story.append(KeepTogether([img, P(caption, styles["caption"], rtl=rtl)]))


def make_table(headers, rows, styles, col_widths):
    data = [[Paragraph(h, styles["th"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), styles["td"]) for c in row])
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def build_english(body: str, bold: str, out_path: Path) -> None:
    styles = build_styles(body, bold, rtl=False)
    story = []

    story.append(P("IPS Journey & Paper Snapshot", styles["title"], rtl=False))
    story.append(
        P(
            "Updated 2026-07-26 · Phases 0–6 complete (P6 as negative ablations)\n"
            "Source: docs/JOURNEY_AND_PAPER_EN.md · RESULTS.md",
            styles["subtitle"],
            rtl=False,
        )
    )

    story.append(P("1. Goal", styles["h1"], rtl=False))
    story.append(
        P(
            "Learning-based VIO toward indoor positioning (IPS) with camera + IMU only: "
            "reproduce VIFT on KITTI, adapt EuRoC → TUM-VI, refresh frontend (Phase 5), "
            "and test lightweight geometric backends (Phase 6). "
            "Claim: staged domain adaptation + honest evaluation — not EuRoC absolute SOTA.",
            styles["body"],
            rtl=False,
        )
    )

    story.append(P("2. Full pipeline", styles["h1"], rtl=False))
    add_image(story, PIPELINE_PNG, "Figure 1 — IPS pipeline (Phases 0 → 6)", styles, rtl=False)

    story.append(P("Phase map (results filled)", styles["h2"], rtl=False))
    story.append(
        make_table(
            ["Phase", "What", "Key result"],
            [
                ["0", "Setup + pretrained encoders", "Reproducible codebase"],
                ["1–2", "Frozen encoder + EuRoC latents", "Origin ATE 27.59 m (test)"],
                ["3", "Joint EuRoC raw FT", "Origin ATE 16.71 m (−39%)"],
                ["4", "TUM-VI adaptation", "slides1 27.77→2.70 m; EuRoC 15.25 m"],
                ["5 @100", "MultiHorizon + SE3/Sim3 protocol", "Origin 16.56 / SE3 5.76; path recovered"],
                ["6 A–D", "IMU / BA / ORB pose-graph offline", "No mean SE3 win — negative ablations"],
            ],
            styles,
            [22 * mm, 70 * mm, 68 * mm],
        )
    )
    story.append(Spacer(1, 6))

    story.append(P("3. Architecture", styles["h1"], rtl=False))
    add_image(
        story,
        ARCH_PNG,
        "Figure 2 — Frontend / train / eval / Phase 6 backend",
        styles,
        rtl=False,
    )

    story.append(P("4. Headline numbers", styles["h1"], rtl=False))
    story.append(
        make_table(
            ["Setting", "Metric", "Value", "Note"],
            [
                ["KITTI frozen 200 ep", "mean t_rel", "2.10%", "paper 1.96%"],
                ["P2 EuRoC", "origin ATE", "27.59 m", "MH_05+V2_02"],
                ["P3 EuRoC joint", "origin ATE", "16.71 m", "joint FT helps"],
                ["P4 after TUM", "origin ATE EuRoC", "15.25 m", "limited forgetting"],
                ["P4 TUM-FT", "slides1 ATE", "2.70 m", "from 27.77 zero-shot"],
                ["P4 epoch_024", "SE3 / path MH_05", "4.35 m / ~2.8 m", "SE3 strong, scale collapsed"],
                ["P5 @100", "origin / SE3", "16.56 / 5.76 m", "preferred frontend"],
                ["P6 safer loops", "SE3 Δ MH_05", "+0.03 m", "harmless no-op"],
                ["P6 default loops", "SE3 Δ MH_05", "+6.48 m", "hurts — bad scale lock"],
            ],
            styles,
            [40 * mm, 34 * mm, 36 * mm, 50 * mm],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        P(
            "Important: epoch_024 SE3 looks best (4.35) but estimated path collapses (~2.8 vs GT 161). "
            "Phase 5 restores path (~110 m) with SE3 5.76 — use that for reporting / Phase 6 dumps.",
            styles["note"],
            rtl=False,
        )
    )

    story.append(P("5. Phase 6 geometric backends", styles["h1"], rtl=False))
    add_image(story, P6_PNG, "Figure 3 — Phase 6 offline flow (scaffold family)", styles, rtl=False)
    story.append(
        make_table(
            ["Stage", "vs raw SE3", "Ship as better?"],
            [
                ["6-A rot/scale", "worse", "No"],
                ["6-B open-loop gravity", "much worse (~20 m)", "No"],
                ["6-C sliding-window BA", "mixed; mean worse", "No"],
                ["6-D ORB default", "MH_05 much worse", "No"],
                ["6-D safer retune", "≈ flat", "No (no-op)"],
            ],
            styles,
            [50 * mm, 70 * mm, 40 * mm],
        )
    )

    story.append(P("6. vs other papers (honest)", styles["h1"], rtl=False))
    story.append(
        make_table(
            ["Family", "Example †", "Their #", "Ours", "Write as"],
            [
                ["Same backbone", "VIFT KITTI", "1.96% t_rel", "2.10%", "Faithful reproduction"],
                ["Classical", "ORB-SLAM3 / VINS", "~0.04–0.3 m", "~5.8 m SE3", "Upper bound"],
                ["Learning hybrid", "SelfVIO / Adaptive", "~0.1–0.2 m", "~5.8 m SE3", "Not our claim"],
                ["Our transfer", "TUM slides1", "—", "27.8→2.7 m", "Main IPS story"],
            ],
            styles,
            [32 * mm, 40 * mm, 32 * mm, 28 * mm, 28 * mm],
        )
    )

    story.append(P("7. Paper abstract (current)", styles["h1"], rtl=False))
    story.append(
        P(
            "We study staged domain adaptation of a causal-transformer VIO backbone (VIFT) from "
            "aerial (EuRoC) toward handheld indoor (TUM-VI) motion. We reproduce VIFT on KITTI "
            "(2.10% vs 1.96% mean t_rel), show joint EuRoC fine-tuning improves origin ATE "
            "(27.59→16.71 m), and adapt to TUM-VI (slides1 27.77→2.70 m) with limited EuRoC "
            "forgetting (15.25 m). Phase 5 adds fair origin/SE(3)/Sim(3) reporting; @100 epochs "
            "yields 16.56 / 5.76 m with recovered scale. Phase 6 lightweight backends do not "
            "improve mean SE(3). Contribution: transfer + evaluation study, not EuRoC SOTA.",
            styles["body"],
            rtl=False,
        )
    )

    story.append(P("8. Remaining & claims", styles["h1"], rtl=False))
    for b in [
        "• Remaining: KITTI joint t_rel; slides1 after Phase 5; optional classical re-run",
        "• OK to claim: VIFT reproduction; joint FT helps; TUM transfer works; P6 negative",
        "• Do NOT claim: beat ORB-SLAM3 / Adaptive / SelfVIO on absolute EuRoC ATE",
    ]:
        story.append(P(b, styles["bullet"], rtl=False))

    story.append(Spacer(1, 12))
    story.append(
        P(
            "IPS · JOURNEY_AND_PAPER_EN.md · RESULTS.md · PHASE5.md · PHASE6.md · assets/canva/",
            styles["footer"],
            rtl=False,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="IPS Journey & Paper (English)",
        author="IPS",
    )
    doc.build(story)


def build_arabic(body: str, bold: str, out_path: Path) -> None:
    styles = build_styles(body, bold, rtl=True)
    story = []
    rtl = True

    story.append(P("رحلة IPS ومسودة الورقة", styles["title"], rtl=rtl))
    story.append(
        P(
            "محدّث 2026-07-26 · المراحل 0–6 (Phase 6 تجارب سالبة)\n"
            "المصدر: docs/JOURNEY_AND_PAPER_AR.md · RESULTS.md",
            styles["subtitle"],
            rtl=rtl,
        )
    )

    story.append(P("١. الهدف", styles["h1"], rtl=rtl))
    story.append(
        P(
            "مسار VIO بالتعلّم نحو التموضع الداخلي (IPS) بكاميرا وIMU فقط: إعادة إنتاج VIFT على KITTI، "
            "تكيّف EuRoC → TUM-VI، تحديث الواجهة (Phase 5)، وتجربة backends هندسية خفيفة (Phase 6). "
            "الادعاء: تكيّف عبر الدومينات + تقييم صادق — وليس تفوّقاً مطلقاً على EuRoC.",
            styles["body"],
            rtl=rtl,
        )
    )

    story.append(P("٢. خط الأنابيب الكامل", styles["h1"], rtl=rtl))
    add_image(story, PIPELINE_PNG, "شكل ١ — خط أنابيب IPS (المراحل ٠ → ٦)", styles, rtl=rtl)

    story.append(P("خريطة المراحل (نتائج مملوءة)", styles["h2"], rtl=rtl))
    story.append(
        make_table(
            [ar(h) for h in ["المرحلة", "ماذا", "أبرز نتيجة"]],
            [
                [ar("٠"), ar("إعداد + مشفرات مسبقة"), ar("كود قابل للتكرار")],
                [ar("١–٢"), ar("مشفر مجمّد + latents EuRoC"), ar("origin ATE ٢٧٫٥٩ م")],
                [ar("٣"), ar("تدريب مشترك EuRoC"), ar("origin ATE ١٦٫٧١ م (−٣٩٪)")],
                [ar("٤"), ar("تكيّف TUM-VI"), ar("slides1 ٢٧٫٧٧→٢٫٧٠ م؛ EuRoC ١٥٫٢٥ م")],
                [ar("٥ @١٠٠"), ar("MultiHorizon + بروتوكول SE3"), ar("origin ١٦٫٥٦ / SE3 ٥٫٧٦؛ مقياس مستعاد")],
                [ar("٦ أ–د"), ar("IMU / BA / ORB أوفلاين"), ar("لا فوز متوسط SE3 — تجارب سالبة")],
            ],
            styles,
            [22 * mm, 70 * mm, 68 * mm],
        )
    )
    story.append(Spacer(1, 6))

    story.append(P("٣. المعمارية", styles["h1"], rtl=rtl))
    add_image(
        story,
        ARCH_PNG,
        "شكل ٢ — الواجهة / التدريب / التقييم / خلفية المرحلة ٦",
        styles,
        rtl=rtl,
    )

    story.append(P("٤. الأرقام الرئيسية", styles["h1"], rtl=rtl))
    story.append(
        make_table(
            [ar(h) for h in ["الإعداد", "المقياس", "القيمة", "ملاحظة"]],
            [
                [ar("KITTI مجمّد ٢٠٠"), ar("mean t_rel"), ar("٢٫١٠٪"), ar("الورقة ١٫٩٦٪")],
                [ar("P2 EuRoC"), ar("origin ATE"), ar("٢٧٫٥٩ م"), ar("MH_05+V2_02")],
                [ar("P3 مشترك"), ar("origin ATE"), ar("١٦٫٧١ م"), ar("التدريب المشترك يساعد")],
                [ar("P4 بعد TUM"), ar("origin EuRoC"), ar("١٥٫٢٥ م"), ar("نسيان محدود")],
                [ar("P4 TUM-FT"), ar("slides1"), ar("٢٫٧٠ م"), ar("من ٢٧٫٧٧ صفر-لقطة")],
                [ar("P4 epoch_024"), ar("SE3 / مسار MH_05"), ar("٤٫٣٥ / ~٢٫٨ م"), ar("SE3 قوي، مقياس منهار")],
                [ar("P5 @١٠٠"), ar("origin / SE3"), ar("١٦٫٥٦ / ٥٫٧٦ م"), ar("الواجهة المفضّلة")],
                [ar("P6 حلقات آمنة"), ar("Δ SE3 MH_05"), ar("+٠٫٠٣ م"), ar("بلا ضرر ولا فائدة")],
                [ar("P6 حلقات افتراضي"), ar("Δ SE3 MH_05"), ar("+٦٫٤٨ م"), ar("يضر — قفل مقياس خاطئ")],
            ],
            styles,
            [40 * mm, 34 * mm, 36 * mm, 50 * mm],
        )
    )
    story.append(Spacer(1, 4))
    story.append(
        P(
            "مهم: SE3 عند epoch_024 يبدو الأفضل (٤٫٣٥) لكن المسار ينهار (~٢٫٨ مقابل ١٦١). "
            "Phase 5 يستعيد المسار (~١١٠ م) مع SE3 ٥٫٧٦ — هذا الأنسب للنشر وdumps المرحلة ٦.",
            styles["note"],
            rtl=rtl,
        )
    )

    story.append(P("٥. خلفية Phase 6 الهندسية", styles["h1"], rtl=rtl))
    add_image(story, P6_PNG, "شكل ٣ — تدفق المرحلة ٦ أوفلاين", styles, rtl=rtl)
    story.append(
        make_table(
            [ar(h) for h in ["المرحلة", "مقابل SE3 الخام", "تُنشَر كتحسين؟"]],
            [
                [ar("٦-أ دوران/مقياس"), ar("أسوأ"), ar("لا")],
                [ar("٦-ب مفتوح الحلقة"), ar("أسوأ بكثير (~٢٠ م)"), ar("لا")],
                [ar("٦-ج BA نافذة"), ar("مختلط؛ المتوسط أسوأ"), ar("لا")],
                [ar("٦-د ORB افتراضي"), ar("MH_05 أسوأ بكثير"), ar("لا")],
                [ar("٦-د آمن"), ar("≈ ثابت"), ar("لا (بلا أثر)")],
            ],
            styles,
            [50 * mm, 70 * mm, 40 * mm],
        )
    )

    story.append(P("٦. مقابل الأوراق الأخرى (بصدق)", styles["h1"], rtl=rtl))
    story.append(
        make_table(
            [ar(h) for h in ["العائلة", "مثال †", "رقمهم", "عندنا", "الصياغة"]],
            [
                [ar("نفس العمود"), ar("VIFT KITTI"), ar("١٫٩٦٪"), ar("٢٫١٠٪"), ar("إعادة إنتاج أمينة")],
                [ar("كلاسيكي"), ar("ORB / VINS"), ar("~٠٫٠٤–٠٫٣ م"), ar("~٥٫٨ م SE3"), ar("حد علوي")],
                [ar("تعلّم هجين"), ar("SelfVIO / Adaptive"), ar("~٠٫١–٠٫٢ م"), ar("~٥٫٨ م SE3"), ar("ليس ادعاءنا")],
                [ar("نقلنا"), ar("TUM slides1"), ar("—"), ar("٢٧٫٨→٢٫٧ م"), ar("قصة IPS")],
            ],
            styles,
            [32 * mm, 40 * mm, 32 * mm, 28 * mm, 28 * mm],
        )
    )

    story.append(P("٧. ملخص الورقة (الحالي)", styles["h1"], rtl=rtl))
    story.append(
        P(
            "ندرس التكيّف المرحلي لعمود VIO بمحوّل سببي (VIFT) من الجوي (EuRoC) نحو اليدوي الداخلي (TUM-VI). "
            "نعيد إنتاج VIFT على KITTI (٢٫١٠٪ مقابل ١٫٩٦٪)، ونبيّن أن التدريب المشترك على EuRoC يحسّن origin ATE "
            "(٢٧٫٥٩→١٦٫٧١ م)، وأن التكيّف إلى TUM يخفّض slides1 (٢٧٫٧٧→٢٫٧٠ م) مع نسيان EuRoC محدود (١٥٫٢٥ م). "
            "Phase 5 يضيف إبلاغاً عادلاً origin/SE3/Sim3؛ عند ١٠٠ عصر: ١٦٫٥٦ / ٥٫٧٦ م مع استعادة المقياس. "
            "Phase 6 لا يحسّن متوسط SE3. المساهمة: دراسة نقل وتقييم، وليست SOTA على EuRoC.",
            styles["body"],
            rtl=rtl,
        )
    )

    story.append(P("٨. المتبقي والادعاءات", styles["h1"], rtl=rtl))
    for b in [
        "• المتبقي: KITTI joint؛ slides1 بعد Phase 5؛ اختياري كلاسيكي",
        "• مسموح: إعادة إنتاج VIFT؛ التدريب المشترك يساعد؛ نقل TUM ينجح؛ P6 سالب",
        "• ممنوع: ادعاء التفوّق على ORB-SLAM3 / Adaptive / SelfVIO في ATE المطلق",
    ]:
        story.append(P(b, styles["bullet"], rtl=rtl))

    story.append(Spacer(1, 12))
    story.append(
        P(
            "IPS · JOURNEY_AND_PAPER_AR.md · RESULTS.md · PHASE5.md · PHASE6.md · assets/canva/",
            styles["footer"],
            rtl=rtl,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="رحلة IPS ومسودة الورقة (عربي)",
        author="IPS",
    )
    doc.build(story)


def main() -> int:
    body, bold = register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for p in (PIPELINE_PNG, ARCH_PNG, P6_PNG):
        if not p.is_file():
            raise FileNotFoundError(f"Missing diagram PNG: {p}")
    en = OUT_DIR / "JOURNEY_AND_PAPER_EN.pdf"
    ar_path = OUT_DIR / "JOURNEY_AND_PAPER_AR.pdf"
    build_english(body, bold, en)
    build_arabic(body, bold, ar_path)
    print(f"Wrote {en}")
    print(f"Wrote {ar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
