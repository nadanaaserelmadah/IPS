"""Generate Arabic PDF: IPS contributions / solution / improvements."""
from __future__ import annotations

from pathlib import Path

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)

OUT = Path(__file__).resolve().parents[1] / "docs" / "IPS_Contributions_AR.pdf"
FONT = "C:/Windows/Fonts/tahoma.ttf"
FONT_B = "C:/Windows/Fonts/tahomabd.ttf"

pdfmetrics.registerFont(TTFont("Tahoma", FONT))
if Path(FONT_B).is_file():
    pdfmetrics.registerFont(TTFont("TahomaBold", FONT_B))
else:
    pdfmetrics.registerFont(TTFont("TahomaBold", FONT))


def ar(text: str) -> str:
    """Reshape + bidi for Arabic in ReportLab Paragraphs."""
    return get_display(arabic_reshaper.reshape(text))


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUT),
        pagesize=A4,
        rightMargin=1.8 * cm,
        leftMargin=1.8 * cm,
        topMargin=1.6 * cm,
        bottomMargin=1.6 * cm,
        title="مساهمات نظام IPS / VIFT",
        author="IPS Project",
    )

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "ARTitle",
        parent=styles["Title"],
        fontName="TahomaBold",
        fontSize=16,
        leading=22,
        alignment=TA_CENTER,
        textColor=HexColor("#1a1a1a"),
        spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "ARH1",
        parent=styles["Heading1"],
        fontName="TahomaBold",
        fontSize=13,
        leading=18,
        alignment=TA_RIGHT,
        textColor=HexColor("#0b3d5c"),
        spaceBefore=14,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "ARH2",
        parent=styles["Heading2"],
        fontName="TahomaBold",
        fontSize=11,
        leading=15,
        alignment=TA_RIGHT,
        textColor=HexColor("#1f5f7a"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "ARBody",
        parent=styles["Normal"],
        fontName="Tahoma",
        fontSize=10,
        leading=15,
        alignment=TA_RIGHT,
        textColor=HexColor("#222222"),
        spaceAfter=4,
    )
    bullet = ParagraphStyle(
        "ARBullet",
        parent=body,
        leftIndent=8,
        rightIndent=12,
        spaceAfter=3,
    )
    note = ParagraphStyle(
        "ARNote",
        parent=body,
        fontSize=9,
        textColor=HexColor("#444444"),
        alignment=TA_CENTER,
        spaceBefore=6,
        spaceAfter=10,
    )
    eng = ParagraphStyle(
        "ENBody",
        parent=styles["Normal"],
        fontName="Tahoma",
        fontSize=9,
        leading=13,
        alignment=TA_CENTER,
        textColor=HexColor("#333333"),
        spaceBefore=4,
        spaceAfter=4,
    )

    def P(text: str, style=body):
        return Paragraph(ar(text), style)

    def B(text: str):
        return Paragraph(ar("• " + text), bullet)

    story = []

    story.append(P("نظام التموضع الداخلي IPS المبني على VIFT", title))
    story.append(
        P(
            "شرح المساهمات · الحل المقترح · أوجه التحسين",
            note,
        )
    )
    story.append(
        Paragraph(
            "Based on project results: KITTI replication, EuRoC Phase 3, TUM-VI Phase 4",
            eng,
        )
    )
    story.append(Spacer(1, 6))

    # ---- 1 ----
    story.append(P("1) السياق: ما المشكلة أصلاً؟", h1))
    story.append(
        P(
            "نهدف إلى بناء نظام تموضع داخلي (Indoor Positioning System) يعتمد على "
            "كاميرا ووحدة قياس بالقصور الذاتي (IMU) — كما في الهاتف — دون الاعتماد على GPS."
        )
    )
    story.append(
        P(
            "الأنظمة القوية تقليدياً هي أنظمة SLAM الكلاسيكية مثل VINS-Mono و ORB-SLAM3: "
            "تتبع نقاط هندسية وتحسين عاملي، وغالباً تحقق أخطاء أقل من متر على EuRoC."
        )
    )
    story.append(
        P(
            "مسار هذا المشروع مختلف: تعلّم عميق للـ Visual-Inertial Odometry (VIO)، "
            "حيث تتعلم الشبكة توقع الحركة النسبية من الصور والـ IMU ثم تُركَّب في مسار."
        )
    )
    story.append(
        P(
            "الأساس الأكاديمي هو ورقة VIFT (Kurt et al., 2024) — Transformer سببي لدمج "
            "الرؤية والـ IMU. الورقة الأصلية ركّزت على KITTI (سيارات في الخارج)، "
            "وليست على التموضع الداخلي أو النقل بين البيئات. هنا تبدأ مساهمة المشروع."
        )
    )

    # ---- 2 ----
    story.append(P("2) ما الحل الذي نقدمه؟", h1))
    story.append(
        P(
            "الحل ليس نظام SLAM كلاسيكي جديد، بل منهج متعدد المراحل فوق عمارة VIFT:"
        )
    )
    story.append(
        P(
            "خط تعلّم عميق للـ VIO + تحسين تدريجي للتدريب + نقل معرفة من حركة طيران "
            "(EuRoC) إلى حركة يد داخلية (TUM-VI)."
        )
    )

    story.append(P("أ) العمارة المشتركة (من VIFT)", h2))
    for line in [
        "كاميرا + IMU",
        "Encoder من Visual-Selective-VIO → ميزات رؤية وميزات IMU",
        "دمج الخصائص (Concat، بُعد 768)",
        "Causal Transformer (PoseTransformer)",
        "توقع حركة نسبية بست درجات حرية (6-DoF)",
        "تكامل المسار ثم التقييم بمقياس ATE أو trel/rrel",
    ]:
        B(line)
        story.append(Spacer(1, 1))

    story.append(P("ب) مراحل المشروع (ما أُضيف على طريقة الاستخدام)", h2))

    story.append(P("Phase 2 — أسلوب VIFT على EuRoC", h2))
    story.append(
        P(
            "Encoder مجمّد، تخزين latents، تدريب الـ fusion فقط. "
            "النتيجة التقريبية على EuRoC حوالي 27 متراً ATE. "
            "المعنى: تطبيق منهج الورقة كما هو على بيانات الدرون غير كافٍ لمسار IPS."
        )
    )

    story.append(P("Phase 3 — التحسين الأول (Joint Fine-tune)", h2))
    story.append(
        P(
            "تدريب Encoder و Fusion معاً على صور EuRoC الخام. "
            "يُجمَّد الـ Encoder أولاً ثم يُفتح بمعدل تعلّم أصغر (حوالي 0.1×). "
            "نتيجة اختبار EuRoC: 16.71 متراً (متوسط MH_05 و V2_02). "
            "المعنى: الـ Encoder يحتاج تكييفاً للبيانات الجديدة وليس فقط الـ Transformer."
        )
    )

    story.append(P("Phase 4 — التحسين الثاني (Domain Adaptation)", h2))
    story.append(
        P(
            "البداية من Checkpoint مرحلة 3، ثم Fine-tune على TUM-VI (حركة يد داخل المباني). "
            "على التسلسل المحجوز slides1: من 27.77 متراً (zero-shot) إلى 2.70 متراً. "
            "بعد التكييف يبقى EuRoC عند 15.25 متراً تقريباً (لا انهيار / لا نسيان كارثي)."
        )
    )

    story.append(P("ج) التحقق من صحة تطبيق VIFT", h2))
    story.append(
        P(
            "على KITTI بمقدار 200 دورة تدريب: متوسط خطأ الانتقال trel حوالي 2.10٪ "
            "مقابل 1.96٪ في الورقة الأصلية. هذا دليل مصداقية للأساس قبل ادعاء التحسينات."
        )
    )

    # ---- 3 ----
    story.append(P("3) المساهمات بصيغة ورقة علمية", h1))

    story.append(P("المساهمة 1 — إعادة الإنتاج وخط الأنابيب", h2))
    story.append(
        P(
            "إخراج VIFT من سياق KITTI وحده إلى منصة قابلة لإعادة الإنتاج تشمل "
            "EuRoC و TUM، مع تدريب وتقييم موحّد وجداول نتائج واضحة."
        )
    )

    story.append(P("المساهمة 2 — تدريب مشترك للـ Encoder والـ Fusion على EuRoC", h2))
    story.append(
        P(
            "إظهار أن الأسلوب بـ Encoder مجمّد (~27 م) يتحسن إلى حوالي 16.7 م "
            "بعد التدريب المشترك. المساهمة منهجية: متى نجمّد الـ Encoder ومتى ندرّبه."
        )
    )

    story.append(P("المساهمة 3 — التكييف عبر المجالات إلى TUM (الأهم لـ IPS)", h2))
    story.append(
        P(
            "التكييف من بيانات جوّية إلى حركة يد داخلية يخفض خطأ المسار على تسلسل "
            "محجوز (slides1) بحوالي عشرة أضعاف، مع الحفاظ تقريباً على أداء EuRoC. "
            "هذه أقوى نقطة لقصة التموضع الداخلي."
        )
    )

    # ---- 4 ----
    story.append(P("4) أوجه التحسين", h1))
    story.append(P("أ) تحسين رقمي", h2))

    data = [
        [
            Paragraph(ar("المقارنة"), ParagraphStyle("th", fontName="TahomaBold", fontSize=8, alignment=TA_CENTER)),
            Paragraph(ar("قبل"), ParagraphStyle("th2", fontName="TahomaBold", fontSize=8, alignment=TA_CENTER)),
            Paragraph(ar("بعد"), ParagraphStyle("th3", fontName="TahomaBold", fontSize=8, alignment=TA_CENTER)),
            Paragraph(ar("المعنى"), ParagraphStyle("th4", fontName="TahomaBold", fontSize=8, alignment=TA_CENTER)),
        ],
        [
            Paragraph(ar("EuRoC Phase2 → 3"), body),
            Paragraph("~27 m", eng),
            Paragraph("16.7 m", eng),
            Paragraph(ar("التدريب المشترك مفيد"), body),
        ],
        [
            Paragraph(ar("TUM slides1"), body),
            Paragraph("27.8 m", eng),
            Paragraph("2.7 m", eng),
            Paragraph(ar("تكييف المجال ناجح"), body),
        ],
        [
            Paragraph(ar("EuRoC بعد TUM"), body),
            Paragraph("16.7 m", eng),
            Paragraph("15.3 m", eng),
            Paragraph(ar("لا نسيان كارثي"), body),
        ],
        [
            Paragraph(ar("KITTI vs ورقة VIFT"), body),
            Paragraph("1.96%", eng),
            Paragraph("2.10%", eng),
            Paragraph(ar("الأساس مطابق تقريباً"), body),
        ],
    ]
    t = Table(data, colWidths=[4.2 * cm, 2.6 * cm, 2.6 * cm, 5.2 * cm])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), HexColor("#e8f1f5")),
                ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(KeepTogether([t, Spacer(1, 8)]))

    story.append(P("ب) تحسين مفهومي", h2))
    B("من معيار واحد (KITTI) إلى سلسلة: KITTI + EuRoC + TUM.")
    B("من نموذج ثابت إلى منهج تكيفي مرحلي: درّب → انقل → كيّف.")
    B("من دقة عامة إلى دقة مرتبطة بـ IPS الداخلي (TUM).")

    # ---- 5 ----
    story.append(P("5) ما الذي ليس ادّعاء المشروع؟", h1))
    story.append(
        P(
            "لا يُدعى التفوق على ORB-SLAM3 أو VINS على EuRoC (نتائجنا حوالي 15 م "
            "وهم غالباً أقل من متر). ولا يُدعى أن المشروع State-of-the-Art عالمياً. "
            "ولا تُقارن أرقام KITTI (نسب مئوية) مباشرة بأرقام EuRoC/TUM (أمتار)."
        )
    )
    story.append(
        P(
            "الادّعاء الصحيح: نبني على VIFT ونوسّعه؛ التدريب المشترك يحسّن EuRoC "
            "مقارنة بالـ latent-only؛ تكييف TUM يحسّن الأداء الداخلي دون تدمير EuRoC."
        )
    )

    # ---- 6 ----
    story.append(P("6) الخلاصة في جملة واحدة", h1))
    story.append(
        P(
            "نقدّم نظام تموضع بصري-قصوري قائم على VIFT، نثبته أولاً بإعادة إنتاج نتائج "
            "KITTI، ثم نوسّعه إلى EuRoC عبر تدريب مشترك للـ Encoder والـ Fusion، ثم "
            "نكيّفه إلى بيئة داخلية (TUM-VI) فنحصل على تحسن كبير على تسلسل محجوز مع "
            "الحفاظ على أداء EuRoC؛ ويكون التحسين موجّهاً لمسار التموضع الداخلي وليس "
            "لمنافسة SLAM الكلاسيكي على EuRoC."
        )
    )

    # ---- English abstract ----
    story.append(PageBreak())
    story.append(Paragraph("English summary (for the paper)", title))
    story.append(Spacer(1, 8))

    en_body = ParagraphStyle(
        "ENFull",
        parent=styles["Normal"],
        fontName="Tahoma",
        fontSize=10,
        leading=14,
        alignment=TA_CENTER,
        spaceAfter=8,
    )
    en_h = ParagraphStyle(
        "ENH",
        parent=en_body,
        fontName="TahomaBold",
        fontSize=11,
        textColor=HexColor("#0b3d5c"),
        spaceBefore=10,
        spaceAfter=4,
    )

    story.append(Paragraph("<b>Contributions</b>", en_h))
    story.append(
        Paragraph(
            "(1) We reproduce VIFT on KITTI with near-paper translational error "
            "(mean trel 2.10% vs 1.96%). (2) We extend the pipeline to EuRoC with "
            "joint encoder–fusion fine-tuning, reducing test ATE from ~27 m to 16.7 m "
            "versus a frozen-encoder baseline. (3) We perform cross-domain adaptation "
            "to TUM-VI, reducing held-out slides1 ATE from 27.8 m to 2.7 m while "
            "preserving EuRoC test accuracy (15.3 m vs 16.7 m).",
            en_body,
        )
    )
    story.append(Paragraph("<b>Solution</b>", en_h))
    story.append(
        Paragraph(
            "A staged learning pipeline on a shared VIFT backbone: latent fusion, "
            "then joint fine-tuning on EuRoC, then target-domain fine-tuning for "
            "indoor handheld motion on TUM-VI.",
            en_body,
        )
    )
    story.append(Paragraph("<b>Improvement focus</b>", en_h))
    story.append(
        Paragraph(
            "Not classical SOTA on EuRoC, but reliable transfer toward indoor "
            "visual–inertial positioning with measurable gains on TUM-VI and limited "
            "catastrophic forgetting.",
            en_body,
        )
    )

    story.append(Spacer(1, 16))
    story.append(
        Paragraph(
            "Source: IPS docs/RESULTS.md · Figures in docs/figures/",
            eng,
        )
    )

    doc.build(story)
    print(f"Wrote: {OUT}")


if __name__ == "__main__":
    main()
