#!/usr/bin/env python3
"""Export bilingual IPS journey PDFs (Arabic + English) with system diagrams.

Usage (from IPS root, with reportlab + arabic-reshaper + python-bidi + pillow):
  python scripts/export_ips_journey_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from arabic_reshaper import reshape
from bidi.algorithm import get_display
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_RIGHT, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
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
    """Shape Arabic for ReportLab (visual order)."""
    return get_display(reshape(text))


def build_styles(body: str, bold: str, *, rtl: bool) -> dict:
    align = TA_RIGHT if rtl else TA_LEFT
    justify = TA_JUSTIFY
    styles = getSampleStyleSheet()
    out = {
        "title": ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontName=bold,
            fontSize=18,
            leading=24,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=styles["Normal"],
            fontName=body,
            fontSize=11,
            leading=15,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "h1",
            parent=styles["Heading1"],
            fontName=bold,
            fontSize=14,
            leading=20,
            alignment=align,
            textColor=ACCENT,
            spaceBefore=14,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "h2",
            parent=styles["Heading2"],
            fontName=bold,
            fontSize=12,
            leading=16,
            alignment=align,
            textColor=colors.HexColor("#1e293b"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "body": ParagraphStyle(
            "body",
            parent=styles["Normal"],
            fontName=body,
            fontSize=10,
            leading=15,
            alignment=justify if not rtl else TA_RIGHT,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=styles["Normal"],
            fontName=body,
            fontSize=10,
            leading=14,
            alignment=align,
            leftIndent=12 if not rtl else 0,
            rightIndent=12 if rtl else 0,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=styles["Normal"],
            fontName=body,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=MUTED,
            spaceBefore=4,
            spaceAfter=12,
        ),
        "note": ParagraphStyle(
            "note",
            parent=styles["Normal"],
            fontName=body,
            fontSize=9,
            leading=13,
            alignment=align,
            textColor=WARN,
            backColor=colors.HexColor("#fffbeb"),
            borderPadding=6,
            spaceBefore=6,
            spaceAfter=10,
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
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.white,
        ),
        "td": ParagraphStyle(
            "td",
            parent=styles["Normal"],
            fontName=body,
            fontSize=8.5,
            leading=11,
            alignment=TA_CENTER,
        ),
    }
    return out


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
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 1), (-1, -1), LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return t


def build_english(body: str, bold: str, out_path: Path) -> None:
    styles = build_styles(body, bold, rtl=False)
    story = []

    story.append(P("IPS Project Journey", styles["title"], rtl=False))
    story.append(
        P(
            "From the original VIFT paper to our Phase 0–6 pipeline\n"
            "Indoor Positioning System · Learning-based Visual–Inertial Odometry",
            styles["subtitle"],
            rtl=False,
        )
    )

    story.append(P("1. The original paper (VIFT) — what it is", styles["h1"], rtl=False))
    story.append(
        P(
            "VIFT (Kurt et al., 2024) is a learning-based Visual–Inertial Odometry (VIO) method. "
            "It takes a sliding window of camera frames and IMU measurements, encodes them into "
            "latent features, and uses a causal Transformer to predict relative 6-DoF poses "
            "(rotation + translation). Poses are accumulated into a trajectory.",
            styles["body"],
            rtl=False,
        )
    )
    story.append(P("Key idea of the original paper:", styles["h2"], rtl=False))
    for b in [
        "• Visual + IMU encoder (Visual-Selective-VIO) → latent features",
        "• Causal Transformer fusion → relative pose between frames",
        "• Strong results on KITTI (driving): mean t_rel ≈ 1.96%",
        "• Original training recipe: freeze the encoder, train only the fusion on cached latents",
    ]:
        story.append(P(b, styles["bullet"], rtl=False))

    story.append(P("2. Limitations of the original paper (why we need IPS)", styles["h1"], rtl=False))
    story.append(
        P(
            "VIFT works well on KITTI (cars, mostly forward/planar motion). It was not designed "
            "or evaluated for indoor positioning with a phone/handheld device. Main gaps:",
            styles["body"],
            rtl=False,
        )
    )
    for b in [
        "• Single domain: KITTI driving only — not EuRoC (aerial) or TUM-VI (handheld indoor)",
        "• Frozen encoder: the visual features stay tuned to the original domain",
        "• No cross-domain adaptation study (source → target transfer)",
        "• No hybrid geometric backend (IMU preintegration + BA) like Adaptive VIO / classical VIO",
        "• Evaluation on KITTI uses t_rel/r_rel — not directly comparable to EuRoC/TUM ATE (meters)",
        "• Absolute accuracy on EuRoC remains far from classical systems (ORB-SLAM3, VINS-Mono ~0.05–0.2 m)",
    ]:
        story.append(P(b, styles["bullet"], rtl=False))
    story.append(
        P(
            "Our goal is not to invent a new network, but to study: how far can this backbone "
            "transfer from aerial → indoor, with fair evaluation, toward a practical IPS stack.",
            styles["note"],
            rtl=False,
        )
    )

    story.append(P("3. Our solution — staged pipeline (Phases 0–6)", styles["h1"], rtl=False))
    story.append(
        P(
            "We keep the VIFT backbone and add a staged domain-adaptation + evaluation + backend roadmap.",
            styles["body"],
            rtl=False,
        )
    )
    add_image(
        story,
        PIPELINE_PNG,
        "Figure 1 — Full IPS pipeline (Phases 0 → 6)",
        styles,
        rtl=False,
    )

    story.append(P("Phase-by-phase map", styles["h2"], rtl=False))
    story.append(
        make_table(
            ["Phase", "What we did", "Problem it solves", "Key result"],
            [
                [
                    "0",
                    "Setup: VIFT fork, env, pretrained encoders",
                    "Reproducible baseline",
                    "Ready codebase",
                ],
                [
                    "1–2",
                    "Latent cache + fusion train on EuRoC (frozen encoder)",
                    "Apply VIFT recipe to EuRoC",
                    "EuRoC test ATE ≈ 27.59 m (origin)",
                ],
                [
                    "3",
                    "Joint encoder + fusion fine-tune on EuRoC raw images",
                    "Frozen encoder too weak on new domain",
                    "EuRoC test ATE ≈ 16.71 m (−39%)",
                ],
                [
                    "4",
                    "Adapt P3 → TUM-VI (handheld indoor)",
                    "Need indoor / phone-like motion",
                    "slides1: 27.77 → 2.70 m; EuRoC kept ~15.25 m",
                ],
                [
                    "5",
                    "Fair metrics (origin/SE3/Sim3) + multi-horizon loss + clean split",
                    "Origin ATE not peer-comparable; scale issues",
                    "TUM-FT SE3 baseline ≈ 4.35 m; P5 train in progress",
                ],
                [
                    "6-A",
                    "IMU fusion scaffold (ΔR blend + scale cue)",
                    "No geometric correction yet",
                    "Code ready; not BA / not SOTA claim",
                ],
                [
                    "6-B/C",
                    "Real preintegration + sliding-window BA (planned)",
                    "Need Adaptive/SelfVIO-level accuracy",
                    "Future work (weeks)",
                ],
            ],
            styles,
            [18 * mm, 48 * mm, 48 * mm, 46 * mm],
        )
    )
    story.append(Spacer(1, 8))

    story.append(P("4. Model architecture (frontend → eval → backend)", styles["h1"], rtl=False))
    add_image(
        story,
        ARCH_PNG,
        "Figure 2 — Architecture: Deep VIO frontend, training, evaluation, Phase 6 backend",
        styles,
        rtl=False,
    )
    for b in [
        "• Frontend (P3–P5): images + IMU → Encoder → Causal Transformer → relative 6-DoF pose",
        "• Training (P5): MultiHorizonPoseLoss (Smooth-L1 translation + geodesic rotation + multi-step SE3 consistency)",
        "• Evaluation (P5): report origin / SE(3) / Sim(3) ATE separately; also path length & RPE",
        "• Backend (P6-A): offline post-process of frozen poses with simplified IMU fusion",
    ]:
        story.append(P(b, styles["bullet"], rtl=False))

    story.append(P("5. Checkpoint chain (what each famous checkpoint is)", styles["h1"], rtl=False))
    story.append(
        make_table(
            ["Checkpoint", "Epochs", "Meaning", "Use for"],
            [
                [
                    "KITTI 200 ep",
                    "200",
                    "VIFT paper replication (driving)",
                    "Paper baseline only — NOT for EuRoC/TUM path",
                ],
                [
                    "P3 last",
                    "50",
                    "EuRoC joint fine-tune",
                    "Source before TUM adaptation",
                ],
                [
                    "epoch_024 (TUM-FT)",
                    "25",
                    "TUM-VI domain adaptation from P3",
                    "Warm-start for Phase 5; production indoor ckpt",
                ],
                [
                    "Phase 5 (50/100)",
                    "50→100",
                    "EuRoC refresh with new loss/metrics",
                    "Frontend for Phase 6 after SE3+scale check",
                ],
            ],
            styles,
            [40 * mm, 22 * mm, 55 * mm, 43 * mm],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Important: the 200-epoch checkpoint is a separate KITTI branch. "
            "The main IPS chain is: encoders → P2 → P3 → epoch_024 → Phase 5 → Phase 6.",
            styles["note"],
            rtl=False,
        )
    )

    story.append(P("6. Current results snapshot (honest)", styles["h1"], rtl=False))
    story.append(
        make_table(
            ["Setting", "Metric", "Value", "Note"],
            [
                ["KITTI replication", "mean t_rel", "2.10%", "paper 1.96%"],
                ["P2 EuRoC (frozen enc.)", "origin ATE", "27.59 m", "test MH_05+V2_02"],
                ["P3 EuRoC joint", "origin ATE", "16.71 m", "better domain fit"],
                ["TUM-FT / epoch_024", "origin ATE EuRoC", "15.25 m", "no catastrophic forgetting"],
                ["TUM-FT / epoch_024", "SE3 ATE EuRoC", "4.35 m", "fairer peer metric"],
                ["TUM-FT / epoch_024", "est path MH_05", "2.8 m vs GT 161 m", "severe scale collapse"],
                ["Phase 5 @50", "SE3 ATE", "7.77 m", "worse SE3 but path recovered (112 m)"],
                ["SelfVIO / Adaptive †", "SE3 ATE", "~0.19 / 0.11 m", "need hybrid BA — Phase 6-C"],
            ],
            styles,
            [42 * mm, 32 * mm, 38 * mm, 48 * mm],
        )
    )
    story.append(Spacer(1, 6))
    story.append(
        P(
            "Reading the numbers: epoch_024 looks better on SE3 (4.35) but the estimated path "
            "almost collapses (2.8 m). Phase 5 @50 recovers path length (~112 m) at the cost of "
            "higher SE3. Continue Phase 5 to 100, then pick the frontend with best scale+SE3 "
            "trade-off for Phase 6.",
            styles["note"],
            rtl=False,
        )
    )

    story.append(P("7. Phase 6 — geometric backend plan", styles["h1"], rtl=False))
    add_image(
        story,
        P6_PNG,
        "Figure 3 — Phase 6-A offline flow (scaffold, not full BA)",
        styles,
        rtl=False,
    )
    for b in [
        "• 6-A (done in code): bias-free IMU ΔR blend + optional scale cue + light xyz smooth",
        "• 6-B (days): proper preintegration with bias + gravity",
        "• 6-C (weeks): sliding-window factor graph / BA — path to SelfVIO/Adaptive-level claims",
        "• Honesty: 6-A alone will not reach ~0.1 m SE3; do not claim SOTA from the scaffold",
    ]:
        story.append(P(b, styles["bullet"], rtl=False))

    story.append(P("8. Where we are now & next step", styles["h1"], rtl=False))
    for b in [
        "• Done: VIFT replication, EuRoC joint FT, TUM adaptation, Phase 5 protocol + code, Phase 6-A scaffold",
        "• In progress: Phase 5 training to epoch 100 (monitor est_path_len + SE3 together)",
        "• Next: choose best frontend checkpoint → run Phase 6-A (prefer rotation-only first) → decide 6-B/C",
        "• Paper stance: transfer + fair eval study; absolute EuRoC SOTA is future hybrid work",
    ]:
        story.append(P(b, styles["bullet"], rtl=False))

    story.append(Spacer(1, 16))
    story.append(
        P(
            "IPS · docs/SYSTEM_DIAGRAM.md · docs/PHASE5.md · docs/PHASE6.md · diagrams in docs/assets/canva/",
            styles["footer"],
            rtl=False,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="IPS Project Journey (English)",
        author="IPS",
    )
    doc.build(story)


def build_arabic(body: str, bold: str, out_path: Path) -> None:
    styles = build_styles(body, bold, rtl=True)
    story = []
    rtl = True

    story.append(P("رحلة مشروع IPS — من الورقة الأصلية إلى النظام الحالي", styles["title"], rtl=rtl))
    story.append(
        P(
            "من ورقة VIFT إلى مراحلنا 0–6\n"
            "نظام التموضع الداخلي · تقدير الحركة البصري–القصوري بالتعلم العميق",
            styles["subtitle"],
            rtl=rtl,
        )
    )

    story.append(P("١. الورقة الأصلية (VIFT) — ما هي؟", styles["h1"], rtl=rtl))
    story.append(
        P(
            "VIFT (Kurt وآخرون، ٢٠٢٤) طريقة تعلم عميق لتقدير الحركة البصري–القصوري (VIO). "
            "تأخذ نافذة منزلقة من صور الكاميرا وقراءات IMU، وتحوّلها إلى تمثيلات كامنة (latents)، "
            "ثم يستخدم Transformer سببي للتنبؤ بالوضعية النسبية بست درجات حرية (دوران + إزاحة). "
            "تُراكم الوضعيات لتكوين المسار.",
            styles["body"],
            rtl=rtl,
        )
    )
    story.append(P("فكرة الورقة الأصلية:", styles["h2"], rtl=rtl))
    for b in [
        "• مشفّر صور + IMU → ميزات كامنة",
        "• دمج عبر Causal Transformer → وضعية نسبية بين الإطارات",
        "• نتائج قوية على KITTI (قيادة سيارات): خطأ نسبي ≈ ١٫٩٦٪",
        "• وصفة التدريب الأصلية: تجميد المشفر وتدريب الدمج فقط على latents مخزّنة",
    ]:
        story.append(P(b, styles["bullet"], rtl=rtl))

    story.append(P("٢. عيوب / حدود الورقة الأصلية (لماذا نحتاج IPS؟)", styles["h1"], rtl=rtl))
    story.append(
        P(
            "VIFT ناجح على KITTI (حركة أمامية شبه مستوية للسيارة)، لكنه لم يُصمَّم أو يُقيَّم "
            "للتموضع الداخلي بهاتف أو جهاز محمول باليد. أهم الثغرات:",
            styles["body"],
            rtl=rtl,
        )
    )
    for b in [
        "• مجال واحد فقط: KITTI — بدون EuRoC (جوي) أو TUM-VI (داخلي محمول)",
        "• مشفر مجمّد: الميزات البصرية تبقى على مجال التدريب الأصلي",
        "• لا دراسة تكيّف بين المجالات (نقل من مصدر → هدف)",
        "• لا خلفية هندسية هجينة (تكامل IMU مسبق + BA) كما في Adaptive VIO أو الأنظمة الكلاسيكية",
        "• مقاييس KITTI (t_rel/r_rel) ليست مباشرة كمقارنة مع ATE بالمتر على EuRoC/TUM",
        "• الدقة المطلقة على EuRoC بعيدة عن ORB-SLAM3 / VINS-Mono (~٠٫٠٥–٠٫٢ م)",
    ]:
        story.append(P(b, styles["bullet"], rtl=rtl))
    story.append(
        P(
            "هدفنا ليس ابتكار شبكة جديدة، بل دراسة: إلى أي مدى يمكن نقل هذا العمود الفقري "
            "من الجوي → الداخلي، بتقييم عادل، نحو نظام IPS عملي.",
            styles["note"],
            rtl=rtl,
        )
    )

    story.append(P("٣. حلولنا — خط أنابيب مرحلي (المراحل ٠–٦)", styles["h1"], rtl=rtl))
    story.append(
        P(
            "نحتفظ بعمود VIFT ونضيف تكيّفًا مرحليًا للمجال + بروتوكول تقييم + خلفية هندسية مخطط لها.",
            styles["body"],
            rtl=rtl,
        )
    )
    add_image(
        story,
        PIPELINE_PNG,
        "شكل ١ — خط أنابيب IPS الكامل (المراحل ٠ → ٦)",
        styles,
        rtl=rtl,
    )

    story.append(P("خريطة المراحل", styles["h2"], rtl=rtl))
    # Keep table cells in English short labels + Arabic via reshape for headers/cells
    headers = [ar(h) for h in ["المرحلة", "ماذا فعلنا", "المشكلة التي تحلها", "أبرز نتيجة"]]
    rows_ar = [
        [ar("٠"), ar("إعداد: تفريع VIFT والبيئة والمشفرات"), ar("خط أساس قابل للتكرار"), ar("كود جاهز")],
        [
            ar("١–٢"),
            ar("تخزين latents + تدريب الدمج على EuRoC (مشفر مجمّد)"),
            ar("تطبيق وصفة VIFT على EuRoC"),
            ar("ATE ≈ ٢٧٫٥٩ م (origin)"),
        ],
        [
            ar("٣"),
            ar("ضبط مشترك للمشفر والدمج على صور EuRoC"),
            ar("المشفر المجمّد ضعيف على مجال جديد"),
            ar("ATE ≈ ١٦٫٧١ م (تحسّن ~٣٩٪)"),
        ],
        [
            ar("٤"),
            ar("نقل P3 → TUM-VI (داخلي محمول)"),
            ar("حاجة لحركة داخلية شبيهة بالهاتف"),
            ar("slides1: ٢٧٫٧٧ → ٢٫٧٠ م؛ EuRoC محفوظ"),
        ],
        [
            ar("٥"),
            ar("مقاييس عادلة + خسارة هندسية متعددة الأفق + تقسيم نظيف"),
            ar("origin غير قابل للمقارنة العادلة؛ مشاكل مقياس"),
            ar("أساس SE3 ≈ ٤٫٣٥ م؛ تدريب جارٍ"),
        ],
        [
            ar("٦-أ"),
            ar("سكافولد دمج IMU (خلط دوران + إشارة مقياس)"),
            ar("لا تصحيح هندسي بعد"),
            ar("كود جاهز؛ ليس BA وليس SOTA"),
        ],
        [
            ar("٦-ب/ج"),
            ar("تكامل مسبق حقيقي + BA (مخطط)"),
            ar("للوصول لدقة Adaptive/SelfVIO"),
            ar("عمل مستقبلي (أسابيع)"),
        ],
    ]
    story.append(make_table(headers, rows_ar, styles, [18 * mm, 48 * mm, 48 * mm, 46 * mm]))
    story.append(Spacer(1, 8))

    story.append(P("٤. معمارية النموذج (الواجهة → التقييم → الخلفية)", styles["h1"], rtl=rtl))
    add_image(
        story,
        ARCH_PNG,
        "شكل ٢ — المعمارية: واجهة VIO العميقة، التدريب، التقييم، وخلفية المرحلة ٦",
        styles,
        rtl=rtl,
    )
    for b in [
        "• الواجهة (٣–٥): صور + IMU → مشفر → Causal Transformer → وضعية نسبية ٦ درجات",
        "• التدريب (٥): MultiHorizonPoseLoss (إزاحة Smooth-L1 + دوران جيوديسي + اتساق SE3)",
        "• التقييم (٥): نبلغ origin / SE(3) / Sim(3) بشكل منفصل + طول المسار و RPE",
        "• الخلفية (٦-أ): معالجة لاحقة أوفلاين للوضعيات المجمّدة مع دمج IMU مبسّط",
    ]:
        story.append(P(b, styles["bullet"], rtl=rtl))

    story.append(P("٥. سلسلة الـ Checkpoints — كل واحد لأي شيء؟", styles["h1"], rtl=rtl))
    headers2 = [ar(h) for h in ["الـ Checkpoint", "العصور", "المعنى", "يُستخدم لـ"]]
    rows2 = [
        [
            ar("KITTI ٢٠٠ عصر"),
            ar("٢٠٠"),
            ar("تكرار ورقة VIFT (قيادة)"),
            ar("مرجع الورقة فقط — ليس لمسار EuRoC/TUM"),
        ],
        [
            ar("P3 last"),
            ar("٥٠"),
            ar("ضبط مشترك على EuRoC"),
            ar("المصدر قبل التكيّف على TUM"),
        ],
        [
            ar("epoch_024 (TUM-FT)"),
            ar("٢٥"),
            ar("تكيّف على TUM-VI من P3"),
            ar("بداية Phase 5؛ إنتاج داخلي"),
        ],
        [
            ar("Phase 5 (٥٠/١٠٠)"),
            ar("٥٠→١٠٠"),
            ar("تحديث EuRoC بخسارة ومقاييس جديدة"),
            ar("واجهة لـ Phase 6 بعد فحص المقياس و SE3"),
        ],
    ]
    story.append(make_table(headers2, rows2, styles, [40 * mm, 22 * mm, 55 * mm, 43 * mm]))
    story.append(Spacer(1, 6))
    story.append(
        P(
            "مهم: الـ ٢٠٠ عصر فرع KITTI منفصل. السلسلة الرئيسية لـ IPS هي: "
            "مشفرات → P2 → P3 → epoch_024 → Phase 5 → Phase 6.",
            styles["note"],
            rtl=rtl,
        )
    )

    story.append(P("٦. لقطة نتائج حالية (بصدق)", styles["h1"], rtl=rtl))
    headers3 = [ar(h) for h in ["الإعداد", "المقياس", "القيمة", "ملاحظة"]]
    rows3 = [
        [ar("تكرار KITTI"), ar("mean t_rel"), ar("٢٫١٠٪"), ar("الورقة ١٫٩٦٪")],
        [ar("P2 EuRoC مشفر مجمّد"), ar("origin ATE"), ar("٢٧٫٥٩ م"), ar("اختبار MH_05+V2_02")],
        [ar("P3 ضبط مشترك"), ar("origin ATE"), ar("١٦٫٧١ م"), ar("ملاءمة أفضل للمجال")],
        [ar("TUM-FT / epoch_024"), ar("origin EuRoC"), ar("١٥٫٢٥ م"), ar("لا نسيان كارثي")],
        [ar("TUM-FT / epoch_024"), ar("SE3 EuRoC"), ar("٤٫٣٥ م"), ar("مقياس أدق للمقارنة")],
        [ar("TUM-FT / epoch_024"), ar("طول مسار MH_05"), ar("٢٫٨ م مقابل ١٦١ م"), ar("انهيار مقياس شديد")],
        [ar("Phase 5 @٥٠"), ar("SE3 ATE"), ar("٧٫٧٧ م"), ar("SE3 أسوأ لكن المسار عاد (~١١٢ م)")],
        [ar("SelfVIO / Adaptive †"), ar("SE3 ATE"), ar("~٠٫١٩ / ٠٫١١ م"), ar("يحتاج BA — المرحلة ٦-ج")],
    ]
    story.append(make_table(headers3, rows3, styles, [42 * mm, 32 * mm, 38 * mm, 48 * mm]))
    story.append(Spacer(1, 6))
    story.append(
        P(
            "قراءة الأرقام: epoch_024 يبدو أفضل على SE3 (٤٫٣٥) لكن المسار المقدّر ينهار (٢٫٨ م). "
            "Phase 5 عند ٥٠ يسترجع طول المسار (~١١٢ م) مقابل SE3 أعلى. أكمل Phase 5 إلى ١٠٠، "
            "ثم اختر الواجهة ذات أفضل توازن مقياس+SE3 لـ Phase 6.",
            styles["note"],
            rtl=rtl,
        )
    )

    story.append(P("٧. المرحلة ٦ — خطة الخلفية الهندسية", styles["h1"], rtl=rtl))
    add_image(
        story,
        P6_PNG,
        "شكل ٣ — مسار المرحلة ٦-أ أوفلاين (سكافولد، ليس BA كامل)",
        styles,
        rtl=rtl,
    )
    for b in [
        "• ٦-أ (موجود بالكود): خلط دوران IMU بدون bias + إشارة مقياس اختيارية + تنعيم خفيف",
        "• ٦-ب (أيام): تكامل مسبق صحيح مع bias والجاذبية",
        "• ٦-ج (أسابيع): نافذة منزلقة / BA — الطريق لمستوى SelfVIO/Adaptive",
        "• صدق: ٦-أ وحده لن يصل ~٠٫١ م SE3؛ لا تدّعِ SOTA من السكافولد",
    ]:
        story.append(P(b, styles["bullet"], rtl=rtl))

    story.append(P("٨. أين نحن الآن والخطوة التالية", styles["h1"], rtl=rtl))
    for b in [
        "• تم: تكرار VIFT، ضبط EuRoC، تكيّف TUM، بروتوكول وكود Phase 5، سكافولد Phase 6-A",
        "• جارٍ: تدريب Phase 5 إلى العصر ١٠٠ (راقب طول المسار + SE3 معًا)",
        "• التالي: اختر أفضل واجهة → شغّل ٦-أ (يفضّل دوران فقط أولًا) → قرّر ٦-ب/ج",
        "• موقف الورقة: دراسة نقل وتقييم عادل؛ SOTA على EuRoC عمل هجين مستقبلي",
    ]:
        story.append(P(b, styles["bullet"], rtl=rtl))

    story.append(Spacer(1, 16))
    story.append(
        P(
            "IPS · docs/SYSTEM_DIAGRAM.md · docs/PHASE5.md · docs/PHASE6.md · الرسوم في docs/assets/canva/",
            styles["footer"],
            rtl=rtl,
        )
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="رحلة مشروع IPS (عربي)",
        author="IPS",
    )
    doc.build(story)


def main() -> int:
    body, bold = register_fonts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    en = OUT_DIR / "IPS_JOURNEY_EN.pdf"
    ar_path = OUT_DIR / "IPS_JOURNEY_AR.pdf"
    for p in (PIPELINE_PNG, ARCH_PNG, P6_PNG):
        if not p.is_file():
            raise FileNotFoundError(f"Missing diagram PNG: {p}")
    build_english(body, bold, en)
    build_arabic(body, bold, ar_path)
    print(f"Wrote {en}")
    print(f"Wrote {ar_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
