from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
except Exception:
    arabic_reshaper = None
    get_display = None


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
]


def _font_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _register_fonts():
    regular = _font_path(FONT_CANDIDATES)
    bold = _font_path(BOLD_CANDIDATES) or regular
    if regular:
        try:
            pdfmetrics.registerFont(TTFont("PFArabic", regular))
            pdfmetrics.registerFont(TTFont("PFArabicBold", bold))
            return "PFArabic", "PFArabicBold"
        except Exception:
            pass
    return "Helvetica", "Helvetica-Bold"


def _ar(value: Any) -> str:
    s = "-" if value is None else str(value)
    if arabic_reshaper and get_display and any("\u0600" <= ch <= "\u06ff" for ch in s):
        try:
            return get_display(arabic_reshaper.reshape(s))
        except Exception:
            return s
    return s


def _fmt(v, decimals=0):
    if v is None:
        return "-"
    try:
        return f"{float(v):,.{decimals}f}"
    except Exception:
        return str(v)


def build_comparison_pdf(report: dict[str, Any], title: str = "تقرير مراجعة الخدمات العامة") -> bytes:
    font, bold = _register_fonts()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        rightMargin=12 * mm,
        leftMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Public Facilities Review Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "PFTitle", parent=styles["Title"], fontName=bold, fontSize=18, leading=24,
        alignment=TA_RIGHT, spaceAfter=8,
    )
    body_style = ParagraphStyle(
        "PFBody", parent=styles["BodyText"], fontName=font, fontSize=9, leading=13,
        alignment=TA_RIGHT,
    )
    small_style = ParagraphStyle(
        "PFSmall", parent=body_style, fontSize=7.3, leading=9,
    )

    project = report.get("project", {})
    summary = report.get("summary", {})
    story = [
        Paragraph(_ar(title), title_style),
        Paragraph(_ar("Public Facilities Compliance Review"), ParagraphStyle(
            "PFSub", parent=body_style, fontName=bold, fontSize=11, textColor=colors.HexColor("#475467")
        )),
        Spacer(1, 5 * mm),
    ]

    meta_data = [
        [_ar("مساحة المشروع"), _ar(f"{_fmt(project.get('project_area_m2'))} م²"),
         _ar("عدد السكان"), _ar(_fmt(project.get('population'))),
         _ar("فئة الكثافة"), _ar(project.get("density_category"))],
        [_ar("الكثافة"), _ar(f"{_fmt(project.get('density_person_per_ha'), 2)} فرد/هكتار"),
         _ar("أعلى مستوى خدمي"), _ar(project.get("max_service_level")),
         _ar("نسبة الاستيفاء"), _ar(f"{summary.get('compliance_percent', 0)}%")],
    ]
    meta = Table(meta_data, colWidths=[28*mm, 45*mm, 28*mm, 45*mm, 28*mm, 55*mm])
    meta.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,-1), font),
        ("FONTNAME", (0,0), (-1,-1), font),
        ("BACKGROUND", (0,0), (-1,-1), colors.HexColor("#F2F4F7")),
        ("BOX", (0,0), (-1,-1), 0.5, colors.HexColor("#D0D5DD")),
        ("INNERGRID", (0,0), (-1,-1), 0.25, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "RIGHT"),
        ("TOPPADDING", (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    story += [meta, Spacer(1, 5*mm)]

    story.append(Paragraph(_ar(
        f"الخدمات المطلوبة: {summary.get('required_services', 0)} | "
        f"مستوفى: {summary.get('compliant_services', 0)} | "
        f"يوجد عجز: {summary.get('deficit_services', 0)}"
    ), body_style))
    story.append(Spacer(1, 4*mm))

    headers = [
        "الحالة", "المستوى", "الخدمة", "المطلوب", "المقدم", "عجز العدد",
        "الأرض المطلوبة", "الأرض المقدمة", "عجز الأرض", "GFA المطلوب", "GFA المقدم", "ملاحظات"
    ]
    data = [[Paragraph(_ar(h), small_style) for h in headers]]
    row_statuses = []
    for x in report.get("comparisons", []):
        ok = x.get("status") == "مستوفى"
        row_statuses.append(ok)
        status = "✓ مستوفى" if ok else "✗ عجز"
        vals = [
            status,
            x.get("level"),
            x.get("service"),
            _fmt(x.get("required_count")),
            _fmt(x.get("provided_count")),
            _fmt(x.get("count_deficit")),
            _fmt(x.get("required_land_area_m2")),
            _fmt(x.get("provided_land_area_m2")),
            _fmt(x.get("land_deficit_m2")),
            _fmt(x.get("required_gfa_m2")),
            _fmt(x.get("provided_gfa_m2")),
            x.get("notes"),
        ]
        data.append([Paragraph(_ar(v), small_style) for v in vals])

    widths = [20, 29, 37, 17, 17, 18, 23, 23, 22, 22, 22, 48]
    table = Table(data, repeatRows=1, colWidths=[w*mm for w in widths])
    ts = [
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#101828")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("FONTNAME", (0,0), (-1,0), bold),
        ("FONTNAME", (0,1), (-1,-1), font),
        ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#D0D5DD")),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN", (0,0), (-1,-1), "RIGHT"),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]
    for i, ok in enumerate(row_statuses, start=1):
        bg = colors.HexColor("#ECFDF3") if ok else colors.HexColor("#FEF3F2")
        fg = colors.HexColor("#027A48") if ok else colors.HexColor("#B42318")
        ts.extend([
            ("BACKGROUND", (0,i), (-1,i), bg),
            ("TEXTCOLOR", (0,i), (0,i), fg),
            ("FONTNAME", (0,i), (0,i), bold),
        ])
    table.setStyle(TableStyle(ts))
    story.append(table)

    story += [Spacer(1, 5*mm), Paragraph(_ar(
        "ملاحظة: تتم المقارنة آليًا مع الخدمات الإلزامية المحسوبة للمشروع. "
        "الخدمات الاختيارية والبنود غير المطلوبة داخل المشروع التطويري لا تُعامل كعجز إلزامي. "
        "هذا التقرير أداة مراجعة أولية ولا يمثل اعتمادًا نهائيًا."
    ), small_style)]

    doc.build(story)
    return buffer.getvalue()
