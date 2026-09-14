from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math
import re

LEVEL_ORDER = [
    "خدمات المجاورة السكنية",
    "خدمات المنطقة السكنية",
    "خدمات الحي السكني",
    "خدمات القطاع",
]

RULES_FILE = Path(__file__).with_name("public_facilities_rules.json")


@dataclass
class FacilityResult:
    level: str
    service: str
    population_standard: float | None
    facility_count: int | None
    min_site_area_m2: float | None
    per_capita_area_m2: float | None
    service_radius_m: Any
    height: Any
    building_ratio: Any
    note_type: str
    required_land_area_m2: float | None
    required_gfa_m2: float | None
    land_formula_source: str | None
    gfa_formula_source: str | None


def density_category(project_area_m2: float, population: int) -> tuple[float, str]:
    if project_area_m2 <= 0:
        raise ValueError("مساحة المشروع يجب أن تكون أكبر من صفر.")
    if population < 0:
        raise ValueError("عدد السكان لا يمكن أن يكون سالبًا.")

    hectares = project_area_m2 / 10000.0
    density = population / hectares

    # Exact Excel logic from Project information!B5:B6
    if density < 70:
        category = "منخفضة"
    elif density <= 220:
        category = "متوسطة"
    else:
        category = "مرتفعة"
    return density, category


def _note_type(service_name: str) -> str:
    if "❷" in service_name:
        return "اختياري"
    if "❸" in service_name:
        return "غير مطلوب داخل المشروع التطويري"
    if "❶" in service_name:
        return "مطلوب - ويمكن توفيره ضمن مبنى متعدد الاستعمال"
    return "مطلوب"


def _clean_service_name(service_name: str) -> str:
    return (
        service_name.replace("❶", "")
        .replace("❷", "")
        .replace("❸", "")
        .strip()
    )


def load_rules() -> dict[str, list[dict[str, Any]]]:
    with RULES_FILE.open("r", encoding="utf-8") as f:
        return json.load(f)


def determine_max_level(rules: list[dict[str, Any]], population: int) -> str | None:
    """
    Business rule requested by the user:
    a higher service level becomes active only AFTER its starting population is exceeded.

    Example: if the first neighborhood/district row starts at 10,000,
    population = 10,000 does NOT activate that higher level.
    """
    starts: dict[str, float] = {}
    for level in LEVEL_ORDER:
        vals = [
            float(r["population_standard"])
            for r in rules
            if r["level"] == level and isinstance(r.get("population_standard"), (int, float))
        ]
        if vals:
            starts[level] = min(vals)

    active = None
    for level in LEVEL_ORDER:
        start = starts.get(level)
        if start is not None and population > start:
            active = level
        else:
            break
    return active


def _facility_count(
    land_area_m2: float | None,
    min_site_area_m2: float | None,
) -> int | None:
    """
    Number of facilities is derived from the Excel-calculated total land demand
    divided by the Excel minimum site area, rounded up.

    This keeps the facility count tied to the workbook's own land-area equations
    instead of inventing a second population-capacity equation that is not present
    in the source workbook.
    """
    if land_area_m2 is None or not min_site_area_m2 or min_site_area_m2 <= 0:
        return None
    return max(1, int(math.ceil(land_area_m2 / min_site_area_m2)))


def _eval_land_formula(rule: dict[str, Any], population: int) -> float | None:
    """
    Evaluates the exact calculation pattern stored in the source Excel J column.

    Supported source patterns in the workbook:
      Project population * per-capita area
      Project population * per-capita area * 0.86
      Minimum site area (e.g. =E7)
    """
    formula = rule.get("land_formula")
    per_capita = rule.get("per_capita_area_m2")
    min_site = rule.get("min_site_area_m2")

    if formula:
        if "'Project information'!$B$3" in formula and re.search(r"\*F\d+", formula):
            if not isinstance(per_capita, (int, float)):
                return None
            value = population * float(per_capita)
            # Preserve any numeric multiplier after the F-cell, such as *0.86.
            m = re.search(r"\*F\d+((?:\*[0-9.]+)*)$", formula.replace(" ", ""))
            if m and m.group(1):
                for factor in re.findall(r"\*([0-9.]+)", m.group(1)):
                    value *= float(factor)
            return value

        if re.fullmatch(r"=E\d+", formula.replace(" ", "")):
            return float(min_site) if isinstance(min_site, (int, float)) else None

    # Safety fallback: only used if the source row has no readable formula.
    if isinstance(per_capita, (int, float)):
        return population * float(per_capita)
    if isinstance(min_site, (int, float)):
        return float(min_site)
    return None


def _eval_gfa_formula(rule: dict[str, Any], land_area: float | None) -> float | None:
    """
    Evaluates the exact source Excel K-column formula whenever it exists.
    Examples:
       =J11*0.45*2
       =J15*1.45
       =J6*2

    If Excel intentionally has no GFA formula and its stored value is zero,
    GFA is reported as zero. If the workbook has a fixed numeric value in K,
    that fixed source value is retained.
    """
    if land_area is None:
        return None

    formula = rule.get("gfa_formula")
    if formula:
        normalized = formula.replace(" ", "")
        if re.fullmatch(r"=J\d+", normalized):
            return land_area

        m = re.fullmatch(r"=J\d+((?:\*[0-9.]+)+)", normalized)
        if m:
            value = land_area
            for factor in re.findall(r"\*([0-9.]+)", m.group(1)):
                value *= float(factor)
            return value

    cached = rule.get("gfa_cached")
    if isinstance(cached, (int, float)):
        # Zero rows (parks/open spaces) intentionally have no GFA formula.
        # Non-zero fixed values are also preserved exactly as entered in Excel.
        return float(cached)
    return None


def review_public_facilities(
    project_area_m2: float,
    population: int,
) -> dict[str, Any]:
    all_rules = load_rules()
    density, category = density_category(project_area_m2, population)
    rules = all_rules[category]
    max_level = determine_max_level(rules, population)

    base = {
        "project_area_m2": float(project_area_m2),
        "project_area_ha": round(float(project_area_m2) / 10000.0, 4),
        "population": int(population),
        "density_person_per_ha": round(density, 2),
        "density_category": category,
        "max_service_level": max_level,
        "required": [],
        "optional": [],
        "not_required_in_development": [],
    }
    if max_level is None:
        return base

    max_idx = LEVEL_ORDER.index(max_level)
    allowed_levels = set(LEVEL_ORDER[:max_idx + 1])

    required: list[FacilityResult] = []
    optional: list[FacilityResult] = []
    not_required: list[FacilityResult] = []

    for rule in rules:
        if rule["level"] not in allowed_levels:
            continue

        standard = rule.get("population_standard")
        # Same strict threshold rule for the individual facility.
        if isinstance(standard, (int, float)) and population <= standard:
            continue

        land = _eval_land_formula(rule, population)
        gfa = _eval_gfa_formula(rule, land)
        note = _note_type(rule["service"])

        item = FacilityResult(
            level=rule["level"],
            service=_clean_service_name(rule["service"]),
            population_standard=float(standard) if isinstance(standard, (int, float)) else None,
            facility_count=_facility_count(
                land,
                float(rule["min_site_area_m2"]) if isinstance(rule.get("min_site_area_m2"), (int, float)) else None,
            ),
            min_site_area_m2=float(rule["min_site_area_m2"]) if isinstance(rule.get("min_site_area_m2"), (int, float)) else None,
            per_capita_area_m2=float(rule["per_capita_area_m2"]) if isinstance(rule.get("per_capita_area_m2"), (int, float)) else None,
            service_radius_m=rule.get("service_radius_m"),
            height=rule.get("height"),
            building_ratio=rule.get("building_ratio"),
            note_type=note,
            required_land_area_m2=round(land, 2) if land is not None else None,
            required_gfa_m2=round(gfa, 2) if gfa is not None else None,
            land_formula_source=rule.get("land_formula"),
            gfa_formula_source=rule.get("gfa_formula"),
        )

        if note == "اختياري":
            optional.append(item)
        elif note == "غير مطلوب داخل المشروع التطويري":
            not_required.append(item)
        else:
            required.append(item)

    base["required"] = [asdict(x) for x in required]
    base["optional"] = [asdict(x) for x in optional]
    base["not_required_in_development"] = [asdict(x) for x in not_required]

    base["totals"] = {
        "required_facilities_count": sum(x.facility_count or 0 for x in required),
        "required_land_area_m2": round(sum(x.required_land_area_m2 or 0 for x in required), 2),
        "required_gfa_m2": round(sum(x.required_gfa_m2 or 0 for x in required), 2),
    }
    return base


def _fmt(value: float | int | None, decimals: int = 0) -> str:
    if value is None:
        return "-"
    return f"{value:,.{decimals}f}"


def build_customer_reply(result: dict[str, Any]) -> str:
    level = result["max_service_level"] or "أقل من مستوى المجاورة السكنية"

    lines = [
        "بناءً على بيانات المشروع المدخلة ومعايير الخدمات العامة:",
        f"- مساحة المشروع: {_fmt(result['project_area_m2'])} م² ({_fmt(result['project_area_ha'], 2)} هكتار)",
        f"- عدد السكان: {result['population']:,} نسمة",
        f"- الكثافة السكانية: {_fmt(result['density_person_per_ha'], 2)} فرد/هكتار",
        f"- فئة الكثافة: {result['density_category']}",
        f"- أعلى مستوى خدمي منطبق: {level}",
        "",
        "الخدمات العامة المطلوبة:",
    ]

    if not result["required"]:
        lines.append("- لا توجد خدمات إلزامية منطبقة وفق الحدود السكانية الحالية.")
    else:
        for lvl in LEVEL_ORDER:
            group = [x for x in result["required"] if x["level"] == lvl]
            if not group:
                continue
            lines.append(f"\n{lvl}:")
            for x in group:
                count = x["facility_count"] if x["facility_count"] is not None else "-"
                lines.append(
                    f"- {x['service']}: عدد المرافق {count}، "
                    f"مساحة الأرض المطلوبة {_fmt(x['required_land_area_m2'])} م²، "
                    f"GFA {_fmt(x['required_gfa_m2'])} م²."
                )
                if "متعدد الاستعمال" in x["note_type"]:
                    lines.append("  يمكن توفير هذه الخدمة ضمن المباني متعددة الاستعمال.")

    if result["optional"]:
        lines.append("\nخدمات اختيارية:")
        for x in result["optional"]:
            lines.append(
                f"- {x['service']} ({x['level']}): "
                f"عدد تقديري {x['facility_count'] or '-'}، "
                f"أرض {_fmt(x['required_land_area_m2'])} م²."
            )

    if result.get("totals"):
        t = result["totals"]
        lines.extend([
            "",
            "إجمالي الخدمات الإلزامية المحتسبة:",
            f"- إجمالي عدد المرافق: {t['required_facilities_count']:,}",
            f"- إجمالي مساحة الأرض: {_fmt(t['required_land_area_m2'])} م²",
            f"- إجمالي GFA: {_fmt(t['required_gfa_m2'])} م²",
        ])

    lines.extend([
        "",
        "ملاحظة: لا يتم احتساب خدمات المستويات الأعلى من المستوى السكاني المنطبق على المشروع. "
        "كما أن الخدمات المعلّمة في المرجع بأنها غير مطلوبة ضمن المشاريع التطويرية لا يتم طلبها من المتعامل."
    ])
    return "\n".join(lines)


def result_rows(result: dict[str, Any], include_optional: bool = True) -> list[dict[str, Any]]:
    """Rows ready for a Streamlit dataframe."""
    items = list(result["required"])
    if include_optional:
        items += list(result["optional"])

    return [{
        "مستوى الخدمة": x["level"],
        "الخدمة": x["service"],
        "الحالة": x["note_type"],
        "معيار السكان/مرفق": x["population_standard"],
        "عدد المرافق": x["facility_count"],
        "مساحة الأرض المطلوبة (م²)": x["required_land_area_m2"],
        "GFA (م²)": x["required_gfa_m2"],
        "نطاق الخدمة (م)": x["service_radius_m"],
        "الارتفاع": x["height"],
    } for x in items]
