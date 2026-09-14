from __future__ import annotations

import re
from typing import Any

from public_facilities_engine_v2 import review_public_facilities, result_rows


INTENT_KEYWORDS = (
    "الخدمات العامة", "الخدمات المطلوب", "الخدمات المطلوبة", "حساب الخدمات",
    "احسب الخدمات", "احسبلي الخدمات", "احسب لي الخدمات", "مرافق عامة",
    "public facilities", "facility calculation", "calculate facilities",
)

RESET_KEYWORDS = (
    "ابدأ من جديد", "ابدأ حساب جديد", "حساب جديد", "امسح الحساب",
    "reset facilities", "new calculation",
)

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")


def new_flow_state() -> dict[str, Any]:
    return {
        "active": False,
        "awaiting": None,  # area | population | None
        "project_area_m2": None,
        "population": None,
        "last_result": None,
    }


def normalize_number_text(text: str) -> str:
    return text.translate(ARABIC_DIGITS).replace("،", ",")


def _numbers(text: str) -> list[float]:
    text = normalize_number_text(text)
    vals: list[float] = []
    for raw in re.findall(r"(?<!\w)(\d+(?:[.,]\d+)?)", text):
        # Thousands separators are allowed, but decimal comma is also tolerated.
        cleaned = raw.replace(",", "") if raw.count(",") == 1 and len(raw.split(",")[-1]) == 3 else raw.replace(",", ".")
        try:
            vals.append(float(cleaned))
        except ValueError:
            pass
    return vals


def detect_public_facilities_intent(text: str) -> bool:
    t = normalize_number_text(text).lower()
    return any(k in t for k in INTENT_KEYWORDS)


def detect_reset(text: str) -> bool:
    t = normalize_number_text(text).lower()
    return any(k in t for k in RESET_KEYWORDS)


def extract_area_m2(text: str) -> float | None:
    """Extract project area from Arabic/English text and normalize to m²."""
    t = normalize_number_text(text).lower()

    # Explicit area expression first.
    patterns = [
        r"(?:مساحة(?:\s+(?:الأرض|الارض|المشروع))?|area|land\s*area)\s*(?:هي|=|:)?\s*([\d.,]+)\s*(هكتار|hectares?|ha|م2|م²|متر\s*مربع|sqm|sq\.?\s*m)?",
        r"([\d.,]+)\s*(هكتار|hectares?|ha|م2|م²|متر\s*مربع|sqm|sq\.?\s*m)\b",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            value = _numbers(m.group(1))[0]
            unit = (m.group(2) or "m2").strip()
            if unit in {"هكتار", "hectare", "hectares", "ha"}:
                value *= 10000.0
            return value if value > 0 else None
    return None


def extract_population(text: str) -> int | None:
    t = normalize_number_text(text).lower()
    patterns = [
        r"(?:عدد\s*السكان|السكان|population|residents?)\s*(?:هو|هي|=|:)?\s*([\d.,]+)",
        r"([\d.,]+)\s*(?:نسمة|شخص|سكان|people|persons?|residents?)\b",
    ]
    for p in patterns:
        m = re.search(p, t)
        if m:
            vals = _numbers(m.group(1))
            if vals:
                value = int(round(vals[0]))
                return value if value >= 0 else None
    return None


def _bare_value_for_expected_field(text: str, expected: str | None) -> float | int | None:
    """When the assistant just asked for one field, accept a bare numeric reply."""
    vals = _numbers(text)
    if len(vals) != 1:
        return None
    if expected == "area":
        value = vals[0]
        # If the user wrote hectare with a bare number, convert it.
        low = normalize_number_text(text).lower()
        if re.search(r"\b(?:ha|hectares?)\b|هكتار", low):
            value *= 10000.0
        return value if value > 0 else None
    if expected == "population":
        return max(0, int(round(vals[0])))
    return None


def _level_short(level: str | None) -> str:
    if not level:
        return "غير منطبق"
    return level.replace("خدمات ", "")


def build_chat_summary(result: dict[str, Any]) -> str:
    required = result.get("required", [])
    optional = result.get("optional", [])
    t = result.get("totals") or {}

    if result.get("max_service_level") is None:
        return (
            f"تم الحساب. الكثافة السكانية هي **{result['density_person_per_ha']:,.2f} فرد/هكتار** "
            f"وتصنيفها **{result['density_category']}**. وفق الحدود السكانية الحالية لا يوجد مستوى خدمي منطبق بعد."
        )

    msg = (
        f"تم حساب الخدمات العامة للمشروع. المساحة **{result['project_area_m2']:,.0f} م²** "
        f"({result['project_area_ha']:,.2f} هكتار)، وعدد السكان **{result['population']:,} نسمة**، "
        f"والكثافة **{result['density_person_per_ha']:,.2f} فرد/هكتار** وتصنيفها **{result['density_category']}**.\n\n"
        f"أعلى مستوى خدمي منطبق هو **{_level_short(result['max_service_level'])}**؛ لذلك تم احتساب الخدمات "
        "تراكميًا حتى هذا المستوى فقط، ولم يتم إدراج خدمات المستويات الأعلى."
    )
    if t:
        msg += (
            f"\n\n**الإجمالي للخدمات الإلزامية:** {t.get('required_facilities_count', 0):,} مرفق، "
            f"مساحة أرض {t.get('required_land_area_m2', 0):,.0f} م²، "
            f"وGFA {t.get('required_gfa_m2', 0):,.0f} م²."
        )
    if optional:
        msg += f"\n\nيوجد أيضًا **{len(optional)}** بند/بنود اختيارية موضحة في الجدول بشكل منفصل."
    if required:
        msg += "\n\nالجدول التالي يوضح الخدمات المطلوبة والحسابات التفصيلية طبقًا لمعادلات ملف الخدمات العامة:"
    return msg


def process_public_facilities_turn(
    user_message: str,
    flow_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Deterministic chat flow for public-facilities calculation.

    Returns:
      handled: whether this turn belongs to the facilities workflow
      message: assistant text
      table_rows: rows to render in chat when calculation completes
      result: full calculator result when completed
      state: updated flow state
    """
    state = dict(flow_state or new_flow_state())

    if detect_reset(user_message) and (state.get("active") or state.get("last_result")):
        state = new_flow_state()
        state["active"] = True
        state["awaiting"] = "area"
        return {
            "handled": True,
            "message": "أكيد. نبدأ حسابًا جديدًا. ما مساحة أرض المشروع بالمتر المربع؟",
            "table_rows": None,
            "result": None,
            "state": state,
        }

    is_intent = detect_public_facilities_intent(user_message)
    if not is_intent and not state.get("active"):
        return {
            "handled": False,
            "message": None,
            "table_rows": None,
            "result": None,
            "state": state,
        }

    state["active"] = True

    area = extract_area_m2(user_message)
    population = extract_population(user_message)

    # During a prompted step, accept a reply containing only the requested number.
    expected = state.get("awaiting")
    if area is None and expected == "area":
        bare = _bare_value_for_expected_field(user_message, "area")
        if isinstance(bare, (int, float)):
            area = float(bare)
    if population is None and expected == "population":
        bare = _bare_value_for_expected_field(user_message, "population")
        if isinstance(bare, int):
            population = bare

    if area is not None:
        state["project_area_m2"] = float(area)
    if population is not None:
        state["population"] = int(population)

    if state.get("project_area_m2") is None:
        state["awaiting"] = "area"
        return {
            "handled": True,
            "message": "لحساب الخدمات العامة، ما **مساحة أرض المشروع بالمتر المربع**؟",
            "table_rows": None,
            "result": None,
            "state": state,
        }

    if state.get("population") is None:
        state["awaiting"] = "population"
        return {
            "handled": True,
            "message": (
                f"تم تسجيل مساحة المشروع: **{state['project_area_m2']:,.0f} م²**. "
                "كم **عدد السكان المتوقع** للمشروع؟"
            ),
            "table_rows": None,
            "result": None,
            "state": state,
        }

    try:
        result = review_public_facilities(
            project_area_m2=float(state["project_area_m2"]),
            population=int(state["population"]),
        )
    except ValueError as e:
        # Keep the flow alive so the user can correct the input in chat.
        state["awaiting"] = "area" if float(state.get("project_area_m2") or 0) <= 0 else "population"
        return {
            "handled": True,
            "message": f"لا يمكن إكمال الحساب: {e}",
            "table_rows": None,
            "result": None,
            "state": state,
        }

    state["awaiting"] = None
    state["last_result"] = result
    # End the active intake flow after a complete calculation. A fresh intent starts a new one.
    state["active"] = False

    return {
        "handled": True,
        "message": build_chat_summary(result),
        "table_rows": result_rows(result, include_optional=True),
        "result": result,
        "state": state,
    }
