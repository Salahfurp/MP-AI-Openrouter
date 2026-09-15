from __future__ import annotations

import io
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from typing import Any, Callable

import pandas as pd


SERVICE_ALIASES = (
    "الخدمة", "اسم الخدمة", "الخدمات", "نوع الخدمة", "المرفق", "اسم المرفق",
    "service", "service name", "facility", "facility name", "facility type",
    "public facility", "public facilities", "public facility type", "amenity", "amenity type",
)
COUNT_ALIASES = (
    "العدد", "عدد", "عدد المرافق", "الكمية", "العدد المقدم", "المتوفر", "المتوفر فعليا",
    "quantity", "qty", "count", "no", "no.", "number", "provided", "provided count",
    "number of facilities", "no of facilities", "no. of facilities", "facility count", "units",
)
LAND_ALIASES = (
    "مساحة الارض", "مساحة الأرض", "ارض", "الأرض", "مساحة الموقع", "اجمالي مساحة الارض",
    "land area", "site area", "plot area", "total land area", "total site area", "provided land area",
    "land area m2", "land area sqm", "site area m2", "site area sqm", "plot area sqm", "site area sq.m",
)
GFA_ALIASES = (
    "gfa", "مساحة البناء", "المساحة البنائية", "اجمالي المساحة البنائية", "إجمالي المساحة البنائية",
    "gross floor area", "built up area", "built-up area", "bua", "total gfa", "provided gfa",
)
LEVEL_ALIASES = (
    "المستوى", "مستوى الخدمة", "المستوى الخدمي", "level", "service level", "hierarchy", "facility level",
)

# Official Arabic service name -> common English/Arabic variants seen in consultant schedules.
SERVICE_NAME_ALIASES: dict[str, tuple[str, ...]] = {
    "مسجد أوقات": ("daily mosque", "local mosque", "neighborhood mosque", "neighbourhood mosque", "local masjid", "community mosque", "prayer mosque", "masjid", "mosque daily prayer"),
    "محلات تجارية": ("retail shops", "shops", "local shops", "commercial shops", "convenience retail", "retail"),
    "مجمع صناديق بريد": ("post box cluster", "po box cluster", "p.o. box cluster", "mailbox cluster", "post boxes"),
    "ساحة عامة": ("public plaza", "public square", "community plaza", "civic plaza"),
    "ساحة عائلية": ("family plaza", "family square"),
    "حديقة عائلية": ("family park", "family garden"),
    "مسجد جمعة": ("friday mosque", "juma mosque", "jumaa mosque", "jumuah mosque", "jumu'ah mosque", "congregational mosque"),
    "مركز تجاري": ("commercial center", "commercial centre", "retail center", "retail centre", "shopping center", "shopping centre"),
    "حضانة أطفال": ("nursery", "daycare", "day care", "childcare", "child care", "nursery school"),
    "روضة أطفال": ("kindergarten", "kg", "kg school", "pre school", "preschool"),
    "مدرسة ابتدائي": ("primary school", "elementary school", "primary education", "primary education school", "primary boys school", "primary girls school"),
    "عيادة خاصة تخصص عام": ("private general clinic", "general clinic", "private clinic general", "general medical clinic"),
    "عيادة خاصة تخصصية": ("private specialist clinic", "private specialized clinic", "specialist clinic", "specialized clinic"),
    "مركز طبي خاص": ("private medical center", "private medical centre", "medical center", "medical centre"),
    "ملاعب رياضية": ("sports fields", "sports field", "sports facilities", "sports courts", "play fields", "playing fields"),
    "حديقة منطقة": ("area park", "community park", "district park", "area garden"),
    "مدرسة إعدادي": ("preparatory school", "middle school", "intermediate school", "prep school", "preparatory education school", "middle education school"),
    "مدرسة ثانوي": ("secondary school", "high school", "senior school", "secondary education school", "secondary education", "senior secondary school"),
    "مركز طبي جراحي خاص": ("private surgical medical center", "private surgical medical centre", "private surgical center", "surgical medical center"),
    "مكتب بريد": ("post office", "postal office"),
    "مركز صحي حكومي": ("government health center", "government health centre", "public health center", "public health centre", "health center", "health centre"),
    "نقطة تمركز خدمات الإسعاف": ("ambulance station", "ambulance point", "ems station", "emergency medical services station", "ambulance service point"),
    "حديقة حي": ("neighborhood park", "neighbourhood park", "community neighborhood park", "community neighbourhood park"),
    "مركز دفاع مدني": ("civil defense center", "civil defence centre", "civil defence center", "civil defense centre", "fire station"),
    "مصلى عيد": ("eid prayer ground", "eid prayer area", "eid musalla", "eid musalla", "eid prayer hall"),
    "مكتبة عامة": ("public library", "library"),
    "مركز شرطة": ("police station", "police center", "police centre"),
    "استراحة كبار السن": ("senior citizens center", "senior citizens centre", "elderly center", "elderly centre", "senior rest house", "elderly rest house"),
    "مكتب بلدية": ("municipality office", "municipal office"),
    "مركز بلدية": ("municipality center", "municipality centre", "municipal center", "municipal centre"),
    "مستشفى خاص": ("private hospital",),
    "مستشفى حكومي": ("government hospital", "public hospital"),
    "حديقة قطاع": ("sector park", "sector garden"),
}


@dataclass
class SubmittedFacility:
    service: str
    count: float | None = None
    land_area_m2: float | None = None
    gfa_m2: float | None = None
    level: str | None = None
    source_row: str | None = None
    canonical_service: str | None = None


def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,"))
    t = re.sub(r"[❶❷❸]", "", t)
    t = t.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ى":"ي", "ؤ":"و", "ئ":"ي"}))
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = re.sub(r"[ـ_\-–—/\\|:;,.()\[\]{}]+", " ", t)
    t = re.sub(r"\b(sq\s*m|sqm|m2|m²)\b", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _num(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        try:
            if pd.isna(v):
                return None
        except Exception:
            pass
        return float(v)
    s = str(v).strip().translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,"))
    if not s:
        return None
    s = re.sub(r"[^0-9.,\-]", "", s)
    if not s:
        return None
    if s.count(",") == 1 and s.count(".") == 0 and len(s.split(",")[-1]) != 3:
        s = s.replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def _find_col(columns: list[Any], aliases: tuple[str, ...]) -> Any | None:
    norm_cols = {c: _norm(c) for c in columns}
    norm_aliases = [_norm(a) for a in aliases]
    for c, nc in norm_cols.items():
        if nc in norm_aliases:
            return c
    for c, nc in norm_cols.items():
        if any(a and (a in nc or nc in a) for a in norm_aliases):
            return c
    return None


def _header_score(values: list[Any]) -> int:
    cells = [_norm(v) for v in values if str(v).strip() and str(v).lower() != "nan"]
    if not cells:
        return 0
    groups = [SERVICE_ALIASES, COUNT_ALIASES, LAND_ALIASES, GFA_ALIASES, LEVEL_ALIASES]
    score = 0
    for aliases in groups:
        norm_aliases = [_norm(a) for a in aliases]
        if any(any(a == c or a in c or c in a for a in norm_aliases if a) for c in cells):
            score += 1
    return score


def _unique_headers(values: list[Any]) -> list[str]:
    """Create stable unique column names even when Excel repeats labels such as Area/Area."""
    used: dict[str, int] = {}
    out: list[str] = []
    for j, v in enumerate(values):
        base = str(v).strip() if pd.notna(v) else ""
        base = base or f"column_{j+1}"
        key = _norm(base) or f"column_{j+1}"
        used[key] = used.get(key, 0) + 1
        out.append(base if used[key] == 1 else f"{base}__{used[key]}")
    return out


def _cell(row: pd.Series, col: Any) -> Any:
    """Return a scalar even if a malformed/duplicate header would otherwise yield a Series."""
    if col is None:
        return None
    value = row.get(col)
    if isinstance(value, pd.Series):
        non_null = value.dropna()
        return non_null.iloc[0] if not non_null.empty else None
    return value


def _promote_header(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return raw
    best_idx, best_score = 0, -1
    for i in range(min(20, len(raw))):
        score = _header_score(raw.iloc[i].tolist())
        if score > best_score:
            best_idx, best_score = i, score
    if best_score >= 1:
        headers = _unique_headers(raw.iloc[best_idx].tolist())
        df = raw.iloc[best_idx + 1:].copy()
        df.columns = headers
        return df.dropna(how="all")
    return raw


def rows_from_dataframe(df: pd.DataFrame) -> list[SubmittedFacility]:
    if df is None or df.empty:
        return []
    df = df.dropna(how="all").copy()
    if df.empty:
        return []

    service_col = _find_col(list(df.columns), SERVICE_ALIASES)
    count_col = _find_col(list(df.columns), COUNT_ALIASES)
    land_col = _find_col(list(df.columns), LAND_ALIASES)
    gfa_col = _find_col(list(df.columns), GFA_ALIASES)
    level_col = _find_col(list(df.columns), LEVEL_ALIASES)

    if service_col is None and len(df.columns) >= 1:
        # Prefer the text-heavy column as service rather than blindly taking the first.
        best = None
        best_hits = -1
        for c in df.columns[: min(5, len(df.columns))]:
            vals = df[c].tolist()
            text_hits = sum(bool(re.search(r"[A-Za-z\u0600-\u06FF]", str(v))) for v in vals if pd.notna(v))
            if text_hits > best_hits:
                best, best_hits = c, text_hits
        service_col = best or df.columns[0]
    if count_col is None and len(df.columns) >= 2:
        for cand in df.columns:
            if cand == service_col:
                continue
            numeric_hits = sum(_num(v) is not None for v in df[cand].tolist())
            if numeric_hits >= max(1, len(df) // 2):
                count_col = cand
                break

    out: list[SubmittedFacility] = []
    for _, row in df.iterrows():
        service_value = _cell(row, service_col)
        service = str(service_value).strip() if service_value is not None else ""
        if not service or service.lower() == "nan" or _norm(service) in {_norm(x) for x in SERVICE_ALIASES}:
            continue
        # Drop obvious total/subtotal rows.
        if _norm(service) in {"total", "subtotal", "grand total", "الاجمالي", "اجمالي", "المجموع"}:
            continue
        out.append(SubmittedFacility(
            service=service,
            count=_num(_cell(row, count_col)) if count_col is not None else None,
            land_area_m2=_num(_cell(row, land_col)) if land_col is not None else None,
            gfa_m2=_num(_cell(row, gfa_col)) if gfa_col is not None else None,
            level=(str(_cell(row, level_col)).strip() if level_col is not None and _cell(row, level_col) is not None and not pd.isna(_cell(row, level_col)) else None),
            source_row=" | ".join(str(v) for v in row.tolist() if pd.notna(v)),
        ))
    return out


def parse_excel_bytes(data: bytes, filename: str = "submission.xlsx") -> list[SubmittedFacility]:
    # Read header=None first so consultant title rows / merged headings do not break detection.
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    all_rows: list[SubmittedFacility] = []
    for _, raw in sheets.items():
        df = _promote_header(raw)
        rows = rows_from_dataframe(df)
        if rows:
            all_rows.extend(rows)
    return all_rows


def parse_csv_or_text_table(text: str) -> list[SubmittedFacility]:
    text = (text or "").strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Markdown table.
    md_lines = [ln for ln in lines if "|" in ln]
    if len(md_lines) >= 2:
        rows = []
        for ln in md_lines:
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if parts and all(re.fullmatch(r":?-{2,}:?", p.replace(" ", "")) for p in parts if p):
                continue
            rows.append(parts)
        if len(rows) >= 2:
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            parsed = rows_from_dataframe(pd.DataFrame(rows[1:], columns=rows[0]))
            if parsed:
                return parsed

    # TSV / CSV / semicolon.
    for sep in ["\t", ",", ";"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
            if len(df.columns) >= 2:
                parsed = rows_from_dataframe(df)
                if parsed:
                    return parsed
        except Exception:
            pass

    # Fixed-width copy/paste from PDF/Word/Excel.
    try:
        df = pd.read_fwf(io.StringIO(text))
        if len(df.columns) >= 2:
            parsed = rows_from_dataframe(df)
            if parsed:
                return parsed
    except Exception:
        pass

    # Loose lines: service + numeric values. First numeric is count, second land, third GFA.
    out: list[SubmittedFacility] = []
    for ln in lines:
        if re.search(r"^(الخدمة|service|facility)\b", _norm(ln)):
            continue
        nums = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", ln.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,")))
        service = re.sub(r"[-+]?\d[\d,]*(?:\.\d+)?", " ", ln).strip(" |-:\t")
        service = re.sub(r"\s+", " ", service)
        if not service:
            continue
        vals = [_num(n) for n in nums]
        out.append(SubmittedFacility(
            service=service,
            count=vals[0] if len(vals) > 0 else None,
            land_area_m2=vals[1] if len(vals) > 1 else None,
            gfa_m2=vals[2] if len(vals) > 2 else None,
            source_row=ln,
        ))
    return out


def extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        if txt.strip():
            parts.append(txt)
    return "\n".join(parts)


def parse_pdf_bytes(data: bytes, ai_json_parser: Callable[[str], list[dict[str, Any]]] | None = None) -> tuple[list[SubmittedFacility], str]:
    text = extract_pdf_text(data)
    if not text.strip():
        return [], text

    rows = parse_csv_or_text_table(text)
    plausible = [r for r in rows if best_official_service_match(r.service)[1] >= 0.72]
    if len(plausible) >= 2:
        return rows, text

    if ai_json_parser is not None:
        raw_rows = ai_json_parser(text)
        out = []
        for x in raw_rows or []:
            service = str(x.get("service", "")).strip()
            if not service:
                continue
            out.append(SubmittedFacility(
                service=service,
                count=_num(x.get("count")),
                land_area_m2=_num(x.get("land_area_m2")),
                gfa_m2=_num(x.get("gfa_m2")),
                level=x.get("level"),
                source_row=json.dumps(x, ensure_ascii=False),
            ))
        return out, text

    return rows, text


def _text_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        shorter = min(len(na), len(nb))
        if shorter >= 4:
            return 0.97
    sa, sb = set(na.split()), set(nb.split())
    jaccard = len(sa & sb) / len(sa | sb) if sa and sb else 0.0
    seq = SequenceMatcher(None, na, nb).ratio()
    return max(seq, jaccard * 0.96)


def best_official_service_match(text: str) -> tuple[str | None, float]:
    best_name, best_score = None, 0.0
    for official, aliases in SERVICE_NAME_ALIASES.items():
        candidates = (official,) + tuple(aliases)
        score = max(_text_similarity(text, c) for c in candidates)
        if score > best_score:
            best_name, best_score = official, score
    return best_name, best_score


def _service_similarity(required_official: str, submitted_name: str) -> float:
    required_clean = re.sub(r"[❶❷❸]", "", required_official).strip()
    # First classify the submitted Arabic/English label to one official service.
    # This prevents a generic word such as "school" or "mosque" from being consumed
    # by the wrong required service before the exact English service is reached.
    canonical, canon_score = best_official_service_match(submitted_name)
    if canonical == required_clean and canon_score >= 0.72:
        return canon_score

    # Direct similarity is useful for Arabic spelling variations and labels that already
    # use the official Arabic wording. Cross-language matching is handled by the alias map above.
    direct = _text_similarity(required_clean, submitted_name)
    if direct >= 0.84:
        return direct
    return 0.0


def match_submission(
    required_service: str,
    submitted: list[SubmittedFacility],
    threshold: float = 0.68,
    semantic_map: dict[str, tuple[str | None, float]] | None = None,
):
    """Match deterministically first, then use an optional AI semantic map for ambiguous bilingual labels."""
    best = None
    best_score = 0.0
    required_clean = re.sub(r"[❶❷❸]", "", required_service).strip()
    for row in submitted:
        # Prefer the canonical bilingual translation resolved before comparison.
        if row.canonical_service:
            score = 0.995 if _norm(row.canonical_service) == _norm(required_clean) else 0.0
        else:
            score = _service_similarity(required_service, row.service)
        sem = None
        if semantic_map:
            sem = semantic_map.get(row.service) or semantic_map.get(_norm(row.service))
        if sem:
            mapped_name, confidence = sem
            if mapped_name == required_clean:
                # AI is only a semantic resolver; require reasonable confidence.
                score = max(score, min(float(confidence or 0), 0.99))
        if score > best_score:
            best_score = score
            best = row
    if best_score < threshold:
        return None, best_score
    return best, best_score


def looks_like_facilities_table(text: str) -> bool:
    t = text or ""
    if not t.strip() or len(t.splitlines()) < 2:
        return False
    n = _norm(t)
    header_words = tuple(SERVICE_ALIASES + COUNT_ALIASES + LAND_ALIASES + GFA_ALIASES)
    if any(_norm(a) in n for a in header_words if _norm(a)):
        return True
    rows = parse_csv_or_text_table(t)
    matched = sum(best_official_service_match(r.service)[1] >= 0.72 for r in rows)
    return matched >= 2


def compare_facilities(required_result: dict[str, Any], submitted: list[SubmittedFacility], semantic_map: dict[str, tuple[str | None, float]] | None = None) -> dict[str, Any]:
    comparisons = []
    deficit_count = 0
    ok_count = 0

    available = list(submitted)
    for req in required_result.get("required", []):
        sub, score = match_submission(req["service"], available, semantic_map=semantic_map)
        if sub is not None:
            try:
                available.remove(sub)
            except ValueError:
                pass
        req_count = float(req.get("facility_count") or 0)
        req_land = float(req.get("required_land_area_m2") or 0)
        req_gfa = float(req.get("required_gfa_m2") or 0)

        provided_count = sub.count if sub else None
        provided_land = sub.land_area_m2 if sub else None
        provided_gfa = sub.gfa_m2 if sub else None

        count_deficit = max(0.0, req_count - (provided_count or 0.0)) if req_count else 0.0
        land_deficit = max(0.0, req_land - (provided_land or 0.0)) if req_land and provided_land is not None else None
        gfa_deficit = max(0.0, req_gfa - (provided_gfa or 0.0)) if req_gfa and provided_gfa is not None else None

        is_deficit = sub is None or (req_count > 0 and (provided_count or 0.0) < req_count)
        reasons = []
        if sub is None:
            reasons.append("الخدمة غير موجودة في الجدول المقدم")
        else:
            if req_count > 0 and (provided_count or 0.0) < req_count:
                reasons.append(f"عجز في العدد: {count_deficit:g}")
            if land_deficit is not None and land_deficit > 0.01:
                is_deficit = True
                reasons.append(f"عجز في مساحة الأرض: {land_deficit:,.0f} م²")
            if gfa_deficit is not None and gfa_deficit > 0.01:
                is_deficit = True
                reasons.append(f"عجز في GFA: {gfa_deficit:,.0f} م²")

        status = "عجز" if is_deficit else "مستوفى"
        deficit_count += int(is_deficit)
        ok_count += int(not is_deficit)

        comparisons.append({
            "level": req.get("level"),
            "service": req.get("service"),
            "status": status,
            "required_count": req.get("facility_count"),
            "provided_count": provided_count,
            "count_deficit": round(count_deficit, 2),
            "required_land_area_m2": req.get("required_land_area_m2"),
            "provided_land_area_m2": provided_land,
            "land_deficit_m2": round(land_deficit, 2) if land_deficit is not None else None,
            "required_gfa_m2": req.get("required_gfa_m2"),
            "provided_gfa_m2": provided_gfa,
            "gfa_deficit_m2": round(gfa_deficit, 2) if gfa_deficit is not None else None,
            "matched_submission_service": sub.service if sub else None,
            "match_score": round(score, 3),
            "notes": "؛ ".join(reasons) if reasons else "لا يوجد عجز حسب البيانات المقدمة",
        })

    matched_ids = {id(sub) for sub in []}
    # 'available' already contains unmatched rows only.
    extras = [asdict(x) for x in available]

    return {
        "project": {
            "project_area_m2": required_result.get("project_area_m2"),
            "population": required_result.get("population"),
            "density_person_per_ha": required_result.get("density_person_per_ha"),
            "density_category": required_result.get("density_category"),
            "max_service_level": required_result.get("max_service_level"),
        },
        "summary": {
            "required_services": len(comparisons),
            "compliant_services": ok_count,
            "deficit_services": deficit_count,
            "compliance_percent": round((ok_count / len(comparisons) * 100.0), 1) if comparisons else 100.0,
        },
        "comparisons": comparisons,
        "extras": extras,
    }


def comparison_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for x in report.get("comparisons", []):
        rows.append({
            "الحالة / Status": "✅ مستوفى" if x["status"] == "مستوفى" else "❌ عجز",
            "المستوى / Level": x["level"],
            "الخدمة المطلوبة / Required service": x["service"],
            "الاسم في ملف المتعامل / Submitted name": x.get("matched_submission_service"),
            "العدد المطلوب / Required": x["required_count"],
            "العدد المقدم / Provided": x["provided_count"],
            "عجز العدد / Count deficit": x["count_deficit"],
            "الأرض المطلوبة (م²)": x["required_land_area_m2"],
            "الأرض المقدمة (م²)": x["provided_land_area_m2"],
            "عجز الأرض (م²)": x["land_deficit_m2"],
            "GFA المطلوب (م²)": x["required_gfa_m2"],
            "GFA المقدم (م²)": x["provided_gfa_m2"],
            "عجز GFA (م²)": x["gfa_deficit_m2"],
            "ملاحظات / Notes": x["notes"],
        })
    return rows
