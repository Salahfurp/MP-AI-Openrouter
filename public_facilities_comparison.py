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
    "الخدمة", "اسم الخدمة", "الخدمات", "service", "facility", "facility name",
    "public facility", "public facilities",
)
COUNT_ALIASES = (
    "العدد", "عدد", "عدد المرافق", "الكمية", "quantity", "qty", "count", "no", "no.",
    "number", "number of facilities",
)
LAND_ALIASES = (
    "مساحة الارض", "مساحة الأرض", "ارض", "الأرض", "land area", "site area", "plot area",
    "land area m2", "land area sqm", "site area m2", "site area sqm",
)
GFA_ALIASES = (
    "gfa", "مساحة البناء", "المساحة البنائية", "اجمالي المساحة البنائية", "إجمالي المساحة البنائية",
    "gross floor area", "built up area", "bua",
)
LEVEL_ALIASES = (
    "المستوى", "مستوى الخدمة", "level", "service level",
)


@dataclass
class SubmittedFacility:
    service: str
    count: float | None = None
    land_area_m2: float | None = None
    gfa_m2: float | None = None
    level: str | None = None
    source_row: str | None = None


def _norm(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip().lower()
    t = unicodedata.normalize("NFKC", t)
    t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩٫٬", "0123456789.,"))
    t = re.sub(r"[❶❷❸]", "", t)
    # Light Arabic normalization improves matching without changing meaning.
    t = t.translate(str.maketrans({"أ":"ا", "إ":"ا", "آ":"ا", "ى":"ي", "ؤ":"و", "ئ":"ي"}))
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = re.sub(r"[ـ_\-–—/\\|:;,.()\[\]{}]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _num(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, bool):
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

    # Fallback for simple two-column tables where first column is service.
    if service_col is None and len(df.columns) >= 1:
        service_col = df.columns[0]
    if count_col is None and len(df.columns) >= 2:
        # Only use a fallback second column if it is predominantly numeric.
        cand = df.columns[1]
        numeric_hits = sum(_num(v) is not None for v in df[cand].tolist())
        if numeric_hits >= max(1, len(df) // 2):
            count_col = cand

    out: list[SubmittedFacility] = []
    for _, row in df.iterrows():
        service = str(row.get(service_col, "")).strip() if service_col is not None else ""
        if not service or service.lower() == "nan":
            continue
        out.append(SubmittedFacility(
            service=service,
            count=_num(row.get(count_col)) if count_col is not None else None,
            land_area_m2=_num(row.get(land_col)) if land_col is not None else None,
            gfa_m2=_num(row.get(gfa_col)) if gfa_col is not None else None,
            level=str(row.get(level_col)).strip() if level_col is not None and pd.notna(row.get(level_col)) else None,
            source_row=" | ".join(str(v) for v in row.tolist() if pd.notna(v)),
        ))
    return out


def parse_excel_bytes(data: bytes, filename: str = "submission.xlsx") -> list[SubmittedFacility]:
    bio = io.BytesIO(data)
    sheets = pd.read_excel(bio, sheet_name=None)
    all_rows: list[SubmittedFacility] = []
    for _, df in sheets.items():
        rows = rows_from_dataframe(df)
        if rows:
            all_rows.extend(rows)
    return all_rows


def parse_csv_or_text_table(text: str) -> list[SubmittedFacility]:
    text = (text or "").strip()
    if not text:
        return []

    # Markdown table
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    md_lines = [ln for ln in lines if "|" in ln]
    if len(md_lines) >= 2:
        rows = []
        for ln in md_lines:
            parts = [p.strip() for p in ln.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", p.replace(" ", "")) for p in parts if p):
                continue
            rows.append(parts)
        if len(rows) >= 2:
            width = max(len(r) for r in rows)
            rows = [r + [""] * (width - len(r)) for r in rows]
            df = pd.DataFrame(rows[1:], columns=rows[0])
            parsed = rows_from_dataframe(df)
            if parsed:
                return parsed

    # TSV / CSV / semicolon
    for sep in ["\t", ",", ";"]:
        try:
            df = pd.read_csv(io.StringIO(text), sep=sep)
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


def parse_pdf_bytes(
    data: bytes,
    ai_json_parser: Callable[[str], list[dict[str, Any]]] | None = None,
) -> tuple[list[SubmittedFacility], str]:
    text = extract_pdf_text(data)
    if not text.strip():
        return [], text

    # Try deterministic text parsing first.
    rows = parse_csv_or_text_table(text)
    # If it produced plausible table-like rows, keep them.
    if len(rows) >= 2 and sum(r.count is not None for r in rows) >= 1:
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


def _service_similarity(a: str, b: str) -> float:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if na in nb or nb in na:
        return 0.94
    return SequenceMatcher(None, na, nb).ratio()


def match_submission(required_service: str, submitted: list[SubmittedFacility], threshold: float = 0.80):
    best = None
    best_score = 0.0
    for row in submitted:
        score = _service_similarity(required_service, row.service)
        if score > best_score:
            best_score = score
            best = row
    if best_score < threshold:
        return None, best_score
    return best, best_score


def compare_facilities(required_result: dict[str, Any], submitted: list[SubmittedFacility]) -> dict[str, Any]:
    comparisons = []
    deficit_count = 0
    ok_count = 0

    available = list(submitted)
    for req in required_result.get("required", []):
        sub, score = match_submission(req["service"], available)
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

        # A missing service or insufficient facility count is always a deficit.
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
        if is_deficit:
            deficit_count += 1
        else:
            ok_count += 1

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

    # Extra services supplied by customer but not matched to any mandatory requirement.
    matched_names = {_norm(x["matched_submission_service"]) for x in comparisons if x.get("matched_submission_service")}
    extras = [asdict(x) for x in submitted if _norm(x.service) not in matched_names]

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
            "الحالة": "✅ مستوفى" if x["status"] == "مستوفى" else "❌ عجز",
            "المستوى": x["level"],
            "الخدمة": x["service"],
            "العدد المطلوب": x["required_count"],
            "العدد المقدم": x["provided_count"],
            "عجز العدد": x["count_deficit"],
            "الأرض المطلوبة (م²)": x["required_land_area_m2"],
            "الأرض المقدمة (م²)": x["provided_land_area_m2"],
            "عجز الأرض (م²)": x["land_deficit_m2"],
            "GFA المطلوب (م²)": x["required_gfa_m2"],
            "GFA المقدم (م²)": x["provided_gfa_m2"],
            "عجز GFA (م²)": x["gfa_deficit_m2"],
            "ملاحظات": x["notes"],
        })
    return rows
