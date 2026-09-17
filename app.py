import os
import re
import json
import io
from pathlib import Path

import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

from ai_provider import ai_chat, provider_label
from ai_orchestrator import decide_route, general_answer

from public_facilities_chat import new_flow_state, process_public_facilities_turn, detect_public_facilities_intent
from public_facilities_comparison import (
    SubmittedFacility, compare_facilities, comparison_rows, parse_csv_or_text_table,
    parse_excel_bytes, parse_pdf_bytes, looks_like_facilities_table, best_official_service_match,
)
from public_facilities_report import build_comparison_pdf

APP_DIR = Path(__file__).parent
TEXT_PATH = APP_DIR / "guide_pages.txt"
PF_STATE_KEY = "public_facilities_flow"
PF_PENDING_SUBMISSION_KEY = "pending_public_facilities_submission"
PF_REVIEW_REQUESTED_KEY = "public_facilities_review_requested"
PF_GOAL_KEY = "public_facilities_goal"  # calculate | review
PF_COMPARE_KEYWORDS = (
    "قارن", "مقارنة", "قارنها", "قارنه", "راجع الخدمات", "راجع الجدول", "راجع الملف",
    "العجز", "ترجم وقارن", "ترجمه واعمل المقارنة", "ترجم الجدول", "حلل الجدول",
    "deficit", "compare", "compare it", "comparison", "review facilities", "review the table",
    "check facilities", "compliance", "translate and compare", "translate the table",
)

st.set_page_config(page_title="Master Plan AI Assistant", page_icon="🏙️", layout="wide")

SYSTEM_PROMPT = """You are the Master Plan AI Assistant for Dubai Urban Planning Permits.
Your authoritative source for normal planning Q&A is ONLY the UPPG excerpts supplied in the prompt.
Do not invent planning requirements. If the excerpts do not support an answer, say that clearly.
Answer in the user's language (Arabic or English), while preserving official English planning terms when useful.
Be practical and concise. Distinguish mandatory requirements from conditional/context-dependent items.
Never present the answer as an official approval, legal opinion, or final compliance determination.
Do NOT reveal chain-of-thought, internal reasoning, analysis steps, or a thinking process. Return only the final user-facing answer.
Always end normal UPPG answers with a short 'Source' line listing the UPPG page numbers that support the answer.
"""


# -----------------------------
# UPPG RAG
# -----------------------------
@st.cache_resource
def load_guide():
    raw = TEXT_PATH.read_text(encoding="utf-8", errors="ignore")
    matches = list(re.finditer(r"===== PAGE (\d+) =====", raw))
    pages = []
    for i, m in enumerate(matches):
        page_no = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        text = raw[start:end].strip()
        if text:
            pages.append({"page": page_no, "text": text})

    chunks = []
    for p in pages:
        text = re.sub(r"\s+", " ", p["text"]).strip()
        words = text.split()
        size, overlap = 250, 60
        if len(words) <= size:
            chunks.append({"page": p["page"], "text": text})
        else:
            step = size - overlap
            for j in range(0, len(words), step):
                part = words[j:j + size]
                if len(part) < 40:
                    break
                chunks.append({"page": p["page"], "text": " ".join(part)})

    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return chunks, bm25


def tokenize(text):
    return re.findall(r"[A-Za-z0-9%+.-]+|[\u0600-\u06FF]+", text.lower())


def model_chat(messages, max_tokens=900, temperature=0.15):
    """Provider-independent AI call. Google AI is preferred when configured; OpenRouter remains a fallback."""
    return ai_chat(messages, max_tokens=max_tokens, temperature=temperature)


def clean_ai_answer(text):
    if not isinstance(text, str):
        return text

    cleaned = text.strip()
    patterns = [
        r"(?is)\bfinal answer\s*:\s*(.+)$",
        r"(?is)\banswer\s*:\s*(.+)$",
    ]
    for pattern in patterns:
        m = re.search(pattern, cleaned)
        if m:
            cleaned = m.group(1).strip()
            break

    reasoning_starts = [
        r"(?im)^\s*here(?:'s| is) (?:a|the) thinking process\s*:?\s*$",
        r"(?im)^\s*thinking process\s*:?\s*$",
        r"(?im)^\s*analysis\s*:?\s*$",
        r"(?im)^\s*reasoning\s*:?\s*$",
    ]
    if any(re.search(p, cleaned) for p in reasoning_starts):
        return (
            "I can answer from the UPPG, but the selected AI model returned internal reasoning "
            "instead of a clean final response. Please retry the question."
        )
    return cleaned


def _normalized_question(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


UPPG_SEARCH_PROFILES = {
    "requirements": {
        "triggers": (
            "what documents", "documents required", "required documents", "requirements",
            "submission requirements", "checklist", "متطلبات", "المتطلبات", "المستندات",
            "الوثائق", "الاوراق", "الأوراق", "قائمة التحقق",
        ),
        "terms": (
            "Master Plan Permit required documents submission requirements checklist "
            "application supporting documents property specifications studies attachments"
        ),
        "boost": (
            "checklist", "required", "requirements", "documents", "submission",
            "supporting documents", "property specifications", "master plan permit",
        ),
    },
    "process": {
        "triggers": (
            "approval process", "process", "steps", "procedure", "workflow",
            "اجراءات", "إجراءات", "الخطوات", "عملية الاعتماد", "مسار الاعتماد",
        ),
        "terms": (
            "Master Plan Permit approval process procedure workflow steps stages "
            "submission review approval resubmission"
        ),
        "boost": ("process", "procedure", "steps", "stage", "review", "approval", "submission"),
    },
    "dubai2040": {
        "triggers": ("dubai 2040", "2040", "مواءمة", "دبي 2040"),
        "terms": (
            "Dubai 2040 Urban Master Plan alignment consistency strategic planning "
            "master plan alignment requirement"
        ),
        "boost": ("dubai 2040", "alignment", "urban master plan", "strategic"),
    },
    "modification": {
        "triggers": (
            "modification", "major modification", "minor modification",
            "تعديل", "تعديل جوهري", "تعديل رئيسي",
        ),
        "terms": (
            "Master Plan modification major modification minor modification requirements studies "
            "permit amendment"
        ),
        "boost": ("modification", "major", "minor", "amendment", "study", "studies"),
    },
    "studies": {
        "triggers": ("studies", "study", "traffic", "environment", "دراسات", "دراسة"),
        "terms": (
            "Master Plan supporting studies traffic impact study environmental study "
            "infrastructure study qualified professionals"
        ),
        "boost": ("study", "studies", "traffic", "environment", "supporting"),
    },
}


def detect_uppg_search_profile(question: str) -> str | None:
    q = _normalized_question(question)
    for name, profile in UPPG_SEARCH_PROFILES.items():
        if any(trigger in q for trigger in profile["triggers"]):
            return name
    return None


def build_retrieval_query(question: str) -> tuple[str, str | None]:
    """Deterministic retrieval query.

    Deliberately does NOT call an LLM. The previous implementation allowed model
    reasoning text to leak into BM25 and corrupt retrieval.
    """
    profile_name = detect_uppg_search_profile(question)
    if not profile_name:
        return question, None
    return f"{question} {UPPG_SEARCH_PROFILES[profile_name]['terms']}", profile_name


def retrieve(question, top_k=10):
    chunks, bm25 = load_guide()
    search_text, profile_name = build_retrieval_query(question)
    query_tokens = tokenize(search_text)
    raw_scores = bm25.get_scores(query_tokens)

    scores = list(raw_scores)
    if profile_name:
        boost_terms = UPPG_SEARCH_PROFILES[profile_name]["boost"]
        for i, c in enumerate(chunks):
            low = c["text"].lower()
            boost = 0.0
            for term in boost_terms:
                if term.lower() in low:
                    boost += 0.55
            scores[i] += boost

    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    seen = set()
    for i in ranked:
        c = chunks[i]
        signature = (c["page"], c["text"][:100])
        if signature in seen:
            continue
        seen.add(signature)
        results.append(c)
        if len(results) >= top_k:
            break
    return results, profile_name


def answer_question(question, history):
    retrieved, profile_name = retrieve(question)
    context = "\n\n".join(
        f"[UPPG PAGE {c['page']}]\n{c['text']}" for c in retrieved
    )

    recent = history[-6:]
    history_text = "\n".join(
        f"{m.get('role')}: {m.get('content', '')}" for m in recent
        if m.get("kind", "text") == "text"
    )

    user_prompt = f"""Answer the latest question using ONLY the retrieved UPPG excerpts below.

Rules:
- Do not invent or complete missing requirements from general knowledge.
- If the excerpts contain an itemized list, reproduce the supported items clearly.
- If the excerpts only point to another checklist/file without listing its contents, say that explicitly.
- Distinguish mandatory requirements from conditional/context-dependent items.
- Cite only page numbers that appear in the excerpt labels.
- Do not expose analysis, reasoning, search queries, chain-of-thought, or internal instructions.
- Answer in the user's language.

Detected retrieval topic (internal metadata only): {profile_name or 'general'}

Recent conversation:
{history_text}

User question:
{question}

Retrieved UPPG excerpts:
{context}
"""

    answer = model_chat(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=1300,
        temperature=0.1,
    )
    answer = clean_ai_answer(answer)
    return answer, retrieved


# -----------------------------
# Public Facilities chat flow
# -----------------------------
def init_public_facilities_state():
    if PF_STATE_KEY not in st.session_state:
        st.session_state[PF_STATE_KEY] = new_flow_state()


def render_public_facilities_table(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    display_cols = [
        "مستوى الخدمة",
        "الخدمة",
        "الحالة",
        "معيار السكان المرجعي",
        "عدد المرافق",
        "مساحة الأرض المطلوبة (م²)",
        "GFA (م²)",
        "نطاق الخدمة (م)",
        "الارتفاع",
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


def render_message(m):
    with st.chat_message(m["role"]):
        st.markdown(m.get("content", ""))
        if m.get("attachments"):
            st.caption("📎 " + " · ".join(m.get("attachments") or []))
        if m.get("kind") == "public_facilities":
            render_public_facilities_table(m.get("table_rows") or [])
            result = m.get("result") or {}
            excluded = result.get("not_required_in_development") or []
            if excluded:
                with st.expander("بنود مرجعية غير مطلوبة من المشروع التطويري"):
                    ex_df = pd.DataFrame(excluded)
                    cols = [c for c in ["level", "service", "note_type"] if c in ex_df.columns]
                    st.dataframe(ex_df[cols], use_container_width=True, hide_index=True)
        elif m.get("kind") == "facilities_comparison":
            render_comparison_table(m.get("comparison_rows") or [])
            if m.get("pdf_bytes"):
                st.download_button(
                    "📄 تنزيل تقرير المقارنة PDF",
                    data=m["pdf_bytes"],
                    file_name=m.get("pdf_name", "Public_Facilities_Compliance_Report.pdf"),
                    mime="application/pdf",
                    key=f"pdf_{id(m)}",
                    use_container_width=True,
                )


def handle_public_facilities_turn(question):
    init_public_facilities_state()
    payload = process_public_facilities_turn(
        question,
        st.session_state[PF_STATE_KEY],
    )
    st.session_state[PF_STATE_KEY] = payload["state"]

    if not payload["handled"]:
        return None

    return {
        "role": "assistant",
        "content": payload["message"],
        "kind": "public_facilities" if payload.get("result") else "text",
        "table_rows": payload.get("table_rows") or [],
        "result": payload.get("result"),
    }



# -----------------------------
# Public Facilities comparison / compliance review
# -----------------------------
def detect_facilities_comparison_intent(text):
    t = (text or "").lower()
    return any(k in t for k in PF_COMPARE_KEYWORDS)


def classify_turn(question: str, uploaded_files: list) -> str:
    """AI-first semantic router with deterministic fallback inside ai_orchestrator."""
    state = st.session_state.get(PF_STATE_KEY) or {}
    pending = st.session_state.get(PF_PENDING_SUBMISSION_KEY)
    review_pending = st.session_state.get(PF_GOAL_KEY) == "review" or bool(pending)
    decision = decide_route(
        question, st.session_state.get("messages", []),
        has_files=bool(uploaded_files),
        pf_active=bool(state.get("active")),
        review_pending=review_pending,
        has_facilities_result=bool(state.get("last_result")),
    )
    st.session_state["last_orchestrator_decision"] = {
        "route": decision.route, "confidence": decision.confidence,
        "reason": decision.reason, "suggested_action": decision.suggested_action,
    }
    return decision.route


def ai_parse_pdf_facilities(extracted_text):
    """Strict structure extraction only; comparison/calculation remains deterministic."""
    state = st.session_state.get(PF_STATE_KEY) or {}
    last = state.get("last_result") or {}
    known = [x.get("service") for x in (last.get("required") or []) + (last.get("optional") or []) if x.get("service")]
    known_text = "\n".join(f"- {x}" for x in known)
    prompt = f"""You are extracting a submitted public-facilities schedule from PDF text.
Return ONLY a JSON array. Do not add markdown or explanation.
Each object must use these keys exactly:
service, count, land_area_m2, gfa_m2, level
Rules:
- Extract only values explicitly present in the PDF text. Never invent missing values.
- Use null when count/land/GFA/level is not shown.
- Keep the service name as written, but when clearly equivalent prefer the closest known official service name below.
- Ignore headers, footers, totals and unrelated planning text.

Known service names for this project:
{known_text}

PDF text:
{extracted_text[:18000]}
"""
    raw = model_chat([{"role": "user", "content": prompt}], max_tokens=2200, temperature=0)
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    import json
    data = json.loads(cleaned)
    return data if isinstance(data, list) else []


def ai_parse_excel_facilities(data: bytes, filename: str = "submission.xlsx"):
    """AI fallback for messy bilingual consultant workbooks.

    It does NOT decide compliance. It only converts visible workbook rows to a neutral
    structure when deterministic header detection cannot reliably read the schedule.
    """
    try:
        sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    except Exception:
        return []

    previews = []
    for sheet_name, df in sheets.items():
        if df is None or df.empty:
            continue
        # Keep the prompt bounded while preserving title/header rows and enough schedule rows.
        view = df.iloc[:100, :24].copy()
        view = view.where(pd.notna(view), "")
        previews.append(f"SHEET: {sheet_name}\n" + view.to_csv(index=False, header=False))
    if not previews:
        return []

    prompt = f"""You extract a submitted public-facilities schedule from an Excel workbook.
The workbook may be Arabic, English, bilingual, have title rows, merged-looking headers, abbreviations, or reordered columns.
Return ONLY a JSON array; no markdown and no explanation.
Each object must have exactly these keys:
service, count, land_area_m2, gfa_m2, level
Rules:
- A facility/service row must represent a real public-facility item, not a heading or total.
- Keep the service label as written in the workbook. Do NOT translate it and do NOT decide whether it complies.
- Extract count, total land/site area, and GFA only when explicitly present; otherwise null.
- Understand Arabic and English column names and common variants such as Facility Type, Service, No., Qty, Site Area, Plot Area, Land Area, GFA, BUA, Gross Floor Area.
- Do not invent numbers.

Workbook: {filename}
{chr(10).join(previews)[:26000]}
"""
    try:
        raw = model_chat([{"role": "user", "content": prompt}], max_tokens=3200, temperature=0)
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        data_json = json.loads(cleaned)
        out = []
        if not isinstance(data_json, list):
            return []
        for x in data_json:
            if not isinstance(x, dict):
                continue
            service = str(x.get("service", "")).strip()
            if not service:
                continue
            def n(v):
                if v is None or v == "": return None
                try: return float(str(v).replace(",", ""))
                except Exception: return None
            out.append(SubmittedFacility(
                service=service,
                count=n(x.get("count")),
                land_area_m2=n(x.get("land_area_m2")),
                gfa_m2=n(x.get("gfa_m2")),
                level=(str(x.get("level")).strip() if x.get("level") not in (None, "") else None),
                source_row=json.dumps(x, ensure_ascii=False),
            ))
        return out
    except Exception:
        return []


def _rows_are_plausible(rows):
    if not rows:
        return False
    # Require actual text labels; pure serial-number columns are not a facilities schedule.
    text_rows = [r for r in rows if re.search(r"[A-Za-z\u0600-\u06FF]", str(getattr(r, "service", "")))]
    return len(text_rows) >= 1


def parse_uploaded_facilities(uploaded_file):
    if uploaded_file is None:
        return [], None
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        rows = parse_excel_bytes(data, uploaded_file.name)
        if _rows_are_plausible(rows):
            return rows, None
        # Messy workbook fallback: let the AI identify the table structure only.
        rows = ai_parse_excel_facilities(data, uploaded_file.name)
        return rows, None
    if name.endswith(".csv"):
        return parse_csv_or_text_table(data.decode("utf-8", errors="ignore")), None
    if name.endswith(".txt"):
        return parse_csv_or_text_table(data.decode("utf-8", errors="ignore")), None
    if name.endswith(".pdf"):
        rows, extracted = parse_pdf_bytes(data, ai_json_parser=ai_parse_pdf_facilities)
        return rows, extracted
    raise ValueError("صيغة الملف غير مدعومة. استخدم Excel أو CSV أو TXT أو PDF.")


def _extract_json_array(raw: str):
    """Accept clean JSON or JSON surrounded by provider chatter/thinking."""
    if not raw:
        return []
    cleaned = re.sub(r"^```(?:json)?\s*", "", str(raw).strip(), flags=re.I)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except Exception:
        pass
    # Providers can occasionally prepend text. Extract the outermost JSON array only.
    a, b = cleaned.find("["), cleaned.rfind("]")
    if a >= 0 and b > a:
        try:
            data = json.loads(cleaned[a:b+1])
            return data if isinstance(data, list) else []
        except Exception:
            return []
    return []


def build_semantic_service_map(required_result, submitted_rows):
    """Translate/canonicalize every submitted service before compliance comparison.

    This is deliberately a bilingual terminology task. The AI may ONLY choose from
    the official services calculated for this project. Quantities and compliance are
    never decided by the model.
    """
    required_names = [
        re.sub(r"[❶❷❸]", "", str(x.get("service", ""))).strip()
        for x in (required_result.get("required") or []) if x.get("service")
    ]
    submitted_names = [str(x.service).strip() for x in submitted_rows if str(x.service).strip()]
    if not required_names or not submitted_names:
        return {}

    # Stage 1: deterministic bilingual dictionary/fuzzy matching. This works even if
    # the external AI provider is unavailable.
    out = {}
    unresolved = []
    for name in submitted_names:
        official, confidence = best_official_service_match(name)
        if official in required_names and confidence >= 0.70:
            out[name] = (official, max(confidence, 0.90))
            out[_norm(name)] = out[name]
        else:
            unresolved.append(name)

    # Stage 2: AI semantic translation only for unresolved labels. It sees the entire
    # official list so Arabic <-> English and consultant terminology can be resolved.
    if unresolved:
        prompt = f"""You are a professional Dubai urban-planning bilingual terminology translator and matcher.
Translate the meaning of EACH submitted public-facility label, then map it to exactly ONE equivalent item from ALLOWED_OFFICIAL_SERVICES.
The submitted schedule can be Arabic or English. Handle abbreviations, British/American spelling, reordered words, consultant terminology and small typos.
Preserve subtype distinctions: primary != preparatory/middle != secondary; daily/local mosque != Friday mosque; area park != neighborhood park != sector park; clinic != health center != hospital.
If there is no genuinely equivalent official service, set official to null.
Return ONLY a JSON array. No prose. No markdown. No reasoning.
Schema: [{{"submitted":"original label","arabic_translation":"Arabic meaning","official":"exact allowed Arabic service or null","confidence":0.0}}]
ALLOWED_OFFICIAL_SERVICES={json.dumps(required_names, ensure_ascii=False)}
SUBMITTED_LABELS={json.dumps(unresolved, ensure_ascii=False)}
"""
        try:
            raw = model_chat([{"role": "user", "content": prompt}], max_tokens=2200, temperature=0)
            data = _extract_json_array(raw)
            allowed = set(required_names)
            for item in data:
                if not isinstance(item, dict):
                    continue
                submitted = str(item.get("submitted", "")).strip()
                official = item.get("official")
                official = str(official).strip() if official is not None else None
                try:
                    confidence = float(item.get("confidence", 0) or 0)
                except Exception:
                    confidence = 0.0
                if submitted and official in allowed and confidence >= 0.62:
                    out[submitted] = (official, min(max(confidence, 0.0), 0.99))
                    out[_norm(submitted)] = out[submitted]
        except Exception:
            pass

    # Persist the canonical Arabic translation on the parsed rows. The comparison
    # engine therefore compares canonical concepts, not raw strings.
    for row in submitted_rows:
        mapped = out.get(row.service) or out.get(_norm(row.service))
        if mapped:
            row.canonical_service = mapped[0]
    return out


def build_comparison_chat_event(submitted_rows, source_label="الجدول المقدم"):
    state = st.session_state.get(PF_STATE_KEY) or {}
    required_result = state.get("last_result")
    if not required_result:
        return {
            "role": "assistant",
            "content": "سأحتفظ بالجدول وأكمل مراجعة العجز تلقائيًا. أحتاج فقط **مساحة المشروع** و**عدد السكان** لحساب المتطلبات المرجعية أولًا؛ لن تحتاج إلى إعادة رفع الملف أو طلب المقارنة مرة ثانية.",
            "kind": "text",
        }
    if not submitted_rows:
        return {
            "role": "assistant",
            "content": "لم أستطع العثور على صفوف خدمات قابلة للمقارنة في البيانات المقدمة. تأكد أن الجدول يحتوي على الأقل على **اسم الخدمة** ويفضل **العدد**، ويمكن أيضًا إضافة **مساحة الأرض** و **GFA**.",
            "kind": "text",
        }

    semantic_map = build_semantic_service_map(required_result, submitted_rows)
    report = compare_facilities(required_result, submitted_rows, semantic_map=semantic_map)
    rows = comparison_rows(report)
    summary = report.get("summary", {})
    matched_n = sum(1 for x in report.get("comparisons", []) if x.get("matched_submission_service"))
    deficits = [x for x in report.get("comparisons", []) if x.get("status") == "عجز"]
    if deficits:
        top = "، ".join(x.get("service", "") for x in deficits[:5])
        more = "" if len(deficits) <= 5 else f" وغيرها ({len(deficits)-5} خدمة إضافية)"
        message = (
            f"تمت ترجمة/توحيد أسماء الخدمات في **{source_label}** عربيًا ثم مقارنتها بالخدمات الإلزامية للمشروع. "
            f"تم التعرف على **{matched_n}** خدمة مقدمة وربطها بالمسمى الرسمي. "
            f"الاستيفاء **{summary.get('compliance_percent', 0)}%**: "
            f"**{summary.get('compliant_services', 0)} مستوفاة** و**{summary.get('deficit_services', 0)} بها عجز**.\n\n"
            f"أبرز الخدمات التي تحتاج استكمال: **{top}{more}**. "
            "الجدول أدناه يوضح العجز بالتفصيل، ويمكن تنزيل تقرير PDF احترافي بالألوان."
        )
    else:
        message = (
            f"تمت مراجعة **{source_label}**، وجميع الخدمات الإلزامية الظاهرة في الحساب **مستوفاة حسب البيانات المقدمة**. "
            f"نسبة الاستيفاء **{summary.get('compliance_percent', 100)}%**."
        )

    pdf_bytes = build_comparison_pdf(report)
    return {
        "role": "assistant",
        "content": message,
        "kind": "facilities_comparison",
        "comparison_rows": rows,
        "comparison_report": report,
        "pdf_bytes": pdf_bytes,
        "pdf_name": "Public_Facilities_Compliance_Report.pdf",
    }


def render_comparison_table(rows):
    if not rows:
        return
    df = pd.DataFrame(rows)
    def color_status(v):
        text = str(v)
        if "عجز" in text or "Deficit" in text:
            return "background-color: #FEE4E2; color: #B42318; font-weight: 700"
        if "مستوفى" in text or "Compliant" in text:
            return "background-color: #D1FADF; color: #027A48; font-weight: 700"
        return ""
    try:
        styler = df.style.map(color_status, subset=["الحالة / Status"]).set_table_styles([
            {"selector": "th", "props": [("background-color", "#F2F4F7"), ("color", "#101828"), ("font-weight", "700"), ("border", "1px solid #D0D5DD")]},
            {"selector": "td", "props": [("color", "#101828")]},
        ])
        st.dataframe(styler, use_container_width=True, hide_index=True)
    except Exception:
        st.dataframe(df, use_container_width=True, hide_index=True)

# -----------------------------
# UI
# -----------------------------
st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:800;margin-bottom:.2rem}
.sub{color:#667085;margin-bottom:1.2rem}
.small-note{font-size:.85rem;color:#667085}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏙️ Master Plan AI Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub">Smart UPPG guidance + Public Facilities calculation, file review and compliance reporting in one chat</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
init_public_facilities_state()

with st.sidebar:
    st.header("Assistant")
    st.success("AI + UPPG RAG enabled")
    st.info("Public Facilities Calculator enabled")
    st.caption(f"AI route: {provider_label()}")

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state[PF_STATE_KEY] = new_flow_state()
        st.session_state.pop(PF_PENDING_SUBMISSION_KEY, None)
        st.session_state.pop(PF_REVIEW_REQUESTED_KEY, None)
        st.session_state.pop(PF_GOAL_KEY, None)
        st.rerun()

    st.divider()
    st.markdown("**Try asking:**")
    st.caption(
        "• What documents are required for a Master Plan Permit?\n\n"
        "• متى أحتاج مواءمة مع Dubai 2040؟\n\n"
        "• احسب لي الخدمات العامة المطلوبة للمشروع"
    )
    st.divider()
    st.caption(
        "Prototype only. UPPG answers and public-facilities calculations are preliminary guidance, "
        "not an official approval or final compliance determination."
    )

col1, col2, col3, col4, col5 = st.columns(5)
quick = [
    (col1, "📋 Requirements", "What documents are required for a Master Plan Permit?"),
    (col2, "🔎 Process", "Explain the Master Plan approval process step by step."),
    (col3, "🧭 Dubai 2040", "When is Dubai 2040 alignment required?"),
    (col4, "📝 Checklist", "Create a preliminary checklist for a new master plan submission."),
    (col5, "🏫 Public Facilities", "احسب لي الخدمات العامة المطلوبة للمشروع"),
]
for col, label, q in quick:
    if col.button(label, use_container_width=True):
        st.session_state.pending = q

for m in st.session_state.messages:
    render_message(m)

st.caption("📎 يمكنك إرفاق أو سحب Excel / CSV / TXT / PDF مباشرة إلى صندوق المحادثة. العربية والإنجليزية مدعومتان.")

submission = st.chat_input(
    "اكتب رسالتك أو أرفق جدول الخدمات هنا...",
    accept_file="multiple",
    file_type=["xlsx", "xls", "csv", "txt", "pdf"],
    key="main_chat_input",
)

question = ""
uploaded_files = []
if submission:
    # With accept_file enabled Streamlit returns a ChatInputValue.
    question = getattr(submission, "text", "") or ""
    uploaded_files = list(getattr(submission, "files", []) or [])
if "pending" in st.session_state and not question and not uploaded_files:
    question = st.session_state.pop("pending")

if question or uploaded_files:
    attachment_names = [f.name for f in uploaded_files]
    visible_text = question.strip() or ("📎 " + "، ".join(attachment_names))
    user_event = {
        "role": "user", "content": visible_text, "kind": "text",
        "attachments": attachment_names,
    }
    st.session_state.messages.append(user_event)
    render_message(user_event)

    comparison_event = None
    pf_event = None
    parsed_uploaded_rows = []
    parse_errors = []

    turn_route = classify_turn(question, uploaded_files)
    explicit_review_intent = (turn_route == "facilities_review")
    pending_before = st.session_state.get(PF_PENDING_SUBMISSION_KEY)
    continuing_review = st.session_state.get(PF_GOAL_KEY) == "review" or bool(pending_before)

    # A review/deficit request is a persistent goal, not just a keyword on one turn.
    if explicit_review_intent:
        st.session_state[PF_GOAL_KEY] = "review"
        st.session_state[PF_REVIEW_REQUESTED_KEY] = True
    elif uploaded_files and (continuing_review or not question.strip()):
        # A file dropped while a review is in progress belongs to that review.
        st.session_state[PF_GOAL_KEY] = "review"
        st.session_state[PF_REVIEW_REQUESTED_KEY] = True

    goal = st.session_state.get(PF_GOAL_KEY)

    # 1) Parse attachments FIRST. A failed review attachment must never silently fall through
    # to the calculator or UPPG and produce an unrelated answer.
    for uploaded in uploaded_files:
        try:
            rows, _ = parse_uploaded_facilities(uploaded)
            if rows:
                parsed_uploaded_rows.extend(rows)
            else:
                parse_errors.append(f"{uploaded.name}: لم أستطع تحديد صفوف خدمات عامة داخل الملف")
        except Exception as exc:
            parse_errors.append(f"{uploaded.name}: {exc}")

    if parsed_uploaded_rows:
        st.session_state[PF_PENDING_SUBMISSION_KEY] = {
            "rows": parsed_uploaded_rows,
            "source_label": "، ".join(attachment_names) or "الملف المرفق",
        }
        st.session_state[PF_GOAL_KEY] = "review"
        st.session_state[PF_REVIEW_REQUESTED_KEY] = True
        goal = "review"

    # Hard stop for a review file that could not be read. Do not answer with a cached calculation.
    if uploaded_files and not parsed_uploaded_rows and parse_errors and (goal == "review" or explicit_review_intent):
        event = {
            "role": "assistant",
            "content": (
                "أنا فاهم أن المطلوب **حساب العجز ومقارنة الملف**، وليس إعادة حساب الخدمات فقط. "
                "لكن لم أستطع استخراج جدول خدمات موثوق من المرفق، لذلك لن أعطيك نتيجة غير مرتبطة بطلبك.\n\n"
                "تفاصيل القراءة:\n- " + "\n- ".join(parse_errors) +
                "\n\nجرّب نفس الملف بعد التأكد أن أسماء الخدمات والأعداد موجودة كخلايا داخل Excel، "
                "أو أرسل نسخة PDF/Excel أخرى."
            ),
            "kind": "text",
        }
        render_message(event)
        st.session_state.messages.append(event)
    else:
        # 2) Text tables pasted into chat are also part of the review workflow.
        if not parsed_uploaded_rows and question and (
            looks_like_facilities_table(question) or
            (explicit_review_intent and ("\n" in question or "|" in question or "\t" in question))
        ):
            submitted_rows = parse_csv_or_text_table(question)
            if submitted_rows:
                st.session_state[PF_PENDING_SUBMISSION_KEY] = {
                    "rows": submitted_rows,
                    "source_label": "الجدول النصي المقدم في المحادثة",
                }
                st.session_state[PF_GOAL_KEY] = "review"
                st.session_state[PF_REVIEW_REQUESTED_KEY] = True
                goal = "review"

        pending = st.session_state.get(PF_PENDING_SUBMISSION_KEY)
        current_pf = st.session_state.get(PF_STATE_KEY) or {}
        have_reference = bool(current_pf.get("last_result"))

        # 3) If the goal is a deficit/compliance review and we already have both the
        # reference calculation and a submitted schedule, compare NOW. Never show the
        # generic calculator table first.
        if goal == "review" and pending and pending.get("rows") and have_reference and not current_pf.get("active"):
            comparison_event = build_comparison_chat_event(
                pending["rows"], source_label=pending.get("source_label", "الملف/الجدول المقدم")
            )
            render_message(comparison_event)
            st.session_state.messages.append(comparison_event)
            st.session_state.pop(PF_PENDING_SUBMISSION_KEY, None)
            st.session_state[PF_REVIEW_REQUESTED_KEY] = False
            st.session_state[PF_GOAL_KEY] = None
        else:
            # 4) A review with no reference calculation uses the calculator only as an
            # internal intake step. Ask only for missing area/population.
            should_run_pf = False
            pf_prompt = question
            if goal == "review":
                should_run_pf = True
                # Force entry into the facilities state machine even when the user's wording
                # is simply "العجز" or the turn contains only a number.
                if not current_pf.get("active") and not have_reference:
                    pf_prompt = ((question or "") + "\nاحسب الخدمات العامة").strip()
            elif turn_route == "facilities_calculation" and question:
                should_run_pf = True

            if should_run_pf and pf_prompt:
                pf_event = handle_public_facilities_turn(pf_prompt)

            if pf_event is not None:
                # If this was only an internal calculation step for a review, do not dump the
                # full requirements table into chat. Once complete, move directly to comparison.
                if goal == "review" and pf_event.get("result"):
                    pending = st.session_state.get(PF_PENDING_SUBMISSION_KEY)
                    if pending and pending.get("rows"):
                        comparison_event = build_comparison_chat_event(
                            pending["rows"], source_label=pending.get("source_label", "الملف/الجدول المقدم")
                        )
                        render_message(comparison_event)
                        st.session_state.messages.append(comparison_event)
                        st.session_state.pop(PF_PENDING_SUBMISSION_KEY, None)
                        st.session_state[PF_REVIEW_REQUESTED_KEY] = False
                        st.session_state[PF_GOAL_KEY] = None
                    else:
                        follow = {
                            "role": "assistant",
                            "content": (
                                "تم تحديد المتطلبات المرجعية للمشروع. الآن أرفق أو اسحب **جدول الخدمات التي وفرتها** "
                                "(Excel / CSV / TXT / PDF)، وسأحسب العجز مباشرة وأصدر تقرير المقارنة."
                            ),
                            "kind": "text",
                        }
                        render_message(follow)
                        st.session_state.messages.append(follow)
                else:
                    render_message(pf_event)
                    st.session_state.messages.append(pf_event)
            elif question and goal != "review" and turn_route == "uppg":
                with st.chat_message("assistant"):
                    with st.spinner("Searching the UPPG and preparing the answer…"):
                        try:
                            answer, retrieved = answer_question(question, st.session_state.messages[:-1])
                            st.markdown(answer)
                            with st.expander("Retrieved UPPG evidence"):
                                for c in retrieved:
                                    st.markdown(f"**Page {c['page']}**")
                                    st.write(c["text"][:1200] + ("…" if len(c["text"]) > 1200 else ""))
                                    st.divider()
                        except Exception as e:
                            answer = f"⚠️ AI service is not ready: `{e}`"
                            st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer, "kind": "text"})
            elif question and goal != "review" and turn_route == "general":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking…"):
                        try:
                            answer = clean_ai_answer(general_answer(question, st.session_state.messages[:-1]))
                            st.markdown(answer)
                        except Exception as e:
                            answer = f"⚠️ AI service is not ready: `{e}`"
                            st.error(answer)
                st.session_state.messages.append({"role":"assistant","content":answer,"kind":"text"})

st.divider()
st.caption(
    "Source basis: Urban Planning Permits Guideline (UPPG), 139 pages, plus the Public Facilities Calculator rules "
    "extracted from the supplied Excel workbook. The calculator uses project area and population to determine density, "
    "the applicable service level, and all mandatory facilities up to that level. Submitted service schedules can be compared against the calculated requirements and exported as a PDF review report."
)
