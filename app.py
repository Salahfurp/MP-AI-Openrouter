import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from rank_bm25 import BM25Okapi

from ai_provider import ai_chat, provider_label

from public_facilities_chat import new_flow_state, process_public_facilities_turn
from public_facilities_comparison import (
    SubmittedFacility, compare_facilities, comparison_rows, parse_csv_or_text_table,
    parse_excel_bytes, parse_pdf_bytes, looks_like_facilities_table,
)
from public_facilities_report import build_comparison_pdf

APP_DIR = Path(__file__).parent
TEXT_PATH = APP_DIR / "guide_pages.txt"
PF_STATE_KEY = "public_facilities_flow"
PF_PENDING_SUBMISSION_KEY = "pending_public_facilities_submission"
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


def expand_query(question):
    prompt = f"""Convert the user's question into a compact English search query for an Urban Planning Permits Guideline.
Return only 8-18 useful search terms/phrases, no explanation.
Include likely official terminology and synonyms. If the question is Arabic, translate its planning meaning to English.
User question: {question}"""
    try:
        return model_chat([{"role": "user", "content": prompt}], max_tokens=120, temperature=0)
    except Exception:
        return question


def retrieve(question, top_k=8):
    chunks, bm25 = load_guide()
    expanded = expand_query(question)
    query_tokens = tokenize(question + " " + expanded)
    scores = bm25.get_scores(query_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    seen = set()
    for i in ranked:
        c = chunks[i]
        signature = (c["page"], c["text"][:80])
        if signature in seen:
            continue
        seen.add(signature)
        results.append(c)
        if len(results) >= top_k:
            break
    return results, expanded


def answer_question(question, history):
    retrieved, expanded = retrieve(question)
    context = "\n\n".join(
        f"[UPPG PAGE {c['page']}]\n{c['text']}" for c in retrieved
    )

    recent = history[-6:]
    history_text = "\n".join(
        f"{m.get('role')}: {m.get('content', '')}" for m in recent
        if m.get("kind", "text") == "text"
    )

    user_prompt = f"""Use ONLY the UPPG excerpts below to answer the user's latest question.
If the answer is not established by these excerpts, say that the available retrieved sections do not confirm it and avoid guessing.
Cite page numbers that appear in the excerpt labels.

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
        max_tokens=1100,
        temperature=0.1,
    )
    answer = clean_ai_answer(answer)
    return answer, retrieved, expanded


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


def parse_uploaded_facilities(uploaded_file):
    if uploaded_file is None:
        return [], None
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    if name.endswith((".xlsx", ".xls")):
        return parse_excel_bytes(data, uploaded_file.name), None
    if name.endswith(".csv"):
        return parse_csv_or_text_table(data.decode("utf-8", errors="ignore")), None
    if name.endswith(".txt"):
        return parse_csv_or_text_table(data.decode("utf-8", errors="ignore")), None
    if name.endswith(".pdf"):
        rows, extracted = parse_pdf_bytes(data, ai_json_parser=ai_parse_pdf_facilities)
        return rows, extracted
    raise ValueError("صيغة الملف غير مدعومة. استخدم Excel أو CSV أو TXT أو PDF.")


def build_comparison_chat_event(submitted_rows, source_label="الجدول المقدم"):
    state = st.session_state.get(PF_STATE_KEY) or {}
    required_result = state.get("last_result")
    if not required_result:
        return {
            "role": "assistant",
            "content": "قبل مراجعة جدول الخدمات، أحتاج أولًا حساب المتطلبات للمشروع. اطلب **حساب الخدمات العامة** وأدخل مساحة الأرض وعدد السكان، ثم أعد إرسال الجدول للمقارنة.",
            "kind": "text",
        }
    if not submitted_rows:
        return {
            "role": "assistant",
            "content": "لم أستطع العثور على صفوف خدمات قابلة للمقارنة في البيانات المقدمة. تأكد أن الجدول يحتوي على الأقل على **اسم الخدمة** ويفضل **العدد**، ويمكن أيضًا إضافة **مساحة الأرض** و **GFA**.",
            "kind": "text",
        }

    report = compare_facilities(required_result, submitted_rows)
    rows = comparison_rows(report)
    summary = report.get("summary", {})
    deficits = [x for x in report.get("comparisons", []) if x.get("status") == "عجز"]
    if deficits:
        top = "، ".join(x.get("service", "") for x in deficits[:5])
        more = "" if len(deficits) <= 5 else f" وغيرها ({len(deficits)-5} خدمة إضافية)"
        message = (
            f"تمت مراجعة **{source_label}** مقابل الخدمات الإلزامية المحسوبة للمشروع. "
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
        st.dataframe(df.style.map(color_status, subset=["الحالة / Status"]), use_container_width=True, hide_index=True)
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

    # 1) Attachments are handled inside the chat before any UPPG search.
    for uploaded in uploaded_files:
        try:
            rows, _ = parse_uploaded_facilities(uploaded)
            if rows:
                parsed_uploaded_rows.extend(rows)
            else:
                parse_errors.append(f"{uploaded.name}: لم أجد صفوف خدمات قابلة للقراءة")
        except Exception as exc:
            parse_errors.append(f"{uploaded.name}: {exc}")

    if parsed_uploaded_rows:
        current_pf = st.session_state.get(PF_STATE_KEY) or {}
        if current_pf.get("last_result"):
            comparison_event = build_comparison_chat_event(
                parsed_uploaded_rows,
                source_label="، ".join(attachment_names) or "الملف المرفق",
            )
        else:
            # Keep the uploaded schedule in memory while the assistant asks for area/population.
            st.session_state[PF_PENDING_SUBMISSION_KEY] = {
                "rows": parsed_uploaded_rows,
                "source_label": "، ".join(attachment_names) or "الملف المرفق",
            }
            pf_prompt = (question + "\nاحسب الخدمات العامة").strip()
            pf_event = handle_public_facilities_turn(pf_prompt)

    # 2) Pasted Arabic/English schedules are recognized even if the user did not say the word 'compare'.
    if comparison_event is None and not parsed_uploaded_rows and question:
        pending = st.session_state.get(PF_PENDING_SUBMISSION_KEY)
        if detect_facilities_comparison_intent(question) and pending and pending.get("rows"):
            comparison_event = build_comparison_chat_event(
                pending["rows"], source_label=pending.get("source_label", "الملف المرفق")
            )
            st.session_state.pop(PF_PENDING_SUBMISSION_KEY, None)
        elif looks_like_facilities_table(question) or (detect_facilities_comparison_intent(question) and ("\n" in question or "|" in question or "\t" in question)):
            submitted_rows = parse_csv_or_text_table(question)
            if submitted_rows:
                current_pf = st.session_state.get(PF_STATE_KEY) or {}
                if current_pf.get("last_result"):
                    comparison_event = build_comparison_chat_event(submitted_rows, source_label="الجدول النصي المقدم في المحادثة")
                else:
                    st.session_state[PF_PENDING_SUBMISSION_KEY] = {"rows": submitted_rows, "source_label": "الجدول النصي المقدم في المحادثة"}
                    pf_event = handle_public_facilities_turn("احسب الخدمات العامة")

    # 3) Continue / start calculator before falling back to UPPG.
    if comparison_event is None and pf_event is None and question:
        pf_event = handle_public_facilities_turn(question)

    if comparison_event is not None:
        render_message(comparison_event)
        st.session_state.messages.append(comparison_event)
    elif pf_event is not None:
        render_message(pf_event)
        st.session_state.messages.append(pf_event)

        # If a schedule was attached first, auto-compare immediately after area/population calculation completes.
        if pf_event.get("result"):
            pending = st.session_state.get(PF_PENDING_SUBMISSION_KEY)
            if pending and pending.get("rows"):
                auto_event = build_comparison_chat_event(
                    pending["rows"], source_label=pending.get("source_label", "الملف/الجدول المقدم")
                )
                render_message(auto_event)
                st.session_state.messages.append(auto_event)
                st.session_state.pop(PF_PENDING_SUBMISSION_KEY, None)
    elif parse_errors:
        event = {
            "role": "assistant",
            "content": "تعذر قراءة بعض المرفقات:\n- " + "\n- ".join(parse_errors) +
                       "\n\nتأكد أن الملف يحتوي على جدول خدمات واضح. PDF الممسوح كصورة فقط قد يحتاج نسخة نصية/Excel.",
            "kind": "text",
        }
        render_message(event)
        st.session_state.messages.append(event)
    elif question:
        # Only genuine planning Q&A reaches the UPPG retrieval path. Facility tables/files never fall through here.
        with st.chat_message("assistant"):
            with st.spinner("Understanding the planning question and searching the UPPG…"):
                try:
                    answer, retrieved, expanded = answer_question(question, st.session_state.messages[:-1])
                    st.markdown(answer)
                    with st.expander("Retrieved UPPG evidence"):
                        st.caption(f"Search expansion: {expanded}")
                        for c in retrieved:
                            st.markdown(f"**Page {c['page']}**")
                            st.write(c["text"][:1200] + ("…" if len(c["text"]) > 1200 else ""))
                            st.divider()
                except Exception as e:
                    answer = f"⚠️ AI service is not ready: `{e}`"
                    st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer, "kind": "text"})

st.divider()
st.caption(
    "Source basis: Urban Planning Permits Guideline (UPPG), 139 pages, plus the Public Facilities Calculator rules "
    "extracted from the supplied Excel workbook. The calculator uses project area and population to determine density, "
    "the applicable service level, and all mandatory facilities up to that level. Submitted service schedules can be compared against the calculated requirements and exported as a PDF review report."
)
