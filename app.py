import os
import re
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from rank_bm25 import BM25Okapi

from public_facilities_chat import new_flow_state, process_public_facilities_turn

APP_DIR = Path(__file__).parent
TEXT_PATH = APP_DIR / "guide_pages.txt"
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
API_URL = "https://openrouter.ai/api/v1/chat/completions"
PF_STATE_KEY = "public_facilities_flow"

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


def openrouter_chat(messages, max_tokens=900, temperature=0.15):
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set in Render Environment Variables.")

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://mp-ai-assistant.onrender.com"),
        "X-Title": "Master Plan AI Assistant",
    }

    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "low", "exclude": True},
    }

    last_error = None
    for attempt in range(2):
        try:
            r = requests.post(API_URL, headers=headers, json=payload, timeout=120)
            if not r.ok:
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text
                raise RuntimeError(f"OpenRouter error {r.status_code}: {detail}")

            data = r.json()
            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"OpenRouter returned no choices: {data}")

            message = choices[0].get("message") or {}
            content = message.get("content")

            if isinstance(content, str) and content.strip():
                return content.strip()

            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        value = item.get("text") or item.get("content")
                        if isinstance(value, str):
                            parts.append(value)
                joined = "\n".join(p for p in parts if p).strip()
                if joined:
                    return joined

            finish_reason = choices[0].get("finish_reason")
            provider_error = data.get("error")
            raise RuntimeError(
                "OpenRouter returned an empty answer "
                f"(finish_reason={finish_reason}, error={provider_error})."
            )

        except Exception as exc:
            last_error = exc
            if attempt == 0:
                continue

    raise RuntimeError(
        f"OpenRouter did not return a usable answer after retrying: {last_error}"
    )


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
        return openrouter_chat([{"role": "user", "content": prompt}], max_tokens=120, temperature=0)
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

    answer = openrouter_chat(
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
        "معيار السكان/مرفق",
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
        if m.get("kind") == "public_facilities":
            render_public_facilities_table(m.get("table_rows") or [])
            result = m.get("result") or {}
            excluded = result.get("not_required_in_development") or []
            if excluded:
                with st.expander("بنود مرجعية غير مطلوبة من المشروع التطويري"):
                    ex_df = pd.DataFrame(excluded)
                    cols = [c for c in ["level", "service", "note_type"] if c in ex_df.columns]
                    st.dataframe(ex_df[cols], use_container_width=True, hide_index=True)


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
    '<div class="sub">UPPG guidance + Public Facilities Calculator inside the same chat</div>',
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
init_public_facilities_state()

with st.sidebar:
    st.header("Assistant")
    st.success("AI + UPPG RAG enabled")
    st.info("Public Facilities Calculator enabled")
    st.caption(f"Model route: {MODEL}")

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state[PF_STATE_KEY] = new_flow_state()
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

question = st.chat_input("اسأل عن UPPG أو اطلب حساب الخدمات العامة...")
if "pending" in st.session_state and not question:
    question = st.session_state.pop("pending")

if question:
    user_event = {"role": "user", "content": question, "kind": "text"}
    st.session_state.messages.append(user_event)
    render_message(user_event)

    # IMPORTANT: calculator workflow gets the turn first. If not applicable,
    # the question falls back to the existing UPPG RAG assistant.
    pf_event = handle_public_facilities_turn(question)

    if pf_event is not None:
        render_message(pf_event)
        st.session_state.messages.append(pf_event)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Understanding the question and searching the UPPG…"):
                try:
                    answer, retrieved, expanded = answer_question(
                        question,
                        st.session_state.messages[:-1],
                    )
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

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "kind": "text",
        })

st.divider()
st.caption(
    "Source basis: Urban Planning Permits Guideline (UPPG), 139 pages, plus the Public Facilities Calculator rules "
    "extracted from the supplied Excel workbook. The calculator uses project area and population to determine density, "
    "the applicable service level, and the required facilities up to that level only."
)
