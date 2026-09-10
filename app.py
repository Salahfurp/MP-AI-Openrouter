import os
import re
from pathlib import Path

import requests
import streamlit as st
from rank_bm25 import BM25Okapi

APP_DIR = Path(__file__).parent
TEXT_PATH = APP_DIR / "guide_pages.txt"
MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
API_URL = "https://openrouter.ai/api/v1/chat/completions"

st.set_page_config(page_title="Master Plan AI Assistant", page_icon="🏙️", layout="wide")

SYSTEM_PROMPT = """You are the Master Plan AI Assistant for Dubai Urban Planning Permits.
Your authoritative source is ONLY the UPPG excerpts supplied in the prompt.
Do not invent planning requirements. If the excerpts do not support an answer, say that clearly.
Answer in the user's language (Arabic or English), while preserving official English planning terms when useful.
Be practical and concise. Distinguish mandatory requirements from conditional/context-dependent items.
Never present the answer as an official approval, legal opinion, or final compliance determination.
Always end with a short 'Source' line listing the UPPG page numbers that support the answer.
"""

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
    }
    r = requests.post(API_URL, headers=headers, json=payload, timeout=90)
    if not r.ok:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise RuntimeError(f"OpenRouter error {r.status_code}: {detail}")
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()

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
    history_text = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
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
    return answer, retrieved, expanded

st.markdown("""
<style>
.main-title{font-size:2.2rem;font-weight:800;margin-bottom:.2rem}
.sub{color:#667085;margin-bottom:1.2rem}
.small-note{font-size:.85rem;color:#667085}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏙️ Master Plan AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">AI-powered assistant grounded in the Urban Planning Permits Guideline (UPPG)</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("Assistant")
    st.success("AI + UPPG RAG enabled")
    st.caption(f"Model route: {MODEL}")
    if st.button("Reset conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.markdown("**Try asking:**")
    st.caption("• What documents are required for a Master Plan Permit?\n\n• متى أحتاج مواءمة مع Dubai 2040؟\n\n• What studies may be required for a major modification?")
    st.divider()
    st.caption("Prototype only. Answers are preliminary guidance, not an official approval or compliance determination.")

if "messages" not in st.session_state:
    st.session_state.messages = []

col1, col2, col3, col4 = st.columns(4)
quick = [
    (col1, "📋 Requirements", "What documents are required for a Master Plan Permit?"),
    (col2, "🔎 Process", "Explain the Master Plan approval process step by step."),
    (col3, "🧭 Dubai 2040", "When is Dubai 2040 alignment required?"),
    (col4, "📝 Checklist", "Create a preliminary checklist for a new master plan submission."),
]
for col, label, q in quick:
    if col.button(label, use_container_width=True):
        st.session_state.pending = q

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Ask in Arabic or English about the UPPG...")
if "pending" in st.session_state and not question:
    question = st.session_state.pop("pending")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Understanding the question and searching the UPPG…"):
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
    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption("Source basis: Urban Planning Permits Guideline (UPPG), 139 pages. The assistant retrieves relevant guide excerpts first, then asks the AI to answer only from that evidence.")
