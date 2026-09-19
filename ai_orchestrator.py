from __future__ import annotations
import json, re
from dataclasses import dataclass
from typing import Any
from ai_provider import ai_chat

ROUTES = {"facilities_review","facilities_calculation","uppg","general"}

@dataclass
class Decision:
    route: str
    confidence: float = 0.0
    reason: str = ""
    suggested_action: str = ""

def _json_object(text: str) -> dict[str, Any]:
    text=(text or "").strip()
    text=re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text=re.sub(r"\s*```$", "", text)
    try: return json.loads(text)
    except Exception:
        m=re.search(r"\{.*\}", text, flags=re.S)
        if not m: return {}
        try: return json.loads(m.group(0))
        except Exception: return {}

def decide_route(question: str, history: list[dict], *, has_files=False, pf_active=False,
                 review_pending=False, has_facilities_result=False) -> Decision:
    recent=[]
    for m in history[-8:]:
        if m.get("kind","text") == "text":
            recent.append(f"{m.get('role')}: {m.get('content','')[:700]}")
    context="\n".join(recent)
    prompt=f"""You are the intent orchestrator for a bilingual Arabic/English Master Plan AI Assistant.
Understand meaning, paraphrases and conversational context. Do NOT route by literal keyword matching.
Choose exactly one route:
- facilities_review: compare required public facilities against facilities supplied/provided by the user; deficit, gap, missing facilities, sufficiency/compliance, or explaining a previous facilities comparison.
- facilities_calculation: calculate what public facilities are required from project area/population, without comparing against a submitted schedule.
- uppg: ONLY when the user is asking for an official guideline fact, permit requirement, required document, official process/criterion, Dubai 2040 requirement, modification rule, or study requirement that must be grounded in UPPG. Do NOT choose uppg merely because the conversation is about master planning or public facilities.
- general: planning-assistant conversation, clarification, recommendations, brainstorming, or anything not requiring one of the specialist tools.

Important:
* Arabic and English expressions with the same meaning MUST map to the same route.
* A user may switch topics at any time. Do not force a sequence.
* Attachments do not automatically mean facilities_review; use the user's goal and context.
* Public-facilities calculation/review is handled by the facilities tools, NOT by UPPG retrieval.
* General advice, explanations, suggestions and follow-ups should stay general unless an official UPPG fact is actually needed.
* If a facilities review is already pending and the user supplies a missing number/file, continue facilities_review.
* If the user asks 'what is missing', 'what is the difference between required and provided', 'احسب العجز', 'ايه النقص', these are facilities_review.

Runtime state:
has_files={has_files}
pf_active={pf_active}
review_pending={review_pending}
has_facilities_result={has_facilities_result}

Recent conversation:
{context}

Latest user message:
{question or '[attachment only]'}

Return ONLY JSON:
{{"route":"one route","confidence":0.0,"reason":"short reason","suggested_action":"short next action or empty"}}"""
    try:
        raw=ai_chat([{"role":"user","content":prompt}], max_tokens=220, temperature=0)
        data=_json_object(raw)
        route=str(data.get("route","")).strip()
        if route not in ROUTES: raise ValueError(route)
        return Decision(route, float(data.get("confidence") or 0), str(data.get("reason") or ""), str(data.get("suggested_action") or ""))
    except Exception:
        # Safe semantic-ish fallback; the AI router is preferred, but app remains usable if provider is down.
        q=(question or "").lower()
        review_words=("عجز","نقص","قارن","مقارن","المتوفر","المقدم","provided","missing","deficit","gap","compare","sufficient","compliance")
        calc_words=("احسب","الخدمات المطلوبة","public facilities required","calculate facilities")
        uppg_words=("uppg","permit","master plan","متطلبات","مستندات","اعتماد","اجراءات","إجراءات","dubai 2040","2040","modification","study","دراسة")
        if review_pending or any(x in q for x in review_words): return Decision("facilities_review", .45, "fallback")
        if pf_active or any(x in q for x in calc_words): return Decision("facilities_calculation", .4, "fallback")
        if any(x in q for x in uppg_words): return Decision("uppg", .4, "fallback")
        return Decision("general", .3, "fallback")

GENERAL_SYSTEM="""You are an intelligent bilingual Master Plan Assistant, not a rigid chatbot.
Understand Arabic and English naturally, preserve conversation context, answer practical planning-workflow questions, and proactively suggest useful next actions when appropriate.
Never invent official Dubai Municipality requirements. If an official requirement is needed, tell the user you can check the UPPG tool. Never claim official approval or final compliance.
Do not expose chain-of-thought. Be concise, useful and conversational in the user's language."""

def general_answer(question: str, history: list[dict]) -> str:
    recent=[]
    for m in history[-10:]:
        if m.get("kind","text") == "text": recent.append({"role":m.get("role","user"),"content":m.get("content","")})
    msgs=[{"role":"system","content":GENERAL_SYSTEM}] + recent + [{"role":"user","content":question}]
    return ai_chat(msgs, max_tokens=900, temperature=.25)
