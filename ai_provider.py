from __future__ import annotations

import os
from typing import Any

import requests


def _extract_openrouter_text(data: dict[str, Any]) -> str:
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
    raise RuntimeError(f"OpenRouter returned an empty answer (finish_reason={finish_reason}).")


def _openrouter_chat(messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    model = os.getenv("OPENROUTER_MODEL", "openrouter/free")
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("APP_URL", "https://mp-ai-assistant.onrender.com"),
        "X-Title": "Master Plan AI Assistant",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"effort": "low", "exclude": True},
    }
    last_error = None
    for attempt in range(2):
        try:
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120)
            if not r.ok:
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text
                raise RuntimeError(f"OpenRouter error {r.status_code}: {detail}")
            return _extract_openrouter_text(r.json())
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                continue
    raise RuntimeError(f"OpenRouter did not return a usable answer after retrying: {last_error}")


def _google_chat(messages: list[dict[str, str]], max_tokens: int, temperature: float) -> str:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")
    model = os.getenv("GOOGLE_MODEL", "gemini-3.8-flash")

    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system" and m.get("content")]
    system_text = "\n\n".join(system_parts).strip()
    contents = []
    first_user = True
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        text = m.get("content", "")
        if not text:
            continue
        if role == "assistant":
            g_role = "model"
        else:
            g_role = "user"
            if first_user and system_text:
                text = f"SYSTEM INSTRUCTIONS:\n{system_text}\n\nUSER REQUEST:\n{text}"
                first_user = False
        contents.append({"role": g_role, "parts": [{"text": text}]})

    if not contents:
        raise RuntimeError("No message content supplied to Google AI.")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            # Keep real-time chat concise. Gemini 3.8 Flash supports low/medium/high thinking.
            "thinkingConfig": {"thinkingLevel": os.getenv("GOOGLE_THINKING_LEVEL", "low")},
        },
    }
    headers = {"x-goog-api-key": key, "Content-Type": "application/json"}
    last_error = None
    for attempt in range(2):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=120)
            if not r.ok:
                try:
                    detail = r.json()
                except Exception:
                    detail = r.text
                raise RuntimeError(f"Google AI error {r.status_code}: {detail}")
            data = r.json()
            candidates = data.get("candidates") or []
            if not candidates:
                raise RuntimeError(f"Google AI returned no candidates: {data}")
            parts = ((candidates[0].get("content") or {}).get("parts") or [])
            text = "\n".join(p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")).strip()
            if not text:
                raise RuntimeError(f"Google AI returned an empty response: {data}")
            return text
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                continue
    raise RuntimeError(f"Google AI did not return a usable answer after retrying: {last_error}")


def configured_provider() -> str:
    requested = os.getenv("AI_PROVIDER", "auto").strip().lower()
    google_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if requested in {"google", "gemini", "gemma"}:
        return "google"
    if requested == "openrouter":
        return "openrouter"
    if google_key:
        return "google"
    if openrouter_key:
        return "openrouter"
    return "google" if requested == "google" else "openrouter"


def provider_label() -> str:
    p = configured_provider()
    if p == "google":
        return f"Google AI · {os.getenv('GOOGLE_MODEL', 'gemini-3.8-flash')}"
    return f"OpenRouter · {os.getenv('OPENROUTER_MODEL', 'openrouter/free')}"


def ai_chat(messages: list[dict[str, str]], max_tokens: int = 900, temperature: float = 0.15) -> str:
    provider = configured_provider()
    if provider == "google":
        return _google_chat(messages, max_tokens=max_tokens, temperature=temperature)
    return _openrouter_chat(messages, max_tokens=max_tokens, temperature=temperature)
