from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from ai_consensus_clone.core.config.settings import Settings


def _load_prompt() -> str:
    prompt_path = Path(__file__).parent / "prompts" / "answer_prompt.txt"
    return prompt_path.read_text(encoding="utf-8")


def _build_user_payload(
    question: str,
    evidences: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> str:
    compact_evidences = []
    for ev in evidences:
        compact_evidences.append(
            {
                "paper_id": ev.get("paper_id"),
                "title": ev.get("title"),
                "year": ev.get("year"),
                "venue": ev.get("venue"),
                "doi": ev.get("doi"),
                "full_text_source": ev.get("full_text_source"),
                "score": ev.get("score"),
                "span": ev.get("span"),
            }
        )

    payload = {
        "question": question,
        "evidences": compact_evidences,
        "citations": citations,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_llm_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None

    # intenta parse directo
    try:
        obj = json.loads(text)
        return _validate_llm_output(obj)
    except Exception:
        pass

    # intenta extraer bloque JSON
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            return _validate_llm_output(obj)
        except Exception:
            return None

    return None


def _validate_llm_output(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    conclusion = (obj.get("conclusion") or "").strip()
    confidence = (obj.get("confidence") or "").strip().lower()

    if not conclusion:
        return None

    if confidence not in {"alta", "media", "baja"}:
        return None

    return {
        "conclusion": conclusion,
        "confidence": confidence,
    }


def build_consensus_answer_with_llm(
    question: str,
    evidences: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    settings = Settings()

    if (settings.llm_provider or "none").lower() == "none":
        return None

    if settings.llm_provider.lower() == "openai":
        return _build_with_openai(
            question=question,
            evidences=evidences,
            citations=citations,
            settings=settings,
        )

    return None


def _build_with_openai(
    question: str,
    evidences: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    settings: Settings,
) -> Optional[Dict[str, Any]]:
    try:
        from openai import OpenAI
    except Exception:
        return None

    if not settings.llm_api_key:
        return None

    system_prompt = _load_prompt()
    user_payload = _build_user_payload(
        question=question,
        evidences=evidences,
        citations=citations,
    )

    try:
        client = OpenAI(api_key=settings.llm_api_key, timeout=settings.llm_timeout)

        response = client.chat.completions.create(
            model=settings.llm_model,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_payload},
            ],
            response_format={"type": "json_object"},
        )

        text = response.choices[0].message.content or ""
        return _parse_llm_json(text)

    except Exception:
        return None
