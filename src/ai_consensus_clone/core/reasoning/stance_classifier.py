from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Literal, Any

from ai_consensus_clone.core.domain.paper import Paper
from ai_consensus_clone.core.reasoning.llm_client import LLMClient


StanceLabel = Literal["support", "contradict", "neutral"]
StrengthLabel = Literal["weak", "moderate", "strong"]


@dataclass
class PaperStance:
    paper_id: str
    stance: StanceLabel
    strength: StrengthLabel
    evidence: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _truncate_text(text: str, max_chars: int = 8000) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars]


def _safe_json_load(raw: str) -> dict[str, Any]:
    if not raw:
        return {}

    raw = raw.strip()

    try:
        return json.loads(raw)
    except Exception:
        pass

    try:
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(raw[start:end + 1])
    except Exception:
        pass

    return {}


def _normalize_stance(value: str) -> StanceLabel:
    value = (value or "").strip().lower()
    if value in {"support", "contradict", "neutral"}:
        return value  # type: ignore[return-value]
    return "neutral"


def _normalize_strength(value: str) -> StrengthLabel:
    value = (value or "").strip().lower()
    if value in {"weak", "moderate", "strong"}:
        return value  # type: ignore[return-value]
    return "weak"


def classify_paper_stance(
    llm_client: LLMClient,
    question: str,
    paper: Paper,
    prompt_template: str,
    max_chars: int = 8000,
) -> PaperStance:
    source_text = (
        getattr(paper, "full_text", None)
        or getattr(paper, "abstract", None)
        or ""
    ).strip()

    if not source_text:
        return PaperStance(
            paper_id=getattr(paper, "paper_id", ""),
            stance="neutral",
            strength="weak",
            evidence="",
            rationale="No text available to classify stance.",
        )

    source_text = _truncate_text(source_text, max_chars=max_chars)

    prompt = prompt_template.format(
        question=(question or "").strip(),
        title=(getattr(paper, "title", None) or "").strip(),
        text=source_text,
    )

    raw_output = llm_client.generate(prompt)
    parsed = _safe_json_load(raw_output)

    stance = _normalize_stance(parsed.get("stance", "neutral"))
    strength = _normalize_strength(parsed.get("strength", "weak"))
    evidence = (parsed.get("evidence") or "").strip()
    rationale = (parsed.get("rationale") or "").strip()

    return PaperStance(
        paper_id=getattr(paper, "paper_id", ""),
        stance=stance,
        strength=strength,
        evidence=evidence,
        rationale=rationale,
    )


def classify_papers_stances(
    llm_client: LLMClient,
    question: str,
    papers: list[Paper],
    prompt_template: str,
    max_chars: int = 8000,
) -> list[PaperStance]:
    results: list[PaperStance] = []

    for paper in papers:
        try:
            stance_result = classify_paper_stance(
                llm_client=llm_client,
                question=question,
                paper=paper,
                prompt_template=prompt_template,
                max_chars=max_chars,
            )
        except Exception as exc:
            stance_result = PaperStance(
                paper_id=getattr(paper, "paper_id", ""),
                stance="neutral",
                strength="weak",
                evidence="",
                rationale=f"Stance classification failed: {exc}",
            )

        results.append(stance_result)

    return results
