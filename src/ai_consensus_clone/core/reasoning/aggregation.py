from __future__ import annotations

import math
from typing import Dict, Any, List


STANCE_STRENGTH_WEIGHTS = {
    "weak": 0.75,
    "moderate": 1.0,
    "strong": 1.35,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compute_citation_weight(citation_count: int | None) -> float:
    """
    Weight based on paper citations.

    Examples:
    0 citations   -> 1.0
    10 citations  -> 1.24
    100 citations -> 1.92
    1000 citations -> capped
    """

    if not citation_count or citation_count <= 0:
        return 1.0

    weight = 1.0 + (math.log1p(citation_count) / 5.0)

    return min(weight, 2.5)


def compute_relevance_weight(score: float | None) -> float:
    """
    Converts rerank/retrieval score into usable weight.
    """

    score = _safe_float(score, 0.0)

    if score <= 0:
        return 0.75

    if score >= 5:
        return 1.5

    return min(0.75 + (score / 5.0), 1.5)


def compute_strength_weight(strength: str | None) -> float:
    """
    Maps LLM stance strength to numeric weight.
    """

    if not strength:
        return 1.0

    return STANCE_STRENGTH_WEIGHTS.get(
        str(strength).lower().strip(),
        1.0,
    )


def compute_paper_weight(
    paper_stance: Any,
    citation_count: int | None = None,
    retrieval_score: float | None = None,
) -> float:
    """
    Final paper importance score.
    """

    strength = getattr(paper_stance, "strength", None)

    citation_weight = compute_citation_weight(citation_count)
    relevance_weight = compute_relevance_weight(retrieval_score)
    strength_weight = compute_strength_weight(strength)

    final_weight = (
        citation_weight
        * relevance_weight
        * strength_weight
    )

    return round(final_weight, 4)


def aggregate_weighted_stances(
    paper_stances: List[Any],
    hits: List[Dict[str, Any]],
) -> Dict[str, Any]:

    support_weight = 0.0
    contradict_weight = 0.0
    neutral_weight = 0.0

    support_count = 0
    contradict_count = 0
    neutral_count = 0

    paper_weights: List[Dict[str, Any]] = []

    hit_map = {
        h.get("paper_id"): h
        for h in hits
    }

    for ps in paper_stances:

        stance = getattr(ps, "stance", "neutral")

        paper_id = getattr(ps, "paper_id", None)

        hit = hit_map.get(paper_id, {})

        citation_count = hit.get("citation_count", 0)
        retrieval_score = (
            hit.get("rerank_score")
            or hit.get("score")
            or 0.0
        )

        weight = compute_paper_weight(
            paper_stance=ps,
            citation_count=citation_count,
            retrieval_score=retrieval_score,
        )

        paper_weights.append(
            {
                "paper_id": paper_id,
                "stance": stance,
                "weight": weight,
                "citation_count": citation_count,
                "retrieval_score": retrieval_score,
            }
        )

        if stance == "support":
            support_weight += weight
            support_count += 1

        elif stance == "contradict":
            contradict_weight += weight
            contradict_count += 1

        else:
            neutral_weight += weight
            neutral_count += 1

    effective_total = support_weight + contradict_weight

    if effective_total <= 0:
        dominant_stance = "neutral"
        consensus_score = 0.0

    else:

        support_ratio = support_weight / effective_total
        contradict_ratio = contradict_weight / effective_total

        consensus_score = abs(
            support_ratio - contradict_ratio
        )

        if support_ratio >= 0.66:
            dominant_stance = "support"

        elif contradict_ratio >= 0.66:
            dominant_stance = "contradict"

        else:
            dominant_stance = "mixed"

    return {
        "support": support_count,
        "contradict": contradict_count,
        "neutral": neutral_count,

        "support_weight": round(support_weight, 4),
        "contradict_weight": round(contradict_weight, 4),
        "neutral_weight": round(neutral_weight, 4),

        "dominant_stance": dominant_stance,

        "consensus_score": round(consensus_score, 4),

        "paper_weights": paper_weights,
    }
