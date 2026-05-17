from __future__ import annotations

from typing import Dict, Any, List


def build_consensus_label(
    dominant_stance: str,
    consensus_score: float,
) -> str:

    if dominant_stance == "support":

        if consensus_score >= 0.75:
            return "Strong positive evidence"

        if consensus_score >= 0.45:
            return "Moderate positive evidence"

        return "Weak positive evidence"

    if dominant_stance == "contradict":

        if consensus_score >= 0.75:
            return "Strong negative evidence"

        if consensus_score >= 0.45:
            return "Moderate negative evidence"

        return "Weak negative evidence"

    if dominant_stance == "mixed":
        return "Mixed evidence"

    return "Insufficient evidence"


def build_confidence_numeric(confidence: str) -> float:

    mapping = {
        "alta": 0.9,
        "media": 0.65,
        "baja": 0.35,
    }

    return mapping.get(confidence.lower(), 0.35)


def group_evidences_by_stance(
    evidences: List[Dict[str, Any]],
    paper_stances: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:

    paper_stance_map = {}

    for ps in paper_stances:
        pid = ps.get("paper_id")
        stance = ps.get("stance", "neutral")

        if pid:
            paper_stance_map[pid] = stance

    grouped = {
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "neutral_evidence": [],
    }

    for ev in evidences:

        pid = ev.get("paper_id")
        stance = paper_stance_map.get(pid, "neutral")

        formatted = {
            "paper_id": pid,
            "title": ev.get("title"),
            "span": ev.get("span"),
            "score": round(float(ev.get("score", 0.0)), 4),
            "year": ev.get("year"),
            "venue": ev.get("venue"),
        }

        if stance == "support":
            grouped["supporting_evidence"].append(formatted)

        elif stance == "contradict":
            grouped["contradicting_evidence"].append(formatted)

        else:
            grouped["neutral_evidence"].append(formatted)

    return grouped


def build_top_papers(
    paper_weights: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    max_papers: int = 5,
) -> List[Dict[str, Any]]:

    citation_map = {
        c.get("paper_id"): c
        for c in citations
    }

    sorted_papers = sorted(
        paper_weights,
        key=lambda x: float(x.get("weight", 0.0)),
        reverse=True,
    )

    top_papers = []

    for pw in sorted_papers[:max_papers]:

        pid = pw.get("paper_id")
        citation = citation_map.get(pid, {})

        top_papers.append(
            {
                "paper_id": pid,
                "title": citation.get("title"),
                "doi": citation.get("doi"),
                "stance": pw.get("stance"),
                "weight": round(float(pw.get("weight", 0.0)), 4),
                "citation_count": pw.get("citation_count"),
            }
        )

    return top_papers


def build_contradiction_summary(
    support: int,
    contradict: int,
) -> str | None:

    if contradict == 0:
        return None

    if support > contradict:
        return (
            "Some contradictory findings exist, although most studies "
            "support the hypothesis."
        )

    if contradict > support:
        return (
            "Several studies report findings that do not support the hypothesis."
        )

    return (
        "Studies report conflicting findings and no clear consensus emerges."
    )


def build_consensus_api_output(
    *,
    conclusion: str,
    confidence: str,
    confidence_score: float,
    confidence_factors: Dict[str, Any],
    evidence_breakdown: Dict[str, Any],
    evidences: List[Dict[str, Any]],
    paper_stances: List[Dict[str, Any]],
    citations: List[Dict[str, Any]],
    aggregation_result: Dict[str, Any],
) -> Dict[str, Any]:

    dominant_stance = evidence_breakdown.get(
        "dominant_stance",
        "neutral",
    )

    consensus_score = float(
        aggregation_result.get("consensus_score", 0.0)
    )

    grouped_evidences = group_evidences_by_stance(
        evidences=evidences,
        paper_stances=paper_stances,
    )

    support = int(evidence_breakdown.get("support", 0))
    contradict = int(evidence_breakdown.get("contradict", 0))
    neutral = int(evidence_breakdown.get("neutral", 0))

    return {
        "conclusion": conclusion,

        "consensus_label": build_consensus_label(
            dominant_stance=dominant_stance,
            consensus_score=consensus_score,
        ),

        "dominant_stance": dominant_stance,

        "confidence": confidence,

        "confidence_numeric": build_confidence_numeric(
            confidence,
        ),

        "confidence_score": confidence_score,

        "consensus_score": round(consensus_score, 4),

        "contradictions_present": contradict > 0,

        "contradiction_summary": build_contradiction_summary(
            support=support,
            contradict=contradict,
        ),

        "supporting_evidence":
            grouped_evidences["supporting_evidence"],

        "contradicting_evidence":
            grouped_evidences["contradicting_evidence"],

        "neutral_evidence":
            grouped_evidences["neutral_evidence"],

        "top_papers": build_top_papers(
            paper_weights=aggregation_result.get(
                "paper_weights",
                [],
            ),
            citations=citations,
        ),

        "methodology": {
            "papers_analyzed": (
                support + contradict + neutral
            ),

            "supporting_papers": support,

            "contradicting_papers": contradict,

            "neutral_papers": neutral,
        },

        "confidence_factors": confidence_factors,

        "citations": citations,
    }
