from __future__ import annotations

import re
from typing import Dict, Any, List


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

REVIEW_CUES = (
    "systematic review",
    "meta-analysis",
    "meta analysis",
    "review",
)

POSITIVE_STUDY_TYPE_CUES = (
    "randomized",
    "randomised",
    "trial",
    "placebo",
    "double-blind",
    "double blind",
)

AGE_CUES = (
    "older adults",
    "elderly",
    "aging",
    "aged",
)

FULLTEXT_SOURCE_BONUS = {
    "pdf": 0.55,
    "pmc_html": 0.45,
    "pubmed_abstract": 0.18,
    "openalex_abstract": 0.10,
    "existing_full_text": 0.30,
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _overlap_score(query: str, text: str) -> float:
    q = set(_tokenize(query))
    t = set(_tokenize(text))
    if not q or not t:
        return 0.0
    return len(q.intersection(t)) / max(1, len(q))


def _contains_any(text: str, cues: tuple[str, ...]) -> int:
    tl = (text or "").lower()
    return sum(1 for cue in cues if cue in tl)


def rerank_hits(query: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Reordena hits con señales heurísticas adicionales.
    Prioriza más agresivamente:
    - relevancia textual
    - reviews/meta-analysis
    - disponibilidad de full text rico
    """
    reranked: List[Dict[str, Any]] = []

    ql = query.lower()

    for h in hits:
        title = h.get("title") or ""
        abstract = h.get("abstract") or ""
        full_text_preview = h.get("full_text_preview") or ""
        has_full_text = bool(h.get("has_full_text"))
        full_text_source = (h.get("full_text_source") or "").strip().lower()

        title_overlap = _overlap_score(query, title)
        abstract_overlap = _overlap_score(query, abstract)
        fulltext_overlap = _overlap_score(query, full_text_preview)

        review_bonus = 0.20 * _contains_any(title + " " + abstract, REVIEW_CUES)
        study_type_bonus = 0.08 * _contains_any(title + " " + abstract, POSITIVE_STUDY_TYPE_CUES)

        # Full text bonuses
        fulltext_bonus = 0.28 if has_full_text else -0.15
        fulltext_source_bonus = FULLTEXT_SOURCE_BONUS.get(full_text_source, 0.0)

        # Preview length bonus
        preview_len = len(full_text_preview.strip()) if full_text_preview else 0
        preview_bonus = 0.0
        if preview_len >= 20000:
            preview_bonus = 0.30
        elif preview_len >= 8000:
            preview_bonus = 0.22
        elif preview_len >= 3000:
            preview_bonus = 0.18
        elif preview_len >= 800:
            preview_bonus = 0.10

        age_bonus = 0.0
        if any(cue in ql for cue in AGE_CUES):
            age_bonus = 0.12 * _contains_any(title + " " + abstract, AGE_CUES)

        bm25_score = float(h.get("score", 0.0))

        rerank_score = (
            0.42 * bm25_score
            + 1.20 * title_overlap
            + 0.85 * abstract_overlap
            + 0.70 * fulltext_overlap
            + review_bonus
            + study_type_bonus
            + fulltext_bonus
            + fulltext_source_bonus
            + preview_bonus
            + age_bonus
        )

        item = dict(h)
        item["rerank_score"] = float(rerank_score)
        reranked.append(item)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked
