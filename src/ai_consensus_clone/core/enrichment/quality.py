from __future__ import annotations

import re
from typing import Optional


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

BAD_TEXT_CUES = (
    "download pdf",
    "cookie",
    "all rights reserved",
    "access through your institution",
    "buy article",
)

GOOD_TEXT_CUES = (
    "abstract",
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "participants",
    "randomized",
    "trial",
)


def token_count(text: Optional[str]) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def looks_like_meaningful_full_text(text: Optional[str], min_tokens: int = 200) -> bool:
    if not text:
        return False

    tl = text.lower()
    n_tokens = token_count(text)

    if n_tokens < min_tokens:
        return False

    bad_hits = sum(1 for cue in BAD_TEXT_CUES if cue in tl)
    if bad_hits >= 2:
        return False

    return True


def quality_score(text: Optional[str], source: Optional[str]) -> float:
    if not text:
        return 0.0

    tl = text.lower()
    n_tokens = token_count(text)

    score = 0.0

    if n_tokens >= 200:
        score += 0.8
    if n_tokens >= 800:
        score += 0.8
    if n_tokens >= 2000:
        score += 0.8

    good_hits = sum(1 for cue in GOOD_TEXT_CUES if cue in tl)
    score += 0.15 * good_hits

    bad_hits = sum(1 for cue in BAD_TEXT_CUES if cue in tl)
    score -= 0.30 * bad_hits

    if source == "pdf":
        score += 0.35
    elif source == "pmc_html":
        score += 0.30
    elif source == "pubmed_abstract":
        score += 0.10
    elif source == "openalex_abstract":
        score += 0.05

    return max(0.0, round(score, 4))
