from __future__ import annotations

import re
from typing import List, Tuple


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

RESULT_SECTION_CUES = (
    "result",
    "results",
    "finding",
    "findings",
    "conclusion",
    "conclusions",
    "discussion",
    "we found",
    "our findings",
)

POSITIVE_EFFECT_CUES = (
    "increase",
    "increased",
    "improve",
    "improved",
    "improvement",
    "greater",
    "higher",
    "enhanced",
    "enhancement",
    "benefit",
    "beneficial",
    "significant increase",
    "significantly increased",
    "gain",
    "gains",
    "augmented",
    "stronger",
)

NULL_OR_MIXED_CUES = (
    "no effect",
    "no significant",
    "not significant",
    "did not improve",
    "did not increase",
    "no difference",
    "similar between groups",
    "equivocal",
    "mixed",
    "inconsistent",
    "unclear",
    "limited evidence",
)

METHOD_CUES = (
    "methods",
    "method",
    "materials and methods",
    "participants",
    "subjects",
    "randomized",
    "randomly",
    "double-blind",
    "double blinded",
    "trial",
    "we examined",
    "we investigated",
    "the aim of this study",
    "this study evaluated",
    "background",
    "objective",
    "objectives",
)

QUANT_CUES = (
    "p<",
    "p <",
    "95% ci",
    "confidence interval",
    "mean difference",
    "standardized mean difference",
    "smd",
    "effect size",
    "significant",
)

GENERIC_BAD_SPANS = {
    "introduction",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "background",
    "objective",
    "objectives",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p and p.strip()]


def _make_passages(sentences: List[str], window: int = 2, stride: int = 1) -> List[Tuple[int, str]]:
    passages: List[Tuple[int, str]] = []
    n = len(sentences)

    for i in range(0, n, stride):
        chunk = sentences[i:i + window]
        if not chunk:
            continue
        passages.append((i, " ".join(chunk)))
        if i + window >= n:
            break

    return passages


def _count_cues(text_lower: str, cues: tuple[str, ...]) -> int:
    return sum(1 for cue in cues if cue in text_lower)


def _is_bad_generic_span(passage: str) -> bool:
    t = passage.strip().lower()
    if not t:
        return True

    if t in GENERIC_BAD_SPANS:
        return True

    compact = re.sub(r"[^a-z]+", " ", t).strip()
    if compact in GENERIC_BAD_SPANS:
        return True

    if len(_tokenize(t)) <= 4:
        return True

    return False


def _jaccard(a: str, b: str) -> float:
    sa = set(_tokenize(a))
    sb = set(_tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def score_passage(claim: str, passage: str) -> float:
    if _is_bad_generic_span(passage):
        return 0.0

    claim_toks = _tokenize(claim)
    passage_toks = _tokenize(passage)

    if not claim_toks or not passage_toks:
        return 0.0

    claim_set = set(claim_toks)
    passage_set = set(passage_toks)

    overlap = len(claim_set.intersection(passage_set))
    lexical_score = overlap / max(1, len(claim_set))

    long_term_bonus = 0.0
    for t in claim_set:
        if len(t) >= 6 and t in passage_set:
            long_term_bonus += 0.16

    p_lower = passage.lower()

    result_bonus = 0.18 * _count_cues(p_lower, RESULT_SECTION_CUES)
    effect_bonus = 0.16 * _count_cues(p_lower, POSITIVE_EFFECT_CUES)
    null_mixed_bonus = 0.16 * _count_cues(p_lower, NULL_OR_MIXED_CUES)
    quant_bonus = 0.12 * _count_cues(p_lower, QUANT_CUES)
    method_penalty = 0.14 * _count_cues(p_lower, METHOD_CUES)

    length_tokens = len(passage_toks)
    length_bonus = 0.0
    if 12 <= length_tokens <= 90:
        length_bonus = 0.14
    elif 8 <= length_tokens < 12:
        length_bonus = 0.06
    elif length_tokens < 6:
        length_bonus = -0.25

    score = (
        lexical_score
        + long_term_bonus
        + result_bonus
        + effect_bonus
        + null_mixed_bonus
        + quant_bonus
        + length_bonus
        - method_penalty
    )

    return max(0.0, round(score, 6))


def extract_evidence_spans(
    claim: str,
    full_text: str,
    max_spans: int = 4,
    window_sentences: int = 2,
) -> List[Tuple[str, float]]:
    sentences = _split_sentences(full_text)
    passages = _make_passages(sentences, window=window_sentences, stride=1)

    scored: List[Tuple[str, float]] = []
    for _, passage_text in passages:
        sc = score_passage(claim, passage_text)
        if sc > 0:
            scored.append((passage_text, sc))

    scored.sort(key=lambda x: x[1], reverse=True)

    out: List[Tuple[str, float]] = []

    for txt, sc in scored:
        redundant = False
        for kept_txt, _ in out:
            if _jaccard(txt, kept_txt) >= 0.72:
                redundant = True
                break

        if redundant:
            continue

        out.append((txt, sc))

        if len(out) >= max_spans:
            break

    return out
