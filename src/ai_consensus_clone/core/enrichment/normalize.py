from __future__ import annotations

from typing import Optional

from ai_consensus_clone.utils.text import clean_text


def normalize_full_text(text: Optional[str]) -> Optional[str]:
    if not text:
        return None

    text = clean_text(text)
    text = text.replace("\x00", " ")
    text = " ".join(text.split())
    text = text.strip()

    return text or None


def build_preview(text: Optional[str], max_chars: int = 4000) -> str:
    if not text:
        return ""
    text = normalize_full_text(text) or ""
    return text[:max_chars]


def ensure_non_empty_full_text(
    full_text: Optional[str],
    abstract: Optional[str],
) -> Optional[str]:
    ft = normalize_full_text(full_text)
    if ft:
        return ft

    ab = normalize_full_text(abstract)
    if ab:
        return ab

    return None


def infer_fallback_source(
    current_source: Optional[str],
    abstract: Optional[str],
) -> Optional[str]:
    if current_source:
        return current_source
    if abstract:
        return "openalex_abstract"
    return None
