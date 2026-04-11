# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import re

_WS_RE = re.compile(r"\s+")

# Mojibake hints frecuentes
_MOJIBAKE_HINT_RE = re.compile(
    r"(?:[ÃÂ]|â[^\s]{0,3}|�|\x96|\x97|\x91|\x92|\x93|\x94)"
)

MOJIBAKE_TOKENS = (
    "Ã",
    "Â",
    "�",
    "â",
)

# Reemplazos directos comunes y ampliados
MOJIBAKE_REPL = {
    # Smart quotes / dashes / ellipsis habituales
    "â€œ": "“",
    "â€": "”",
    "â€˜": "‘",
    "â€™": "’",
    "â€“": "–",
    "â€”": "—",
    "â€¦": "…",
    "â€": '"',
    "â€¡": "‡",
    "â€¢": "•",
    "â„¢": "™",
    "âˆ’": "−",

    # Variantes rotas muy típicas en PDFs
    "â": "–",
    "â": "—",
    "â": "‐",
    "â": "‘",
    "â": "’",
    "â": "“",
    "â": "”",
    "â¦": "…",

    # Casos cortados/degenerados que a veces aparecen
    "â-": "–",
    "â—": "—",
    "â€™s": "’s",

    # Caracteres de cp1252 vistos como Unicode raro
    "\x91": "‘",
    "\x92": "’",
    "\x93": "“",
    "\x94": "”",
    "\x96": "–",
    "\x97": "—",

    # Ligaduras / símbolos de PDF a veces molestos
    "ﬁ": "fi",
    "ﬂ": "fl",

    # Espacios especiales
    "\u00a0": " ",
    "\u2002": " ",
    "\u2003": " ",
    "\u2009": " ",
    "\u202f": " ",

    # Soft hyphen
    "\u00ad": "",

    # Basura común
    "Â": "",
    "Ã": "",
    "�": "",
}


def _score_mojibake(s: str) -> int:
    return sum(s.count(t) for t in MOJIBAKE_TOKENS)


def _try_redecode(s: str, wrong_text_encoding: str, intended_encoding: str) -> str | None:
    try:
        raw = s.encode(wrong_text_encoding, errors="strict")
        return raw.decode(intended_encoding, errors="strict")
    except Exception:
        return None


def _try_fix_mojibake(s: str) -> str:
    if not s:
        return s

    if not _MOJIBAKE_HINT_RE.search(s):
        return s

    candidates: list[str] = [s]

    for wrong in ("latin-1", "cp1252"):
        fixed = _try_redecode(s, wrong, "utf-8")
        if fixed:
            candidates.append(fixed)

    # A veces hay doble mojibake: intentar una segunda pasada sobre candidatos
    extra_candidates = []
    for cand in candidates:
        for wrong in ("latin-1", "cp1252"):
            fixed2 = _try_redecode(cand, wrong, "utf-8")
            if fixed2:
                extra_candidates.append(fixed2)

    candidates.extend(extra_candidates)

    best = min(candidates, key=_score_mojibake)
    return best if _score_mojibake(best) < _score_mojibake(s) else s


def _strip_html_tags_light(text: str) -> str:
    # Útil para previews que vengan con tags sueltos
    text = re.sub(r"<sub>(.*?)</sub>", r"\1", text, flags=re.I | re.S)
    text = re.sub(r"<sup>(.*?)</sup>", r"\1", text, flags=re.I | re.S)
    text = re.sub(r"<i>(.*?)</i>", r"\1", text, flags=re.I | re.S)
    text = re.sub(r"<b>(.*?)</b>", r"\1", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def _normalize_dashes(text: str) -> str:
    # Arregla algunos rangos numéricos rotos
    text = re.sub(r"(?<=\d)â(?=\d)", "–", text)   # 0.97â1.76 -> 0.97–1.76
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "–", text)
    return text


def clean_text(text: str) -> str:
    if not text:
        return ""

    # 1) Unescape HTML entities
    text = html.unescape(text)

    # 2) Reparación de mojibake
    text = _try_fix_mojibake(text)

    # 3) Reemplazos directos ampliados
    for bad, good in MOJIBAKE_REPL.items():
        text = text.replace(bad, good)

    # 4) Strip HTML ligero
    text = _strip_html_tags_light(text)

    # 5) Normaliza dashes y algunos casos de rangos
    text = _normalize_dashes(text)

    # 6) Limpia espacios raros antes de puntuación
    text = re.sub(r"\s+([,;:.!?])", r"\1", text)

    # 7) Colapsa whitespace
    text = _WS_RE.sub(" ", text).strip()

    return text


if __name__ == "__main__":
    tests = [
        "Español: áéíóú ñ ü",
        "Mojibake: EspaÃ±ol: Ã¡Ã©Ã­Ã³Ãº",
        "Quotes: Itâ€™s really â€” fine",
        "Already fine: “smart quotes” — ok",
        "HTML: Tom &amp; Jerry &quot;test&quot;",
        "CI [0.97â1.76]",
        "CI [0.97â1.76]",
        "D<sub>3</sub>-creatine dilution",
        "heavyâ€resistance doubleâ€blinded",
    ]
    for t in tests:
        print("IN :", t)
        print("OUT:", clean_text(t))
        print()
