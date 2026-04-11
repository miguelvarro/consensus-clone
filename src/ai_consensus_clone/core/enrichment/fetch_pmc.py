from __future__ import annotations

import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ai_consensus_clone.core.enrichment.models import FullTextResult
from ai_consensus_clone.core.enrichment.normalize import normalize_full_text
from ai_consensus_clone.core.enrichment.quality import quality_score, token_count


PMC_RE = re.compile(
    r"^https?://(?:pmc\.ncbi\.nlm\.nih\.gov|(?:www\.)?ncbi\.nlm\.nih\.gov)/pmc/articles/(?:PMC)?\d+/?$"
    r"|^https?://pmc\.ncbi\.nlm\.nih\.gov/articles/(?:PMC)?\d+/?$",
    re.I,
)

ANTI_BOT_CUES = (
    "checking your browser",
    "recaptcha",
    "cloudflare",
    "access denied",
    "not automatically redirected",
    "enable javascript and cookies",
    "cf-browser-verification",
)

SCIENTIFIC_SECTION_CUES = (
    "abstract",
    "introduction",
    "background",
    "methods",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "references",
)

BOILERPLATE_PATTERNS = [
    r"skip to main content",
    r"official websites use \.gov",
    r"a \.gov website belongs to an official government organization in the united states",
    r"secure \.gov websites use https",
    r"share sensitive information only on official, secure websites",
    r"search pmc full[- ]?text archive search in pmc",
    r"search term",
    r"journal list",
    r"copyright notice",
    r"pmc disclaimer",
    r"free full text",
    r"nihpa author manuscript",
    r"author manuscript",
    r"similar articles",
    r"cited by",
    r"publication types",
    r"related information",
    r"linkout - more resources",
    r"supplementary materials?",
    r"associated data",
    r"acknowledg(?:e)?ments?",
]

START_SECTION_PATTERNS = [
    r"\babstract\b",
    r"\bintroduction\b",
    r"\bbackground\b",
    r"\bobjective\b",
    r"\bobjectives\b",
    r"\bmethods\b",
]

END_SECTION_PATTERNS = [
    r"\breferences\b",
    r"\backnowledg(?:e)?ments?\b",
    r"\bsupplementary materials?\b",
    r"\bsimilar articles\b",
    r"\brelated information\b",
    r"\blinkout\b",
]


def normalize_pmc_url(url: str) -> str:
    raw = (url or "").strip()

    m = re.search(r"/pmc/articles/(PMC?\d+)/?$", raw, re.I)
    if m:
        article_id = m.group(1)
        if not article_id.upper().startswith("PMC"):
            article_id = f"PMC{article_id}"
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"

    m = re.search(r"/articles/(PMC?\d+)/?$", raw, re.I)
    if m:
        article_id = m.group(1)
        if not article_id.upper().startswith("PMC"):
            article_id = f"PMC{article_id}"
        return f"https://pmc.ncbi.nlm.nih.gov/articles/{article_id}/"

    return raw


class PMCFetcher:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "ai-consensus-clone/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def looks_like_pmc(self, url: Optional[str]) -> bool:
        if not url:
            return False
        return bool(PMC_RE.match(url.strip()))

    def fetch(self, url: str) -> FullTextResult:
        url = normalize_pmc_url(url)

        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                r = client.get(url)
                r.raise_for_status()

                html = r.text or ""
                if self._looks_like_antibot_page(html):
                    return FullTextResult(
                        text=None,
                        source=None,
                        url=str(r.url),
                        notes=["pmc antibot/protection page detected"],
                    )

                soup = BeautifulSoup(html, "lxml")
                self._remove_bad_nodes(soup)

                chunks: list[str] = []

                article = soup.find("article")
                if article:
                    txt = self._clean_extracted_text(article.get_text(" ", strip=True))
                    if txt:
                        chunks.append(txt)

                selectors = [
                    "main",
                    "div#maincontent",
                    "div.content",
                    "div.tsec",
                    "div.fm",
                    "div.body",
                    "section",
                ]

                for sel in selectors:
                    for node in soup.select(sel):
                        txt = self._clean_extracted_text(node.get_text(" ", strip=True))
                        if txt and len(txt) > 300:
                            chunks.append(txt)

                fallback = self._clean_extracted_text(soup.get_text(" ", strip=True))
                if fallback:
                    chunks.append(fallback)

                best = self._select_best_candidate(chunks)
                if not self._is_valid_scientific_text(best):
                    return FullTextResult(
                        text=None,
                        source=None,
                        url=str(r.url),
                        notes=["pmc page did not pass scientific text validation"],
                    )

                return FullTextResult(
                    text=best,
                    source="pmc_html",
                    url=str(r.url),
                    quality_score=quality_score(best, "pmc_html"),
                    extracted_from="pmc_html",
                )

        except Exception as e:
            return FullTextResult(
                text=None,
                source=None,
                url=url,
                notes=[f"pmc fetch error: {type(e).__name__}"],
            )

    def _remove_bad_nodes(self, soup: BeautifulSoup) -> None:
        selectors = [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "noscript",
            ".usa-banner",
            ".ncbi-alerts",
            ".ncbi-acc-toolbar",
            ".article-citation",
            ".tsec.sec",
            ".ref-list",
            "#site-navigation",
            "#navigation",
            ".navigation",
            ".breadcrumbs",
            ".sharebox",
            ".social-sharing",
            ".supplementary-materials",
            ".license",
            ".permissions",
            ".article-comments",
            ".related-links",
            ".similar-articles",
        ]

        for sel in selectors:
            for node in soup.select(sel):
                node.decompose()

    def _looks_like_antibot_page(self, html: str) -> bool:
        tl = (html or "").lower()
        return any(cue in tl for cue in ANTI_BOT_CUES)

    def _scientific_cue_count(self, text: str) -> int:
        tl = (text or "").lower()
        return sum(1 for cue in SCIENTIFIC_SECTION_CUES if cue in tl)

    def _select_best_candidate(self, chunks: list[str]) -> Optional[str]:
        valid = [c for c in chunks if c and c.strip()]
        if not valid:
            return None

        valid.sort(
            key=lambda t: (
                self._scientific_cue_count(t),
                token_count(t),
                len(t),
            ),
            reverse=True,
        )
        return valid[0]

    def _is_valid_scientific_text(self, text: Optional[str]) -> bool:
        if not text:
            return False

        tl = text.lower()
        n_tokens = token_count(text)

        if n_tokens < 250:
            return False

        if any(cue in tl for cue in ANTI_BOT_CUES):
            return False

        sci_hits = self._scientific_cue_count(text)
        if sci_hits < 2:
            return False

        return True

    def _clean_extracted_text(self, text: str) -> str:
        text = normalize_full_text(text) or ""
        if not text:
            return ""

        tl = text.lower()

        # Si sigue oliendo a antibot, fuera
        if any(cue in tl for cue in ANTI_BOT_CUES):
            return ""

        # Quitar boilerplate típico
        for pat in BOILERPLATE_PATTERNS:
            text = re.sub(pat, " ", text, flags=re.I)

        # Recortar desde la primera sección científica útil
        start_idx = None
        for pat in START_SECTION_PATTERNS:
            m = re.search(pat, text, flags=re.I)
            if m:
                idx = m.start()
                if start_idx is None or idx < start_idx:
                    start_idx = idx

        if start_idx is not None and start_idx > 0:
            prefix = text[:start_idx]
            # solo recorta si el prefijo parece claramente cabecera/navegación
            if token_count(prefix) < 250:
                text = text[start_idx:]

        # Cortar antes de secciones finales poco útiles
        end_idx = None
        for pat in END_SECTION_PATTERNS:
            m = re.search(pat, text, flags=re.I)
            if m:
                idx = m.start()
                if idx > 1000:
                    end_idx = idx if end_idx is None else min(end_idx, idx)

        if end_idx is not None:
            text = text[:end_idx]

        # Limpiezas extra
        text = re.sub(r"\bpmc\b", " ", text, flags=re.I)
        text = re.sub(r"\bpubmed\b", " ", text, flags=re.I)
        text = re.sub(r"\bnih\b", " ", text, flags=re.I)
        text = re.sub(r"\s+", " ", text).strip()

        return normalize_full_text(text) or ""
