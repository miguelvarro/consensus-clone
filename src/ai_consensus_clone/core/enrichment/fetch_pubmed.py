from __future__ import annotations

import re
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from ai_consensus_clone.core.enrichment.models import FullTextResult
from ai_consensus_clone.core.enrichment.normalize import normalize_full_text
from ai_consensus_clone.core.enrichment.quality import quality_score


PUBMED_RE = re.compile(r"^https?://pubmed\.ncbi\.nlm\.nih\.gov/(\d+)/?$", re.I)


class PubMedFetcher:
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "ai-consensus-clone/1.0",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        }

    def looks_like_pubmed(self, url: Optional[str]) -> bool:
        if not url:
            return False
        return bool(PUBMED_RE.match(url.strip()))

    def resolve_pmc_url(self, pubmed_url: str) -> Optional[str]:
        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                r = client.get(pubmed_url)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "lxml")

                for a in soup.select("a[href]"):
                    href = (a.get("href") or "").strip()

                    if "pmc.ncbi.nlm.nih.gov/articles/" in href:
                        return href if href.startswith("http") else f"https://{href.lstrip('/')}"

                    if "ncbi.nlm.nih.gov/pmc/articles/" in href:
                        return href if href.startswith("http") else f"https://www.ncbi.nlm.nih.gov{href}"

                    if href.startswith("/pmc/articles/"):
                        return f"https://www.ncbi.nlm.nih.gov{href}"

        except Exception:
            return None

        return None

    def fetch_abstract(self, pubmed_url: str) -> FullTextResult:
        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                r = client.get(pubmed_url, params={"format": "abstract"})
                if r.status_code == 200 and r.text:
                    text = self._extract_abstract_from_html(r.text)
                    text = normalize_full_text(text)
                    return FullTextResult(
                        text=text,
                        source="pubmed_abstract" if text else None,
                        url=str(r.url),
                        quality_score=quality_score(text, "pubmed_abstract"),
                        extracted_from="pubmed_html",
                    )
        except Exception:
            pass

        try:
            with httpx.Client(timeout=self.timeout, headers=self.headers, follow_redirects=True) as client:
                r = client.get(pubmed_url, params={"format": "pubmed"})
                if r.status_code == 200 and r.text:
                    text = self._extract_abstract_from_medline(r.text)
                    text = normalize_full_text(text)
                    return FullTextResult(
                        text=text,
                        source="pubmed_abstract" if text else None,
                        url=str(r.url),
                        quality_score=quality_score(text, "pubmed_abstract"),
                        extracted_from="pubmed_medline",
                    )
        except Exception:
            pass

        return FullTextResult(text=None, source=None, url=pubmed_url)

    def _extract_abstract_from_html(self, html: str) -> Optional[str]:
        soup = BeautifulSoup(html, "lxml")

        div = soup.find("div", class_=re.compile(r"abstract-content|abstract", re.I))
        if div:
            return div.get_text(" ", strip=True)

        sec = soup.find(lambda tag: tag.name in ("section", "div") and "abstract" in " ".join(tag.get("class", [])).lower())
        if sec:
            return sec.get_text(" ", strip=True)

        h2 = soup.find(
            lambda tag: tag.name in ("h2", "h3") and "abstract" in tag.get_text(strip=True).lower()
        )
        if h2:
            texts = []
            for sib in h2.find_all_next(["p", "div"], limit=10):
                t = sib.get_text(" ", strip=True)
                if t:
                    texts.append(t)
            return " ".join(texts).strip() or None

        return None

    def _extract_abstract_from_medline(self, text: str) -> Optional[str]:
        lines = [ln.rstrip() for ln in text.splitlines()]
        ab_lines = []
        in_ab = False

        for ln in lines:
            if ln.startswith("AB  -"):
                in_ab = True
                ab_lines.append(ln.replace("AB  -", "").strip())
                continue

            if in_ab:
                if re.match(r"^[A-Z]{2}\s{2}-", ln):
                    break
                ab_lines.append(ln.strip())

        merged = " ".join(t for t in ab_lines if t).strip()
        return merged or None
