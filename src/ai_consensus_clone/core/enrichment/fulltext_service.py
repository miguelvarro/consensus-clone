from __future__ import annotations

from typing import Optional, List

from ai_consensus_clone.core.domain.paper import Paper
from ai_consensus_clone.core.enrichment.fetch_pdf import PDFFetcher
from ai_consensus_clone.core.enrichment.fetch_pmc import PMCFetcher
from ai_consensus_clone.core.enrichment.fetch_pubmed import PubMedFetcher
from ai_consensus_clone.core.enrichment.models import FullTextResult, EnrichmentDecision
from ai_consensus_clone.core.enrichment.normalize import (
    normalize_full_text,
    ensure_non_empty_full_text,
)
from ai_consensus_clone.core.enrichment.quality import looks_like_meaningful_full_text, quality_score


class FullTextService:
    """
    Orden de prioridad:
    1) PDF
    2) PMC HTML directo
    3) PubMed -> PMC HTML
    4) PubMed abstract
    5) OpenAlex abstract fallback
    """

    def __init__(
        self,
        pdf_fetcher: Optional[PDFFetcher] = None,
        pmc_fetcher: Optional[PMCFetcher] = None,
        pubmed_fetcher: Optional[PubMedFetcher] = None,
    ):
        self.pdf_fetcher = pdf_fetcher or PDFFetcher()
        self.pmc_fetcher = pmc_fetcher or PMCFetcher()
        self.pubmed_fetcher = pubmed_fetcher or PubMedFetcher()

    def enrich_paper(self, paper: Paper) -> Paper:
        if (paper.full_text or "").strip():
            paper.full_text = ensure_non_empty_full_text(paper.full_text, paper.abstract)
            if not paper.full_text_source and paper.full_text:
                paper.full_text_source = "existing_full_text"
            return paper

        decision = self.resolve(paper)

        selected = decision.selected
        if selected and selected.ok:
            paper.full_text = ensure_non_empty_full_text(selected.text, paper.abstract)
            paper.full_text_source = selected.source
            if selected.source == "pmc_html":
                paper.pmc_url = selected.url
            return paper

        fallback = ensure_non_empty_full_text(None, paper.abstract)
        if fallback:
            paper.full_text = fallback
            paper.full_text_source = "openalex_abstract"

        return paper

    def resolve(self, paper: Paper) -> EnrichmentDecision:
        candidates: List[FullTextResult] = []

        if paper.pdf_url:
            res = self.pdf_fetcher.fetch(paper.pdf_url)
            if res.ok:
                candidates.append(res)
                if looks_like_meaningful_full_text(res.text, min_tokens=250):
                    return EnrichmentDecision(selected=res, candidates=candidates)

        if paper.oa_url and self.pmc_fetcher.looks_like_pmc(paper.oa_url):
            res = self.pmc_fetcher.fetch(paper.oa_url)
            if res.ok:
                candidates.append(res)
                if looks_like_meaningful_full_text(res.text, min_tokens=250):
                    return EnrichmentDecision(selected=res, candidates=candidates)

        if paper.oa_url and self.pubmed_fetcher.looks_like_pubmed(paper.oa_url):
            pmc_url = self.pubmed_fetcher.resolve_pmc_url(paper.oa_url)
            if pmc_url:
                res = self.pmc_fetcher.fetch(pmc_url)
                if res.ok:
                    candidates.append(res)
                    if looks_like_meaningful_full_text(res.text, min_tokens=250):
                        return EnrichmentDecision(selected=res, candidates=candidates)

            abs_res = self.pubmed_fetcher.fetch_abstract(paper.oa_url)
            if abs_res.ok:
                candidates.append(abs_res)

        if paper.abstract:
            text = normalize_full_text(paper.abstract)
            candidates.append(
                FullTextResult(
                    text=text,
                    source="openalex_abstract" if text else None,
                    url=paper.oa_url,
                    quality_score=quality_score(text, "openalex_abstract"),
                    extracted_from="openalex",
                )
            )

        selected = self._select_best_candidate(candidates)
        return EnrichmentDecision(selected=selected, candidates=candidates)

    def _select_best_candidate(self, candidates: List[FullTextResult]) -> Optional[FullTextResult]:
        valid = [c for c in candidates if c.ok]
        if not valid:
            return None

        source_priority = {
            "pdf": 5,
            "pmc_html": 4,
            "pubmed_abstract": 2,
            "openalex_abstract": 1,
        }

        valid.sort(
            key=lambda c: (
                source_priority.get(c.source or "", 0),
                c.quality_score,
                len((c.text or "").strip()),
            ),
            reverse=True,
        )
        return valid[0]
