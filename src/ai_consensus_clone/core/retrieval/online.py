from __future__ import annotations

import re
from typing import List, Optional, Tuple

from ai_consensus_clone.core.domain.paper import Paper
from ai_consensus_clone.core.enrichment.fulltext_service import FullTextService
from ai_consensus_clone.core.ingestion.connectors.openalex import (
    OpenAlexClient,
    extract_abstract,
    extract_doi,
    extract_oa_landing_url,
    extract_oa_pdf_url,
)
from ai_consensus_clone.utils.text import clean_text


_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ0-9]+")

STOPWORDS = {
    "does", "do", "is", "are", "can", "the", "a", "an", "of", "and", "in",
    "on", "for", "to", "with", "during", "plus", "via", "using", "use",
    "effect", "effects", "evidence", "study", "studies", "paper", "review",
    "older", "adults",
}

CORE_TERMS = {
    "creatine",
    "monohydrate",
    "supplementation",
    "supplement",
    "strength",
    "muscle",
    "resistance",
    "training",
    "exercise",
    "performance",
    "sarcopenia",
}


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _content_terms(text: str) -> set[str]:
    toks = _tokenize(text)
    return {t for t in toks if len(t) >= 4 and t not in STOPWORDS}


class OnlinePaperRetriever:
    """
    Recuperador online:
    - busca works en OpenAlex
    - normaliza a Paper
    - filtra ruido temático con scoring suave
    - enriquece full_text con pipeline robusto
    """

    def __init__(
        self,
        client: Optional[OpenAlexClient] = None,
        fulltext_service: Optional[FullTextService] = None,
    ):
        self.client = client or OpenAlexClient()
        self.fulltext_service = fulltext_service or FullTextService()

    def search_openalex(
        self,
        query: str,
        n: int = 10,
        oa_only: bool = True,
    ) -> List[Paper]:
        filters = "is_oa:true" if oa_only else None

        # Pedimos más para permitir filtrado y ranking temático
        per_page = max(n * 5, 20)
        works = self.client.search_works(query, per_page=per_page, filters=filters)

        scored_candidates: List[Tuple[float, Paper]] = []

        for w in works:
            paper = self._work_to_paper(w)
            if paper is None:
                continue

            thematic_score = self._thematic_score(query, paper)
            if thematic_score <= 0:
                continue

            scored_candidates.append((thematic_score, paper))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        papers: List[Paper] = []

        for thematic_score, paper in scored_candidates[: max(n * 2, 10)]:
            try:
                paper = self.fulltext_service.enrich_paper(paper)
            except Exception:
                if paper.abstract and not paper.full_text:
                    paper.full_text = clean_text(paper.abstract)
                    paper.full_text_source = "openalex_abstract"

            if not (paper.full_text or "").strip() and paper.abstract:
                paper.full_text = clean_text(paper.abstract)
                paper.full_text_source = "openalex_abstract"

            papers.append(paper)

            if len(papers) >= n:
                break

        return papers

    def _work_to_paper(self, work: dict) -> Optional[Paper]:
        paper_id_raw = work.get("id") or ""
        paper_id = paper_id_raw.split("/")[-1].strip() if paper_id_raw else ""

        title = clean_text(work.get("title") or "")
        abstract = clean_text(extract_abstract(work) or "") or None
        year = work.get("publication_year")
        venue = (work.get("primary_location") or {}).get("source", {}).get("display_name")
        doi = extract_doi(work)
        oa_url = extract_oa_landing_url(work)
        pdf_url = extract_oa_pdf_url(work)

        authors = [
            (a.get("author") or {}).get("display_name")
            for a in (work.get("authorships") or [])
            if (a.get("author") or {}).get("display_name")
        ]

        if not paper_id and not title:
            return None

        return Paper(
            paper_id=paper_id or title[:50],
            title=title,
            abstract=abstract,
            oa_url=oa_url,
            pdf_url=pdf_url,
            full_text=None,
            year=year,
            venue=venue,
            doi=doi,
            authors=authors,
            full_text_source=None,
            pmc_url=None,
            citation_count=work.get("cited_by_count"),
        )

    def _thematic_score(self, query: str, paper: Paper) -> float:
        q = clean_text(query).lower()
        title = clean_text(paper.title or "")
        abstract = clean_text(paper.abstract or "")
        text = f"{title} {abstract}".strip().lower()

        if not text:
            return 0.0

        q_terms = _content_terms(q)
        p_terms = _content_terms(text)

        # Regla dura mínima: si la query contiene "creatine", el documento debe contenerlo
        if "creatine" in q and "creatine" not in p_terms and "creatine" not in text:
            return 0.0

        overlap = len(q_terms.intersection(p_terms))
        core_hits = sum(1 for t in CORE_TERMS if t in p_terms)

        score = 0.0

        # overlap general
        score += 1.2 * overlap

        # bonus núcleo temático
        score += 0.4 * core_hits

        # bonus por presencia explícita de creatina
        if "creatine" in text:
            score += 2.0

        # bonus por fuerza en título
        title_l = title.lower()
        if "creatine" in title_l:
            score += 1.2
        if "strength" in title_l:
            score += 0.8
        if "resistance training" in title_l:
            score += 0.8
        if "older adults" in title_l or "older" in title_l:
            score += 0.5

        # bonus review/meta-analysis
        if any(cue in text for cue in ("systematic review", "meta-analysis", "meta analysis", "review")):
            score += 0.7

        # penalización ligera si es demasiado genérico
        generic_penalties = (
            "guideline",
            "guidelines",
            "consensus",
            "cardiovascular",
            "myocardial",
            "hepatitis",
            "sars-cov-2",
        )
        if any(cue in text for cue in generic_penalties):
            score -= 2.0

        # umbral suave
        return score if score >= 2.0 else 0.0
