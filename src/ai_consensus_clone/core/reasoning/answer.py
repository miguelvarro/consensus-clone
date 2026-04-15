from __future__ import annotations

from collections import Counter
from typing import Dict, Any, List, Tuple, Optional

from ai_consensus_clone.core.domain.paper import Paper
from ai_consensus_clone.core.ranking.reranker import rerank_hits
from ai_consensus_clone.core.reasoning.evidence_extractor import extract_evidence_spans
from ai_consensus_clone.core.retrieval.bm25 import BM25Search
from ai_consensus_clone.core.retrieval.online import OnlinePaperRetriever
from ai_consensus_clone.utils.text import clean_text
from ai_consensus_clone.core.reasoning.prompt_loader import load_prompt
from ai_consensus_clone.core.reasoning.stance_classifier import classify_papers_stances
from ai_consensus_clone.core.reasoning.llm_client import LLMClient

try:
    from ai_consensus_clone.core.reasoning.llm_client import build_consensus_answer_with_llm
except Exception:
    build_consensus_answer_with_llm = None


POSITIVE_CUES = (
    "increase",
    "increased",
    "improve",
    "improved",
    "improvement",
    "greater",
    "enhanced",
    "enhancement",
    "benefit",
    "beneficial",
    "significant increase",
    "significantly increased",
    "higher",
    "gain",
    "gains",
    "augments",
    "augment",
    "stronger",
)

NEGATIVE_OR_NULL_CUES = (
    "no effect",
    "no significant",
    "not significant",
    "did not improve",
    "did not increase",
    "no difference",
    "similar between groups",
    "equivocal",
    "inconsistent",
    "unclear",
    "limited evidence",
)

UNCERTAINTY_CUES = (
    "may",
    "might",
    "suggest",
    "suggests",
    "potential",
    "possible",
    "preliminary",
)

MIXED_QUESTION_CUES = (
    "mixed",
    "inconclusive",
    "equivocal",
    "unclear",
    "inconsistent",
)

POSITIVE_QUESTION_CUES = (
    "increase",
    "improve",
    "benefit",
    "help",
    "enhance",
)


class AnswerService:
    def __init__(
        self,
        search: BM25Search,
        online_retriever: Optional[OnlinePaperRetriever] = None,
    ):
        self.search = search
        self.online_retriever = online_retriever

    def _question_mode(self, q: str) -> str:
        ql = clean_text(q).lower()

        if any(cue in ql for cue in MIXED_QUESTION_CUES):
            return "mixed_check"

        if ql.startswith(("does ", "is ", "can ", "do ", "are ")):
            return "yes_no"

        if any(cue in ql for cue in POSITIVE_QUESTION_CUES):
            return "effect_check"

        return "generic"

    def _classify_span(self, text: str) -> str:
        t = clean_text(text).lower()

        pos_hits = sum(1 for cue in POSITIVE_CUES if cue in t)
        neg_hits = sum(1 for cue in NEGATIVE_OR_NULL_CUES if cue in t)
        unc_hits = sum(1 for cue in UNCERTAINTY_CUES if cue in t)

        if pos_hits >= 1 and neg_hits == 0:
            return "positive"
        if neg_hits >= 1:
            return "mixed"
        if unc_hits >= 1:
            return "mixed"
        return "insufficient"

    def _select_best_evidences(
        self,
        evidences: List[Dict[str, Any]],
        max_total: int = 6,
        max_per_paper: int = 2,
    ) -> List[Dict[str, Any]]:
        evidences = sorted(evidences, key=lambda e: e["score"], reverse=True)

        selected: List[Dict[str, Any]] = []
        per_paper_count: Dict[str, int] = {}

        for ev in evidences:
            pid = ev["paper_id"]

            if per_paper_count.get(pid, 0) >= max_per_paper:
                continue

            redundant = False
            ev_key = clean_text(ev["span"]).lower()

            for kept in selected:
                kept_key = clean_text(kept["span"]).lower()
                if ev_key[:180] == kept_key[:180]:
                    redundant = True
                    break

            if redundant:
                continue

            selected.append(ev)
            per_paper_count[pid] = per_paper_count.get(pid, 0) + 1

            if len(selected) >= max_total:
                break

        return selected

    def _compute_confidence(
        self,
        n_hits: int,
        n_evidences: int,
        n_unique_papers: int,
        avg_score: float,
        positive: int,
        mixed: int,
        insufficient: int,
    ) -> str:
        if n_evidences == 0 or n_unique_papers == 0:
            return "baja"

        if (
            n_hits >= 5
            and n_unique_papers >= 3
            and n_evidences >= 4
            and avg_score >= 0.75
            and positive >= 3
            and positive > mixed
        ):
            return "alta"

        if n_unique_papers >= 2 and n_evidences >= 2 and avg_score >= 0.35:
            return "media"

        if insufficient >= max(2, n_evidences // 2):
            return "baja"

        return "baja"

    def _paper_to_hit(self, paper: Paper, score: float = 0.0) -> Dict[str, Any]:
        full_text = clean_text(paper.full_text or "")
        abstract = clean_text(paper.abstract or "")

        return {
            "paper_id": paper.paper_id,
            "title": clean_text(paper.title or ""),
            "year": paper.year,
            "venue": paper.venue,
            "doi": paper.doi,
            "citation_count": paper.citation_count,
            "score": float(score),
            "abstract": abstract,
            "has_full_text": bool(full_text.strip()),
            "full_text_preview": full_text[:4000] if full_text else "",
            "oa_url": paper.oa_url,
            "full_text_source": paper.full_text_source,
        }

    def _hits_to_papers(self, hits: List[Dict[str, Any]]) -> List[Paper]:
        papers: List[Paper] = []

        for h in hits:
            papers.append(
                Paper(
                    paper_id=h.get("paper_id", ""),
                    title=clean_text(h.get("title") or ""),
                    abstract=clean_text(h.get("abstract") or "") or None,
                    full_text=clean_text(h.get("full_text_preview") or "") or None,
                    year=h.get("year"),
                    venue=h.get("venue"),
                    doi=h.get("doi"),
                    authors=[],
                    oa_url=h.get("oa_url"),
                    pdf_url=None,
                    full_text_source=h.get("full_text_source"),
                    pmc_url=None,
                    citation_count=h.get("citation_count"),
                )
            )

        return papers

    def _aggregate_paper_stances(self, paper_stances: List[Any]) -> Dict[str, Any]:
        counts = Counter()

        for ps in paper_stances:
            stance = getattr(ps, "stance", None)
            if stance in {"support", "contradict", "neutral"}:
                counts[stance] += 1

        support = counts.get("support", 0)
        contradict = counts.get("contradict", 0)
        neutral = counts.get("neutral", 0)

        dominant_stance = "neutral"
        max_count = max(support, contradict, neutral)

        tied = [
            label for label, value in {
                "support": support,
                "contradict": contradict,
                "neutral": neutral,
            }.items()
            if value == max_count
        ]

        if max_count == 0:
            dominant_stance = "neutral"
        elif len(tied) == 1:
            dominant_stance = tied[0]
        else:
            dominant_stance = "mixed"

        return {
            "support": support,
            "contradict": contradict,
            "neutral": neutral,
            "dominant_stance": dominant_stance,
        }

    def _build_conclusion_from_stance_breakdown(
        self,
        q: str,
        evidence_breakdown: Dict[str, Any],
        evidences: List[Dict[str, Any]],
        n_hits: int,
    ) -> Tuple[str, str]:
        support = int(evidence_breakdown.get("support", 0) or 0)
        contradict = int(evidence_breakdown.get("contradict", 0) or 0)
        neutral = int(evidence_breakdown.get("neutral", 0) or 0)
        dominant_stance = evidence_breakdown.get("dominant_stance", "neutral")

        labels = [self._classify_span(ev["span"]) for ev in evidences]
        counts = Counter(labels)

        positive = counts.get("positive", 0)
        mixed = counts.get("mixed", 0)
        insufficient = counts.get("insufficient", 0)

        unique_papers = len({ev["paper_id"] for ev in evidences})
        avg_score = sum(ev["score"] for ev in evidences) / max(1, len(evidences)) if evidences else 0.0

        confidence = self._compute_confidence(
            n_hits=n_hits,
            n_evidences=len(evidences),
            n_unique_papers=unique_papers,
            avg_score=avg_score,
            positive=positive,
            mixed=mixed,
            insufficient=insufficient,
        )

        mode = self._question_mode(q)

        if support == 0 and contradict == 0 and neutral == 0:
            return (
                clean_text(
                    "Con el dataset actual no se han encontrado artículos suficientemente informativos para responder con solidez a la pregunta."
                ),
                "baja",
            )

        if dominant_stance == "support":
            if contradict >= 1:
                return (
                    clean_text(
                        "En conjunto, la mayoría de los artículos recuperados apoyan la hipótesis planteada, aunque existe cierta evidencia no concluyente o parcialmente contradictoria."
                    ),
                    confidence,
                )
            return (
                clean_text(
                    "En conjunto, la mayoría de los artículos recuperados apoyan la hipótesis planteada."
                ),
                confidence,
            )

        if dominant_stance == "contradict":
            if support >= 1:
                return (
                    clean_text(
                        "En conjunto, la mayoría de los artículos recuperados no apoyan la hipótesis planteada, aunque existe alguna evidencia a favor."
                    ),
                    confidence,
                )
            return (
                clean_text(
                    "En conjunto, la mayoría de los artículos recuperados no apoyan la hipótesis planteada."
                ),
                confidence,
            )

        if dominant_stance == "mixed":
            if mode == "mixed_check":
                return (
                    clean_text(
                        "La evidencia recuperada es mixta: los artículos no apuntan todos en la misma dirección y el conjunto no permite una conclusión uniforme."
                    ),
                    confidence,
                )
            return (
                clean_text(
                    "La evidencia recuperada es mixta o inconsistente: distintos artículos apuntan en direcciones diferentes."
                ),
                confidence,
            )

        if neutral > 0 and support == 0 and contradict == 0:
            return (
                clean_text(
                    "La mayoría de los artículos recuperados no responden de forma directa o concluyente a la pregunta planteada."
                ),
                "baja",
            )

        return (
            clean_text(
                "Con el dataset actual, la evidencia recuperada no permite una conclusión clara y fiable."
            ),
            confidence,
        )

    def _dedupe_hits(self, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not hits:
            return []

        source_priority = {
            "pdf": 5,
            "pmc_html": 4,
            "pubmed_abstract": 3,
            "openalex_abstract": 2,
            "existing_full_text": 2,
            "": 0,
            None: 0,
        }

        def title_key(hit: Dict[str, Any]) -> str:
            return clean_text(hit.get("title") or "").lower().strip()

        def paper_key(hit: Dict[str, Any]) -> str:
            pid = (hit.get("paper_id") or "").strip()
            if pid:
                return f"pid::{pid}"
            tk = title_key(hit)
            return f"title::{tk}"

        def hit_rank_tuple(hit: Dict[str, Any]) -> tuple:
            source = (hit.get("full_text_source") or "").strip().lower()
            has_full_text = bool(hit.get("has_full_text"))
            preview_len = len((hit.get("full_text_preview") or "").strip())
            rerank_score = float(hit.get("rerank_score", 0.0) or 0.0)
            score = float(hit.get("score", 0.0) or 0.0)

            return (
                source_priority.get(source, 0),
                1 if has_full_text else 0,
                preview_len,
                rerank_score,
                score,
            )

        def better_hit(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            best = a if hit_rank_tuple(a) >= hit_rank_tuple(b) else b
            other = b if best is a else a

            merged = dict(best)

            for field in (
                "abstract",
                "oa_url",
                "doi",
                "venue",
                "year",
                "title",
                "paper_id",
                "citation_count",
            ):
                if merged.get(field) in (None, "", []) and other.get(field) not in (None, "", []):
                    merged[field] = other.get(field)

            if not (merged.get("full_text_preview") or "").strip() and (other.get("full_text_preview") or "").strip():
                merged["full_text_preview"] = other.get("full_text_preview")

            merged["has_full_text"] = bool((merged.get("full_text_preview") or "").strip()) or bool(merged.get("has_full_text"))

            best_source = (merged.get("full_text_source") or "").strip().lower()
            other_source = (other.get("full_text_source") or "").strip().lower()

            best_source_rank = source_priority.get(best_source, 0)
            other_source_rank = source_priority.get(other_source, 0)

            best_preview_len = len((merged.get("full_text_preview") or "").strip())
            other_preview_len = len((other.get("full_text_preview") or "").strip())

            if other_source_rank > best_source_rank and other_preview_len > 0:
                merged["full_text_source"] = other.get("full_text_source")
                merged["full_text_preview"] = other.get("full_text_preview")
                merged["has_full_text"] = bool(other.get("has_full_text"))

            return merged

        grouped: Dict[str, Dict[str, Any]] = {}

        for hit in hits:
            key = paper_key(hit)
            if key not in grouped:
                grouped[key] = dict(hit)
            else:
                grouped[key] = better_hit(grouped[key], hit)

        deduped = list(grouped.values())
        deduped.sort(
            key=lambda h: float(h.get("rerank_score", h.get("score", 0.0)) or 0.0),
            reverse=True,
        )
        return deduped

    def _needs_online_fallback(self, local_hits: List[Dict[str, Any]]) -> bool:
        if len(local_hits) < 4:
            return True

        top_scores = [float(h.get("score", 0.0)) for h in local_hits[:5]]
        if not top_scores:
            return True

        avg_top_score = sum(top_scores[:3]) / max(1, min(3, len(top_scores)))

        strong_fulltext_sources = {"pdf", "pmc_html", "existing_full_text"}
        weak_sources = {"", "openalex_abstract", "pubmed_abstract"}

        strong_count = 0
        weak_or_missing_count = 0

        for h in local_hits[:5]:
            source = (h.get("full_text_source") or "").strip().lower()
            if source in strong_fulltext_sources:
                strong_count += 1
            if source in weak_sources or source not in strong_fulltext_sources:
                weak_or_missing_count += 1

        if avg_top_score < 1.5:
            return True
        if weak_or_missing_count >= 3:
            return True
        if strong_count < 2:
            return True

        return False

    def _fetch_online_hits(self, q: str, n: int = 8) -> List[Dict[str, Any]]:
        if self.online_retriever is None:
            return []

        try:
            papers = self.online_retriever.search_openalex(query=q, n=n, oa_only=True)
        except Exception:
            return []

        return [self._paper_to_hit(p, score=0.15) for p in papers]

    def _get_candidate_hits(self, q: str, k: int) -> List[Dict[str, Any]]:
        local_hits = self.search.search(q, k=max(k * 3, 15), include_text=True)
        local_hits = rerank_hits(q, local_hits)

        combined = list(local_hits)

        if self._needs_online_fallback(local_hits):
            online_hits = self._fetch_online_hits(q, n=max(k * 2, 8))
            combined.extend(online_hits)

        combined = self._dedupe_hits(combined)
        combined = rerank_hits(q, combined)

        return combined[:k]

    def _extract_evidences_from_hits(self, q: str, hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_evidences: List[Dict[str, Any]] = []

        for h in hits:
            base_text = h.get("full_text_preview") or h.get("abstract") or ""
            base_text = clean_text(base_text)

            if not base_text.strip():
                continue

            spans = extract_evidence_spans(
                claim=q,
                full_text=base_text,
                max_spans=3,
                window_sentences=2,
            )

            for span_text, score in spans:
                cleaned_span = clean_text(span_text)
                if not cleaned_span:
                    continue

                all_evidences.append(
                    {
                        "paper_id": h["paper_id"],
                        "doi": h.get("doi"),
                        "title": clean_text(h.get("title") or ""),
                        "year": h.get("year"),
                        "venue": h.get("venue"),
                        "full_text_source": h.get("full_text_source"),
                        "span": cleaned_span,
                        "score": float(score),
                    }
                )

        return self._select_best_evidences(
            all_evidences,
            max_total=6,
            max_per_paper=2,
        )

    def _try_llm_conclusion(
        self,
        q: str,
        evidences: List[Dict[str, Any]],
        citations: List[Dict[str, Any]],
        fallback_conclusion: str,
        fallback_confidence: str,
    ) -> Tuple[str, str]:
        try:
            llm_result = build_consensus_answer_with_llm(
                question=q,
                evidences=evidences,
                citations=citations,
            )
        except Exception:
            llm_result = None

        if not llm_result:
            return fallback_conclusion, fallback_confidence

        conclusion = clean_text(llm_result.get("conclusion") or fallback_conclusion)
        confidence = str(llm_result.get("confidence") or fallback_confidence).lower()

        if confidence not in {"alta", "media", "baja"}:
            confidence = fallback_confidence

        return conclusion, confidence

    def answer(self, q: str, k: int = 8) -> Dict[str, Any]:
        hits = self._get_candidate_hits(q, k=k)

        if not hits:
            return {
                "q": q,
                "conclusion": clean_text("No se han encontrado artículos relevantes con el dataset actual."),
                "confidence": "baja",
                "citations": [],
                "evidences": [],
                "paper_stances": [],
                "evidence_breakdown": {
                    "support": 0,
                    "contradict": 0,
                    "neutral": 0,
                    "dominant_stance": "neutral",
                },
            }

        citations: List[Dict[str, Any]] = [
            {
                "paper_id": h["paper_id"],
                "doi": h.get("doi"),
                "title": clean_text(h.get("title") or ""),
                "citation_count": h.get("citation_count"),
            }
            for h in hits
        ]

        evidences = self._extract_evidences_from_hits(q, hits)

        stance_prompt = load_prompt("stance_classification_prompt.txt")
        papers = self._hits_to_papers(hits)
        llm_client = LLMClient()
        paper_stances = classify_papers_stances(
            llm_client=llm_client,
            question=q,
            papers=papers,
            prompt_template=stance_prompt,
        )

        evidence_breakdown = self._aggregate_paper_stances(paper_stances)

        conclusion, confidence = self._build_conclusion_from_stance_breakdown(
            q=q,
            evidence_breakdown=evidence_breakdown,
            evidences=evidences,
            n_hits=len(hits),
        )

        conclusion, confidence = self._try_llm_conclusion(
            q=q,
            evidences=evidences,
            citations=citations,
            fallback_conclusion=conclusion,
            fallback_confidence=confidence,
        )

        return {
            "q": q,
            "conclusion": conclusion,
            "confidence": confidence,
            "citations": citations,
            "evidences": evidences,
            "paper_stances": [s.to_dict() for s in paper_stances],
            "evidence_breakdown": evidence_breakdown,
        }
