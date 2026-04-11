from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os
import json

import orjson
from rank_bm25 import BM25Okapi

from ai_consensus_clone.core.domain.paper import Paper
from ai_consensus_clone.core.retrieval.filters import year_filter

def _tokenize(text: str) -> List[str]:
    return [t for t in text.lower().replace("\n", " ").split(" ") if t.strip()]

@dataclass
class BM25Search:
    papers: List[Paper]
    bm25: BM25Okapi
    corpus_tokens: List[List[str]]

    @staticmethod
    def build_from_jsonl(jsonl_path: str) -> "BM25Search":
        papers: List[Paper] = []
        corpus_tokens: List[List[str]] = []
        with open(jsonl_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = orjson.loads(line)
                p = Paper(**obj)
                papers.append(p)
                text = f"{p.title} {(p.abstract or '')} {(p.full_text or '')}"
                corpus_tokens.append(_tokenize(text))
        bm25 = BM25Okapi(corpus_tokens)
        return BM25Search(papers=papers, bm25=bm25, corpus_tokens=corpus_tokens)

    def save_to_disk(self, index_dir: str) -> None:
        os.makedirs(index_dir, exist_ok=True)
        meta_path = os.path.join(index_dir, "papers.jsonl")
        with open(meta_path, "wb") as f:
            for p in self.papers:
                f.write(orjson.dumps(p.model_dump()) + b"\n")

        # Guardamos solo metadatos + corpus tokens si quieres acelerar.
        tokens_path = os.path.join(index_dir, "tokens.jsonl")
        with open(tokens_path, "w", encoding="utf-8") as f:
            for toks in self.corpus_tokens:
                f.write(json.dumps(toks) + "\n")

    @staticmethod
    def from_disk(index_dir: str) -> "BM25Search":
        meta_path = os.path.join(index_dir, "papers.jsonl")
        tokens_path = os.path.join(index_dir, "tokens.jsonl")
        papers: List[Paper] = []
        corpus_tokens: List[List[str]] = []

        with open(meta_path, "rb") as f:
            for line in f:
                if not line.strip():
                    continue
                papers.append(Paper(**orjson.loads(line)))

        with open(tokens_path, "r", encoding="utf-8") as f:
            for line in f:
                corpus_tokens.append(json.loads(line))

        bm25 = BM25Okapi(corpus_tokens)
        return BM25Search(papers=papers, bm25=bm25, corpus_tokens=corpus_tokens)

    def search(
        self,
        query: str,
        k: int = 10,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        include_text: bool = True,
        full_text_preview_chars: int = 4000,
    ):
        q_tokens = _tokenize(query)
        scores = self.bm25.get_scores(q_tokens)

        ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        hits = []

        for i in ranked_idx:
            p = self.papers[i]
            if not year_filter(p, year_from, year_to):
                continue

            hit = {
                "paper_id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "venue": p.venue,
                "doi": p.doi,
                "score": float(scores[i]),
            }

            if include_text:
                abstract = p.abstract or ""
                full_text = p.full_text or ""
                hit["abstract"] = abstract
                hit["has_full_text"] = bool(full_text.strip())
                hit["full_text_preview"] = full_text[:full_text_preview_chars] if full_text else ""
                hit["full_text_source"] = getattr(p, "full_text_source", None)

            hits.append(hit)

            if len(hits) >= k:
                break

        return hits
