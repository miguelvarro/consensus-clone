from fastapi import APIRouter, Depends

from ai_consensus_clone.app.api.schemas.search import SearchRequest, SearchResponse
from ai_consensus_clone.app.api.deps import get_bm25_search
from ai_consensus_clone.core.ranking.reranker import rerank_hits
from ai_consensus_clone.core.retrieval.bm25 import BM25Search

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, bm25: BM25Search = Depends(get_bm25_search)):
    raw_hits = bm25.search(
        req.q,
        k=max(req.k * 3, 15),
        year_from=req.year_from,
        year_to=req.year_to,
    )
    hits = rerank_hits(req.q, raw_hits)[: req.k]
    return SearchResponse(q=req.q, k=req.k, hits=hits)
