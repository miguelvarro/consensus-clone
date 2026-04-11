from fastapi import APIRouter, Depends, HTTPException
from ai_consensus_clone.app.api.deps import get_bm25_search
from ai_consensus_clone.core.retrieval.bm25 import BM25Search

router = APIRouter()

@router.get("/papers/{paper_id}")
def get_paper(paper_id: str, bm25: BM25Search = Depends(get_bm25_search)):
    for p in bm25.papers:
        if p.paper_id == paper_id:
            full_text = getattr(p, "full_text", None)
            oa_url = getattr(p, "oa_url", None)

            return {
                "paper_id": p.paper_id,
                "title": p.title,
                "year": p.year,
                "venue": p.venue,
                "doi": p.doi,
                "authors": getattr(p, "authors", []),
                "oa_url": oa_url,
                "has_full_text": bool(full_text),
                "abstract": p.abstract,
                "full_text_preview": (full_text[:2000] + "...") if full_text else None,
            }

    raise HTTPException(status_code=404, detail="paper_id no encontrado")

