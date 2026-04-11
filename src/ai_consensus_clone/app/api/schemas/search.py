from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    q: str = Field(..., min_length=2)
    k: int = Field(10, ge=1, le=50)
    year_from: Optional[int] = None
    year_to: Optional[int] = None

class PaperHit(BaseModel):
    paper_id: str
    title: str
    year: Optional[int] = None
    venue: Optional[str] = None
    doi: Optional[str] = None
    score: float

class SearchResponse(BaseModel):
    q: str
    k: int
    hits: List[PaperHit]

