from pydantic import BaseModel, Field
from typing import List, Optional


class AnswerRequest(BaseModel):
    q: str = Field(..., min_length=2)
    k: int = Field(8, ge=3, le=20)


class Citation(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    title: str


class Evidence(BaseModel):
    paper_id: str
    doi: Optional[str] = None
    title: str
    span: str
    score: float


class AnswerResponse(BaseModel):
    q: str
    conclusion: str
    confidence: str
    citations: List[Citation]
    evidences: List[Evidence] = []

