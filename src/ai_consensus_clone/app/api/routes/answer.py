from fastapi import APIRouter, Depends
from ai_consensus_clone.app.api.schemas.answer import AnswerRequest, AnswerResponse
from ai_consensus_clone.app.api.deps import get_answer_service
from ai_consensus_clone.core.reasoning.answer import AnswerService

router = APIRouter()

@router.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, svc: AnswerService = Depends(get_answer_service)):
    result = svc.answer(req.q, k=req.k)
    return result

