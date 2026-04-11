from functools import lru_cache

from ai_consensus_clone.core.config.settings import Settings
from ai_consensus_clone.core.reasoning.answer import AnswerService
from ai_consensus_clone.core.retrieval.bm25 import BM25Search
from ai_consensus_clone.core.retrieval.online import OnlinePaperRetriever


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_bm25_search() -> BM25Search:
    s = get_settings()
    return BM25Search.from_disk(index_dir=s.bm25_index_dir)


@lru_cache
def get_online_retriever() -> OnlinePaperRetriever:
    return OnlinePaperRetriever()


@lru_cache
def get_answer_service() -> AnswerService:
    return AnswerService(
        search=get_bm25_search(),
        online_retriever=get_online_retriever(),
    )
