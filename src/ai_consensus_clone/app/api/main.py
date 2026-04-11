from fastapi import FastAPI
from ai_consensus_clone.app.api.utf8_json import UTF8JSONResponse
from ai_consensus_clone.app.api.routes.health import router as health_router
from ai_consensus_clone.app.api.routes.search import router as search_router
from ai_consensus_clone.app.api.routes.answer import router as answer_router
from ai_consensus_clone.app.api.routes.papers import router as papers_router


app = FastAPI(
    title="AI Consensus Clone",
    default_response_class=UTF8JSONResponse,
)

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class ForceUTF8JSONMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp: Response = await call_next(request)
        ct = resp.headers.get("content-type", "")
        if ct.startswith("application/json") and "charset=" not in ct:
            resp.headers["content-type"] = "application/json; charset=utf-8"
        return resp

app.add_middleware(ForceUTF8JSONMiddleware)
app.include_router(health_router, tags=["health"])
app.include_router(search_router, tags=["search"])
app.include_router(answer_router, tags=["answer"])
app.include_router(papers_router, tags=["papers"])


