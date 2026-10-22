from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.recipes.agent_rag.agent_rag_service import (
    AgentRagService,
    AgentSearchResult,
)
from private_gpt.server.utils.auth import authenticated

agent_rag_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class AgentSearchBody(BaseModel):
    query: str = Field(examples=["What were the main risks mentioned across the reports?"])
    max_iterations: int = Field(default=3, ge=1, le=6)
    queries_per_step: int = Field(default=3, ge=1, le=6)
    retrieve_limit: int = Field(default=4, ge=1, le=20)
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    include_steps: bool = True


@agent_rag_router.post("/agent/search", tags=["Recipes"])
def agent_search(request: Request, body: AgentSearchBody) -> AgentSearchResult:
    """Answer a question with an agentic RAG loop over the ingested documents.

    Unlike `/chat/completions` with `use_context=true` (which retrieves once),
    this endpoint lets the model plan several searches, inspect what comes
    back, search again to fill gaps (bounded by `max_iterations`), and then
    answer with inline `[n]` citations into the returned `sources`.

    The `steps` field returns the agent's trace (plan / retrieve / reflect /
    answer) for transparency; set `include_steps=false` to omit it.
    """
    service = request.state.injector.get(AgentRagService)
    result = service.search(
        query=body.query,
        max_iterations=body.max_iterations,
        queries_per_step=body.queries_per_step,
        retrieve_limit=body.retrieve_limit,
        context_filter=body.context_filter,
    )
    return AgentSearchResult(
        response=result.response,
        sources=result.sources if body.include_sources else [],
        steps=result.steps if body.include_steps else [],
    )
