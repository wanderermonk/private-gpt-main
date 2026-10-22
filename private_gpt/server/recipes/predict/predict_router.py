from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.recipes.predict.predict_service import (
    PredictResult,
    PredictService,
)
from private_gpt.server.utils.auth import authenticated

predict_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class PredictBody(BaseModel):
    metric: str = Field(
        examples=["quarterly revenue", "monthly active users", "defect rate"]
    )
    horizon: int = Field(default=3, ge=1, le=24)
    retrieve_limit: int = Field(default=8, ge=1, le=30)
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    include_chart: bool = True


@predict_router.post("/predict", tags=["Recipes"])
def predict(request: Request, body: PredictBody) -> PredictResult:
    """Predictive analysis over the ingested documents.

    Retrieves evidence for the requested metric, extracts a historical numeric
    series, computes a local forecast (`horizon` steps ahead), and returns a
    cited written outlook. If no numeric series is found it returns a
    qualitative, cited outlook instead. A base64 PNG chart is included when a
    forecast was produced (set `include_chart=false` to omit it).
    """
    service = request.state.injector.get(PredictService)
    result = service.predict(
        metric=body.metric,
        horizon=body.horizon,
        retrieve_limit=body.retrieve_limit,
        context_filter=body.context_filter,
    )
    return PredictResult(
        metric=result.metric,
        history=result.history,
        forecast=result.forecast,
        method=result.method,
        fit_error=result.fit_error,
        narrative=result.narrative,
        sources=result.sources if body.include_sources else [],
        chart_png_base64=result.chart_png_base64 if body.include_chart else None,
        steps=result.steps,
    )
