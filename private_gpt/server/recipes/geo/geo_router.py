from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.recipes.geo.geo_service import GeoResult, GeoService
from private_gpt.server.utils.auth import authenticated

geo_router = APIRouter(prefix="/v1", dependencies=[Depends(authenticated)])


class GeoBody(BaseModel):
    query: str = Field(
        default="",
        examples=["incidents", "office locations", "events by city"],
    )
    retrieve_limit: int = Field(default=12, ge=1, le=40)
    context_filter: ContextFilter | None = None
    include_sources: bool = True
    include_static_image: bool = True


@geo_router.post("/geo/map", tags=["Recipes"])
def geo_map(request: Request, body: GeoBody) -> GeoResult:
    """Extract located places (lat/long) + their events from the documents.

    Returns the places, an offline static map (base64 PNG), and the path to a
    standalone interactive Leaflet map written to disk (opens in a browser;
    that map fetches generic basemap tiles from the internet).
    """
    service = request.state.injector.get(GeoService)
    result = service.map_places(
        query=body.query,
        retrieve_limit=body.retrieve_limit,
        context_filter=body.context_filter,
    )
    return GeoResult(
        query=result.query,
        places=result.places,
        static_png_base64=(
            result.static_png_base64 if body.include_static_image else None
        ),
        interactive_html_path=result.interactive_html_path,
        sources=result.sources if body.include_sources else [],
        steps=result.steps,
    )
