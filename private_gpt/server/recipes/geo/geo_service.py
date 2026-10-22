"""Map recipe: pull places (with lat/long) and their events out of the
ingested documents and visualise them.

Pipeline:
    retrieve evidence -> extract {name, lat, lon, event} via the LLM
    -> merge in any explicit decimal coordinates found by regex
    -> render an OFFLINE static map (matplotlib, embedded in chat)
    -> ALSO write a standalone INTERACTIVE Leaflet map file (uses online
       map tiles; opened in a browser) for pan/zoom with event popups.

Only the extracted coordinates/labels ever touch the interactive map file;
the documents themselves never leave the machine. The interactive map does
fetch generic basemap tiles from the internet when opened - the static map
is fully offline.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import re
import time
from pathlib import Path

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.paths import local_data_path
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.recipes.agent_rag.agent_rag_service import AgentStep
from private_gpt.server.recipes.geo.coords import extract_decimal_pairs, valid_coord
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


class GeoPlace(BaseModel):
    name: str
    lat: float
    lon: float
    event: str = ""
    source: str = ""


class GeoResult(BaseModel):
    query: str
    places: list[GeoPlace]
    static_png_base64: str | None = None
    interactive_html_path: str | None = None
    sources: list[Chunk]
    steps: list[AgentStep]


_EXTRACT_PROMPT = (
    "From the evidence below, extract every place that has a geographic "
    "location, for the topic: \"{query}\".\n"
    "Return ONLY a JSON array; each item: "
    '{{"name": "<place>", "lat": <decimal>, "lon": <decimal>, '
    '"event": "<what happened there, short>"}}.\n'
    "Use decimal degrees (south/west negative). Skip places without "
    "coordinates. If none, return [].\n\n"
    "Evidence:\n{evidence}\n\n"
    "JSON:"
)


@singleton
class GeoService:
    @inject
    def __init__(
        self,
        settings: Settings,
        llm_component: LLMComponent,
        chunks_service: ChunksService,
    ) -> None:
        self.settings = settings
        self.llm_component = llm_component
        self.chunks_service = chunks_service

    # ------------------------------ helpers ------------------------------- #
    def _complete(self, prompt: str) -> str:
        resp = self.llm_component.llm.complete(prompt)
        return (getattr(resp, "text", None) or str(resp)).strip()

    def _format_evidence(self, chunks: list[Chunk], max_chars: int = 7000) -> str:
        lines, used = [], 0
        for i, c in enumerate(chunks, start=1):
            meta = c.document.doc_metadata or {}
            label = meta.get("file_name") or c.document.doc_id
            text = c.text.strip().replace("\n", " ")
            budget = max(0, max_chars - used)
            if budget <= 0:
                break
            snippet = text[:budget]
            used += len(snippet)
            lines.append(f"[{i}] (source: {label}) {snippet}")
        return "\n".join(lines) if lines else "(no evidence found)"

    def _parse_places(self, raw: str) -> list[GeoPlace]:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except (json.JSONDecodeError, TypeError):
            return []
        out: list[GeoPlace] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            try:
                lat = float(item.get("lat"))
                lon = float(item.get("lon"))
            except (TypeError, ValueError):
                continue
            if not valid_coord(lat, lon):
                continue
            out.append(
                GeoPlace(
                    name=str(item.get("name", "")).strip() or "(unnamed)",
                    lat=round(lat, 6),
                    lon=round(lon, 6),
                    event=str(item.get("event", "")).strip(),
                )
            )
        return out

    @staticmethod
    def _merge_regex(
        places: list[GeoPlace], chunks: list[Chunk]
    ) -> list[GeoPlace]:
        """Add decimal coords found verbatim in text that the LLM missed."""
        known = {(round(p.lat, 3), round(p.lon, 3)) for p in places}
        for c in chunks:
            meta = c.document.doc_metadata or {}
            src = meta.get("file_name") or c.document.doc_id
            for lat, lon in extract_decimal_pairs(c.text):
                key = (round(lat, 3), round(lon, 3))
                if key not in known:
                    known.add(key)
                    places.append(
                        GeoPlace(
                            name=f"{lat:.4f}, {lon:.4f}",
                            lat=lat,
                            lon=lon,
                            event="(coordinates found in text)",
                            source=str(src),
                        )
                    )
        return places

    @staticmethod
    def _static_map(places: list[GeoPlace]) -> str | None:
        """Offline scatter map (no basemap, no network) as a base64 PNG."""
        if not places:
            return None
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.info("Static map skipped (matplotlib unavailable): %s", exc)
            return None
        try:
            lons = [p.lon for p in places]
            lats = [p.lat for p in places]
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.scatter(lons, lats, c="#5b4bdc", s=60, edgecolors="white", zorder=3)
            for p in places:
                ax.annotate(
                    p.name,
                    (p.lon, p.lat),
                    textcoords="offset points",
                    xytext=(6, 4),
                    fontsize=8,
                )
            # pad bounds a little so labels aren't clipped
            if len(places) > 1:
                dx = (max(lons) - min(lons)) or 1.0
                dy = (max(lats) - min(lats)) or 1.0
                ax.set_xlim(min(lons) - dx * 0.15, max(lons) + dx * 0.3)
                ax.set_ylim(min(lats) - dy * 0.15, max(lats) + dy * 0.2)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title("Places found in documents (offline view)")
            ax.grid(True, alpha=0.3, zorder=0)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning("Static map render failed: %s", exc)
            return None

    @staticmethod
    def _interactive_html(places: list[GeoPlace]) -> str | None:
        """Write a standalone Leaflet map (uses online tiles) and return path."""
        if not places:
            return None
        markers = [
            {
                "lat": p.lat,
                "lon": p.lon,
                "name": p.name,
                "event": p.event,
                "source": p.source,
            }
            for p in places
        ]
        avg_lat = sum(p.lat for p in places) / len(places)
        avg_lon = sum(p.lon for p in places) / len(places)
        data_json = html.escape(json.dumps(markers), quote=True)
        page = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width, initial-scale=1'>"
            "<title>Document map</title>"
            "<link rel='stylesheet' "
            "href='https://unpkg.com/leaflet@1.9.4/dist/leaflet.css'/>"
            "<script src='https://unpkg.com/leaflet@1.9.4/dist/leaflet.js'>"
            "</script><style>html,body,#map{height:100%;margin:0}</style>"
            "</head><body><div id='map'></div><script>"
            f"var pts=JSON.parse(\"{data_json}\");"
            f"var map=L.map('map').setView([{avg_lat},{avg_lon}],5);"
            "L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',"
            "{attribution:'© OpenStreetMap'}).addTo(map);"
            "var g=[];pts.forEach(function(p){"
            "var m=L.marker([p.lat,p.lon]).addTo(map);"
            "m.bindPopup('<b>'+p.name+'</b><br>'+(p.event||'')+"
            "(p.source?('<br><i>'+p.source+'</i>'):''));g.push(m);});"
            "if(g.length>1){map.fitBounds(L.featureGroup(g).getBounds()"
            ".pad(0.2));}"
            "</script></body></html>"
        )
        out_dir = Path(local_data_path) / "maps"
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"map_{int(time.time())}.html"
            path.write_text(page, encoding="utf-8")
            return str(path)
        except OSError as exc:
            logger.warning("Could not write interactive map: %s", exc)
            return None

    # ------------------------------- core --------------------------------- #
    def _map_gen(
        self, query: str, retrieve_limit: int, context_filter: ContextFilter | None
    ):
        steps: list[AgentStep] = []
        try:
            chunks = self.chunks_service.retrieve_relevant(
                text=query or "places locations coordinates events",
                context_filter=context_filter,
                limit=retrieve_limit,
            )
        except Exception as exc:
            logger.warning("Geo retrieval failed: %s", exc)
            chunks = []
        step = AgentStep(
            kind="retrieve", detail="gathered evidence", num_new_chunks=len(chunks)
        )
        steps.append(step)
        yield step

        evidence = self._format_evidence(chunks)
        raw = self._complete(_EXTRACT_PROMPT.format(query=query, evidence=evidence))
        places = self._parse_places(raw)
        places = self._merge_regex(places, chunks)
        step = AgentStep(kind="reflect", detail=f"found {len(places)} located place(s)")
        steps.append(step)
        yield step

        static_png = self._static_map(places)
        interactive = self._interactive_html(places)
        step = AgentStep(kind="answer", detail="map rendered")
        steps.append(step)
        yield step

        return GeoResult(
            query=query,
            places=places,
            static_png_base64=static_png,
            interactive_html_path=interactive,
            sources=chunks,
            steps=steps,
        )

    # ------------------------------ public -------------------------------- #
    def map_places(
        self,
        query: str,
        retrieve_limit: int = 12,
        context_filter: ContextFilter | None = None,
    ) -> GeoResult:
        gen = self._map_gen(query, retrieve_limit, context_filter)
        result: GeoResult | None = None
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                result = stop.value
                break
        assert result is not None
        return result

    def map_stream(
        self,
        query: str,
        retrieve_limit: int = 12,
        context_filter: ContextFilter | None = None,
    ):
        gen = self._map_gen(query, retrieve_limit, context_filter)
        while True:
            try:
                yield next(gen)
            except StopIteration as stop:
                yield stop.value
                return
