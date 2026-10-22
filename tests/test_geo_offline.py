"""Offline end-to-end test for GeoService (real coords + real matplotlib)."""

import importlib.util
import sys
import tempfile
import types
from pathlib import Path

import pydantic

inj = types.ModuleType("injector")
inj.inject = lambda f: f
inj.singleton = lambda c: c
sys.modules["injector"] = inj


def _mk(n):
    m = types.ModuleType(n)
    sys.modules[n] = m
    return m


for pkg in [
    "private_gpt", "private_gpt.components", "private_gpt.components.llm",
    "private_gpt.open_ai", "private_gpt.open_ai.extensions",
    "private_gpt.server", "private_gpt.server.chunks",
    "private_gpt.server.recipes", "private_gpt.server.recipes.agent_rag",
    "private_gpt.server.recipes.geo", "private_gpt.settings", "private_gpt.paths",
]:
    _mk(pkg)


class LLMComponent:
    pass


class ContextFilter(pydantic.BaseModel):
    docs_ids: list[str] | None = None


class _Doc(pydantic.BaseModel):
    doc_id: str
    doc_metadata: dict | None = None


class Chunk(pydantic.BaseModel):
    text: str
    document: _Doc


class ChunksService:
    pass


class Settings:
    pass


class AgentStep(pydantic.BaseModel):
    kind: str
    detail: str
    queries: list[str] | None = None
    num_new_chunks: int | None = None


sys.modules["private_gpt.components.llm.llm_component"] = _mk(
    "private_gpt.components.llm.llm_component"
)
sys.modules["private_gpt.components.llm.llm_component"].LLMComponent = LLMComponent
sys.modules["private_gpt.open_ai.extensions.context_filter"] = _mk(
    "private_gpt.open_ai.extensions.context_filter"
)
sys.modules["private_gpt.open_ai.extensions.context_filter"].ContextFilter = ContextFilter
sys.modules["private_gpt.server.chunks.chunks_service"] = _mk(
    "private_gpt.server.chunks.chunks_service"
)
sys.modules["private_gpt.server.chunks.chunks_service"].Chunk = Chunk
sys.modules["private_gpt.server.chunks.chunks_service"].ChunksService = ChunksService
sys.modules["private_gpt.settings.settings"] = _mk("private_gpt.settings.settings")
sys.modules["private_gpt.settings.settings"].Settings = Settings
agent_mod = _mk("private_gpt.server.recipes.agent_rag.agent_rag_service")
agent_mod.AgentStep = AgentStep
paths_mod = _mk("private_gpt.paths")
paths_mod.local_data_path = Path(tempfile.mkdtemp())

_here = Path(__file__).resolve().parent


def _find(rel):
    for base in (_here, _here / "geo", _here.parent):
        for p in (base / rel,
                  base / "private_gpt" / "server" / "recipes" / "geo" / rel):
            if p.exists():
                return p
    raise FileNotFoundError(rel)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_load("private_gpt.server.recipes.geo.coords", _find("coords.py"))
svc = _load("private_gpt.server.recipes.geo.geo_service", _find("geo_service.py"))


class _Resp:
    def __init__(self, t):
        self.text = t


class FakeLLM:
    def __init__(self, s):
        self.s = list(s)

    def complete(self, p):
        return _Resp(self.s.pop(0))


class FakeLLMComponent:
    def __init__(self, llm):
        self.llm = llm


class FakeChunks:
    def __init__(self, c):
        self._c = c

    def retrieve_relevant(self, text, context_filter=None, limit=12):
        return self._c


def make(scripted, chunks):
    s = svc.GeoService.__new__(svc.GeoService)
    s.settings = Settings()
    s.llm_component = FakeLLMComponent(FakeLLM(scripted))
    s.chunks_service = FakeChunks(chunks)
    return s


EV = [
    Chunk(
        text="HQ in Bengaluru 12.9716, 77.5946 opened in 2021. "
        "Incident near 1.2921 S, 36.8219 E reported.",
        document=_Doc(doc_id="d1", doc_metadata={"file_name": "ops.pdf"}),
    )
]


def test_map_extraction_and_render():
    service = make(
        scripted=[
            '[{"name":"Bengaluru HQ","lat":12.9716,"lon":77.5946,'
            '"event":"HQ opened 2021"}]'
        ],
        chunks=EV,
    )
    r = service.map_places("locations", retrieve_limit=5)
    names = [p.name for p in r.places]
    # LLM place present
    assert any("Bengaluru" in n for n in names), names
    # regex-found Nairobi coord (1.2921 S) merged as an extra place
    assert any(abs(p.lat + 1.2921) < 1e-3 for p in r.places), r.places
    # offline static map produced
    assert r.static_png_base64 and len(r.static_png_base64) > 100
    # interactive html written to disk
    assert r.interactive_html_path and Path(r.interactive_html_path).exists()
    html = Path(r.interactive_html_path).read_text()
    assert "leaflet" in html.lower() and "Bengaluru HQ" in html
    print("  places:", [(p.name, p.lat, p.lon) for p in r.places])
    print("PASS test_map_extraction_and_render")


def test_no_places():
    service = make(scripted=["[]"], chunks=[
        Chunk(text="No coordinates here at all.",
              document=_Doc(doc_id="d9", doc_metadata={"file_name": "x.pdf"}))
    ])
    r = service.map_places("locations")
    assert r.places == []
    assert r.static_png_base64 is None
    assert r.interactive_html_path is None
    print("PASS test_no_places")


def test_invalid_coords_rejected():
    service = make(
        scripted=['[{"name":"bad","lat":999,"lon":10,"event":"x"}]'],
        chunks=[Chunk(text="nothing", document=_Doc(doc_id="d", doc_metadata={}))],
    )
    r = service.map_places("x")
    assert r.places == [], r.places  # 999 latitude rejected
    print("PASS test_invalid_coords_rejected")


if __name__ == "__main__":
    test_map_extraction_and_render()
    test_no_places()
    test_invalid_coords_rejected()
    print("\nALL GEO TESTS PASSED")
