"""Offline end-to-end test for PredictService.

Runs the REAL predict_service.py and the REAL forecast.py, stubbing only the
heavy private_gpt dependencies and the LLM. Proves:

  1. a numeric series is extracted from evidence and forecast locally,
  2. forecast periods continue the years in the history,
  3. a base64 PNG chart is produced (matplotlib),
  4. when no numbers exist, it degrades to a qualitative outlook,
  5. messy values ('$1,200', '1.3k') are parsed.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pydantic

# --- stub injector ---------------------------------------------------------- #
inj = types.ModuleType("injector")
inj.inject = lambda f: f
inj.singleton = lambda c: c
sys.modules["injector"] = inj


def _mk(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


# --- stub private_gpt tree -------------------------------------------------- #
for pkg in [
    "private_gpt",
    "private_gpt.components",
    "private_gpt.components.llm",
    "private_gpt.open_ai",
    "private_gpt.open_ai.extensions",
    "private_gpt.server",
    "private_gpt.server.chunks",
    "private_gpt.server.recipes",
    "private_gpt.server.recipes.agent_rag",
    "private_gpt.server.recipes.predict",
    "private_gpt.settings",
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

# --- load the REAL forecast + predict_service by path ----------------------- #
_here = Path(__file__).resolve().parent


def _find(rel):
    for base in (_here, _here.parent, _here / "predict"):
        p = base / rel
        if p.exists():
            return p
        p2 = base / "private_gpt" / "server" / "recipes" / "predict" / rel
        if p2.exists():
            return p2
    raise FileNotFoundError(rel)


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_load("private_gpt.server.recipes.predict.forecast", _find("forecast.py"))
svc = _load(
    "private_gpt.server.recipes.predict.predict_service", _find("predict_service.py")
)


# --- fakes ------------------------------------------------------------------ #
class _Resp:
    def __init__(self, t):
        self.text = t


class FakeLLM:
    def __init__(self, scripted):
        self.scripted = list(scripted)

    def complete(self, prompt):
        return _Resp(self.scripted.pop(0))


class FakeLLMComponent:
    def __init__(self, llm):
        self.llm = llm


class FakeChunks:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve_relevant(self, text, context_filter=None, limit=8):
        return self._chunks


def make_service(scripted, chunks):
    s = svc.PredictService.__new__(svc.PredictService)
    s.settings = Settings()
    s.llm_component = FakeLLMComponent(FakeLLM(scripted))
    s.chunks_service = FakeChunks(chunks)
    return s


EV = [Chunk(text="Revenue was 100 in 2020, 120 in 2021, 140 in 2022.",
            document=_Doc(doc_id="d1", doc_metadata={"file_name": "fin.pdf"}))]


def test_numeric_forecast_and_chart():
    service = make_service(
        scripted=[
            '[{"period":"2020","value":100},{"period":"2021","value":120},'
            '{"period":"2022","value":140}]',
            "Revenue is trending up ~20/yr and should keep rising [1].",
        ],
        chunks=EV,
    )
    r = service.predict("annual revenue", horizon=2)
    assert [p.value for p in r.history] == [100.0, 120.0, 140.0]
    assert len(r.forecast) == 2
    # years should continue: 2023, 2024
    assert [p.period for p in r.forecast] == ["2023", "2024"]
    # upward trend -> next value above last history
    assert r.forecast[0].value > 140
    assert r.method in ("linear", "holt")
    assert r.chart_png_base64 and len(r.chart_png_base64) > 100
    assert "[1]" in r.narrative
    assert r.sources
    print("  forecast:", [(p.period, p.value) for p in r.forecast], "method", r.method)
    print("PASS test_numeric_forecast_and_chart")


def test_qualitative_fallback():
    service = make_service(
        scripted=["[]", "No hard numbers, but sentiment is positive [1]."],
        chunks=EV,
    )
    r = service.predict("market outlook", horizon=3)
    assert r.history == []
    assert r.forecast == []
    assert r.method == "none"
    assert r.chart_png_base64 is None
    assert "[1]" in r.narrative
    print("PASS test_qualitative_fallback")


def test_messy_value_parsing():
    service = make_service(
        scripted=[
            '[{"period":"Q1","value":"$1,200"},{"period":"Q2","value":"1.3k"},'
            '{"period":"Q3","value":1500}]',
            "Up and to the right [1].",
        ],
        chunks=EV,
    )
    r = service.predict("bookings", horizon=1)
    assert [p.value for p in r.history] == [1200.0, 1300.0, 1500.0], r.history
    # non-year labels -> generic forecast label
    assert r.forecast[0].period.startswith("Forecast")
    print("  parsed history:", [p.value for p in r.history])
    print("PASS test_messy_value_parsing")


def test_stream_yields_steps_then_result():
    service = make_service(
        scripted=['[{"period":"2020","value":1},{"period":"2021","value":2}]',
                  "Outlook [1]."],
        chunks=EV,
    )
    items = list(service.predict_stream("x", horizon=1))
    *steps, final = items
    assert all(isinstance(s, svc.AgentStep) for s in steps)
    assert isinstance(final, svc.PredictResult)
    print("  step kinds:", [s.kind for s in steps])
    print("PASS test_stream_yields_steps_then_result")


if __name__ == "__main__":
    test_numeric_forecast_and_chart()
    test_qualitative_fallback()
    test_messy_value_parsing()
    test_stream_yields_steps_then_result()
    print("\nALL PREDICT TESTS PASSED")
