"""Offline functional test for AgentRagService.

Runs the REAL agent_rag_service.py loop, but stubs the heavy dependencies
(injector, llama_index, and the private_gpt components) so it can execute
without a model or a vector store. Proves:

  1. the planner's JSON query list is parsed,
  2. retrieval is actually called (as the agent's tool),
  3. a reflection "DONE" ends the loop,
  4. follow-up queries trigger a second retrieval round,
  5. evidence is de-duplicated,
  6. the final answer + sources + step trace come back.
"""

import sys
import types
from pathlib import Path

import pydantic

# --------------------------------------------------------------------------- #
# 1. Stub `injector` (inject/singleton are no-op passthroughs here).
# --------------------------------------------------------------------------- #
injector_mod = types.ModuleType("injector")
injector_mod.inject = lambda f: f
injector_mod.singleton = lambda c: c
sys.modules["injector"] = injector_mod

# --------------------------------------------------------------------------- #
# 2. Stub the private_gpt modules the service imports.
# --------------------------------------------------------------------------- #


def _mk(name):
    m = types.ModuleType(name)
    sys.modules[name] = m
    return m


_mk("private_gpt")
_mk("private_gpt.components")
_mk("private_gpt.components.llm")
llm_mod = _mk("private_gpt.components.llm.llm_component")
_mk("private_gpt.open_ai")
_mk("private_gpt.open_ai.extensions")
cf_mod = _mk("private_gpt.open_ai.extensions.context_filter")
_mk("private_gpt.server")
_mk("private_gpt.server.chunks")
chunks_mod = _mk("private_gpt.server.chunks.chunks_service")
_mk("private_gpt.settings")
settings_mod = _mk("private_gpt.settings.settings")


class LLMComponent:  # placeholder type for the import
    pass


class ContextFilter(pydantic.BaseModel):
    docs_ids: list[str] | None = None


class _Doc(pydantic.BaseModel):
    doc_id: str
    doc_metadata: dict | None = None


class Chunk(pydantic.BaseModel):
    text: str
    document: _Doc


class ChunksService:  # placeholder type for the import
    pass


class Settings:  # placeholder type for the import
    pass


llm_mod.LLMComponent = LLMComponent
cf_mod.ContextFilter = ContextFilter
chunks_mod.Chunk = Chunk
chunks_mod.ChunksService = ChunksService
settings_mod.Settings = Settings

# --------------------------------------------------------------------------- #
# 3. Import the REAL service under test.
# --------------------------------------------------------------------------- #
import importlib.util  # noqa: E402

_here = Path(__file__).resolve().parent
_candidates = [
    _here / "private_gpt" / "server" / "recipes" / "agent_rag" / "agent_rag_service.py",
    _here.parent / "private_gpt" / "server" / "recipes" / "agent_rag" / "agent_rag_service.py",
    _here / "agent_rag" / "agent_rag_service.py",
]
_target = next((p for p in _candidates if p.exists()), None)
if _target is None:
    raise FileNotFoundError("agent_rag_service.py not found near " + str(_here))
_spec = importlib.util.spec_from_file_location("agent_rag_service_under_test", _target)
svc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svc)


# --------------------------------------------------------------------------- #
# 4. Fakes that drive the loop.
# --------------------------------------------------------------------------- #
class _Resp:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    """Scripted responses: plan -> reflect -> answer."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return _Resp(self.scripted.pop(0))


class FakeLLMComponent:
    def __init__(self, llm):
        self.llm = llm


class FakeChunks:
    """Returns different chunks per query; records what it was asked."""

    def __init__(self):
        self.calls = []

    def retrieve_relevant(self, text, context_filter=None, limit=4):
        self.calls.append(text)
        if "revenue" in text.lower():
            return [
                Chunk(text="Revenue grew 20% in 2023.", document=_Doc(doc_id="d1", doc_metadata={"file_name": "report.pdf"})),
                Chunk(text="Revenue grew 20% in 2023.", document=_Doc(doc_id="d1")),  # dup
            ]
        if "risk" in text.lower():
            return [Chunk(text="Main risk: supply chain.", document=_Doc(doc_id="d2", doc_metadata={"file_name": "risks.pdf"}))]
        return [Chunk(text="Generic context.", document=_Doc(doc_id="d3"))]


def make_service(scripted):
    s = svc.AgentRagService.__new__(svc.AgentRagService)
    s.settings = Settings()
    s.llm_component = FakeLLMComponent(FakeLLM(scripted))
    s.chunks_service = FakeChunks()
    return s


# --------------------------------------------------------------------------- #
# 5. Tests.
# --------------------------------------------------------------------------- #
def test_two_rounds_then_done():
    # plan -> 2 queries ; reflect -> follow-up ; (last iter, no reflect) ; answer
    service = make_service(
        scripted=[
            '["revenue 2023", "revenue drivers"]',   # plan
            '["risk factors"]',                       # reflect round 1 -> follow up
            "Revenue grew 20% [1]; main risk is supply chain [3].",  # answer
        ]
    )
    result = service.search("How did revenue and risks look?", max_iterations=2)

    kinds = [s.kind for s in result.steps]
    assert kinds[0] == "plan"
    assert "retrieve" in kinds
    assert "reflect" in kinds
    assert kinds[-1] == "answer"

    # de-dup: the duplicate revenue chunk must appear once.
    texts = [c.text for c in result.sources]
    assert texts.count("Revenue grew 20% in 2023.") == 1
    # follow-up actually searched the risk query
    assert any("risk" in q.lower() for q in service.chunks_service.calls)
    assert "supply chain" in " ".join(c.text for c in result.sources)
    assert result.response.strip() != ""
    print("  calls:", service.chunks_service.calls)
    print("  sources:", texts)
    print("PASS test_two_rounds_then_done")


def test_done_short_circuits():
    service = make_service(
        scripted=[
            '["revenue 2023"]',   # plan
            "DONE",               # reflect -> stop immediately
            "Revenue grew 20% [1].",  # answer
        ]
    )
    result = service.search("revenue?", max_iterations=3)
    # only one retrieve round (one query) should have happened
    assert service.chunks_service.calls == ["revenue 2023"], service.chunks_service.calls
    assert any(s.kind == "reflect" and "answering" in s.detail for s in result.steps)
    print("PASS test_done_short_circuits")


def test_plan_fallback_on_garbage():
    # planner returns non-JSON -> falls back to line parsing / raw query
    service = make_service(
        scripted=[
            "I think we should look at: revenue",  # messy plan, no JSON
            "DONE",
            "Answer [1].",
        ]
    )
    result = service.search("revenue?", max_iterations=2)
    assert service.chunks_service.calls, "retrieval should still run on fallback query"
    assert result.response.strip() != ""
    print("  fallback queried:", service.chunks_service.calls)
    print("PASS test_plan_fallback_on_garbage")


def test_search_stream_yields_steps_then_result():
    service = make_service(
        scripted=['["revenue 2023"]', "DONE", "Revenue grew 20% [1]."]
    )
    items = list(service.search_stream("revenue?", max_iterations=3))
    # Every item but the last is a step; the last is the final result.
    *step_items, final = items
    assert all(isinstance(s, svc.AgentStep) for s in step_items)
    assert isinstance(final, svc.AgentSearchResult)
    assert step_items[0].kind == "plan"
    assert any(s.kind == "retrieve" for s in step_items)
    assert final.response.strip() != ""
    print("  streamed step kinds:", [s.kind for s in step_items])
    print("PASS test_search_stream_yields_steps_then_result")


if __name__ == "__main__":
    test_two_rounds_then_done()
    test_done_short_circuits()
    test_plan_fallback_on_garbage()
    test_search_stream_yields_steps_then_result()
    print("\nALL TESTS PASSED")
