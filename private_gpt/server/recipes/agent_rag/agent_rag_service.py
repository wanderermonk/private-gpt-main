"""Agentic RAG recipe.

Where the standard ChatService does *one* retrieval and then answers, this
service runs a small agent loop over the ingested documents:

    plan  ->  retrieve (tool)  ->  reflect  ->  (retrieve again)  ->  answer

The only "tool" the agent is given is document retrieval, implemented by
reusing ``ChunksService.retrieve_relevant``. Every step is a plain
``llm.complete(prompt)`` call, so this works with any configured backend
(local llama.cpp / Ollama / OpenAI / ...) without needing a native
function-calling API. The amount of evidence fed back into the model is
capped so it stays within small local context windows.

This file is purely additive - it does not modify any existing behaviour.
"""

import json
import logging
import re
from dataclasses import dataclass, field

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Public result models
# --------------------------------------------------------------------------- #
class AgentStep(BaseModel):
    """One observable step of the agent loop, returned for transparency."""

    kind: str  # "plan" | "retrieve" | "reflect" | "answer"
    detail: str
    queries: list[str] | None = None
    num_new_chunks: int | None = None


class AgentSearchResult(BaseModel):
    response: str
    sources: list[Chunk]
    steps: list[AgentStep]


@dataclass
class _Evidence:
    """Accumulated, de-duplicated chunks gathered across iterations."""

    chunks: list[Chunk] = field(default_factory=list)
    _seen: set[str] = field(default_factory=set)

    def add(self, new_chunks: list[Chunk]) -> int:
        added = 0
        for c in new_chunks:
            key = c.text.strip()
            if key and key not in self._seen:
                self._seen.add(key)
                self.chunks.append(c)
                added += 1
        return added


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #
_PLAN_PROMPT = (
    "You are a research planner. Break the user's question into at most "
    "{max_q} short, standalone search queries that, when looked up in a "
    "document collection, would gather the facts needed to answer it. "
    "If the question is simple, a single query is fine.\n"
    'Respond with ONLY a JSON array of strings, e.g. ["query one", "query two"].\n\n'
    "Question: {question}\n"
    "JSON:"
)

_REFLECT_PROMPT = (
    "You are a research assistant deciding whether enough evidence has been "
    "gathered to answer the question.\n"
    "Question: {question}\n\n"
    "Evidence gathered so far:\n{evidence}\n\n"
    "If this is enough to answer well, respond with exactly: DONE\n"
    "If something important is still missing, respond with ONLY a JSON array "
    "of at most {max_q} new search queries that would fill the gap. Do not "
    "repeat earlier queries.\n"
    "Answer:"
)

_ANSWER_PROMPT = (
    "Answer the question using ONLY the numbered evidence below. "
    "Cite the evidence you rely on inline using its number in square brackets, "
    "like [1] or [2]. If the evidence does not contain the answer, say that "
    "you could not find it in the documents.\n\n"
    "Question: {question}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Answer:"
)


@singleton
class AgentRagService:
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

    # ----------------------------- LLM helper ----------------------------- #
    def _complete(self, prompt: str) -> str:
        response = self.llm_component.llm.complete(prompt)
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        return text.strip()

    @staticmethod
    def _parse_query_list(raw: str, limit: int) -> list[str]:
        """Best-effort extraction of a list of queries from an LLM reply.

        Tries JSON first (the format we ask for), then falls back to
        line/number splitting so a chatty local model still works.
        """
        raw = raw.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                queries = [str(q).strip() for q in data if str(q).strip()]
                if queries:
                    return queries[:limit]
            except (json.JSONDecodeError, TypeError):
                pass
        # Fallback: split lines, strip bullets / numbering / quotes.
        lines = []
        for line in raw.splitlines():
            line = line.strip().strip('"').strip()
            line = re.sub(r"^[\-\*\d\.\)\s]+", "", line).strip()
            if line and not line.lower().startswith("json"):
                lines.append(line)
        return lines[:limit]

    def _retrieve(
        self,
        queries: list[str],
        evidence: _Evidence,
        retrieve_limit: int,
        context_filter: ContextFilter | None,
    ):
        """Run retrieval for each query, yielding one AgentStep per query."""
        for q in queries:
            try:
                chunks = self.chunks_service.retrieve_relevant(
                    text=q,
                    context_filter=context_filter,
                    limit=retrieve_limit,
                )
            except Exception as exc:  # retrieval must not crash the whole loop
                logger.warning("Retrieval failed for query %r: %s", q, exc)
                yield AgentStep(
                    kind="retrieve", detail=f"query failed: {q}", queries=[q]
                )
                continue
            added = evidence.add(chunks)
            yield AgentStep(
                kind="retrieve",
                detail=f"searched: {q}",
                queries=[q],
                num_new_chunks=added,
            )

    def _format_evidence(self, evidence: _Evidence, max_chars: int) -> str:
        """Number and truncate evidence so it fits a small context window."""
        lines: list[str] = []
        used = 0
        for i, chunk in enumerate(evidence.chunks, start=1):
            source = chunk.document.doc_metadata or {}
            label = source.get("file_name") or chunk.document.doc_id
            text = chunk.text.strip().replace("\n", " ")
            budget = max(0, max_chars - used)
            if budget <= 0:
                break
            snippet = text[:budget]
            used += len(snippet)
            lines.append(f"[{i}] (source: {label}) {snippet}")
        return "\n".join(lines) if lines else "(no evidence found)"

    # ------------------------------- Core loop ---------------------------- #
    def _search_gen(
        self,
        query: str,
        max_iterations: int,
        queries_per_step: int,
        retrieve_limit: int,
        max_evidence_chars: int,
        context_filter: ContextFilter | None,
    ):
        """The agent loop as a generator.

        Yields each ``AgentStep`` as it happens (so a UI can show live
        progress) and ``return``\\s the final ``AgentSearchResult``. Callers
        that only want the result use ``search``; callers that want progress
        use ``search_stream``.
        """
        steps: list[AgentStep] = []
        evidence = _Evidence()

        # 1. PLAN -------------------------------------------------------- #
        plan_raw = self._complete(
            _PLAN_PROMPT.format(max_q=queries_per_step, question=query)
        )
        queries = self._parse_query_list(plan_raw, queries_per_step) or [query]
        step = AgentStep(kind="plan", detail="initial search plan", queries=queries)
        steps.append(step)
        yield step

        # 2. ACT + REFLECT loop ----------------------------------------- #
        for iteration in range(max_iterations):
            for r_step in self._retrieve(
                queries, evidence, retrieve_limit, context_filter
            ):
                steps.append(r_step)
                yield r_step

            # Last allowed iteration: stop, go straight to answering.
            if iteration >= max_iterations - 1:
                break

            reflect_raw = self._complete(
                _REFLECT_PROMPT.format(
                    question=query,
                    evidence=self._format_evidence(evidence, max_evidence_chars),
                    max_q=queries_per_step,
                )
            )
            # The model either signals DONE or hands back follow-up queries.
            if "DONE" in reflect_raw.upper():
                step = AgentStep(kind="reflect", detail="enough evidence; answering")
                steps.append(step)
                yield step
                break
            follow_ups = self._parse_query_list(reflect_raw, queries_per_step)
            if not follow_ups:
                step = AgentStep(kind="reflect", detail="no further queries; answering")
                steps.append(step)
                yield step
                break
            queries = follow_ups
            step = AgentStep(
                kind="reflect", detail="gap found; searching further", queries=queries
            )
            steps.append(step)
            yield step

        # 3. ANSWER ------------------------------------------------------ #
        answer = self._complete(
            _ANSWER_PROMPT.format(
                question=query,
                evidence=self._format_evidence(evidence, max_evidence_chars),
            )
        )
        step = AgentStep(kind="answer", detail="final synthesis")
        steps.append(step)
        yield step

        return AgentSearchResult(
            response=answer, sources=evidence.chunks, steps=steps
        )

    # ------------------------------- Public ------------------------------- #
    def search(
        self,
        query: str,
        max_iterations: int = 3,
        queries_per_step: int = 3,
        retrieve_limit: int = 4,
        max_evidence_chars: int = 6000,
        context_filter: ContextFilter | None = None,
    ) -> AgentSearchResult:
        """Run the agentic RAG loop and return an answer with its sources.

        Args:
            query: the user's natural-language question.
            max_iterations: hard cap on plan+reflect rounds (prevents loops).
            queries_per_step: max search queries issued per round.
            retrieve_limit: chunks pulled per individual query.
            max_evidence_chars: cap on evidence text fed to the model, to
                respect small local context windows.
            context_filter: optional restriction to specific document ids.
        """
        gen = self._search_gen(
            query,
            max_iterations,
            queries_per_step,
            retrieve_limit,
            max_evidence_chars,
            context_filter,
        )
        # Drain the generator; its return value is the final result.
        result: AgentSearchResult | None = None
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                result = stop.value
                break
        assert result is not None
        return result

    def search_stream(
        self,
        query: str,
        max_iterations: int = 3,
        queries_per_step: int = 3,
        retrieve_limit: int = 4,
        max_evidence_chars: int = 6000,
        context_filter: ContextFilter | None = None,
    ):
        """Like ``search`` but yields ``AgentStep`` progress, then the final
        ``AgentSearchResult`` as the last item."""
        gen = self._search_gen(
            query,
            max_iterations,
            queries_per_step,
            retrieve_limit,
            max_evidence_chars,
            context_filter,
        )
        while True:
            try:
                yield next(gen)
            except StopIteration as stop:
                yield stop.value
                return
