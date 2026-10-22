"""This file should be imported if and only if you want to run the UI locally."""

import logging
import time
from collections.abc import Iterable
from enum import Enum
from pathlib import Path
from typing import Any

import gradio as gr  # type: ignore
from fastapi import FastAPI
from gradio.themes.utils.colors import slate  # type: ignore
from injector import inject, singleton
from llama_index.core.llms import ChatMessage, ChatResponse, MessageRole
from llama_index.core.types import TokenGen
from pydantic import BaseModel

from private_gpt.constants import PROJECT_ROOT_PATH
from private_gpt.di import global_injector
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chat.chat_service import ChatService, CompletionGen
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.ingest.ingest_service import IngestService
from private_gpt.server.recipes.agent_rag.agent_rag_service import (
    AgentRagService,
    AgentSearchResult,
    AgentStep,
)
from private_gpt.server.recipes.geo.geo_service import GeoResult, GeoService
from private_gpt.server.recipes.predict.predict_service import (
    PredictResult,
    PredictService,
)
from private_gpt.server.recipes.summarize.summarize_service import SummarizeService
from private_gpt.settings.settings import settings

logger = logging.getLogger(__name__)

THIS_DIRECTORY_RELATIVE = Path(__file__).parent.relative_to(PROJECT_ROOT_PATH)
# Should be "private_gpt/ui/avatar-bot.ico"
AVATAR_BOT = THIS_DIRECTORY_RELATIVE / "avatar-bot.ico"

UI_TAB_TITLE = "My Private GPT"

SOURCES_SEPARATOR = "<hr>Sources: \n"


class Modes(str, Enum):
    RAG_MODE = "RAG"
    AGENTIC_MODE = "Agentic"
    PREDICT_MODE = "Predict"
    MAP_MODE = "Map"
    SEARCH_MODE = "Search"
    BASIC_CHAT_MODE = "Basic"
    SUMMARIZE_MODE = "Summarize"


MODES: list[Modes] = [
    Modes.RAG_MODE,
    Modes.AGENTIC_MODE,
    Modes.PREDICT_MODE,
    Modes.MAP_MODE,
    Modes.SEARCH_MODE,
    Modes.BASIC_CHAT_MODE,
    Modes.SUMMARIZE_MODE,
]


def _format_agent_step(step: "AgentStep") -> str:
    """Render one agent step as a short markdown progress line."""
    if step.kind == "plan":
        qs = "; ".join(step.queries or [])
        return f"🧭 **Planned searches:** {qs}"
    if step.kind == "retrieve":
        q = (step.queries or ["?"])[0]
        n = step.num_new_chunks
        suffix = f" (+{n} new)" if n is not None else ""
        return f"🔎 Searched: _{q}_{suffix}"
    if step.kind == "reflect":
        return f"🤔 {step.detail}"
    if step.kind == "answer":
        return "✍️ Writing the answer…"
    return f"• {step.detail}"


def _format_predict_result(result: "PredictResult") -> str:
    """Render a PredictResult as chat markdown: outlook + table + chart."""
    parts: list[str] = [result.narrative.strip(), ""]

    if result.history or result.forecast:
        parts.append("\n**Data & projection**\n")
        parts.append("| Period | Value | Type |")
        parts.append("| --- | --- | --- |")
        for p in result.history:
            parts.append(f"| {p.period} | {p.value:g} | history |")
        for p in result.forecast:
            parts.append(f"| {p.period} | {p.value:g} | **forecast** |")
        if result.method and result.method != "none":
            err = (
                f", backtest error {result.fit_error}"
                if result.fit_error is not None
                else ""
            )
            parts.append(f"\n_Method: {result.method}{err}._")

    if result.chart_png_base64:
        parts.append(
            f"\n![forecast chart](data:image/png;base64,"
            f"{result.chart_png_base64})"
        )

    if result.sources:
        parts.append(SOURCES_SEPARATOR)
        cur_sources = Source.curate_sources(result.sources)
        used = set()
        sources_text = "\n\n\n"
        for index, source in enumerate(cur_sources, start=1):
            key = f"{source.file}-{source.page}"
            if key not in used:
                sources_text += f"{index}. {source.file} (page {source.page}) \n\n"
                used.add(key)
        sources_text += "<hr>\n\n"
        parts.append(sources_text)

    return "\n".join(parts)


def _format_geo_result(result: "GeoResult") -> str:
    """Render a GeoResult as chat markdown: table + offline map + map file."""
    if not result.places:
        return (
            "No places with coordinates were found in the selected documents. "
            "Make sure the documents contain latitude/longitude values."
        )
    parts: list[str] = [f"**Found {len(result.places)} located place(s).**", ""]
    parts.append("| # | Place | Lat | Lon | Event |")
    parts.append("| --- | --- | --- | --- | --- |")
    for i, p in enumerate(result.places, start=1):
        event = (p.event or "").replace("|", "/")
        parts.append(f"| {i} | {p.name} | {p.lat:.4f} | {p.lon:.4f} | {event} |")

    if result.static_png_base64:
        parts.append(
            f"\n![map](data:image/png;base64,{result.static_png_base64})"
        )

    if result.interactive_html_path:
        parts.append(
            f"\n🗺️ **Interactive map saved to:** `{result.interactive_html_path}`\n"
            "\nOpen that file in a browser for a pan/zoom map with event "
            "popups. _(The interactive map loads basemap tiles from the "
            "internet; the offline image above makes no network calls.)_"
        )
    return "\n".join(parts)


class Source(BaseModel):
    file: str
    page: str
    text: str

    class Config:
        frozen = True

    @staticmethod
    def curate_sources(sources: list[Chunk]) -> list["Source"]:
        curated_sources = []

        for chunk in sources:
            doc_metadata = chunk.document.doc_metadata

            file_name = doc_metadata.get("file_name", "-") if doc_metadata else "-"
            page_label = doc_metadata.get("page_label", "-") if doc_metadata else "-"

            source = Source(file=file_name, page=page_label, text=chunk.text)
            curated_sources.append(source)
            curated_sources = list(
                dict.fromkeys(curated_sources).keys()
            )  # Unique sources only

        return curated_sources


@singleton
class PrivateGptUi:
    @inject
    def __init__(
        self,
        ingest_service: IngestService,
        chat_service: ChatService,
        chunks_service: ChunksService,
        summarizeService: SummarizeService,
        agent_rag_service: AgentRagService,
        predict_service: PredictService,
        geo_service: GeoService,
    ) -> None:
        self._ingest_service = ingest_service
        self._chat_service = chat_service
        self._chunks_service = chunks_service
        self._summarize_service = summarizeService
        self._agent_rag_service = agent_rag_service
        self._predict_service = predict_service
        self._geo_service = geo_service

        # Cache the UI blocks
        self._ui_block = None

        self._selected_filename = None

        # Initialize system prompt based on default mode
        default_mode_map = {mode.value: mode for mode in Modes}
        self._default_mode = default_mode_map.get(
            settings().ui.default_mode, Modes.RAG_MODE
        )
        self._system_prompt = self._get_default_system_prompt(self._default_mode)

        # Model selection (GGUF files in models/ + any pulled Ollama models).
        # Reached via the chat service's LLM component so we can switch at
        # runtime without restarting the app.
        self._llm_component = self._chat_service.llm_component
        self._model_id_by_label: dict[str, str] = {}
        self._refresh_model_choices()

    # ------------------------- model selection ---------------------------- #
    def _refresh_model_choices(self) -> list[str]:
        models = self._llm_component.list_models()
        self._model_id_by_label = {m["label"]: m["id"] for m in models}
        return list(self._model_id_by_label.keys())

    def _set_model(self, label: str) -> Any:
        model_id = self._model_id_by_label.get(label)
        if not model_id:
            return gr.update()
        try:
            new_label = self._llm_component.switch_model(model_id)
            gr.Info(f"Model switched to {new_label}")
        except Exception as e:  # keep the UI alive on a bad switch
            logger.error("Model switch failed: %s", e)
            gr.Warning(f"Could not switch model: {e}")
        return gr.update()

    def _run_single(
        self,
        message: str,
        history: list[list[str]],
        mode: Modes,
        files_selected: list[str] | None,
    ) -> Any:
        def yield_deltas(completion_gen: CompletionGen) -> Iterable[str]:
            full_response: str = ""
            stream = completion_gen.response
            for delta in stream:
                if isinstance(delta, str):
                    full_response += str(delta)
                elif isinstance(delta, ChatResponse):
                    full_response += delta.delta or ""
                yield full_response
                time.sleep(0.02)

            if completion_gen.sources:
                full_response += SOURCES_SEPARATOR
                cur_sources = Source.curate_sources(completion_gen.sources)
                sources_text = "\n\n\n"
                used_files = set()
                for index, source in enumerate(cur_sources, start=1):
                    if f"{source.file}-{source.page}" not in used_files:
                        sources_text = (
                            sources_text
                            + f"{index}. {source.file} (page {source.page}) \n\n"
                        )
                        used_files.add(f"{source.file}-{source.page}")
                sources_text += "<hr>\n\n"
                full_response += sources_text
            yield full_response

        def yield_tokens(token_gen: TokenGen) -> Iterable[str]:
            full_response: str = ""
            for token in token_gen:
                full_response += str(token)
                yield full_response

        def build_history() -> list[ChatMessage]:
            history_messages: list[ChatMessage] = []

            for interaction in history:
                history_messages.append(
                    ChatMessage(content=interaction[0], role=MessageRole.USER)
                )
                if len(interaction) > 1 and interaction[1] is not None:
                    history_messages.append(
                        ChatMessage(
                            # Remove from history content the Sources information
                            content=interaction[1].split(SOURCES_SEPARATOR)[0],
                            role=MessageRole.ASSISTANT,
                        )
                    )

            # max 20 messages to try to avoid context overflow
            return history_messages[:20]

        new_message = ChatMessage(content=message, role=MessageRole.USER)
        all_messages = [*build_history(), new_message]
        # If a system prompt is set, add it as a system message
        if self._system_prompt:
            all_messages.insert(
                0,
                ChatMessage(
                    content=self._system_prompt,
                    role=MessageRole.SYSTEM,
                ),
            )
        match mode:
            case Modes.RAG_MODE:
                # Restrict to the selected files (empty selection = all files)
                context_filter = self._context_filter(files_selected)

                query_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=True,
                    context_filter=context_filter,
                )
                yield from yield_deltas(query_stream)
            case Modes.AGENTIC_MODE:
                # Agentic RAG: the model plans searches, retrieves, reflects,
                # and may search again before answering with citations.
                # Restrict to the selected files (empty selection = all files)
                context_filter = self._context_filter(files_selected)

                progress = "🤖 **Agent working…**\n\n"
                final_result: AgentSearchResult | None = None
                for item in self._agent_rag_service.search_stream(
                    query=message, context_filter=context_filter
                ):
                    if isinstance(item, AgentSearchResult):
                        final_result = item
                        break
                    # item is an AgentStep -> show live progress
                    progress += _format_agent_step(item) + "\n\n"
                    yield progress

                # Build the final message: answer + sources + collapsed trace.
                if final_result is None:
                    yield "The agent did not return a result."
                else:
                    full_response = final_result.response
                    if final_result.sources:
                        full_response += SOURCES_SEPARATOR
                        cur_sources = Source.curate_sources(final_result.sources)
                        used_files = set()
                        sources_text = "\n\n\n"
                        for index, source in enumerate(cur_sources, start=1):
                            if f"{source.file}-{source.page}" not in used_files:
                                sources_text += (
                                    f"{index}. {source.file} "
                                    f"(page {source.page}) \n\n"
                                )
                                used_files.add(f"{source.file}-{source.page}")
                        sources_text += "<hr>\n\n"
                        full_response += sources_text
                    trace = "\n".join(
                        _format_agent_step(s) for s in final_result.steps
                    )
                    full_response += (
                        f"\n\n<details><summary>🧠 Agent steps</summary>\n\n"
                        f"{trace}\n\n</details>"
                    )
                    yield full_response
            case Modes.PREDICT_MODE:
                # Predictive analysis: extract a numeric series from the docs,
                # forecast it locally, and narrate a cited outlook + chart.
                context_filter = self._context_filter(files_selected)

                progress = "📈 **Predictive analysis…**\n\n"
                predict_result: PredictResult | None = None
                for item in self._predict_service.predict_stream(
                    metric=message, context_filter=context_filter
                ):
                    if isinstance(item, PredictResult):
                        predict_result = item
                        break
                    progress += _format_agent_step(item) + "\n\n"
                    yield progress

                if predict_result is None:
                    yield "The prediction did not return a result."
                else:
                    final = _format_predict_result(predict_result)
                    trace = "\n".join(
                        _format_agent_step(s) for s in predict_result.steps
                    )
                    final += (
                        f"\n\n<details><summary>🧠 Analysis steps</summary>\n\n"
                        f"{trace}\n\n</details>"
                    )
                    yield final
            case Modes.MAP_MODE:
                # Extract located places (+events) and plot them on a map.
                context_filter = self._context_filter(files_selected)
                progress = "🗺️ **Mapping places…**\n\n"
                geo_result: GeoResult | None = None
                for item in self._geo_service.map_stream(
                    query=message, context_filter=context_filter
                ):
                    if isinstance(item, GeoResult):
                        geo_result = item
                        break
                    progress += _format_agent_step(item) + "\n\n"
                    yield progress

                if geo_result is None:
                    yield "The map step did not return a result."
                else:
                    out = _format_geo_result(geo_result)
                    trace = "\n".join(
                        _format_agent_step(s) for s in geo_result.steps
                    )
                    out += (
                        f"\n\n<details><summary>🧠 Steps</summary>\n\n"
                        f"{trace}\n\n</details>"
                    )
                    yield out
            case Modes.BASIC_CHAT_MODE:
                llm_stream = self._chat_service.stream_chat(
                    messages=all_messages,
                    use_context=False,
                )
                yield from yield_deltas(llm_stream)

            case Modes.SEARCH_MODE:
                response = self._chunks_service.retrieve_relevant(
                    text=message,
                    context_filter=self._context_filter(files_selected),
                    limit=4,
                    prev_next_chunks=0,
                )

                sources = Source.curate_sources(response)

                yield "\n\n\n".join(
                    f"{index}. **{source.file} "
                    f"(page {source.page})**\n "
                    f"{source.text}"
                    for index, source in enumerate(sources, start=1)
                )
            case Modes.SUMMARIZE_MODE:
                # Summarize the selected files (empty selection = all files)
                context_filter = self._context_filter(files_selected)

                summary_stream = self._summarize_service.stream_summarize(
                    use_context=True,
                    context_filter=context_filter,
                    instructions=message,
                )
                yield from yield_tokens(summary_stream)

    # ---------------------- multi-file / multi-model ---------------------- #
    def _ingested_file_names(self) -> list[str]:
        names = set()
        for doc in self._ingest_service.list_ingested():
            md = doc.doc_metadata
            if md:
                names.add(md.get("file_name", "[FILE NAME MISSING]"))
        return sorted(names)

    def _context_filter(self, files_selected: list[str] | None):
        """Build a ContextFilter for the chosen files. Empty = all files."""
        if not files_selected:
            return None
        wanted = set(files_selected)
        docs_ids = []
        for doc in self._ingest_service.list_ingested():
            md = doc.doc_metadata or {}
            if md.get("file_name") in wanted:
                docs_ids.append(doc.doc_id)
        return ContextFilter(docs_ids=docs_ids) if docs_ids else None

    def _ensure_model(self, model_id: str) -> None:
        if getattr(self._llm_component, "current_model_id", None) != model_id:
            self._llm_component.switch_model(model_id)

    # Upload / delete handlers that refresh the multi-select file checkbox.
    def _upload_and_refresh(self, files: list[str]) -> Any:
        self._upload_file(files)
        return gr.update(choices=self._ingested_file_names(), value=[])

    def _delete_checked_files(self, checked: list[str] | None) -> Any:
        wanted = set(checked or [])
        for doc in list(self._ingest_service.list_ingested()):
            md = doc.doc_metadata or {}
            if md.get("file_name") in wanted:
                self._ingest_service.delete(doc.doc_id)
        return gr.update(choices=self._ingested_file_names(), value=[])

    def _delete_all_and_refresh(self) -> Any:
        for doc in list(self._ingest_service.list_ingested()):
            self._ingest_service.delete(doc.doc_id)
        return gr.update(choices=self._ingested_file_names(), value=[])

    def _chat(
        self,
        message: str,
        history: list[list[str]],
        mode: Modes,
        models_selected: list[str] | None,
        files_selected: list[str] | None,
        *_: Any,
    ) -> Any:
        """Entry point from the chat box.

        One model selected -> normal (streamed) answer. Several selected ->
        run the same query on each and stack the answers for comparison."""
        labels = models_selected or []
        model_ids: list[str] = []
        for lbl in labels:
            mid = self._model_id_by_label.get(lbl)
            if mid and mid not in model_ids:
                model_ids.append(mid)

        # Single (or no) model -> stream normally.
        if len(model_ids) <= 1:
            if model_ids:
                try:
                    self._ensure_model(model_ids[0])
                except Exception as e:
                    yield f"⚠️ Could not load model: {e}"
                    return
            yield from self._run_single(message, history, mode, files_selected)
            return

        # Multiple models -> run each and compare (no per-token streaming).
        combined = (
            f"**Comparing {len(model_ids)} models** on the same question.\n"
        )
        for mid in model_ids:
            label = self._llm_component._label_for(mid)
            header = f"\n\n## 🧠 {label}\n\n"
            try:
                self._ensure_model(mid)
            except Exception as e:
                combined += header + f"_could not load this model: {e}_\n"
                yield combined
                continue
            yield combined + header + "_running…_"
            last = ""
            for chunk in self._run_single(message, history, mode, files_selected):
                last = chunk
            combined += header + (last or "_(no output)_") + "\n\n<hr>\n"
            yield combined

    # On initialization and on mode change, this function set the system prompt
    # to the default prompt based on the mode (and user settings).
    @staticmethod
    def _get_default_system_prompt(mode: Modes) -> str:
        p = ""
        match mode:
            # For query chat mode, obtain default system prompt from settings
            case Modes.RAG_MODE:
                p = settings().ui.default_query_system_prompt
            # For chat mode, obtain default system prompt from settings
            case Modes.BASIC_CHAT_MODE:
                p = settings().ui.default_chat_system_prompt
            # For summarization mode, obtain default system prompt from settings
            case Modes.SUMMARIZE_MODE:
                p = settings().ui.default_summarization_system_prompt
            # For any other mode, clear the system prompt
            case _:
                p = ""
        return p

    @staticmethod
    def _get_default_mode_explanation(mode: Modes) -> str:
        match mode:
            case Modes.RAG_MODE:
                return "Get contextualized answers from selected files."
            case Modes.AGENTIC_MODE:
                return (
                    "Agent plans multiple searches, reflects on what it finds, "
                    "and searches again before answering with citations."
                )
            case Modes.PREDICT_MODE:
                return (
                    "Forecast a metric from your documents: type the metric "
                    "(e.g. 'quarterly revenue'). Extracts the numbers, projects "
                    "ahead, and writes a cited outlook with a chart."
                )
            case Modes.MAP_MODE:
                return (
                    "Plot places from your documents on a map. Type a topic "
                    "(e.g. 'incidents') or leave blank for all located places. "
                    "Returns a map + table + an interactive map file."
                )
            case Modes.SEARCH_MODE:
                return "Find relevant chunks of text in selected files."
            case Modes.BASIC_CHAT_MODE:
                return "Chat with the LLM using its training data. Files are ignored."
            case Modes.SUMMARIZE_MODE:
                return "Generate a summary of the selected files. Prompt to customize the result."
            case _:
                return ""

    def _set_system_prompt(self, system_prompt_input: str) -> None:
        logger.info(f"Setting system prompt to: {system_prompt_input}")
        self._system_prompt = system_prompt_input

    def _set_explanatation_mode(self, explanation_mode: str) -> None:
        self._explanation_mode = explanation_mode

    def _set_current_mode(self, mode: Modes) -> Any:
        self.mode = mode
        self._set_system_prompt(self._get_default_system_prompt(mode))
        self._set_explanatation_mode(self._get_default_mode_explanation(mode))
        interactive = self._system_prompt is not None
        return [
            gr.update(placeholder=self._system_prompt, interactive=interactive),
            gr.update(value=self._explanation_mode),
        ]

    def _list_ingested_files(self) -> list[list[str]]:
        files = set()
        for ingested_document in self._ingest_service.list_ingested():
            if ingested_document.doc_metadata is None:
                # Skipping documents without metadata
                continue
            file_name = ingested_document.doc_metadata.get(
                "file_name", "[FILE NAME MISSING]"
            )
            files.add(file_name)
        return [[row] for row in files]

    def _upload_file(self, files: list[str]) -> None:
        logger.debug("Loading count=%s files", len(files))
        paths = [Path(file) for file in files]

        # remove all existing Documents with name identical to a new file upload:
        file_names = [path.name for path in paths]
        doc_ids_to_delete = []
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"] in file_names
            ):
                doc_ids_to_delete.append(ingested_document.doc_id)
        if len(doc_ids_to_delete) > 0:
            logger.info(
                "Uploading file(s) which were already ingested: %s document(s) will be replaced.",
                len(doc_ids_to_delete),
            )
            for doc_id in doc_ids_to_delete:
                self._ingest_service.delete(doc_id)

        self._ingest_service.bulk_ingest([(str(path.name), path) for path in paths])

    def _delete_all_files(self) -> Any:
        ingested_files = self._ingest_service.list_ingested()
        logger.debug("Deleting count=%s files", len(ingested_files))
        for ingested_document in ingested_files:
            self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _delete_selected_file(self) -> Any:
        logger.debug("Deleting selected %s", self._selected_filename)
        # Note: keep looping for pdf's (each page became a Document)
        for ingested_document in self._ingest_service.list_ingested():
            if (
                ingested_document.doc_metadata
                and ingested_document.doc_metadata["file_name"]
                == self._selected_filename
            ):
                self._ingest_service.delete(ingested_document.doc_id)
        return [
            gr.List(self._list_ingested_files()),
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _deselect_selected_file(self) -> Any:
        self._selected_filename = None
        return [
            gr.components.Button(interactive=False),
            gr.components.Button(interactive=False),
            gr.components.Textbox("All files"),
        ]

    def _selected_a_file(self, select_data: gr.SelectData) -> Any:
        self._selected_filename = select_data.value
        return [
            gr.components.Button(interactive=True),
            gr.components.Button(interactive=True),
            gr.components.Textbox(self._selected_filename),
        ]

    def _build_ui_blocks(self) -> gr.Blocks:
        logger.debug("Creating the UI blocks")
        with gr.Blocks(
            title=UI_TAB_TITLE,
            theme=gr.themes.Soft(
                primary_hue=slate,
                neutral_hue="slate",
                # System fonts only — no web-font fetch, so it stays offline.
                font=[
                    "system-ui",
                    "-apple-system",
                    "Segoe UI",
                    "Roboto",
                    "Helvetica",
                    "sans-serif",
                ],
                font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
            ),
            css=(
                # Fill the whole window; no outer page scroll.
                ".gradio-container { max-width:100% !important;"
                "  height:100vh !important; padding:10px 16px 4px !important;"
                "  overflow:hidden !important; }"
                "footer { display:none !important; }"  # hide default gradio footer
                # Header
                ".app-header { background:linear-gradient(90deg,#5b4bdc,#8E7BFF);"
                "  color:#fff; padding:14px 20px; border-radius:14px;"
                "  box-shadow:0 4px 16px rgba(91,75,220,.28); }"
                ".app-header h1 { margin:0; font-size:20px; font-weight:750; }"
                ".app-header p { margin:3px 0 0; font-size:12.5px; opacity:.92; }"
                # Main area fills remaining height; columns manage their own scroll
                "#main-row { gap:14px; align-items:stretch;"
                "  height:calc(100vh - 118px); }"
                "#left-col { height:calc(100vh - 118px); overflow-y:auto;"
                "  padding-right:6px; }"
                "#chat-col { height:100%; display:flex; flex-direction:column; }"
                "#chatbot { flex:1 1 auto !important; min-height:0 !important;"
                "  height:auto !important; border-radius:14px; }"
                # Card-like groups
                ".gr-group, .block.gr-group { border-radius:14px !important; }"
                ".prose h3 { margin:.1em 0 .45em; font-size:14px; }"
                "hr { margin:1em 0; border:0;"
                "  border-top:1px solid var(--border-color-primary); }"
                ".app-footer { text-align:center; padding:6px 0 2px;"
                "  font-size:12px; color:var(--body-text-color-subdued); }"
                # Phone / small windows: let it stack and scroll
                "@media (max-width:820px){"
                "  .gradio-container{ height:auto !important; overflow:auto !important; }"
                "  #main-row{ height:auto; flex-direction:column; }"
                "  #chatbot{ height:60vh !important; } }"
            ),
        ) as blocks:
            # --- Header (your branding) ---------------------------------- #
            with gr.Row():
                gr.HTML(
                    "<div class='app-header'>"
                    "<h1>Private GPT · Fighter</h1>"
                    "<p>100% local &amp; private — your documents never leave "
                    "this machine. Ask, reason over, and forecast from your "
                    "files.</p>"
                    "</div>"
                )

            with gr.Row(equal_height=False, elem_id="main-row"):
                # ===================== LEFT: controls ===================== #
                with gr.Column(scale=3, elem_id="left-col"):
                    # ---- Documents (upload + multi-select) ------------ #
                    with gr.Group():
                        gr.Markdown("### 📄 Documents")
                        upload_button = gr.components.UploadButton(
                            "📤 Upload file(s) — click or drag & drop",
                            type="filepath",
                            file_count="multiple",
                            size="sm",
                        )
                        files_checkbox = gr.CheckboxGroup(
                            choices=self._ingested_file_names(),
                            value=[],
                            label="Query these files (tick none = use ALL)",
                        )
                        with gr.Row():
                            refresh_files_btn = gr.Button("↻ Refresh", size="sm")
                            delete_checked_btn = gr.Button(
                                "🗑️ Delete ticked",
                                size="sm",
                                visible=settings().ui.delete_file_button_enabled,
                            )
                            delete_all_btn = gr.Button(
                                "⚠️ Delete ALL",
                                size="sm",
                                visible=settings().ui.delete_all_files_button_enabled,
                            )

                    # ---- Model picker (multi-select) ------------------ #
                    with gr.Group():
                        gr.Markdown("### 🧠 Model(s)")
                        model_multiselect = gr.Dropdown(
                            choices=self._refresh_model_choices(),
                            value=[self._llm_component.current_model_label],
                            label="Pick 1+ models (several = compare answers)",
                            multiselect=True,
                            interactive=True,
                        )
                        refresh_models_btn = gr.Button(
                            "↻ Refresh model list", size="sm"
                        )

                    # ---- Mode selector -------------------------------- #
                    with gr.Group():
                        gr.Markdown("### 🎛️ Mode")
                        default_mode = self._default_mode
                        mode = gr.Radio(
                            [mode.value for mode in MODES],
                            show_label=False,
                            value=default_mode,
                        )
                        explanation_mode = gr.Textbox(
                            value=self._get_default_mode_explanation(default_mode),
                            show_label=False,
                            max_lines=3,
                            interactive=False,
                        )

                    system_prompt_input = gr.Textbox(
                        placeholder=self._system_prompt,
                        label="System prompt",
                        lines=2,
                        interactive=True,
                        render=False,
                    )

                    # ---- wiring --------------------------------------- #
                    refresh_models_btn.click(
                        lambda: gr.update(choices=self._refresh_model_choices()),
                        outputs=model_multiselect,
                    )
                    upload_button.upload(
                        self._upload_and_refresh,
                        inputs=upload_button,
                        outputs=files_checkbox,
                    )
                    refresh_files_btn.click(
                        lambda: gr.update(choices=self._ingested_file_names()),
                        outputs=files_checkbox,
                    )
                    delete_checked_btn.click(
                        self._delete_checked_files,
                        inputs=files_checkbox,
                        outputs=files_checkbox,
                    )
                    delete_all_btn.click(
                        self._delete_all_and_refresh,
                        outputs=files_checkbox,
                    )
                    mode.change(
                        self._set_current_mode,
                        inputs=mode,
                        outputs=[system_prompt_input, explanation_mode],
                    )
                    system_prompt_input.blur(
                        self._set_system_prompt,
                        inputs=system_prompt_input,
                    )

                # ===================== RIGHT: chat ======================== #
                with gr.Column(scale=7, elem_id="chat-col"):
                    label_text = f"Model: {self._llm_component.current_model_label}"
                    _ = gr.ChatInterface(
                        self._chat,
                        chatbot=gr.Chatbot(
                            label=label_text,
                            show_copy_button=True,
                            elem_id="chatbot",
                            render=False,
                            avatar_images=(None, AVATAR_BOT),
                        ),
                        additional_inputs=[
                            mode,
                            model_multiselect,
                            files_checkbox,
                            system_prompt_input,
                        ],
                    )

            with gr.Row():
                gr.HTML(
                    "<div class='app-footer'>© 2026 Fighter · Runs fully offline"
                    " · Built on PrivateGPT (Apache-2.0)</div>"
                )

        return blocks

    def get_ui_blocks(self) -> gr.Blocks:
        if self._ui_block is None:
            self._ui_block = self._build_ui_blocks()
        return self._ui_block

    def mount_in_app(self, app: FastAPI, path: str) -> None:
        blocks = self.get_ui_blocks()
        blocks.queue()
        logger.info("Mounting the gradio UI, at path=%s", path)
        gr.mount_gradio_app(app, blocks, path=path, favicon_path=AVATAR_BOT)


if __name__ == "__main__":
    ui = global_injector.get(PrivateGptUi)
    _blocks = ui.get_ui_blocks()
    _blocks.queue()
    _blocks.launch(debug=False, show_api=False)
