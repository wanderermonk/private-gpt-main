"""Predictive-analysis recipe (quantitative + qualitative, combined).

Pipeline for a question like "forecast next quarter's revenue":

    retrieve  ->  extract numeric series (LLM)  ->  forecast (numpy)
              ->  chart (matplotlib)            ->  cited written outlook (LLM)

The forecast is computed deterministically by ``forecast.py`` (no cloud, no
heavy ML deps). The LLM is used only for the two things it's good at: pulling
a messy series out of document text, and narrating what the numbers imply -
always with [n] citations into the retrieved evidence. If the documents don't
contain a usable numeric series, it degrades gracefully to a purely
qualitative, cited outlook.

Everything runs locally; nothing here makes a network call.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import re

from injector import inject, singleton
from pydantic import BaseModel

from private_gpt.components.llm.llm_component import LLMComponent
from private_gpt.open_ai.extensions.context_filter import ContextFilter
from private_gpt.server.chunks.chunks_service import Chunk, ChunksService
from private_gpt.server.recipes.agent_rag.agent_rag_service import AgentStep
from private_gpt.server.recipes.predict.forecast import choose_and_forecast
from private_gpt.settings.settings import Settings

logger = logging.getLogger(__name__)


class DataPoint(BaseModel):
    period: str
    value: float


class PredictResult(BaseModel):
    metric: str
    history: list[DataPoint]
    forecast: list[DataPoint]
    method: str
    fit_error: float | None = None
    narrative: str
    sources: list[Chunk]
    chart_png_base64: str | None = None
    steps: list[AgentStep]


_EXTRACT_PROMPT = (
    "From the evidence below, extract the historical numeric time series for: "
    '"{metric}".\n'
    "Return ONLY a JSON array in chronological order, each item "
    '{{"period": "<label>", "value": <number>}}. Use plain numbers (no '
    "currency symbols, no commas, no units). If no such series is present, "
    "return [].\n\n"
    "Evidence:\n{evidence}\n\n"
    "JSON:"
)

_NARRATIVE_PROMPT = (
    "You are writing a short predictive outlook for the metric \"{metric}\".\n"
    "Historical values: {history}\n"
    "Computed projection ({method}): {forecast}\n\n"
    "Using the numbered evidence for supporting facts, write 3-6 sentences: "
    "state the trend, the projected direction and magnitude, the main drivers "
    "or risks, and explicitly note this is an uncertain projection. Cite "
    "evidence inline as [n].\n\n"
    "Evidence:\n{evidence}\n\n"
    "Outlook:"
)

_QUALITATIVE_PROMPT = (
    "No reliable numeric series for \"{metric}\" was found in the documents. "
    "Using ONLY the numbered evidence, give a short qualitative outlook (3-6 "
    "sentences): the apparent direction, likely drivers and risks, and a clear "
    "note that this is a judgement, not a quantitative forecast. Cite evidence "
    "inline as [n]. If the evidence is insufficient, say so.\n\n"
    "Question: {metric}\n\n"
    "Evidence:\n{evidence}\n\n"
    "Outlook:"
)


@singleton
class PredictService:
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

    # ----------------------------- helpers -------------------------------- #
    def _complete(self, prompt: str) -> str:
        response = self.llm_component.llm.complete(prompt)
        text = getattr(response, "text", None)
        if text is None:
            text = str(response)
        return text.strip()

    @staticmethod
    def _to_float(raw: object) -> float | None:
        """Parse a possibly-messy numeric value ('1,234', '$5.2M', '7%')."""
        if isinstance(raw, (int, float)):
            return float(raw)
        if not isinstance(raw, str):
            return None
        s = raw.strip().lower().replace(",", "").replace("$", "").replace("%", "")
        mult = 1.0
        if s.endswith("k"):
            mult, s = 1e3, s[:-1]
        elif s.endswith("m"):
            mult, s = 1e6, s[:-1]
        elif s.endswith("b"):
            mult, s = 1e9, s[:-1]
        try:
            return float(s) * mult
        except ValueError:
            return None

    def _parse_series(self, raw: str) -> list[DataPoint]:
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return []
        points: list[DataPoint] = []
        for item in data if isinstance(data, list) else []:
            if not isinstance(item, dict):
                continue
            value = self._to_float(item.get("value"))
            if value is None:
                continue
            period = str(item.get("period", f"t{len(points) + 1}"))
            points.append(DataPoint(period=period, value=value))
        return points

    @staticmethod
    def _future_periods(history: list[DataPoint], horizon: int) -> list[str]:
        """Best-effort labels for forecast periods."""
        labels_int = []
        for p in history:
            m = re.search(r"(19|20)\d{2}", p.period)  # a year
            labels_int.append(int(m.group(0)) if m else None)
        if history and all(v is not None for v in labels_int):
            last = labels_int[-1]
            return [str(last + i) for i in range(1, horizon + 1)]
        return [f"Forecast {i}" for i in range(1, horizon + 1)]

    def _format_evidence(self, chunks: list[Chunk], max_chars: int = 6000) -> str:
        lines, used = [], 0
        for i, chunk in enumerate(chunks, start=1):
            meta = chunk.document.doc_metadata or {}
            label = meta.get("file_name") or chunk.document.doc_id
            text = chunk.text.strip().replace("\n", " ")
            budget = max(0, max_chars - used)
            if budget <= 0:
                break
            snippet = text[:budget]
            used += len(snippet)
            lines.append(f"[{i}] (source: {label}) {snippet}")
        return "\n".join(lines) if lines else "(no evidence found)"

    @staticmethod
    def _make_chart(
        metric: str, history: list[DataPoint], forecast: list[DataPoint]
    ) -> str | None:
        """Render a history+forecast line chart to a base64 PNG, or None."""
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless, no display
            import matplotlib.pyplot as plt
        except Exception as exc:  # matplotlib optional
            logger.info("Chart skipped (matplotlib unavailable): %s", exc)
            return None
        try:
            hx = [p.period for p in history]
            hy = [p.value for p in history]
            fx = [p.period for p in forecast]
            fy = [p.value for p in forecast]

            fig, ax = plt.subplots(figsize=(7, 3.5))
            ax.plot(hx, hy, marker="o", label="history")
            if forecast:
                # connect last history point to first forecast point
                ax.plot(
                    [hx[-1], *fx],
                    [hy[-1], *fy],
                    marker="o",
                    linestyle="--",
                    color="#C7762F",
                    label="forecast",
                )
            ax.set_title(metric)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout()
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.warning("Chart rendering failed: %s", exc)
            return None

    # ------------------------------ core ---------------------------------- #
    def _predict_gen(
        self,
        metric: str,
        horizon: int,
        retrieve_limit: int,
        context_filter: ContextFilter | None,
    ):
        steps: list[AgentStep] = []

        # 1. RETRIEVE evidence for the metric.
        try:
            chunks = self.chunks_service.retrieve_relevant(
                text=metric, context_filter=context_filter, limit=retrieve_limit
            )
        except Exception as exc:
            logger.warning("Predict retrieval failed: %s", exc)
            chunks = []
        step = AgentStep(
            kind="retrieve",
            detail=f"gathered evidence for: {metric}",
            queries=[metric],
            num_new_chunks=len(chunks),
        )
        steps.append(step)
        yield step
        evidence = self._format_evidence(chunks)

        # 2. EXTRACT numeric series.
        extract_raw = self._complete(
            _EXTRACT_PROMPT.format(metric=metric, evidence=evidence)
        )
        history = self._parse_series(extract_raw)
        step = AgentStep(
            kind="reflect",
            detail=f"extracted {len(history)} historical data point(s)",
        )
        steps.append(step)
        yield step

        # 3. FORECAST (or degrade to qualitative).
        values = [p.value for p in history]
        forecast_points: list[DataPoint] = []
        method = "none"
        fit_error: float | None = None
        chart_b64: str | None = None
        try:
            result = choose_and_forecast(values, horizon=horizon)
            method = result.method
            fit_error = result.fit_error
            future_labels = self._future_periods(history, len(result.forecast))
            forecast_points = [
                DataPoint(period=lbl, value=val)
                for lbl, val in zip(future_labels, result.forecast)
            ]
            step = AgentStep(
                kind="reflect",
                detail=f"forecast via {method} (fit error {fit_error})",
            )
            steps.append(step)
            yield step
            chart_b64 = self._make_chart(metric, history, forecast_points)
        except ValueError:
            step = AgentStep(
                kind="reflect",
                detail="no usable numeric series; qualitative outlook only",
            )
            steps.append(step)
            yield step

        # 4. NARRATE (cited).
        if forecast_points:
            narrative = self._complete(
                _NARRATIVE_PROMPT.format(
                    metric=metric,
                    history=[(p.period, p.value) for p in history],
                    method=method,
                    forecast=[(p.period, p.value) for p in forecast_points],
                    evidence=evidence,
                )
            )
        else:
            narrative = self._complete(
                _QUALITATIVE_PROMPT.format(metric=metric, evidence=evidence)
            )
        step = AgentStep(kind="answer", detail="final outlook written")
        steps.append(step)
        yield step

        return PredictResult(
            metric=metric,
            history=history,
            forecast=forecast_points,
            method=method,
            fit_error=fit_error,
            narrative=narrative,
            sources=chunks,
            chart_png_base64=chart_b64,
            steps=steps,
        )

    # ----------------------------- public --------------------------------- #
    def predict(
        self,
        metric: str,
        horizon: int = 3,
        retrieve_limit: int = 8,
        context_filter: ContextFilter | None = None,
    ) -> PredictResult:
        gen = self._predict_gen(metric, horizon, retrieve_limit, context_filter)
        result: PredictResult | None = None
        while True:
            try:
                next(gen)
            except StopIteration as stop:
                result = stop.value
                break
        assert result is not None
        return result

    def predict_stream(
        self,
        metric: str,
        horizon: int = 3,
        retrieve_limit: int = 8,
        context_filter: ContextFilter | None = None,
    ):
        gen = self._predict_gen(metric, horizon, retrieve_limit, context_filter)
        while True:
            try:
                yield next(gen)
            except StopIteration as stop:
                yield stop.value
                return
