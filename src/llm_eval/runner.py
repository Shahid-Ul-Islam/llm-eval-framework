"""Evaluation runner: dataset x models x templates -> generate -> score -> store."""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections.abc import Callable, Sequence

from llm_eval.client import LLMClient
from llm_eval.models import Example, PromptTemplate, RunRecord, ScoreResult
from llm_eval.prompts import render
from llm_eval.scorers.base import BaseScorer, ScoringContext
from llm_eval.storage import ResultStore, RunInfo, new_run_id

log = logging.getLogger(__name__)


class Runner:
    def __init__(
        self,
        client: LLMClient,
        store: ResultStore,
        scorers: Sequence[BaseScorer],
        *,
        concurrency: int | None = None,
        max_tokens: int = 16000,
        temperature: float | None = None,
        on_record: Callable[[RunRecord], None] | None = None,
    ):
        self.client = client
        self.store = store
        self.scorers = list(scorers)
        self.concurrency = concurrency or client.settings.concurrency
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.on_record = on_record
        self.ctx = ScoringContext(client=client, settings=client.settings)

    async def run(
        self,
        examples: Sequence[Example],
        models: Sequence[str],
        templates: Sequence[PromptTemplate],
        *,
        dataset: str | None = None,
        run_id: str | None = None,
        notes: str | None = None,
    ) -> RunInfo:
        info = RunInfo(
            run_id=run_id or new_run_id(),
            dataset=dataset,
            models=list(models),
            templates=[t.name for t in templates],
            scorers=[s.name for s in self.scorers],
            notes=notes,
        )
        self.store.start_run(info)
        sem = asyncio.Semaphore(self.concurrency)

        async def worker(ex: Example, spec: str, tpl: PromptTemplate) -> None:
            async with sem:
                record = await self.evaluate_one(info.run_id, ex, spec, tpl)
            self.store.add_record(record)
            if self.on_record:
                self.on_record(record)

        jobs = itertools.product(examples, models, templates)
        try:
            await asyncio.gather(*(worker(ex, m, t) for ex, m, t in jobs))
        except BaseException:
            self.store.finish_run(info.run_id, status="aborted")
            raise
        return self.store.finish_run(info.run_id)

    async def evaluate_one(
        self, run_id: str, ex: Example, spec: str, tpl: PromptTemplate
    ) -> RunRecord:
        prompt = render(tpl, ex)
        record = RunRecord(
            run_id=run_id,
            example_id=ex.id,
            model=spec,
            template=tpl.name,
            prompt=prompt,
            expected=ex.expected,
            tags=ex.tags,
        )
        try:
            resp = await self.client.generate(
                spec,
                prompt,
                system=tpl.system,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            log.warning("generation failed for %s/%s/%s: %s", ex.id, spec, tpl.name, e)
            record.error = f"{type(e).__name__}: {e}"
            return record

        record.output = resp.text
        record.usage = resp.usage
        record.latency_s = resp.latency_s
        record.cached = resp.cached
        record.scores = await self.score(ex, resp.text)
        return record

    async def score(self, ex: Example, output: str) -> list[ScoreResult]:
        results = []
        for scorer in self.scorers:
            if not scorer.applicable(ex):
                continue
            try:
                results.append(await scorer.score(ex, output, self.ctx))
            except Exception as e:
                log.warning("scorer %s crashed on %s: %s", scorer.name, ex.id, e)
                results.append(
                    ScoreResult(scorer=scorer.name, score=0.0, error=f"{type(e).__name__}: {e}")
                )
        return results
