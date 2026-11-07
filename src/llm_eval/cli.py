"""Command-line interface: `llm-eval run | compare | report | runs`."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated

import typer

from llm_eval.client import LLMClient
from llm_eval.compare import DIMENSIONS, compare, render_comparison_markdown, variants
from llm_eval.config import Settings, get_settings
from llm_eval.datasets import DatasetError, load_dataset, resolve_dataset_path
from llm_eval.models import RunRecord
from llm_eval.prompts import TemplateError, load_templates, select_templates
from llm_eval.providers.base import ProviderError, get_provider_class, parse_model_spec
from llm_eval.report import (
    build_report,
    markdown_table,
    render_html,
    render_markdown,
    render_overall_table,
)
from llm_eval.runner import Runner
from llm_eval.scorers import available_scorers, get_scorer
from llm_eval.storage import ResultStore

app = typer.Typer(
    name="llm-eval",
    help="Benchmark and prompt evaluation framework for comparing LLM outputs.",
    no_args_is_help=True,
    add_completion=False,
)


def _csv(values: list[str] | None) -> list[str]:
    """Accept both `-m a,b` and `-m a -m b`."""
    out: list[str] = []
    for v in values or []:
        out += [p.strip() for p in v.split(",") if p.strip()]
    return out


def _fail(msg: str) -> typer.Exit:
    typer.secho(f"Error: {msg}", fg=typer.colors.RED, err=True)
    return typer.Exit(code=2)


def _settings(results_dir: Path | None, **overrides) -> Settings:
    update = {k: v for k, v in overrides.items() if v is not None}
    if results_dir is not None:
        update["results_dir"] = results_dir
    return get_settings().model_copy(update=update)


ResultsDirOpt = Annotated[
    Path | None, typer.Option("--results-dir", help="Results directory (default: ./results).")
]
RunsOpt = Annotated[
    list[str] | None,
    typer.Option("--run", "-r", help="Run id(s), comma-separated or repeated. Default: latest."),
]


def _resolve_runs(store: ResultStore, runs: list[str] | None, need: int = 1) -> list[str]:
    ids = _csv(runs)
    if not ids:
        latest = [r.run_id for r in store.list_runs(limit=need)]
        if len(latest) < need:
            raise _fail(f"need {need} run(s) in {store.dir}, found {len(latest)}")
        ids = list(reversed(latest))
    for rid in ids:
        try:
            store.get_run(rid)
        except KeyError as e:
            raise _fail(str(e)) from None
    return ids


# --------------------------------------------------------------------------- run


@app.command()
def run(
    dataset: Annotated[str, typer.Option("--dataset", "-d", help="JSONL path or name.")],
    models: Annotated[
        list[str], typer.Option("--models", "-m", help="provider:model, e.g. openai:gpt-4o-mini")
    ],
    scorers: Annotated[
        list[str] | None,
        typer.Option("--scorers", "-s", help=f"Scorers (default: exact). {available_scorers()}"),
    ] = None,
    templates: Annotated[
        list[str] | None,
        typer.Option("--templates", "-t", help="Prompt template names (default: baseline)."),
    ] = None,
    templates_file: Annotated[
        Path | None, typer.Option(help="Templates YAML (default: prompts/templates.yaml).")
    ] = None,
    tags: Annotated[
        list[str] | None, typer.Option("--tags", help="Only examples with any of these tags.")
    ] = None,
    limit: Annotated[int | None, typer.Option(help="Max examples to evaluate.")] = None,
    concurrency: Annotated[int | None, typer.Option(help="Parallel requests.")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass response cache.")] = False,
    max_tokens: Annotated[int, typer.Option(help="Max output tokens per call.")] = 16000,
    temperature: Annotated[
        float | None, typer.Option(help="Sampling temperature (providers that support it).")
    ] = None,
    judge_model: Annotated[str | None, typer.Option(help="Model spec for llm_judge.")] = None,
    notes: Annotated[str | None, typer.Option(help="Free-text note stored with the run.")] = None,
    results_dir: ResultsDirOpt = None,
) -> None:
    """Run a dataset through models x prompt templates and score the outputs."""
    settings = _settings(results_dir, concurrency=concurrency, judge_model=judge_model)
    model_specs = _csv(models)
    scorer_names = _csv(scorers) or ["exact"]
    try:
        for spec in model_specs:
            get_provider_class(parse_model_spec(spec)[0])
        scorer_objs = [get_scorer(s) for s in scorer_names]
        tpl_objs = select_templates(load_templates(templates_file), _csv(templates))
        ds_path = resolve_dataset_path(dataset)
        examples = load_dataset(ds_path, tags=_csv(tags) or None)
    except (ProviderError, ValueError, DatasetError, TemplateError) as e:
        raise _fail(str(e)) from None
    if limit:
        examples = examples[:limit]
    if not examples:
        raise _fail("no examples selected")
    if "llm_judge" in scorer_names:
        parse_model_spec(settings.judge_model)

    total = len(examples) * len(model_specs) * len(tpl_objs)
    typer.echo(
        f"Evaluating {len(examples)} examples x {len(model_specs)} model(s) x "
        f"{len(tpl_objs)} template(s) = {total} calls; scorers: {', '.join(scorer_names)}"
    )
    done = 0

    def progress(record: RunRecord) -> None:
        nonlocal done
        done += 1
        mark = "E" if record.error else ("c" if record.cached else ".")
        sys.stderr.write(mark)
        if done % 50 == 0 or done == total:
            sys.stderr.write(f" {done}/{total}\n")
        sys.stderr.flush()

    async def _go():
        client = LLMClient(settings, use_cache=not no_cache)
        store = ResultStore(settings.results_dir)
        runner = Runner(
            client,
            store,
            scorer_objs,
            max_tokens=max_tokens,
            temperature=temperature,
            on_record=progress,
        )
        try:
            info = await runner.run(
                examples, model_specs, tpl_objs, dataset=ds_path.stem, notes=notes
            )
            return info, store.load_records(info.run_id), client
        finally:
            await client.aclose()

    info, records, client = asyncio.run(_go())
    data = build_report([info], records)
    typer.echo("")
    typer.echo(render_overall_table(data))

    typer.echo("\nUsage (including judge calls):")
    rows = [
        [
            spec,
            str(s.calls),
            str(s.cache_hits),
            str(s.retries),
            str(s.errors),
            f"{s.usage.input_tokens}/{s.usage.output_tokens}",
            f"${s.usage.cost_usd or 0:.4f}",
        ]
        for spec, s in sorted(client.stats.items())
    ]
    typer.echo(
        markdown_table(
            ["Model", "API calls", "Cache hits", "Retries", "Errors", "Tokens in/out", "Cost"], rows
        )
    )
    errors = [r for r in records if r.error]
    if errors:
        typer.secho(
            f"\n{len(errors)} generation error(s); first: {errors[0].error}", fg=typer.colors.YELLOW
        )
    typer.echo(f"\nRun id: {info.run_id}")
    typer.echo(
        f"Results: {settings.results_dir / (info.run_id + '.jsonl')} and "
        f"{settings.results_dir / 'results.db'}"
    )


# --------------------------------------------------------------------------- compare


@app.command("compare")
def compare_cmd(
    by: Annotated[str, typer.Option("--by", help=f"Dimension: {', '.join(DIMENSIONS)}")] = (
        "template"
    ),
    a: Annotated[str | None, typer.Option("--a", help="Variant A (baseline).")] = None,
    b: Annotated[str | None, typer.Option("--b", help="Variant B (candidate).")] = None,
    runs: RunsOpt = None,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Write Markdown.")] = None,
    results_dir: ResultsDirOpt = None,
) -> None:
    """A/B compare two templates, models or runs on paired examples."""
    if by not in DIMENSIONS:
        raise _fail(f"--by must be one of {DIMENSIONS}")
    settings = _settings(results_dir)
    store = ResultStore(settings.results_dir)
    run_ids = _resolve_runs(store, runs, need=2 if by == "run" else 1)
    records = store.load_records(run_ids)
    available = variants(records, by)
    if a is None or b is None:
        if len(available) != 2:
            raise _fail(f"specify --a and --b; available {by} values: {available}")
        a, b = available
    for v in (a, b):
        if v not in available:
            raise _fail(f"{by} {v!r} not found; available: {available}")
    md = render_comparison_markdown(compare(records, by, a, b), by)
    typer.echo(md)
    if output:
        output.write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {output}")


# --------------------------------------------------------------------------- report


@app.command()
def report(
    runs: RunsOpt = None,
    fmt: Annotated[str, typer.Option("--format", "-f", help="md, html or both")] = "both",
    output_dir: Annotated[
        Path | None, typer.Option(help="Where to write reports (default: <results>/reports).")
    ] = None,
    results_dir: ResultsDirOpt = None,
) -> None:
    """Write a Markdown and/or HTML report with per-model and per-tag scores."""
    if fmt not in ("md", "html", "both"):
        raise _fail("--format must be md, html or both")
    settings = _settings(results_dir)
    store = ResultStore(settings.results_dir)
    run_ids = _resolve_runs(store, runs)
    data = build_report([store.get_run(r) for r in run_ids], store.load_records(run_ids))
    out_dir = output_dir or settings.results_dir / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = run_ids[0] if len(run_ids) == 1 else f"{run_ids[0]}_plus{len(run_ids) - 1}"
    md = render_markdown(data)
    typer.echo(md)
    if fmt in ("md", "both"):
        (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
        typer.echo(f"Wrote {out_dir / (stem + '.md')}")
    if fmt in ("html", "both"):
        (out_dir / f"{stem}.html").write_text(render_html(data), encoding="utf-8")
        typer.echo(f"Wrote {out_dir / (stem + '.html')}")


# --------------------------------------------------------------------------- runs


@app.command("runs")
def list_runs(
    limit: Annotated[int, typer.Option(help="How many recent runs to show.")] = 20,
    results_dir: ResultsDirOpt = None,
) -> None:
    """List recent runs."""
    store = ResultStore(_settings(results_dir).results_dir)
    rows = [
        [
            r.run_id,
            r.created_at.strftime("%Y-%m-%d %H:%M"),
            r.status,
            str(r.dataset),
            ",".join(r.models),
            ",".join(r.templates),
            str(r.n_records),
            "n/a" if r.total_cost_usd is None else f"${r.total_cost_usd:.4f}",
        ]
        for r in store.list_runs(limit=limit)
    ]
    typer.echo(
        markdown_table(
            ["Run", "Created (UTC)", "Status", "Dataset", "Models", "Templates", "N", "Cost"], rows
        )
    )


if __name__ == "__main__":
    app()
