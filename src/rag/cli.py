"""CLI: `python -m rag <command>`."""
from __future__ import annotations

import json
from pathlib import Path

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .ask import ask as ask_question
from .eval import evaluate_strategy
from .ingest import ingest_directory
from .retrieve import RetrievalConfig


load_dotenv()
console = Console()


@click.group()
def cli():
    """RAG Pipeline CLI."""


@cli.command(name="ingest")
@click.option("--root", default="corpus/fastapi", help="Directory of documents to ingest")
@click.option("--strategy", default="recursive",
              type=click.Choice(["fixed", "recursive", "semantic"]))
@click.option("--reset", is_flag=True, help="Delete existing chunks of this strategy first")
@click.option("--no-dedup", is_flag=True, help="Skip near-duplicate detection")
def ingest_cmd(root: str, strategy: str, reset: bool, no_dedup: bool):
    stats = ingest_directory(Path(root), strategy=strategy, reset=reset, skip_duplicates=not no_dedup)
    console.print_json(json.dumps(stats))


@cli.command()
@click.argument("question")
@click.option("--strategy", default="recursive",
              type=click.Choice(["fixed", "recursive", "semantic"]))
@click.option("--mode", default="hybrid", type=click.Choice(["hybrid", "dense", "sparse"]))
@click.option("--no-rerank", is_flag=True)
@click.option("--no-verify", is_flag=True, help="Skip citation verification (faster for demos)")
def ask(question: str, strategy: str, mode: str, no_rerank: bool, no_verify: bool):
    """Ask a question against the indexed corpus."""
    config = RetrievalConfig(strategy=strategy, mode=mode, use_rerank=not no_rerank)
    bundle = ask_question(question, config=config, verify=not no_verify)

    console.print(f"\n[bold cyan]Question:[/] {question}")
    console.print(f"[bold cyan]Mode:[/] {mode} · strategy: {strategy} · rerank: {not no_rerank}")
    if bundle.refused:
        console.print(f"\n[yellow]REFUSED[/]: {bundle.answer}")
    else:
        console.print(f"\n[bold]Answer:[/]\n{bundle.answer}")

    console.print(f"\n[bold]Confidence[/]: composite={bundle.confidence.composite:.2f}, "
                  f"retrieval={bundle.confidence.retrieval:.2f}, "
                  f"citation_coverage={bundle.confidence.citation_coverage:.2f}")

    t = Table(title="Retrieved chunks", show_header=True, header_style="bold")
    t.add_column("#"); t.add_column("Source"); t.add_column("Score"); t.add_column("Preview")
    for i, c in enumerate(bundle.chunks, start=1):
        score = c.rerank_score or c.rrf_score or c.dense_score
        preview = c.text[:100].replace("\n", " ") + ("…" if len(c.text) > 100 else "")
        t.add_row(str(i), c.source, f"{score:.3f}", preview)
    console.print(t)


@cli.command(name="eval")
@click.option("--strategy", default="recursive")
@click.option("--no-verify", is_flag=True, help="Skip citation verification to save tokens (loses citation-accuracy metric)")
def eval_cmd(strategy: str, no_verify: bool):
    """Run the eval suite against one strategy."""
    report = evaluate_strategy(strategy, verify=not no_verify)
    t = Table(title=f"Eval — strategy={strategy}", show_header=True, header_style="bold")
    t.add_column("Metric"); t.add_column("Score", justify="right")
    t.add_row("Cases", str(report.n_cases))
    t.add_row("Mean correctness (1-5)", f"{report.mean_correctness:.2f}")
    t.add_row("Mean faithfulness", f"{report.mean_faithfulness:.2f}")
    t.add_row("Mean retrieval relevance", f"{report.mean_retrieval_relevance:.2f}")
    t.add_row("Mean citation accuracy", f"{report.mean_citation_accuracy:.2f}")
    console.print(t)


@cli.command(name="compare-strategies")
def compare_strategies():
    """Run the eval suite against all 3 chunking strategies and print a comparison."""
    reports = []
    for s in ("fixed", "recursive", "semantic"):
        console.print(f"\n[bold]Evaluating strategy={s}…[/]")
        reports.append(evaluate_strategy(s))

    t = Table(title="Chunking strategy comparison", show_header=True, header_style="bold")
    t.add_column("Strategy")
    for col in ["Correctness", "Faithfulness", "Retr. Relevance", "Citation Accuracy"]:
        t.add_column(col, justify="right")
    for r in reports:
        t.add_row(
            r.strategy,
            f"{r.mean_correctness:.2f}",
            f"{r.mean_faithfulness:.2f}",
            f"{r.mean_retrieval_relevance:.2f}",
            f"{r.mean_citation_accuracy:.2f}",
        )
    console.print(t)


def main():
    cli()


if __name__ == "__main__":
    main()
