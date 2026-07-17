from pathlib import Path

import typer

from personal_ai.config import load_config

app = typer.Typer(no_args_is_help=True, help="Personal Telegram AI pipeline.")


@app.command("authorize-calendar")
def authorize_calendar(
    credentials: Path = Path("private_data/google_calendar_credentials.json"),
    token: Path = Path("private_data/google_calendar_token.json"),
) -> None:
    """Authorize read-only access to the owner's Google Calendar."""
    from personal_ai.google_calendar import authorize_google_calendar

    saved = authorize_google_calendar(credentials, token)
    typer.echo(f"Google Calendar authorized: {saved}")


@app.command("prepare-data")
def prepare_data(config: Path = Path("config.yaml")) -> None:
    """Build tokenizer-budgeted train/validation/test data from complete sessions."""
    from personal_ai.data import prepare_dataset
    from personal_ai.modeling import load_tokenizer

    app_config = load_config(config)
    tokenizer = load_tokenizer(app_config.model.base_model)
    manifest = prepare_dataset(app_config, tokenizer)
    typer.echo(f"Prepared dataset: {manifest['counts']}")


@app.command("build-rag")
def build_rag(config: Path = Path("config.yaml")) -> None:
    """Index private knowledge files and cleaned Telegram sessions for local RAG."""
    from personal_ai.retrieval import build_retrieval_index

    stats = build_retrieval_index(load_config(config))
    typer.echo(f"Built RAG index: {stats}")


@app.command("search-rag")
def search_rag(
    query: str,
    config: Path = Path("config.yaml"),
) -> None:
    """Search the private RAG index from the terminal."""
    import json

    from personal_ai.retrieval import search_retrieval

    app_config = load_config(config)
    results = search_retrieval(
        app_config.retrieval.database,
        query,
        app_config.retrieval.max_results,
    )
    typer.echo(json.dumps(results, ensure_ascii=False, indent=2))


@app.command()
def train(
    config: Path = Path("config.yaml"),
    smoke: bool = typer.Option(False, help="Run one worst-case training step without evaluation."),
    resume: str | None = typer.Option(None, help="Checkpoint path, or 'last'."),
    fresh: bool = typer.Option(
        False, help="Delete the previous run's adapters/checkpoints and start from the base model."
    ),
) -> None:
    """Train the QLoRA adapter, automatically resuming the latest checkpoint."""
    from personal_ai.training import train as run_training

    run_training(load_config(config), smoke=smoke, resume=resume, fresh=fresh)


@app.command()
def evaluate(config: Path = Path("config.yaml")) -> None:
    """Compare the base model and available adapters and write evaluation JSON."""
    from personal_ai.evaluation import evaluate_checkpoints

    report = evaluate_checkpoints(load_config(config))
    typer.echo(f"Evaluation written to {report}")


if __name__ == "__main__":
    app()
