from pathlib import Path

import typer

from personal_ai.config import load_config

app = typer.Typer(no_args_is_help=True, help="Personal Telegram AI pipeline.")


@app.command("prepare-data")
def prepare_data(config: Path = Path("config.yaml")) -> None:
    """Build tokenizer-budgeted train/validation/test data from complete sessions."""
    from transformers import AutoTokenizer

    from personal_ai.data import prepare_dataset

    app_config = load_config(config)
    tokenizer = AutoTokenizer.from_pretrained(app_config.model.base_model, use_fast=True)
    manifest = prepare_dataset(app_config, tokenizer)
    typer.echo(f"Prepared dataset: {manifest['counts']}")


@app.command()
def train(
    config: Path = Path("config.yaml"),
    smoke: bool = typer.Option(False, help="Use at most 20 train/eval examples."),
    resume: str | None = typer.Option(None, help="Checkpoint path, or 'last'."),
    fresh: bool = typer.Option(False, help="Ignore existing checkpoints and start from the base model."),
) -> None:
    """Train the QLoRA adapter, automatically resuming the latest checkpoint."""
    from personal_ai.training import train as run_training

    run_training(load_config(config), smoke=smoke, resume=resume, fresh=fresh)


def _not_implemented(stage: str) -> None:
    typer.echo(f"{stage} is scaffolded but not implemented yet.", err=True)
    raise typer.Exit(code=2)


@app.command("build-index")
def build_index() -> None:
    """Build the private SQLite/NumPy retrieval index (deferred stage)."""
    _not_implemented("build-index")


@app.command()
def evaluate(config: Path = Path("config.yaml")) -> None:
    """Compare the base model and available adapters and write evaluation JSON."""
    from personal_ai.evaluation import evaluate_checkpoints

    report = evaluate_checkpoints(load_config(config))
    typer.echo(f"Evaluation written to {report}")


@app.command("export-model")
def export_model() -> None:
    """Merge the adapter and invoke the GGUF export workflow (future stage)."""
    _not_implemented("export-model")


@app.command("run-bot")
def run_bot() -> None:
    """Run the approved-user Telegram bot (future stage)."""
    _not_implemented("run-bot")


if __name__ == "__main__":
    app()
