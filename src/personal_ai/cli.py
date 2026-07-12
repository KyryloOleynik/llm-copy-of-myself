from pathlib import Path

import typer

from personal_ai.config import load_config

app = typer.Typer(no_args_is_help=True, help="Personal Telegram AI pipeline.")


@app.command("prepare-data")
def prepare_data(config: Path = Path("config.yaml")) -> None:
    """Split the existing dataset and create its manifest (transitional v1 step)."""
    from personal_ai.training import prepare_existing_dataset

    manifest = prepare_existing_dataset(load_config(config))
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
    """Build the sanitized SQLite/NumPy retrieval index (future stage)."""
    _not_implemented("build-index")


@app.command()
def evaluate() -> None:
    """Run style, appropriateness, and privacy evaluation (future stage)."""
    _not_implemented("evaluate")


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
