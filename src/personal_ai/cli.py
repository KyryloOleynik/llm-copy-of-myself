from pathlib import Path

import typer

from personal_ai.config import load_config

app = typer.Typer(no_args_is_help=True, help="Personal Telegram AI pipeline.")


@app.command("prepare-data")
def prepare_data(config: Path = Path("config.yaml")) -> None:
    """Build tokenizer-budgeted train/validation/test data from complete sessions."""
    from personal_ai.data import prepare_dataset
    from personal_ai.modeling import load_tokenizer

    app_config = load_config(config)
    tokenizer = load_tokenizer(app_config.model.base_model)
    manifest = prepare_dataset(app_config, tokenizer)
    typer.echo(f"Prepared dataset: {manifest['counts']}")


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
