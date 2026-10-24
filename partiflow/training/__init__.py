"""VAE and flow-matching training entry points."""


def run(args) -> int:
    from .cli import run as run_training

    return run_training(args)
