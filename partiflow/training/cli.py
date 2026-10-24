"""Public training command adapter. Heavy dependencies load only for training."""

from __future__ import annotations

from pathlib import Path
import re

from .config import load_config


def resolve_resume(config, checkpoint_dir: Path) -> str | None:
    request = config.get("resume", "auto")
    if request is False or request in {None, "none"}:
        return None
    if request == "auto":
        candidates = []
        for path in checkpoint_dir.glob("last*.ckpt"):
            match = re.fullmatch(r"last(?:-v(\d+))?\.ckpt", path.name)
            if match:
                candidates.append((int(match.group(1) or 0), path))
        return str(max(candidates, key=lambda item: item[0])[1]) if candidates else None
    path = Path(str(request))
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return str(path)


def run(args) -> int:
    from omegaconf import OmegaConf

    config = load_config(
        args.config,
        getattr(args, "override", []),
        stage=args.stage,
        model_directory=getattr(args, "model_dir", None),
    )
    if getattr(args, "print_config", False) or getattr(args, "show_config", False):
        print(OmegaConf.to_yaml(config, resolve=True))
        return 0
    from .runner import train

    train(config, stage=args.stage)
    return 0
