"""Resolve packaged configuration separately from local model assets."""

from __future__ import annotations
import os
from pathlib import Path


def checkout_root() -> Path | None:
    root = Path(__file__).resolve().parent.parent
    return (
        root
        if (root / "pyproject.toml").is_file() and (root / "configs").is_dir()
        else None
    )


def resource_root() -> Path:
    return checkout_root() or Path(__file__).resolve().parent / "_resources"


def config_dir() -> Path:
    return resource_root() / "configs"


def model_root(value=None) -> Path:
    selected = value or os.environ.get("PARTIFLOW_MODEL_DIR")
    return (
        Path(selected).expanduser().resolve()
        if selected
        else (checkout_root() or Path.cwd()).resolve()
    )


def input_path(value, invocation_dir=None) -> Path:
    path = Path(value).expanduser()
    return (
        path if path.is_absolute() else Path(invocation_dir or Path.cwd()) / path
    ).resolve()
