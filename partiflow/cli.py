"""The public command surface; parsing and help use only the standard library."""

from __future__ import annotations
import argparse
import importlib
from pathlib import Path


def boolean(value):
    value = str(value).lower()
    if value not in {"true", "false"}:
        raise argparse.ArgumentTypeError("expected true or false")
    return value == "true"


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def parser():
    root = argparse.ArgumentParser(
        prog="partiflow",
        description="PartiFlow training, design, and structure workflows.",
    )
    root.add_argument("--version", action="version", version="partiflow 0.1.0")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Train a VAE or flow-matching model")
    train.add_argument("stage", choices=("vae", "fm"))
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    train.add_argument("--show-config", dest="print_config", action="store_true")
    train.add_argument("--model-dir", type=Path)
    design = commands.add_parser("design", help="Run named tasks from a task YAML")
    design.add_argument("task_file", type=Path, metavar="TASK.yaml")
    design.add_argument("--config", type=Path)
    design.add_argument("--task", action="append")
    design.add_argument("--ckpt", type=Path)
    design.add_argument("--out", type=Path)
    design.add_argument("--run-name")
    design.add_argument("--seed", type=int)
    design.add_argument("--batch-size", type=positive)
    design.add_argument("--num-workers", type=nonnegative)
    design.add_argument("--compile", type=boolean, metavar="{true,false}")
    design.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    design.add_argument("--show-config", dest="print_config", action="store_true")
    design.add_argument("--model-dir", type=Path)
    validate = commands.add_parser(
        "validate", help="Run or inspect paired AF2-IG validation"
    )
    validate.add_argument("paths", nargs="+", type=Path)
    validate.add_argument("--config", type=Path)
    validate.add_argument("--task", action="append")
    validate.add_argument("--source", action="append", metavar="SOURCE:COUNT")
    validate.add_argument("--criteria", action="append")
    validate.add_argument("--presets", type=Path)
    validate.add_argument("--limit", type=positive)
    validate.add_argument("--rerun", action="store_true")
    modes = validate.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--refilter", action="store_true")
    modes.add_argument("--recompute-metrics", action="store_true")
    validate.add_argument("--model-dir", type=Path)
    postprocess = commands.add_parser(
        "postprocess", help="Relax structures and optionally apply quality criteria"
    )
    postprocess.add_argument("paths", nargs="+", type=Path)
    postprocess.add_argument(
        "--filter", type=boolean, default=None, metavar="{true,false}"
    )
    postprocess.add_argument("--config", type=Path)
    postprocess.add_argument("--task", action="append")
    postprocess.add_argument("--criteria", action="append")
    postprocess.add_argument("--presets", type=Path)
    postprocess.add_argument("--rerun", action="store_true")
    postprocess.add_argument("--plan", action="store_true")
    postprocess.add_argument("--platform", choices=("CPU", "CUDA", "OpenCL"))
    postprocess.add_argument("--model-dir", type=Path)
    cluster = commands.add_parser(
        "cluster", help="Cluster binder structures with US-align"
    )
    cluster.add_argument("paths", nargs="+", type=Path)
    cluster.add_argument("--task", action="append")
    cluster.add_argument("--criteria", action="append")
    cluster.add_argument("--tm-threshold", type=float, default=0.5)
    cluster.add_argument("--workers", type=positive, default=16)
    cluster.add_argument("--usalign", type=Path)
    cluster.add_argument("--plan", action="store_true")
    check = commands.add_parser(
        "check", help="Read resource status, or explicitly load/smoke-test models"
    )
    check.add_argument("task_file", nargs="?", type=Path, metavar="TASK.yaml")
    selection = check.add_mutually_exclusive_group()
    selection.add_argument("--model", action="append")
    selection.add_argument("--all", action="store_true")
    operation = check.add_mutually_exclusive_group()
    operation.add_argument("--load", action="store_true")
    operation.add_argument("--smoke", action="store_true")
    check.add_argument("--model-dir", type=Path)
    check.add_argument("--config", type=Path)
    check.add_argument("--data-dir", type=Path)
    download = commands.add_parser(
        "download", help="Download configured assets or import existing local files"
    )
    download.add_argument("targets", nargs="*")
    download.add_argument("--list", action="store_true")
    download.add_argument("--all", action="store_true")
    download.add_argument("--force", action="store_true")
    source = download.add_mutually_exclusive_group()
    source.add_argument("--from-local", type=Path)
    source.add_argument("--url")
    download.add_argument("--model-dir", type=Path)
    download.add_argument("--data-dir", type=Path)
    download.add_argument("--config", type=Path)
    return root


DISPATCH = {
    "train": "partiflow.training",
    "design": "partiflow.design",
    "validate": "partiflow.workflows.validate",
    "postprocess": "partiflow.postprocess",
    "cluster": "partiflow.cluster",
    "check": "partiflow.checking",
    "download": "partiflow.download",
}


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        result = importlib.import_module(DISPATCH[args.command]).run(args)
        return int(result or 0)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        print(f"{args.command} failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
