"""The public command surface; model dependencies load only on dispatch."""

from __future__ import annotations
import argparse
import importlib
from pathlib import Path
import sys


class CLIParser(argparse.ArgumentParser):
    def print_help(self, file=None):
        from .ui import print_help

        print_help(self, file)


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


def _model_directory(group):
    group.add_argument(
        "--model-dir", type=Path,
        help="Model root: this flag, then PARTIFLOW_MODEL_DIR, then checkout/current directory",
    )


def parser():
    root = CLIParser(
        prog="partiflow",
        description="PartiFlow training, design, and structure workflows.",
        epilog="Use partiflow COMMAND --help for command options and examples.",
    )
    root.add_argument("--version", action="version", version="partiflow 0.1.0",
                      help="Show the installed CLI version and exit")
    commands = root.add_subparsers(dest="command", required=True, title="commands")

    train = commands.add_parser(
        "train", help="Train a VAE or flow-matching model",
        description="Train a VAE or flow-matching model from a training YAML.",
        epilog="Example: partiflow train vae --config configs/training_ae.yaml --show-config",
    )
    train.add_argument("stage", choices=("vae", "fm"), help="Training stage: VAE or flow matching")
    config = train.add_argument_group("configuration")
    config.add_argument("--config", type=Path, required=True, help="Training configuration YAML (required)")
    config.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Hydra configuration override (repeatable)")
    config.add_argument("--show-config", dest="print_config", action="store_true",
                        help="Print the resolved configuration and exit without training")
    _model_directory(train.add_argument_group("resources"))

    design = commands.add_parser(
        "design", help="Run named tasks from a task YAML",
        description="Generate peptide structures from named tasks in a task YAML.",
        epilog="Example: partiflow design TASK.yaml --show-config",
    )
    design.add_argument("task_file", type=Path, metavar="TASK.yaml", help="YAML mapping of named design tasks")
    inputs = design.add_argument_group("task selection")
    inputs.add_argument("--config", type=Path, help="Design configuration overlay YAML")
    inputs.add_argument("--task", action="append", help="Select a named task (repeatable; default: all tasks)")
    generation = design.add_argument_group("generation")
    generation.add_argument("--ckpt", type=Path, help="Flow-matching checkpoint path")
    generation.add_argument("--seed", type=int, help="Random seed; otherwise use configuration")
    generation.add_argument("--batch-size", type=positive, help="Generation batch size (positive integer)")
    generation.add_argument("--num-workers", type=nonnegative, help="Data-loader worker count (zero or greater)")
    generation.add_argument("--compile", type=boolean, metavar="{true,false}",
                            help="Enable or disable configured model compilation")
    output = design.add_argument_group("output")
    output.add_argument("--out", type=Path, help="Design output directory")
    output.add_argument("--run-name", help="Run name within the configured output root")
    config = design.add_argument_group("configuration")
    config.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                        help="Hydra configuration override (repeatable)")
    config.add_argument("--show-config", dest="print_config", action="store_true",
                        help="Print the resolved configuration and exit without generation")
    _model_directory(design.add_argument_group("resources"))

    validate = commands.add_parser(
        "validate", help="Run or inspect paired AF2-IG validation",
        description="Run paired AF2-IG validation, or inspect saved validation results.",
        epilog="Example: partiflow validate RUN_DIR --plan",
    )
    validate.add_argument("paths", nargs="+", type=Path, help="Run, task, or structure paths to validate")
    inputs = validate.add_argument_group("inputs")
    inputs.add_argument("--config", type=Path, help="Validation configuration overlay YAML")
    inputs.add_argument("--task", action="append", help="Select a task name (repeatable)")
    inputs.add_argument("--source", action="append", metavar="SOURCE:COUNT",
                        help="Sequence source and sample count (repeatable)")
    filtering = validate.add_argument_group("quality criteria")
    filtering.add_argument("--criteria", action="append", help="Named criteria preset (repeatable)")
    filtering.add_argument("--presets", type=Path, help="YAML defining named quality criteria")
    execution = validate.add_argument_group("execution")
    execution.add_argument("--limit", type=positive, help="Maximum number of input structures")
    execution.add_argument("--rerun", action="store_true", help="Rerun validation instead of reusing completed results")
    modes = execution.add_mutually_exclusive_group()
    modes.add_argument("--plan", action="store_true", help="Inspect the validation plan without running models")
    modes.add_argument("--refilter", action="store_true", help="Reapply criteria to saved validation results")
    modes.add_argument("--recompute-metrics", action="store_true", help="Recompute metrics from saved predictions")
    _model_directory(validate.add_argument_group("resources"))

    postprocess = commands.add_parser(
        "postprocess", help="Relax structures and optionally apply quality criteria",
        description="Relax generated structures and optionally apply quality criteria.",
        epilog="Example: partiflow postprocess RUN_DIR --filter false --plan",
    )
    postprocess.add_argument("paths", nargs="+", type=Path, help="Run, task, or structure paths to postprocess")
    inputs = postprocess.add_argument_group("inputs")
    inputs.add_argument("--config", type=Path, help="Postprocessing configuration overlay YAML")
    inputs.add_argument("--task", action="append", help="Select a task name (repeatable)")
    filtering = postprocess.add_argument_group("quality criteria")
    filtering.add_argument("--filter", type=boolean, default=None, metavar="{true,false}",
                           help="Apply quality criteria after relaxation; otherwise use configuration")
    filtering.add_argument("--criteria", action="append", help="Named criteria preset (repeatable)")
    filtering.add_argument("--presets", type=Path, help="YAML defining named quality criteria")
    execution = postprocess.add_argument_group("execution")
    execution.add_argument("--rerun", action="store_true", help="Rerun postprocessing instead of reusing completed results")
    execution.add_argument("--plan", action="store_true", help="Inspect the postprocessing plan without relaxing structures")
    execution.add_argument("--platform", choices=("CPU", "CUDA", "OpenCL"), help="OpenMM execution platform")
    _model_directory(postprocess.add_argument_group("resources"))

    cluster = commands.add_parser(
        "cluster", help="Cluster binder structures with US-align",
        description="Cluster binder structures by structural similarity with US-align.",
        epilog="Example: partiflow cluster RUN_DIR --plan",
    )
    cluster.add_argument("paths", nargs="+", type=Path, help="Run, task, or structure paths to cluster")
    selection = cluster.add_argument_group("selection")
    selection.add_argument("--task", action="append", help="Select a task name (repeatable)")
    selection.add_argument("--criteria", action="append", help="Select structures passing named criteria (repeatable)")
    execution = cluster.add_argument_group("clustering")
    execution.add_argument("--tm-threshold", type=float, default=0.5,
                            help="TM-score similarity threshold (default: %(default)s)")
    execution.add_argument("--workers", type=positive, default=16,
                            help="Parallel US-align worker count (default: %(default)s)")
    execution.add_argument("--usalign", type=Path, help="US-align executable path")
    execution.add_argument("--plan", action="store_true", help="Inspect clustering inputs without running US-align")

    check = commands.add_parser(
        "check", help="Read resource status, or explicitly load/smoke-test models",
        description="Inspect local resources. Model loading or execution requires --load or --smoke.",
        epilog="Example: partiflow check --all",
    )
    check.add_argument("task_file", nargs="?", type=Path, metavar="TASK.yaml",
                       help="Optional task YAML to check for valid inputs")
    selection = check.add_argument_group("model selection").add_mutually_exclusive_group()
    selection.add_argument("--model", action="append", help="Model target to check (repeatable; default: partiflow, autoencoder)")
    selection.add_argument("--all", action="store_true", help="Check every configured model target")
    operation = check.add_argument_group("operation").add_mutually_exclusive_group()
    operation.add_argument("--load", action="store_true", help="Explicitly load selected models on CPU using strict loaders")
    operation.add_argument("--smoke", action="store_true", help="Explicitly run the configured model smoke test")
    resources = check.add_argument_group("resource paths")
    _model_directory(resources)
    resources.add_argument("--config", type=Path, help="Training YAML used to locate PepInter data")
    resources.add_argument("--data-dir", type=Path, help="PepInter data root; also checks its split and processed files")

    download = commands.add_parser(
        "download", help="Download model weights and training data",
        description="Download model weights and training data from their published sources.",
        epilog="Example: partiflow download --list",
    )
    download.add_argument("targets", nargs="*", help="Resource target names; omit for the terminal selector, or use --list")
    selection = download.add_argument_group("selection")
    selection.add_argument("--list", action="store_true", help="List resource paths and configured download availability")
    selection.add_argument("--all", action="store_true", help="Select all configured model and data targets")
    selection.add_argument("--force", action="store_true", help="Replace existing files when acquiring selected targets")
    source = download.add_argument_group("source")
    source.add_argument("--url", help="Custom HTTP or HTTPS URL for exactly one target")
    resources = download.add_argument_group("resource paths")
    _model_directory(resources)
    resources.add_argument("--data-dir", type=Path, help="Destination root for PepInter data")
    resources.add_argument("--config", type=Path, help="Training YAML used to locate the PepInter destination")
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
        from .ui import show_completion, show_context

        framed = (args.command not in {"check", "download"}
                  and not getattr(args, "print_config", False))
        if framed:
            show_context(args)
        result = importlib.import_module(DISPATCH[args.command]).run(args)
        status = int(result or 0)
        if framed:
            show_completion(args, status)
        return status
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        from .ui import show_error

        show_error(f"{args.command} failed: {exc}")
        return 1
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
