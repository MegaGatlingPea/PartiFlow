"""Optional terminal views; Rich loads only when a terminal needs rendering."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys


ACCENT = "cyan"
RESOURCE_USAGE = {
    "partiflow": "Peptide generation with the base flow-matching model",
    "partiflow-aux": "Auxiliary model for configured autoguidance",
    "partiflow-dock": "Docking-conditioned peptide generation",
    "partiflow-ss": "Secondary-structure-conditioned peptide generation",
    "autoencoder": "Structure encoding and decoding for flow matching",
    "af2": "AF2 structure prediction, validation and reward",
    "proteinmpnn": "ProteinMPNN sequence design",
    "solublempnn": "SolubleMPNN sequence design",
    "pepinter": "Original PepInter structures; preprocess before training",
}
_STATUS_STYLES = {
    "present": "green", "complete": "green", "loaded": "green",
    "downloaded": "green", "skipped": "dim",
    "missing": "yellow", "unavailable": "yellow",
    "failed": "bold red",
}


def _console(stream=None):
    stream = sys.stdout if stream is None else stream
    if (
        not getattr(stream, "isatty", lambda: False)()
        or os.environ.get("TERM") == "dumb"
        or "NO_COLOR" in os.environ
    ):
        return None
    try:
        from rich.console import Console
    except ImportError:
        return None
    return Console(file=stream, force_terminal=True, markup=False, highlight=False)


def _text(value, style=None):
    from rich.text import Text

    return Text(str(value), style=style)


def terminal_output():
    return _console() is not None


def _banner(console, title, root=None, description=None):
    from rich.box import ROUNDED
    from rich.panel import Panel

    label = _text("")
    label.append("PartiFlow", style="bold cyan")
    label.append("  /  " + title, style=ACCENT)
    if description:
        label.append("\n\n" + description, style="default")
    if root is not None:
        label.append("\nModel root: " + str(root), style="dim")
    console.print(Panel(label, box=ROUNDED, border_style=ACCENT, padding=(0, 1)))


def _section(console, title):
    console.print()
    console.print(_text(title, "bold cyan"))


def _help_invocation(action):
    names = ", ".join(action.option_strings)
    if action.nargs == 0:
        return names
    if action.metavar:
        metavar = str(action.metavar)
    elif action.choices is not None:
        metavar = "{" + ",".join(map(str, action.choices)) + "}"
    else:
        metavar = str(action.dest).upper()
    if action.nargs == "?":
        suffix = f"[{metavar}]"
    elif action.nargs == "*":
        suffix = f"[{metavar} ...]"
    elif action.nargs == "+":
        suffix = f"{metavar} [{metavar} ...]"
    elif isinstance(action.nargs, int):
        suffix = " ".join(metavar for _ in range(action.nargs))
    else:
        suffix = metavar
    return f"{names} {suffix}".strip() if names else suffix


def print_help(parser, file=None):
    stream = sys.stdout if file is None else file
    console = _console(stream)
    if console is None:
        print(parser.format_help(), file=stream, end="")
        return
    from rich.box import ROUNDED
    from rich.table import Table

    command = parser.prog.rsplit(" ", 1)[-1]
    _banner(console, "CLI" if command == "partiflow" else command + " help",
            description=parser.description)
    console.print(_text(parser.format_usage().strip(), "dim"))
    formatter = argparse.HelpFormatter(parser.prog, width=console.width)
    groups = sorted(
        parser._action_groups,
        key=lambda group: not any(isinstance(a, argparse._SubParsersAction)
                                  for a in group._group_actions),
    )
    for group in groups:
        actions = [a for a in group._group_actions if a.help != argparse.SUPPRESS]
        if not actions:
            continue
        rows = []
        subcommands = any(isinstance(a, argparse._SubParsersAction) for a in actions)
        for action in actions:
            if isinstance(action, argparse._SubParsersAction):
                rows.extend((choice.dest, choice.help or "") for choice in action._get_subactions())
            else:
                rows.append((_help_invocation(action), formatter._expand_help(action) if action.help else ""))
        _section(console, "Commands" if subcommands else group.title.capitalize())
        compact = console.width < 60
        table = Table(box=ROUNDED, show_edge=True, show_lines=True, expand=True,
                      border_style=ACCENT, padding=(0, 1),
                      show_header=not compact, header_style="bold cyan",
                      )
        if compact:
            table.add_column(overflow="fold")
            for label, description in rows:
                value = _text(label, ACCENT)
                value.append("\n" + description, style="default")
                table.add_row(value)
        else:
            width = min(max(18, max(len(label) for label, _ in rows)), console.width // 2)
            table.add_column("Command" if subcommands else "Argument",
                             style=ACCENT, width=width, overflow="fold", justify="center")
            table.add_column("Description", ratio=1, overflow="fold", justify="center")
            for label, description in rows:
                table.add_row(_text(label, ACCENT), _text(description))
        console.print(table)
    if command in {"download", "check"}:
        _section(console, "Resource targets")
        _fields(console, RESOURCE_USAGE.items(), first_heading="Target", second_heading="Usage")
    if parser.epilog:
        _section(console, "Try next")
        console.print(_text(parser.epilog, "dim"))


def show_error(message):
    console = _console()
    if console is None:
        print(str(message))
        return
    from rich.box import ROUNDED
    from rich.panel import Panel

    console.print(Panel(_text(message), title="Failed", title_align="left",
                        border_style="red", box=ROUNDED, padding=(0, 1)))


def _relative(value, root):
    path = Path(value)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _resource_detail(record, root):
    if record.get("error"):
        return str(record["error"])
    if "files" in record:
        files = record["files"]
        missing = [item for item in files if not item["present"]]
        count = len(files)
        detail = (f"{len(missing)} of {count} files missing" if missing
                  else f"{count} {'file' if count == 1 else 'files'} present")
        selected = missing or files
        if selected:
            detail += "\n" + _relative(selected[0]["path"], root)
        return detail
    detail = _relative(record.get("path", root), root)
    checks = record.get("checks", {})
    if checks:
        detail += "\n" + ", ".join(f"{key}: {'present' if value else 'missing'}" for key, value in checks.items())
    return detail


def _status_table(console, rows):
    from rich.box import ROUNDED
    from rich.table import Table

    compact = console.width < 60
    table = Table(box=ROUNDED, show_edge=True, show_lines=True, expand=True,
                  border_style=ACCENT, padding=(0, 1),
                  show_header=not compact, header_style="bold cyan",
                  )
    if compact:
        table.add_column(overflow="fold")
    else:
        table.add_column("Resource", ratio=2, style=ACCENT, overflow="fold")
        table.add_column("State", width=14, no_wrap=True)
        table.add_column("Detail", ratio=5, overflow="fold")
    for label, status, detail in rows:
        state = _text("[" + str(status) + "]", _STATUS_STYLES.get(status, ACCENT))
        if compact:
            value = _text(label, ACCENT)
            value.append(" ")
            value.append(state)
            value.append("\n" + str(detail), style="default")
            table.add_row(value)
        else:
            table.add_row(_text(label, ACCENT), state, _text(detail))
    console.print(table)


def _fields(console, rows, *, first_heading="Setting", second_heading="Value"):
    from rich.box import ROUNDED
    from rich.table import Table

    compact = console.width < 60
    table = Table(box=ROUNDED, border_style=ACCENT, show_lines=True,
                  expand=True, show_header=not compact, header_style="bold cyan")
    table.add_column(first_heading, style=ACCENT, overflow="fold",
                     ratio=2 if not compact else 1)
    if not compact:
        table.add_column(second_heading, overflow="fold", ratio=5)
    for key, value in rows:
        if compact:
            text = _text(key, ACCENT)
            text.append("\n" + str(value), style="default")
            table.add_row(text)
        else:
            table.add_row(_text(key, ACCENT), _text(value))
    console.print(table)


def show_context(args):
    """Describe an explicit command request without resolving its configuration."""
    console = _console()
    if console is None:
        return False
    mode = ("Plan" if getattr(args, "plan", False) else
            "Refilter saved results" if getattr(args, "refilter", False) else
            "Recompute saved metrics" if getattr(args, "recompute_metrics", False) else
            "Execute")
    _banner(console, args.command)
    rows = [("Operation", mode)]
    for field, label in (("stage", "Stage"), ("task_file", "Task YAML"),
                         ("paths", "Inputs"), ("config", "Configuration file"),
                         ("task", "Selected tasks"), ("out", "Output"),
                         ("run_name", "Run name"), ("model_dir", "Model root override"),
                         ("criteria", "Criteria"), ("source", "Sequence sources"),
                         ("override", "Overrides")):
        value = getattr(args, field, None)
        if value is not None and value != []:
            rows.append((label, "\n".join(map(str, value)) if isinstance(value, list) else str(value)))
    _fields(console, rows)
    return True


def show_completion(args, status):
    console = _console()
    if console is None:
        return
    from rich.box import ROUNDED
    from rich.panel import Panel

    title = "Plan complete" if getattr(args, "plan", False) else "Command complete"
    if status:
        title = "Command failed"
    console.print(Panel(_text(f"{args.command}: exit {status}"), title=title,
                        title_align="left", box=ROUNDED,
                        border_style="red" if status else ACCENT))


def show_report(title, report):
    """Render the same report fields as the redirected JSON, without elision."""
    console = _console()
    if console is None:
        return False

    def label(key):
        return str(key).replace("_", " ").capitalize()

    def display(value):
        if isinstance(value, str):
            return value
        return json.dumps(value, default=str)

    def render(value):
        if not isinstance(value, dict):
            _fields(console, [("Value", display(value))])
            return
        rows, sections = [], []
        for key, child in value.items():
            if isinstance(child, dict) and child:
                sections.append((label(key), child))
            elif isinstance(child, (list, tuple)) and child and all(isinstance(item, dict) for item in child):
                sections.extend((f"{label(key)} {index}", item) for index, item in enumerate(child, 1))
            else:
                rows.append((label(key), display(child)))
        if rows:
            _fields(console, rows, first_heading="Field")
        for heading, child in sections:
            _section(console, heading)
            render(child)

    _section(console, title)
    render(report)
    return True


def message(value, *, file=None, flush=True, title=None, failed=False):
    """Display a CLI event while preserving its existing plain-text fallback."""
    stream = sys.stdout if file is None else file
    console = _console(stream)
    if console is None:
        print(value, file=stream, flush=flush)
    else:
        from rich.box import ROUNDED
        from rich.panel import Panel

        console.print(Panel(_text(value), title=title, title_align="left",
                            border_style="red" if failed else ACCENT, box=ROUNDED))


def _asset_groups(console, records, root, *, listing=False):
    groups = (
        ("PartiFlow models", {"partiflow", "partiflow-aux", "partiflow-dock", "partiflow-ss", "autoencoder"}),
        ("Sequence design", {"proteinmpnn", "solublempnn"}),
        ("Structure prediction", {"af2"}),
        ("Training data", {"pepinter"}),
    )
    for title, targets in groups:
        selected = [item for item in records if item["target"] in targets]
        if not selected:
            continue
        _section(console, title)
        rows = []
        for item in selected:
            detail = _resource_detail(item, root)
            if item["target"] == "pepinter" and "training_data_ready" in item:
                detail += "\nRaw archive; preprocess before training"
            if listing:
                source = item.get("url") or "Use --url to choose a download source"
                detail += "\n" + source
            rows.append((item["target"], item["status"], detail))
        _status_table(console, rows)


def show_inventory(records, root):
    console = _console()
    if console is None:
        return False
    _banner(console, "Resource inventory", root)
    _asset_groups(console, records, root, listing=True)
    console.print()
    console.print(_text("Local presence checks paths only. Use check --load for strict CPU model loading.", "dim"))
    return True


def _execution_detail(record):
    if record.get("error"):
        return record["error"]
    return ", ".join(f"{key}: {json.dumps(value, default=str)}"
                     for key, value in record.items() if key not in {"target", "status"})


def show_check(report):
    console = _console()
    if console is None:
        return False
    root = Path(report["model_root"])
    _section(console, "Check results")
    console.print(_text("Mode: " + report["mode"], "dim"))
    _section(console, "Bundled resources")
    _status_table(console, [(Path(item["path"]).name,
                             "present" if item["present"] else "missing",
                             _relative(item["path"], root)) for item in report["resources"]])
    _asset_groups(console, report["models"], root)
    if "pepinter" in report:
        _asset_groups(console, [report["pepinter"]], root)
    if "task_check" in report:
        _section(console, "Task inputs / basic schema and paths")
        task_check = report["task_check"]
        rows = [(task["task"], task["status"], task["task_type"] + "\n" + task["pdb_path"])
                for task in task_check.get("tasks", [])]
        _status_table(console, rows or [("Task YAML", task_check["status"], task_check.get("error", ""))])
    if "loads" in report:
        _section(console, "Strict model loading")
        _status_table(console, [(item["target"], item["status"], _execution_detail(item))
                               for item in report["loads"]])
    if "smoke" in report:
        _section(console, "Smoke execution")
        item = report["smoke"]
        _status_table(console, [("Smoke test", item["status"], _execution_detail(item))])
    console.print()
    status = report["status"]
    message("Check " + status, title="Result", failed=status == "failed")
    if report["mode"] == "paths":
        console.print(_text("Path checks do not load or execute models.", "dim"))
    return True


def show_check_start(root, selected, mode):
    console = _console()
    if console is None:
        return
    _banner(console, "Check", root)
    _fields(console, [("Operation", mode), ("Selected models", ", ".join(selected))])


@contextmanager
def activity(label):
    console = _console()
    if console is None:
        yield
    else:
        with console.status(_text(label, ACCENT), spinner="line", spinner_style=ACCENT):
            yield


class DownloadDisplay:
    """Show completed target counts without guessing transfer byte counts."""

    def __init__(self, root, targets, *, url=None, force=False):
        self.root = root
        self.targets = list(targets)
        self.console = _console()
        self.progress = None
        self.task = None
        self.records = []
        self.url = url
        self.force = force

    @property
    def active(self):
        return self.console is not None

    def __enter__(self):
        if self.console is not None:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
            from rich.table import Column

            _banner(self.console, "Download", self.root)
            _fields(self.console, [("Selected targets", ", ".join(self.targets)),
                                   ("Source", "Explicit URL" if self.url else "Configured download URLs"),
                                   ("Existing files", "Replace" if self.force else "Skip complete targets")])
            columns = [SpinnerColumn(spinner_name="line", style=ACCENT),
                       TextColumn("{task.description}", markup=False,
                                  table_column=Column(ratio=3, min_width=18, overflow="fold"))]
            if self.console.width >= 60:
                columns.extend([BarColumn(bar_width=None, complete_style=ACCENT,
                                          table_column=Column(ratio=2, min_width=8)),
                                TextColumn("{task.completed:.0f}/{task.total:.0f} targets")])
            self.progress = Progress(*columns, console=self.console, expand=True)
            self.task = self.progress.add_task("Preparing resources", total=len(self.targets))
            self.progress.start()
        return self

    def begin(self, target):
        if self.progress is not None:
            self.progress.update(self.task, description="Acquiring " + str(target), refresh=True)

    def complete(self, record):
        self.records.append(record)
        if self.progress is not None:
            self.progress.update(self.task, advance=1, description="Processed " + record["target"], refresh=True)

    def __exit__(self, exc_type, exc, tb):
        if self.progress is not None:
            self.progress.stop()
        if self.console is not None and self.records:
            _asset_groups(self.console, self.records, self.root)
            failed = sum(item["status"] == "failed" for item in self.records)
            self.console.print()
            self.console.print(_text(f"Processed {len(self.records)} targets; {failed} failed or unavailable",
                                     "bold red" if failed else "green"))
        return False
