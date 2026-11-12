"""Optional terminal views; Rich loads only when a terminal needs rendering."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys


ACCENT = "cyan"
_STATUS_STYLES = {
    "present": "green", "complete": "green", "loaded": "green",
    "downloaded": "green", "imported": "green", "skipped": "dim",
    "missing": "yellow", "unavailable": "yellow", "unconfigured": "yellow",
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


def _banner(console, title, root=None):
    from rich.box import ROUNDED
    from rich.panel import Panel

    label = _text("PartiFlow", "bold cyan")
    label.append("  /  " + title, style=ACCENT)
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
    from rich.table import Table

    command = parser.prog.rsplit(" ", 1)[-1]
    _banner(console, "CLI" if command == "partiflow" else command + " help")
    if parser.description:
        console.print(_text(parser.description))
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
        table = Table(box=None, show_edge=False, pad_edge=False, expand=True,
                      padding=(0, 1), collapse_padding=True,
                      show_header=not compact, header_style="bold cyan",
                      leading=1 if compact else 0)
        if compact:
            table.add_column(overflow="fold")
            for label, description in rows:
                value = _text(label, ACCENT)
                value.append("\n" + description, style="default")
                table.add_row(value)
        else:
            width = min(max(len(label) for label, _ in rows), console.width // 2)
            table.add_column("Command" if subcommands else "Argument",
                             style=ACCENT, width=width, overflow="fold")
            table.add_column("Description", ratio=1, overflow="fold")
            for label, description in rows:
                table.add_row(_text(label, ACCENT), _text(description))
        console.print(table)
    if parser.epilog:
        console.print()
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
    from rich.table import Table

    compact = console.width < 60
    table = Table(box=None, show_edge=False, pad_edge=False, expand=True,
                  padding=(0, 1), collapse_padding=True,
                  show_header=not compact, header_style="bold cyan",
                  leading=1 if compact else 0)
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
            if listing:
                source = "Configured download URL" if item.get("url_configured") else "Supply --from-local or --url"
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
    _banner(console, "Check", root)
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
    console.print(_text("Check " + status, _STATUS_STYLES.get(status, ACCENT)))
    if report["mode"] == "paths":
        console.print(_text("Path checks do not load or execute models.", "dim"))
    return True


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

    def __init__(self, root, targets):
        self.root = root
        self.targets = list(targets)
        self.console = _console()
        self.progress = None
        self.task = None
        self.records = []

    @property
    def active(self):
        return self.console is not None

    def __enter__(self):
        if self.console is not None:
            from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
            from rich.table import Column

            _banner(self.console, "Download / import", self.root)
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
            failed = sum(item["status"] in {"failed", "unconfigured"} for item in self.records)
            self.console.print()
            self.console.print(_text(f"Processed {len(self.records)} targets; {failed} failed or unavailable",
                                     "bold red" if failed else "green"))
        return False
