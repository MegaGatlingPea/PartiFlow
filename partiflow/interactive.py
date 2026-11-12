"""Keyboard-controlled resource selection with no model or network access."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping, Sequence


def can_prompt() -> bool:
    """Require both terminal streams and the optional interactive renderer."""
    return (
        os.environ.get("TERM") != "dumb"
        and bool(getattr(sys.stdin, "isatty", lambda: False)())
        and bool(getattr(sys.stdout, "isatty", lambda: False)())
        and importlib.util.find_spec("prompt_toolkit") is not None
    )


def _literal(value: object) -> str:
    return "".join(char if char.isprintable() else " " for char in str(value))


def select_downloads(
    options: Sequence[Mapping[str, object]], root, *,
    default_values: Sequence[str] = (),
) -> list[str] | None:
    """Return selected names in catalog order; local presence is a separate tick."""
    if not options:
        return None

    from prompt_toolkit.application import Application, get_app
    from prompt_toolkit.filters import has_focus
    from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings
    from prompt_toolkit.key_binding.defaults import load_key_bindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.output import ColorDepth
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Button, CheckboxList, Dialog, Label

    records = {str(item["target"]): item for item in options}
    names = list(records)
    enabled = {name for name, item in records.items() if item.get("selectable")}
    error = [None, ""]
    action = "Download"

    def detail_text(name):
        return "\n".join(_literal(line) for line in str(records[name].get("detail", "")).splitlines())

    class ResourceChoices(CheckboxList):
        def _handle_enter(self):
            name = self.values[self._selected_index][0]
            if name not in enabled:
                lines = detail_text(name).splitlines()
                error[:] = [name, lines[-1] if lines else "Unavailable: choose a configured source."]
                get_app().invalidate()
                return
            error[:] = [None, ""]
            super()._handle_enter()

    try:
        "✓↑↓".encode(sys.stdout.encoding or "utf-8")
        unicode = True
    except (UnicodeError, LookupError):
        unicode = False

    values = []
    for name, item in records.items():
        present = item.get("status") == "present"
        marker = ("✓" if unicode else "+") if present else " "
        fragments = [
            ("class:ready" if present else "class:muted", marker + " "),
            ("class:model" if name in enabled else "class:disabled", _literal(name)),
        ]
        values.append((name, fragments))
    choices = ResourceChoices(values, default_values=[name for name in default_values if name in enabled])
    choices.control.key_bindings.remove("enter")
    choices.window.height = Dimension(min=1, preferred=min(10, len(values)), max=len(values))

    def accept():
        selected = set(choices.current_values) & enabled
        if selected:
            get_app().exit(result=[name for name in names if name in selected])
        else:
            error[:] = [names[choices._selected_index], "Select at least one resource."]

    def cancel():
        get_app().exit(result=None)

    def detail():
        return [("class:muted", detail_text(names[choices._selected_index]))]

    def status():
        name = names[choices._selected_index]
        value = error[1] if error[0] == name else "Selected: " + str(len(choices.current_values))
        return [("class:pending", value)]

    def root_text():
        # Keep a long absolute path from crowding the choices in small terminals.
        width = max(12, get_app().output.get_size().columns - 6)
        value = "Model root: " + _literal(root)
        if len(value) > width:
            value = "Model root: …" + _literal(root)[-(width - 13):]
        return [("class:muted", value)]

    def guide():
        if get_app().output.get_size().columns < 60:
            text = ("↑↓ Move · Space Select" if unicode else "Up/Down Move | Space Select")
            text += "\nEnter " + action + " · Esc Back"
        else:
            text = ("↑↓ Move · Space Select · " if unicode else "Up/Down Move | Space Select | ")
            text += "Enter " + action + " · Esc Back"
        return [("class:muted", text)]

    bindings = KeyBindings()

    @bindings.add("enter", filter=has_focus(choices.control))
    def _accept(event):
        accept()

    @bindings.add("escape", eager=True)
    @bindings.add("c-d", eager=True)
    def _cancel(event):
        cancel()

    @bindings.add("c-c", eager=True)
    def _interrupt(event):
        event.app.exit(exception=KeyboardInterrupt())

    def fitted_label(text):
        label = Label(text=text)

        def height():
            # Labels may otherwise shrink to a single row while a preferred-height
            # checkbox list keeps its rows. Reserve every wrapped text row first;
            # the list can use its existing scrollbar for the space that remains.
            columns = min(100, get_app().output.get_size().columns) - 4
            rows = label.window.content.preferred_height(max(1, columns), 10000, True, None)
            return Dimension.exact(max(1, rows or 1))

        label.window.height = height
        return label

    body = HSplit([
        fitted_label(root_text),
        Window(height=1),
        choices,
        Window(height=1),
        fitted_label(detail),
        fitted_label(status),
        fitted_label(guide),
    ])
    title = "PartiFlow / Resource downloads"
    dialog = Dialog(
        title=title,
        body=body,
        buttons=[Button(text=action, handler=accept), Button(text="Back", handler=cancel)],
        width=Dimension(preferred=100, max=110),
        with_background=True,
    )
    accent = "#67e8f9"
    styles = {
        "dialog": "bg:#102126 #e5f4f5",
        "dialog frame.label": "bold " + accent,
        "dialog frame.border": accent,
        "dialog.body": "bg:#102126 #e5f4f5",
        "dialog shadow": "bg:#071114",
        "button": "bg:#24414a #e5f4f5",
        "button.focused": "bg:" + accent + " #102126 bold",
        "model": accent + " bold",
        "ready": "#86efac",
        "pending": accent,
        "muted": "#b0c3c9",
        "disabled": "#81949b",
        "checkbox": "#e5f4f5",
        "checkbox-selected": "bg:#24414a",
        "checkbox-checked": "#86efac bold",
    }
    app = Application(
        layout=Layout(dialog, focused_element=choices),
        full_screen=True,
        mouse_support=True,
        key_bindings=merge_key_bindings([load_key_bindings(), bindings]),
        style=Style.from_dict(styles),
        color_depth=ColorDepth.MONOCHROME if "NO_COLOR" in os.environ else None,
    )
    try:
        return app.run()
    except EOFError:
        return None
