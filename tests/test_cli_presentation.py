import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from partiflow.cli import DISPATCH, main, parser


_GUARD = """
import importlib.abc
import socket
import sys
BLOCKED = {'torch', 'jax', 'numpy', 'pyrosetta', 'openmm'}
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('Unexpected import: ' + fullname)
sys.meta_path.insert(0, NoModels())
def no_network(*args, **kwargs):
    raise AssertionError('Unexpected network connection')
socket.create_connection = no_network
socket.socket.connect = no_network
"""
_NO_RICH = """
class NoRich(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'rich':
            raise ImportError('Rich intentionally unavailable')
sys.meta_path.insert(0, NoRich())
"""
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_ROOT = Path(__file__).resolve().parents[1]


def _script(args, prefix=""):
    return (_GUARD + prefix + "\nfrom partiflow.cli import main\n"
            + f"result = main({list(args)!r})\n"
            + "assert not (BLOCKED & set(sys.modules))\n"
            + "raise SystemExit(result)\n")


def _environment(width=80, extra=None):
    env = os.environ.copy()
    env.pop("NO_COLOR", None)
    env.update(TERM="xterm-256color", COLUMNS=str(width), LINES="40",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    env.update(extra or {})
    return env


def _pipe(args, prefix=""):
    return subprocess.run([sys.executable, "-B", "-c", _script(args, prefix)],
                          cwd=_ROOT, env=_environment(), capture_output=True,
                          text=True, timeout=20)


def _terminal(args, *, width=80, prefix="", env=None):
    import errno
    import fcntl
    import pty
    import select
    import struct
    import termios
    import time

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 40, width, 0, 0))
    process = None
    chunks = []
    try:
        process = subprocess.Popen(
            [sys.executable, "-B", "-c", _script(args, prefix)],
            cwd=_ROOT, env=_environment(width, env), stdin=subprocess.DEVNULL,
            stdout=slave, stderr=slave,
        )
        os.close(slave)
        slave = None
        deadline = time.monotonic() + 20
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Terminal CLI did not finish")
            ready, _, _ = select.select([master], [], [], min(remaining, 1))
            if not ready:
                if process.poll() is not None:
                    break
                continue
            try:
                chunk = os.read(master, 65536)
            except OSError as exc:
                if exc.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            chunks.append(chunk)
        result = process.wait(timeout=max(0.1, deadline - time.monotonic()))
        return result, b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if slave is not None:
            os.close(slave)
        os.close(master)


class CLIPresentationTests(unittest.TestCase):
    def test_all_public_options_have_descriptions(self):
        root = parser()
        commands = next(action for action in root._actions
                        if isinstance(action, argparse._SubParsersAction))
        self.assertEqual(set(commands.choices), set(DISPATCH))
        for command, command_parser in commands.choices.items():
            self.assertTrue(command_parser.description, command)
            for action in command_parser._actions:
                self.assertTrue(action.help, (command, action.dest))

    def test_help_does_not_dispatch_a_command(self):
        for args in (["--help"], *([name, "--help"] for name in DISPATCH)):
            with (patch("partiflow.cli.importlib.import_module") as dispatch,
                  contextlib.redirect_stdout(io.StringIO()),
                  self.assertRaises(SystemExit) as result):
                main(args)
            self.assertEqual(result.exception.code, 0)
            dispatch.assert_not_called()

    def test_piped_help_stays_plain_and_does_not_import_rich_or_models(self):
        for args in (["--help"], *([name, "--help"] for name in DISPATCH)):
            completed = _pipe(args, "BLOCKED.add('rich')\n")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage: partiflow", completed.stdout)
            self.assertNotIn("\x1b", completed.stdout + completed.stderr)

    def test_config_printing_does_not_load_models_or_use_network(self):
        for args in (
            ["train", "vae", "--config", "configs/training_ae.yaml", "--show-config"],
            ["design", "configs/tasks/example.yaml", "--show-config"],
        ):
            completed = _pipe(args)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(completed.stdout.strip())
            self.assertNotIn("\x1b", completed.stdout + completed.stderr)

    def test_redirected_inventory_and_check_preserve_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "missing-model-root"
            listed = _pipe(["download", "partiflow", "pepinter", "--list", "--model-dir", str(root)],
                           "BLOCKED.add('rich')\n")
            self.assertEqual(listed.returncode, 0, listed.stderr)
            records = [json.loads(line) for line in listed.stdout.splitlines()]
            self.assertEqual([record["target"] for record in records], ["partiflow", "pepinter"])
            self.assertEqual(set(records[0]), {"target", "status", "files", "url_configured"})
            self.assertEqual(records[0]["status"], "missing")
            checked = _pipe(["check", "--model", "partiflow", "--model-dir", str(root)],
                            "BLOCKED.add('rich')\n")
            self.assertEqual(checked.returncode, 1, checked.stderr)
            report = json.loads(checked.stdout)
            self.assertEqual(report["mode"], "paths")
            self.assertEqual(report["models"][0]["status"], "missing")
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("\x1b", listed.stdout + checked.stdout)
            self.assertFalse(root.exists())

    def test_redirected_local_import_keeps_json_and_exact_file_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            source = Path(temporary) / "partiflow.ckpt"
            source.write_bytes(b"small local checkpoint")
            completed = _pipe(["download", "partiflow", "--from-local", str(source),
                               "--model-dir", str(root)], "BLOCKED.add('rich')\n")
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "imported")
            self.assertNotIn("\x1b", completed.stdout + completed.stderr)
            self.assertEqual((root / "ckpts/partiflow.ckpt").read_bytes(), source.read_bytes())

    def test_redirected_inventory_keeps_rows_before_a_later_lookup_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            completed = _pipe(["download", "partiflow", "pepinter", "--list",
                               "--model-dir", str(root), "--config", str(root / "missing.yaml")],
                              "BLOCKED.add('rich')\n")
            self.assertEqual(completed.returncode, 1, completed.stderr)
            lines = completed.stdout.splitlines()
            self.assertEqual(json.loads(lines[0])["target"], "partiflow")
            self.assertIn("download failed:", lines[1])

    def test_bare_download_retains_the_existing_error(self):
        completed = _pipe(["download"])
        self.assertEqual(completed.returncode, 1)
        self.assertEqual(completed.stdout.strip(),
                         "download failed: Provide download targets, --all, or --list")


@unittest.skipUnless(os.name == "posix", "Requires a POSIX terminal")
class CLITerminalTests(unittest.TestCase):
    def assert_width(self, output, width):
        plain = _ANSI.sub("", output)
        self.assertLessEqual(max(map(len, plain.splitlines()), default=0), width, plain)
        return plain

    @unittest.skipUnless(importlib.util.find_spec("rich"), "Optional Rich is not installed")
    def test_every_help_renders_in_a_real_narrow_terminal(self):
        for args in (["--help"], *([name, "--help"] for name in DISPATCH)):
            status, output = _terminal(args, width=38)
            self.assertEqual(status, 0, output)
            plain = self.assert_width(output, 38)
            self.assertIn("PartiFlow", plain)
            self.assertRegex(output, r"\x1b\[[;\d]*36m")
            self.assertNotIn("Traceback", plain)

    @unittest.skipUnless(importlib.util.find_spec("rich"), "Optional Rich is not installed")
    def test_inventory_and_check_render_narrow_without_loading_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            for args, expected in (
                (["download", "--list", "--model-dir", temporary], 0),
                (["check", "--model", "partiflow", "--model-dir", temporary], 1),
            ):
                status, output = _terminal(args, width=38)
                self.assertEqual(status, expected, output)
                plain = self.assert_width(output, 38)
                self.assertIn("partiflow", plain)
                self.assertIn("[missing]", plain)
                self.assertNotIn("Traceback", plain)

    @unittest.skipUnless(importlib.util.find_spec("rich"), "Optional Rich is not installed")
    def test_terminal_import_progress_and_failures_report_actual_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            source = Path(temporary) / "partiflow.ckpt"
            source.write_bytes(b"small local checkpoint")
            status, output = _terminal(["download", "partiflow", "--from-local", str(source),
                                        "--model-dir", str(root)], width=80)
            self.assertEqual(status, 0, output)
            plain = self.assert_width(output, 80)
            self.assertIn("Processed partiflow", plain)
            self.assertIn("[imported]", plain)
            self.assertIn("0 failed or unavailable", plain)
            self.assertEqual((root / "ckpts/partiflow.ckpt").read_bytes(), source.read_bytes())
            prefix = ("from unittest.mock import Mock\nimport partiflow.checking\n"
                      "partiflow.checking._load_model = Mock(side_effect=RuntimeError('loader said no'))\n")
            status, output = _terminal(["check", "--model", "partiflow", "--load",
                                        "--model-dir", str(root)], prefix=prefix)
            self.assertEqual(status, 1, output)
            plain = _ANSI.sub("", output)
            self.assertIn("Strict model loading", plain)
            self.assertIn("loader said no", plain)
            self.assertIn("Check failed", plain)
            status, output = _terminal(["download", "partiflow-aux", "--model-dir", str(root)])
            self.assertEqual(status, 1, output)
            plain = _ANSI.sub("", output)
            self.assertIn("[unconfigured]", plain)
            self.assertIn("1 failed or unavailable", plain)

    def test_missing_rich_and_no_color_fall_back_without_ansi(self):
        for prefix, env in ((_NO_RICH, {}), ("BLOCKED.add('rich')\n", {"NO_COLOR": "1"})):
            status, output = _terminal(["design", "--help"], prefix=prefix, env=env)
            self.assertEqual(status, 0, output)
            self.assertIn("usage: partiflow design", output)
            self.assertIn("--show-config", output)
            self.assertNotIn("\x1b", output)


if __name__ == "__main__":
    unittest.main()
