import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from partiflow.cli import main, parser


class PublicCLITests(unittest.TestCase):
    def parse(self, *args):
        return parser().parse_args(args)

    def test_exact_seven_commands(self):
        commands = next(
            action for action in parser()._actions if action.dest == "command"
        )
        self.assertEqual(
            set(commands.choices),
            {
                "train",
                "design",
                "validate",
                "postprocess",
                "cluster",
                "check",
                "download",
            },
        )

    def test_train_adapter_namespace(self):
        args = self.parse(
            "train",
            "vae",
            "--config",
            "x.yaml",
            "--override",
            "seed=1",
            "--show-config",
        )
        self.assertEqual(args.stage, "vae")
        self.assertEqual(args.config, Path("x.yaml"))
        self.assertEqual(args.override, ["seed=1"])
        self.assertTrue(args.print_config)

    def test_design_explicit_boolean_and_repeatable_tasks(self):
        args = self.parse(
            "design",
            "tasks.yaml",
            "--task",
            "a",
            "--task",
            "b",
            "--compile",
            "false",
            "--num-workers",
            "0",
        )
        self.assertFalse(args.compile)
        self.assertEqual(args.task, ["a", "b"])
        self.assertEqual(args.task_file, Path("tasks.yaml"))
        self.assertEqual(args.num_workers, 0)

    def test_postprocess_filter_default_false_and_explicit_true(self):
        self.assertFalse(self.parse("postprocess", "run").filter)
        self.assertTrue(self.parse("postprocess", "run", "--filter", "true").filter)
        self.assertFalse(self.parse("postprocess", "run", "--filter", "false").filter)

    def test_invalid_boolean_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("postprocess", "run", "--filter", "yes")

    def test_validation_modes_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("validate", "run", "--plan", "--refilter")

    def test_cluster_defaults(self):
        args = self.parse("cluster", "run")
        self.assertEqual(args.tm_threshold, 0.5)
        self.assertEqual(args.workers, 16)

    def test_selected_dispatch_is_lazy(self):
        module = Mock()
        module.run.return_value = 3
        with patch(
            "partiflow.cli.importlib.import_module", return_value=module
        ) as importer:
            self.assertEqual(main(["train", "fm", "--config", "x.yaml"]), 3)
        importer.assert_called_once_with("partiflow.training")
        self.assertEqual(module.run.call_args.args[0].stage, "fm")

    def test_help_does_not_import_models(self):
        script = "from partiflow.cli import parser; import sys; parser().format_help(); assert not ({'torch','jax','numpy','pyrosetta','openmm'} & set(sys.modules))"
        completed = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_every_command_help_works(self):
        for command in (
            "train",
            "design",
            "validate",
            "postprocess",
            "cluster",
            "check",
            "download",
        ):
            with (
                contextlib.redirect_stdout(io.StringIO()),
                self.assertRaises(SystemExit) as result,
            ):
                parser().parse_args([command, "--help"])
            self.assertEqual(result.exception.code, 0)


if __name__ == "__main__":
    unittest.main()
