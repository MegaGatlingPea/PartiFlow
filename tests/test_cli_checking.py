import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from partiflow.checking import run


class CheckModeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "ckpts").mkdir()
        (self.root / "ckpts" / "partiflow.ckpt").write_bytes(b"local-resource")
        self.args = SimpleNamespace(
            model=["partiflow"],
            all=False,
            model_dir=self.root,
            task_file=None,
            config=None,
            data_dir=None,
            load=False,
            smoke=False,
        )

    def tearDown(self):
        self.temp.cleanup()

    def test_default_never_calls_model_loader(self):
        with (
            patch("partiflow.checking._load_model") as load,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            status = run(self.args)
        load.assert_not_called()
        self.assertEqual(json.loads(output.getvalue())["mode"], "paths")
        self.assertEqual(status, 0)
        self.assertEqual(
            (self.root / "ckpts" / "partiflow.ckpt").read_bytes(), b"local-resource"
        )

    def test_explicit_load_failure_is_not_success(self):
        self.args.load = True
        with (
            patch(
                "partiflow.checking._load_model",
                side_effect=RuntimeError("strict load failed"),
            ),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(run(self.args), 1)
        report = json.loads(output.getvalue())
        self.assertEqual(report["loads"][0]["status"], "failed")
        self.assertIn("strict load failed", report["loads"][0]["error"])

    def test_missing_resources_prevent_smoke(self):
        self.args.smoke = True
        (self.root / "ckpts" / "partiflow.ckpt").unlink()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run(self.args), 1)
        self.assertEqual(
            json.loads(output.getvalue())["smoke"]["status"], "unavailable"
        )


if __name__ == "__main__":
    unittest.main()
