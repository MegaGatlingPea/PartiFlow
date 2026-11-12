"""Resource paths use the same lightweight composition and bases as training."""

from contextlib import contextmanager, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from partiflow.cli import main
from partiflow.download import data_root
from partiflow.paths import config_dir
from partiflow.training import config as training_config


@contextmanager
def working_directory(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


class ResourceTrainingPathTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.project = self.base / "project"
        self.project.mkdir()
        self.invocation = self.base / "invocation"
        self.invocation.mkdir()
        self.models = self.base / "models"

    def tearDown(self):
        self.temporary.cleanup()

    def config(self, text, directory=None):
        path = (directory or self.project) / "training.yaml"
        path.write_text(text)
        return path

    def training_root(self, path):
        value = training_config.load_config(path, model_directory=self.models)
        module = value.get("dataset", {}).get("datamodule") or value.get("datamodule")
        return Path(module["processed_dir"]).parent.resolve()

    def test_bundled_vae_and_fm_configs_match_training_from_another_directory(self):
        with working_directory(self.invocation):
            for name in ("training_ae.yaml", "training.yaml"):
                path = config_dir() / name
                with self.subTest(config=name):
                    expected = self.training_root(path)
                    self.assertEqual(data_root(self.models, config=path), expected)
                    self.assertEqual(expected, training_config.default_work_root() / "train_data/pepinter")

    def test_user_defaults_compose_bundled_dataset_and_keep_the_user_base(self):
        path = self.config("defaults:\n  - dataset: pepinter\n  - _self_\n")
        with working_directory(self.invocation):
            expected = self.project / "train_data/pepinter"
            self.assertEqual(self.training_root(path), expected)
            self.assertEqual(data_root(self.models, config=path), expected)

    def test_inline_relative_processed_dir_uses_config_directory(self):
        path = self.config("datamodule:\n  processed_dir: data/processed\n  split_csv: data/indexes/split.csv\n")
        with working_directory(self.invocation):
            self.assertEqual(data_root(self.models, config=path), self.project / "data")
            self.assertEqual(data_root(self.models, config=path), self.training_root(path))

    def test_inline_absolute_paths_match_training(self):
        destination = self.base / "absolute-data"
        path = self.config(f"datamodule:\n  processed_dir: {destination}/processed\n  split_csv: {destination}/indexes/split.csv\n")
        self.assertEqual(data_root(self.models, config=path), destination)
        self.assertEqual(data_root(self.models, config=path), self.training_root(path))

    def test_top_level_data_dir_stays_supported_and_uses_the_user_base(self):
        path = self.config("data_dir: local-data\ndataset:\n  datamodule:\n    processed_dir: ${data_dir}/processed\n")
        with working_directory(self.invocation):
            self.assertEqual(data_root(self.models, config=path), self.project / "local-data")
            self.assertEqual(data_root(self.models, config=path), self.training_root(path))
        path.write_text(f"data_dir: {self.base}/absolute-top-level\n")
        self.assertEqual(data_root(self.models, config=path), self.base / "absolute-top-level")

    def test_explicit_data_dir_precedes_even_an_unreadable_config(self):
        with working_directory(self.invocation):
            self.assertEqual(data_root(self.models, data_dir="chosen", config=self.base / "missing.yaml"),
                             self.invocation / "chosen")

    def test_no_data_setting_or_config_keeps_model_root_fallback(self):
        path = self.config("{}\n")
        expected = self.models / "train_data/pepinter"
        self.assertEqual(data_root(self.models), expected)
        self.assertEqual(data_root(self.models, config=path), expected)

    def test_installed_resource_configs_use_invocation_base_and_user_configs_keep_their_base(self):
        installed = self.base / "installed/_resources/configs"
        shutil.copytree(config_dir(), installed, ignore=shutil.ignore_patterns("*.pt"))
        user = self.config("defaults:\n  - dataset: pepinter\n  - _self_\n")
        with (patch.object(training_config, "config_dir", return_value=installed),
              patch.object(training_config, "checkout_root", return_value=None),
              working_directory(self.invocation)):
            for name in ("training_ae.yaml", "training.yaml"):
                path = installed / name
                with self.subTest(config=name):
                    self.assertEqual(data_root(self.models, config=path), self.invocation / "train_data/pepinter")
                    self.assertEqual(data_root(self.models, config=path), self.training_root(path))
            self.assertEqual(data_root(self.models, config=user), self.project / "train_data/pepinter")
            self.assertEqual(data_root(self.models, config=user), self.training_root(user))

    def test_bundled_relative_top_level_data_dir_uses_the_training_work_root(self):
        bundled = self.base / "bundled-configs"
        bundled.mkdir()
        path = self.config("data_dir: bundled-data\n", directory=bundled)
        with (patch.object(training_config, "config_dir", return_value=bundled),
              patch.object(training_config, "default_work_root", return_value=self.base / "checkout"),
              working_directory(self.invocation)):
            self.assertEqual(data_root(self.models, config=path), self.base / "checkout/bundled-data")

    def test_custom_graph_or_split_layout_requires_an_explicit_standard_destination(self):
        for text in ("datamodule:\n  processed_dir: data/graphs\n",
                     "datamodule:\n  processed_dir: data/processed\n  split_csv: elsewhere/split.csv\n"):
            path = self.config(text)
            with self.subTest(config=text), self.assertRaisesRegex(ValueError, "indexes/ and processed/"):
                data_root(self.models, config=path)
            self.assertEqual(data_root(self.models, data_dir=self.base / "standard", config=path), self.base / "standard")

    def test_public_download_then_check_use_the_training_dataset_destination(self):
        source = self.base / "source"
        (source / "indexes").mkdir(parents=True)
        (source / "processed").mkdir()
        (source / "indexes/split.csv").write_text("id,split\nsample,train\n")
        (source / "processed/sample.pt").write_bytes(b"prepared graph")
        path = self.config("defaults:\n  - dataset: pepinter\n  - _self_\n")
        expected = self.training_root(path)
        with working_directory(self.invocation), redirect_stdout(io.StringIO()) as output:
            status = main(["download", "pepinter", "--from-local", str(source), "--config", str(path),
                           "--model-dir", str(self.models)])
        self.assertEqual(status, 0, output.getvalue())
        self.assertEqual(json.loads(output.getvalue())["path"], str(expected))
        self.assertEqual((expected / "processed/sample.pt").read_bytes(), b"prepared graph")
        (self.models / "ckpts").mkdir(parents=True)
        (self.models / "ckpts/partiflow.ckpt").write_bytes(b"present")
        with working_directory(self.invocation), redirect_stdout(io.StringIO()) as output:
            status = main(["check", "--model", "partiflow", "--config", str(path), "--model-dir", str(self.models)])
        self.assertEqual(status, 0, output.getvalue())
        report = json.loads(output.getvalue())
        self.assertEqual(report["pepinter"]["path"], str(expected))
        self.assertEqual(report["pepinter"]["status"], "present")

    def test_config_resolution_does_not_import_model_runtimes_or_connect_to_network(self):
        path = self.config("defaults:\n  - dataset: pepinter\n  - _self_\n")
        script = f"""
import importlib.abc
import socket
import sys
BLOCKED = {{'torch', 'jax', 'pyrosetta', 'openmm', 'numpy', 'lightning'}}
class NoModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in BLOCKED:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoModels())
def no_network(*args, **kwargs):
    raise AssertionError('Unexpected network connection')
socket.create_connection = no_network
socket.socket.connect = no_network
from pathlib import Path
from partiflow.download import data_root
from partiflow.paths import config_dir
from partiflow.training.config import load_config
for path in (config_dir() / 'training_ae.yaml', config_dir() / 'training.yaml', Path({str(path)!r})):
    value = load_config(path)
    assert data_root(Path({str(self.models)!r}), config=path) == Path(value.dataset.datamodule.processed_dir).parent
assert not (BLOCKED & set(sys.modules))
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
