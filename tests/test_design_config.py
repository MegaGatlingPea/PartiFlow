"""CPU checks of design configuration layers and installed resources."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from partiflow.cli import parser
from partiflow.design.config import load
from partiflow.paths import config_dir, resource_root

ROOT = Path(__file__).resolve().parents[1]


class DesignConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="partiflow-design-config-")
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.invocation = self.directory / "invocation"
        self.invocation.mkdir()
        self.external = self.directory / "external"
        self.external.mkdir()
        self.tasks = self.external / "tasks.yaml"
        self.tasks.write_text(
            "first:\n  task_type: denovo\n  pdb_path: reference.pdb\n"
            "  sample_num: 2\nsecond:\n  task_type: denovo\n  sample_num: null\n"
        )
        previous = Path.cwd()
        os.chdir(self.invocation)
        self.addCleanup(os.chdir, previous)

    def configuration(self, value, name="custom.yaml"):
        path = self.external / name
        path.write_text(yaml.safe_dump(value, sort_keys=False))
        return path

    def args(self, *options):
        return parser().parse_args(
            [
                "design",
                str(self.tasks),
                "--task",
                "first",
                *map(str, options),
            ]
        )

    def test_packaged_defaults_from_another_directory(self):
        cfg = load(self.args())
        self.assertEqual(cfg.seed, 2026)
        self.assertEqual(cfg.datamodule.batch_size, 16)
        self.assertEqual(cfg.generation.args.nsteps, 400)
        self.assertEqual(cfg.generation.args.weight_noise_seed, 2026)
        self.assertIsNone(cfg.search.algorithm)
        self.assertIsNone(cfg.get("reward"))
        self.assertEqual(cfg.task_names, ["first"])
        self.assertEqual(cfg.output.root_dir, str(self.invocation / "inference"))
        self.assertFalse((self.invocation / "inference").exists())

    def test_file_then_value_overrides_then_aliases(self):
        path = self.configuration(
            {
                "seed": 7,
                "sample_num": 3,
                "generation": {"args": {"nsteps": 10}},
                "datamodule": {"batch_size": 5, "num_workers": 6},
                "compile": {"enabled": True},
                "output": {"run_name": "from-file"},
            }
        )
        original = path.read_bytes()
        cfg = load(
            self.args(
                "--config",
                path,
                "--task",
                "second",
                "--override",
                "seed=9",
                "--override",
                "sample_num=4",
                "--override",
                "generation.args.nsteps=20",
                "--override",
                "datamodule.batch_size=3",
                "--num-workers",
                "0",
                "--compile",
                "false",
                "--run-name",
                "alias",
            )
        )
        self.assertEqual(cfg.seed, 9)
        self.assertEqual(cfg.generation.args.weight_noise_seed, 9)
        self.assertEqual(cfg.generation.args.nsteps, 20)
        self.assertEqual(cfg.datamodule.batch_size, 3)
        self.assertEqual(cfg.datamodule.num_workers, 0)
        self.assertFalse(cfg.compile.enabled)
        self.assertEqual(cfg.output.run_name, "alias")
        self.assertEqual(cfg.tasks.first.sample_num, 2)
        self.assertEqual(cfg.tasks.second.sample_num, 4)
        self.assertEqual(path.read_bytes(), original)

    def test_aliases_override_file_values(self):
        path = self.configuration(
            {
                "seed": 7,
                "datamodule": {"batch_size": 5},
                "checkpoint": {"model_ckpt": "./ckpts/file.ckpt"},
            }
        )
        cfg = load(
            self.args(
                "--config",
                path,
                "--seed",
                "11",
                "--batch-size",
                "2",
                "--ckpt",
                "./ckpts/explicit.ckpt",
            )
        )
        self.assertEqual(cfg.seed, 11)
        self.assertEqual(cfg.datamodule.batch_size, 2)
        self.assertEqual(
            cfg.checkpoint.model_ckpt, str(self.invocation / "ckpts/explicit.ckpt")
        )

    def test_duplicate_alias_and_override_is_rejected(self):
        aliases = (
            ("--seed", "7", "seed", "8"),
            ("--batch-size", "2", "datamodule.batch_size", "3"),
            ("--num-workers", "0", "datamodule.num_workers", "1"),
            ("--compile", "true", "compile.enabled", "false"),
            ("--run-name", "named", "output.run_name", "override"),
            ("--ckpt", "named.ckpt", "checkpoint.model_ckpt", "override.ckpt"),
        )
        for option, value, key, override in aliases:
            with self.subTest(option=option):
                with self.assertRaisesRegex(ValueError, "CLI and --override both set"):
                    load(self.args(option, value, "--override", key + "=" + override))
        with self.assertRaisesRegex(ValueError, "CLI and --override both set seed"):
            load(self.args("--seed", "7", "--override", "++seed=8"))

    def test_cli_group_choices_override_plain_file(self):
        path = self.configuration(
            {
                "search": {"algorithm": None, "best_of_n": {"replicas": 2}},
                "reward": None,
            }
        )
        cfg = load(
            self.args(
                "--config",
                path,
                "--override",
                "search=best_of_n",
                "--override",
                "reward=af2",
                "--override",
                "search.best_of_n.replicas=4",
            )
        )
        self.assertEqual(cfg.search.algorithm, "best_of_n")
        self.assertEqual(cfg.search.best_of_n.replicas, 4)
        self.assertEqual(cfg.reward.af2.model, "multimer")

    def test_external_hydra_defaults_use_packaged_groups(self):
        path = self.configuration(
            {
                "defaults": [{"search": "best_of_n"}, {"reward": "af2"}, "_self_"],
                "seed": 7,
                "search": {"best_of_n": {"replicas": 2}},
                "generation": {"args": {"nsteps": 12}},
            }
        )
        cfg = load(
            self.args(
                "--config",
                path,
                "--override",
                "seed=13",
                "--override",
                "search.best_of_n.replicas=3",
            )
        )
        self.assertEqual(cfg.seed, 13)
        self.assertEqual(cfg.generation.args.nsteps, 12)
        self.assertEqual(cfg.datamodule.batch_size, 16)
        self.assertEqual(cfg.search.algorithm, "best_of_n")
        self.assertEqual(cfg.search.best_of_n.replicas, 3)
        self.assertEqual(cfg.reward.sequence.source, "design")

    def test_plain_overlay_and_hydra_defaults_accept_yml_files(self):
        for value in (
            {"seed": 7},
            {
                "defaults": [{"search": "best_of_n"}, {"reward": "af2"}, "_self_"],
                "seed": 7,
            },
        ):
            with self.subTest(value=value):
                path = self.configuration(value, "custom.yml")
                cfg = load(self.args("--config", path, "--override", "seed=13"))
                self.assertEqual(cfg.seed, 13)

    def test_external_defaults_can_extend_packaged_design(self):
        path = self.configuration(
            {
                "defaults": [
                    "design",
                    {"override search": "best_of_n"},
                    {"override reward": "af2"},
                    "_self_",
                ],
                "seed": 7,
            }
        )
        cfg = load(self.args("--config", path, "--override", "seed=13"))
        self.assertEqual(cfg.seed, 13)
        self.assertEqual(cfg.generation.args.nsteps, 400)
        self.assertEqual(cfg.search.algorithm, "best_of_n")

    def test_local_defaults_and_self_order_are_preserved(self):
        self.configuration(
            {"seed": 3, "generation": {"args": {"nsteps": 11}}}, "local.yaml"
        )
        for defaults, expected in ((["local", "_self_"], 7), (["_self_", "local"], 3)):
            with self.subTest(defaults=defaults):
                path = self.configuration({"defaults": defaults, "seed": 7})
                cfg = load(self.args("--config", path))
                self.assertEqual(cfg.seed, expected)
                self.assertEqual(cfg.generation.args.nsteps, 11)

    def test_cli_group_choices_can_use_external_group_files(self):
        directory = self.external / "search"
        directory.mkdir()
        (directory / "local.yaml").write_text(
            "defaults:\n  - default\n  - _self_\n"
            "algorithm: best_of_n\nmax_batch_size: 2\nbest_of_n:\n  replicas: 5\n"
        )
        path = self.configuration({"seed": 7})
        cfg = load(
            self.args(
                "--config",
                path,
                "--override",
                "search=local",
                "--override",
                "reward=af2",
            )
        )
        self.assertEqual(cfg.search.best_of_n.replicas, 5)
        self.assertEqual(cfg.search.max_batch_size, 2)

    def test_cli_can_return_file_best_of_n_to_raw(self):
        path = self.configuration(
            {
                "defaults": [{"search": "best_of_n"}, {"reward": "af2"}, "_self_"],
            }
        )
        cfg = load(
            self.args(
                "--config",
                path,
                "--override",
                "search=default",
                "--override",
                "~reward",
            )
        )
        self.assertIsNone(cfg.search.algorithm)
        self.assertIsNone(cfg.get("reward"))

    def test_explicit_packaged_config_and_task_paths(self):
        args = parser().parse_args(
            [
                "design",
                str(config_dir() / "tasks/example.yaml"),
                "--config",
                str(config_dir() / "design.yaml"),
                "--task",
                "klk2_hotspot",
            ]
        )
        cfg = load(args)
        self.assertEqual(
            cfg.tasks.klk2_hotspot.pdb_path,
            str(resource_root() / "pdbs/design/klk2/klk2.pdb"),
        )
        self.assertTrue(Path(cfg.tasks.klk2_hotspot.pdb_path).is_file())

    def test_user_paths_resolve_from_invocation_directory(self):
        path = self.configuration(
            {
                "checkpoint": {"model_ckpt": "custom.ckpt"},
                "output": {"root_dir": "custom-output"},
            }
        )
        cfg = load(self.args("--config", os.path.relpath(path)))
        self.assertEqual(
            cfg.tasks.first.pdb_path, str(self.invocation / "reference.pdb")
        )
        self.assertEqual(
            cfg.checkpoint.model_ckpt, str(self.invocation / "custom.ckpt")
        )
        self.assertEqual(cfg.output.root_dir, str(self.invocation / "custom-output"))

    def test_model_directory_priority(self):
        environment = self.directory / "environment-models"
        explicit = self.directory / "explicit-models"
        for env, option, expected in (
            (None, None, ROOT),
            (environment, None, environment),
            (environment, explicit, explicit),
        ):
            with self.subTest(env=env, option=option), patch.dict(os.environ):
                os.environ.pop("PARTIFLOW_MODEL_DIR", None)
                if env is not None:
                    os.environ["PARTIFLOW_MODEL_DIR"] = str(env)
                cfg = load(self.args(*(["--model-dir", option] if option else [])))
                self.assertEqual(cfg._model_dir, str(expected))
                self.assertEqual(
                    cfg.checkpoint.model_ckpt, str(expected / "ckpts/partiflow.ckpt")
                )
                self.assertEqual(
                    cfg.checkpoint.autoencoder_ckpt,
                    str(expected / "ckpts/autoencoder.ckpt"),
                )

    def test_out_selects_exact_directory_without_creating_it(self):
        path = self.configuration(
            {"output": {"root_dir": "from-file", "run_name": "file"}}
        )
        cfg = load(
            self.args(
                "--config",
                path,
                "--override",
                "output.run_name=override",
                "--out",
                "runs/exact",
            )
        )
        self.assertEqual(
            Path(cfg.output.root_dir) / cfg.output.run_name,
            self.invocation / "runs/exact",
        )
        self.assertFalse((self.invocation / "runs").exists())

    def test_unknown_root_settings_are_rejected_in_files_and_overrides(self):
        for value in ({"unknown_option": True}, {"model": {"checkpoint": "unused"}}):
            with self.subTest(value=value):
                path = self.configuration(value)
                with self.assertRaisesRegex(
                    ValueError, "Unknown design configuration settings"
                ):
                    load(self.args("--config", path))
        with self.assertRaisesRegex(
            ValueError, "Unknown design configuration settings"
        ):
            load(self.args("--override", "+unknown_option=true"))

    def test_unknown_reward_settings_are_rejected_before_model_import(self):
        from partiflow.workflows.reward import _config

        for key in ("unsupported_backend", "ranking", "_target_"):
            with (
                self.subTest(key=key),
                self.assertRaisesRegex(ValueError, "Unknown reward"),
            ):
                _config({key: "unused"})

    def test_unknown_search_settings_are_rejected_in_files_and_overrides(self):
        path = self.configuration({"search": {"branch_count": 2}})
        with self.assertRaisesRegex(ValueError, "Unknown search settings"):
            load(self.args("--config", path))
        with self.assertRaisesRegex(ValueError, "Unknown search settings"):
            load(self.args("--override", "+search.branch_count=2"))
        path = self.configuration({"search": {"best_of_n": {"extra": 2}}})
        with self.assertRaisesRegex(
            ValueError, "best_of_n accepts only positive replicas"
        ):
            load(self.args("--config", path))

    def test_replicas_require_a_strict_positive_integer(self):
        for value in (None, True, False, 0, -1, 1.5, "2", [], {}):
            with self.subTest(value=value):
                path = self.configuration(
                    {"search": {"best_of_n": {"replicas": value}}}
                )
                with self.assertRaisesRegex(
                    ValueError, "best_of_n accepts only positive replicas"
                ):
                    load(self.args("--config", path))
        cfg = load(self.args("--override", "search.best_of_n.replicas=2"))
        self.assertEqual(cfg.search.best_of_n.replicas, 2)

    def test_max_batch_size_requires_a_strict_positive_integer_or_null(self):
        for value in (True, False, 0, -1, 1.5, "2", [], {}):
            with self.subTest(value=value):
                path = self.configuration({"search": {"max_batch_size": value}})
                with self.assertRaisesRegex(
                    ValueError, "search.max_batch_size must be"
                ):
                    load(self.args("--config", path))
        for value in (None, 1, 8):
            with self.subTest(valid=value):
                path = self.configuration({"search": {"max_batch_size": value}})
                self.assertEqual(
                    load(self.args("--config", path)).search.max_batch_size, value
                )

    def test_public_search_and_reward_constraints(self):
        cases = (
            ({"search": None}, "search must be a mapping"),
            ({"search": {"algorithm": "beam"}}, "search.algorithm must be"),
            (
                {"search": {"selection_scope": "global"}},
                "search.selection_scope must be",
            ),
            ({"reward": {}}, "Raw sampling requires null reward"),
            ({"search": {"algorithm": "best_of_n"}}, "Best-of-N requires AF2 reward"),
            (
                {
                    "search": {"algorithm": "best_of_n"},
                    "reward": {"af2": {"msa_policy": "search"}},
                },
                "offline no-MSA",
            ),
        )
        for value, message in cases:
            with self.subTest(value=value):
                path = self.configuration(value)
                with self.assertRaisesRegex(ValueError, message):
                    load(self.args("--config", path))

    def test_nonmapping_configuration_is_rejected(self):
        for value in ([], ["seed"], "seed"):
            with self.subTest(value=value):
                path = self.configuration(value)
                with self.assertRaisesRegex(
                    ValueError, "Design configuration must be a mapping"
                ):
                    load(self.args("--config", path))

    def test_composition_store_is_cleaned_after_success_and_failure(self):
        from hydra.core.config_store import ConfigStore
        from hydra.errors import ConfigCompositionException

        load(self.args())
        before = set(ConfigStore.instance().repo)
        load(self.args("--override", "seed=8"))
        with self.assertRaises(ConfigCompositionException):
            load(self.args("--override", "missing.setting=1"))
        self.assertEqual(set(ConfigStore.instance().repo), before)

    def test_show_config_stays_lazy_with_reward_and_creates_no_output(self):
        script = """
import contextlib
import importlib.abc
import io
import sys
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'jax', 'pyrosetta', 'openmm', 'lightning'}:
            raise AssertionError('Unexpected model import: ' + fullname)
sys.meta_path.insert(0, BlockModels())
from partiflow.cli import main
output = io.StringIO()
with contextlib.redirect_stdout(output):
    status = main(sys.argv[1:])
assert status == 0, output.getvalue()
assert 'replicas: 3' in output.getvalue(), output.getvalue()
assert not ({'torch', 'jax', 'pyrosetta'} & set(sys.modules))
print('lazy display passed')
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                "design",
                str(self.tasks),
                "--task",
                "first",
                "--override",
                "search=best_of_n",
                "--override",
                "reward=af2",
                "--override",
                "search.best_of_n.replicas=3",
                "--show-config",
            ],
            cwd=self.invocation,
            env=dict(os.environ, PYTHONPATH=str(ROOT)),
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertEqual(completed.stdout.strip(), "lazy display passed")
        self.assertEqual(list(self.invocation.iterdir()), [])

    def test_installed_layout_uses_packaged_groups_and_separate_path_roots(self):
        package = self.directory / "site with spaces/partiflow"
        for relative in (
            "__init__.py",
            "paths.py",
            "cli.py",
            "design/__init__.py",
            "design/config.py",
            "workflows/reward.py",
            "workflows/sequence.py",
            "common/records.py",
        ):
            target = package / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "partiflow" / relative, target)
        shutil.copytree(ROOT / "configs", package / "_resources/configs")
        reference = Path("pdbs/design/klk2/klk2.pdb")
        bundled_pdb = package / "_resources" / reference
        bundled_pdb.parent.mkdir(parents=True)
        shutil.copy2(ROOT / reference, bundled_pdb)
        path = self.configuration(
            {
                "defaults": [{"search": "best_of_n"}, {"reward": "af2"}, "_self_"],
                "seed": 7,
                "search": {"best_of_n": {"replicas": 2}},
            }
        )
        script = """
import json
import sys
from pathlib import Path
from partiflow.cli import parser
from partiflow.design.config import load
from partiflow.paths import checkout_root, config_dir
args = parser().parse_args(['design', str(config_dir() / 'tasks/example.yaml'),
                            '--task', 'klk2_hotspot'])
bundled = load(args)
args = parser().parse_args(['design', sys.argv[1], '--task', 'first',
                            '--config', sys.argv[2], '--override', 'seed=13',
                            '--override', 'search.best_of_n.replicas=3'])
external = load(args)
assert not ({'torch', 'jax', 'pyrosetta'} & set(sys.modules))
print(json.dumps({'checkout': str(checkout_root()), 'config': str(config_dir()),
 'bundled_pdb': bundled.tasks.klk2_hotspot.pdb_path,
 'default_model': bundled.checkpoint.model_ckpt,
 'user_pdb': external.tasks.first.pdb_path, 'seed': external.seed,
 'replicas': external.search.best_of_n.replicas, 'output': external.output.root_dir,
 'reward_model': external.reward.af2.model}))
"""
        environment = dict(
            os.environ, PYTHONPATH=str(self.directory / "site with spaces")
        )
        environment.pop("PARTIFLOW_MODEL_DIR", None)
        completed = subprocess.run(
            [sys.executable, "-c", script, str(self.tasks), str(path)],
            cwd=self.invocation,
            env=environment,
            text=True,
            capture_output=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads(completed.stdout)
        self.assertEqual(
            result,
            {
                "checkout": "None",
                "config": str(package / "_resources/configs"),
                "bundled_pdb": str(bundled_pdb),
                "default_model": str(self.invocation / "ckpts/partiflow.ckpt"),
                "user_pdb": str(self.invocation / "reference.pdb"),
                "seed": 13,
                "replicas": 3,
                "output": str(self.invocation / "inference"),
                "reward_model": "multimer",
            },
        )


if __name__ == "__main__":
    unittest.main()
