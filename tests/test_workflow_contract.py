"""Model-free source, cache and accounting contracts."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from partiflow.common.records import Candidate
from partiflow.workflows.sequence import normalize_sources, backend_settings
from partiflow.workflows.validate import (
    prediction_keys,
    _input_snapshot,
    _slots,
    _valid_prediction,
    _budget,
)


class WorkflowContractTests(unittest.TestCase):
    def test_imports_leave_model_libraries_unloaded(self):
        code = 'import sys; from pathlib import Path; import partiflow.workflows.validate; from partiflow.workflows import reward, smoke; assert reward.evaluate_candidates([],{},Path("/missing"),Path("/missing")) == []; reward.close(); assert not any(name in sys.modules for name in ("torch","jax","colabdesign"))'
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_default_source_shapes_and_aliases(self):
        sources = normalize_sources(
            [{"source": "gt", "count": 1}, {"source": "soluble_mpnn-fi", "count": 2}]
        )
        self.assertEqual(sources[0]["label"], "self")
        self.assertEqual(sources[1]["source"], "solublempnn")
        self.assertTrue(sources[1]["settings"]["fixed_interface"]["enabled"])
        self.assertEqual(
            [(i, s["source"], n) for i, s, n in _slots(sources)],
            [(0, "gt", 1), (1, "solublempnn", 1), (2, "solublempnn", 2)],
        )

    def test_gt_duplicate_and_bad_sources_rejected(self):
        for entries in (
            [{"source": "gt", "count": 2}],
            [{"source": "gt"}, {"source": "gt"}],
            [{"source": "ligand_mpnn"}],
            [
                {
                    "source": "protein_mpnn-fi",
                    "settings": {"fixed_interface": {"enabled": False}},
                }
            ],
        ):
            with self.assertRaises(ValueError):
                normalize_sources(entries)

    def test_named_mpnn_models_have_different_defaults(self):
        vanilla = backend_settings("proteinmpnn", {}, Path("/weights"))
        soluble = backend_settings(
            "auto",
            {
                "proteinmpnn": {
                    "model_type": "protein_mpnn",
                    "checkpoint": None,
                    "temperature": 0.0001,
                },
                "solublempnn_defaults": {
                    "model_type": "soluble_mpnn",
                    "use_soluble_model": True,
                    "temperature": 0.1,
                },
            },
            Path("/weights"),
        )
        self.assertEqual(vanilla["temperature"], 0.0001)
        self.assertEqual(soluble["temperature"], 0.1)
        self.assertIn("/vanilla/", vanilla["checkpoint"])
        self.assertIn("/soluble/", soluble["checkpoint"])

    def test_prediction_keys_are_paired_and_deterministic(self):
        self.assertEqual(
            prediction_keys({"models": [2, 1], "samples": 2}),
            [
                "complex_model2_sample0",
                "complex_model2_sample1",
                "complex_model1_sample0",
                "complex_model1_sample1",
                "monomer_model2_sample0",
                "monomer_model2_sample1",
                "monomer_model1_sample0",
                "monomer_model1_sample1",
            ],
        )
        for config in (
            {"models": [3]},
            {"models": [1, 1]},
            {"samples": True},
            {"recycles": 0},
        ):
            with self.assertRaises(ValueError):
                prediction_keys(config)

    def test_input_snapshot_uses_actual_file_stat_and_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.pdb"
            path.write_text("first")
            candidate = Candidate(
                "sample1",
                "task",
                1,
                1,
                1,
                path,
                "L",
                2,
                "AC",
                fixed_restypes=[{"chain": "L", "res_id": 1, "restype": "ALA"}],
            )
            before = _input_snapshot(candidate, {})
            path.write_text("replacement")
            after = _input_snapshot(candidate, {})
            self.assertNotEqual(before, after)
            self.assertEqual(after["sequence"], "AC")
            self.assertEqual(after["fixed_restypes"], candidate.fixed_restypes)

    def test_prediction_completeness_requires_both_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prediction.pdb"
            path.write_text("ATOM")
            prediction = {"status": "complete", "structure_path": str(path)}
            self.assertFalse(_valid_prediction(prediction))
            arrays = path.with_suffix(".npz")
            arrays.write_bytes(b"array")
            prediction["confidence_path"] = str(arrays)
            self.assertTrue(_valid_prediction(prediction))
            arrays.unlink()
            self.assertFalse(_valid_prediction(prediction))

    def test_auto_does_not_replace_custom_vanilla_settings(self):
        config = {
            "proteinmpnn": {
                "model_type": "protein_mpnn",
                "temperature": 0.3,
                "checkpoint": None,
            },
            "solublempnn_defaults": {
                "model_type": "soluble_mpnn",
                "temperature": 0.1,
                "use_soluble_model": True,
            },
        }
        with self.assertRaisesRegex(ValueError, "explicit proteinmpnn"):
            backend_settings("auto", config, Path("/weights"))
        explicit = backend_settings("proteinmpnn", config, Path("/weights"))
        self.assertEqual(explicit["temperature"], 0.3)
        config["solublempnn"] = {
            "model_type": "soluble_mpnn",
            "temperature": 0.2,
            "use_soluble_model": True,
        }
        self.assertEqual(
            backend_settings("auto", config, Path("/weights"))["temperature"], 0.2
        )

    def test_reward_deadline_is_fatal_and_restores_handler(self):
        import signal, time
        from partiflow.workflows.reward import _deadline
        from partiflow.common.records import WorkflowError

        previous = signal.getsignal(signal.SIGALRM)
        with self.assertRaises(WorkflowError):
            with _deadline(0.01):
                time.sleep(0.05)
        self.assertEqual(signal.getsignal(signal.SIGALRM), previous)

    def test_structure_smoke_model_selection_is_closed(self):
        from partiflow.workflows.smoke import _requested_models

        self.assertEqual(
            _requested_models(["soluble_mpnn", "af2_ig"]), ["solublempnn", "af2"]
        )
        with self.assertRaises(ValueError):
            _requested_models(["unsupported_model"])
        with self.assertRaises(ValueError):
            _requested_models([])


class WorkflowPlanningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.task = Path(self.temporary.name) / "task"
        self.task.mkdir()
        from partiflow.common.results import write_results

        write_results(
            self.task / "result.csv",
            [
                {
                    "filename": "sample1_L.pdb",
                    "design_chain_id": "L",
                    "design_modality": "protein",
                    "base_sample_id": 1,
                    "token_length": 3,
                    "design_sequence": "AGG",
                    "fixed_restypes": "[]",
                    "bond_constraints": "[]",
                }
            ],
        )
        (self.task / "sample1_L.pdb").write_text("plan-only input bytes")
        (self.task / "resolved_config.yaml").write_text(
            "_runtime: {planned_base_samples: 3, replicas_per_base: 4, planned_candidates: 12}\n"
        )
        self.args = SimpleNamespace(
            paths=[str(self.task)],
            config=None,
            task=None,
            source=None,
            criteria=None,
            presets=None,
            limit=None,
            rerun=False,
            plan=True,
            refilter=False,
            recompute_metrics=False,
            model_dir=self.task / "weights",
        )

    def tearDown(self):
        self.temporary.cleanup()

    def _saved_state(self):
        import yaml
        from partiflow.workflows.validate import (
            _load,
            _candidate,
            _sequence_input_files,
        )
        from partiflow.paths import config_dir

        config = _load(self.args)
        _, rows = __import__(
            "partiflow.common.results", fromlist=["read_results"]
        ).read_results(self.task / "result.csv")
        candidate = _candidate(self.task, rows[0], 1)
        return dict(
            config={key: value for key, value in config.items() if key != "criteria"},
            sequence_defaults=yaml.safe_load(
                (config_dir() / "sequence.yaml").read_text()
            ),
            sequence_input_files=_sequence_input_files(
                config,
                yaml.safe_load((config_dir() / "sequence.yaml").read_text()),
                self.args.model_dir,
            ),
            budget=_budget(self.task),
            model_dir=str(self.args.model_dir.resolve()),
            inputs=[_input_snapshot(candidate, rows[0])],
        )

    def test_plan_has_full_budget_and_does_not_create_validation_outputs(self):
        import contextlib, io
        from partiflow.workflows.validate import run

        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(run(self.args), 0)
        task = json.loads(output.getvalue())["tasks"][0]
        self.assertEqual(task["planned_candidates"], 12)
        self.assertEqual(task["planned_sequence_slots"], 24)
        self.assertEqual(task["observed_sequence_slots"], 2)
        self.assertFalse((self.task / "validate").exists())

    def test_changed_saved_settings_fail_before_plan_write(self):
        from partiflow.workflows.validate import run, _atomic_json
        from partiflow.common.records import WorkflowError

        state = self._saved_state()
        state["config"]["af2"]["recycles"] = 99
        root = self.task / "validate" / "af2_ig"
        _atomic_json(root / ".state" / "config.json", state)
        with self.assertRaisesRegex(WorkflowError, "--rerun"):
            run(self.args)
        self.assertFalse((root / ".state" / "plan.json").exists())

    def test_changed_self_sequence_is_rejected_before_plan_write(self):
        from partiflow.workflows.validate import run, _atomic_json
        from partiflow.common.records import WorkflowError

        root = self.task / "validate" / "af2_ig"
        _atomic_json(root / ".state" / "config.json", self._saved_state())
        record = dict(
            method="af2_ig",
            candidate_id="sample1_L",
            seq_index=0,
            source="gt",
            source_label="self",
            source_index=1,
            expected_predictions=prediction_keys({"models": [1]}),
            sequence="AAA",
        )
        _atomic_json(root / "refolds" / "sample1_L" / "seq0.json", record)
        with self.assertRaisesRegex(WorkflowError, "self sequence"):
            run(self.args)
        self.assertFalse((root / ".state" / "plan.json").exists())

    def test_criteria_change_can_be_planned_without_invalidating_models(self):
        import contextlib, io
        from partiflow.workflows.validate import run, _atomic_json

        root = self.task / "validate" / "af2_ig"
        _atomic_json(root / ".state" / "config.json", self._saved_state())
        self.args.criteria = ["af2_ig_peptide"]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(self.args), 0)
        self.assertFalse((root / ".state" / "plan.json").exists())

    def test_failure_rows_are_reported_without_creating_passing_slots(self):
        import csv
        from partiflow.workflows.validate import _report

        root = self.task / "validate" / "af2_ig"
        root.mkdir(parents=True)
        record = dict(
            method="af2_ig",
            candidate_id="sample1",
            base_sample_id=1,
            seq_index=0,
            source="gt",
            source_label="self",
            source_index=1,
            expected_predictions=prediction_keys({}),
            predictions={},
            sequence="",
            status="failed",
            error="input unavailable",
        )
        summary = _report(
            root,
            [(root / "refolds" / "sample1" / "seq0.json", record)],
            [],
            {},
            dict(planned_base_samples=3, replicas_per_base=4, planned_candidates=12),
            1,
        )
        self.assertEqual(summary["planned_sequence_slots"], 12)
        self.assertEqual(summary["complete_sequence_slots"], 0)
        self.assertEqual(summary["passing_criteria_slots"], 0)
        self.assertTrue(summary["metrics_only"])
        with (root / "metrics.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["status"], "failed")
        self.assertEqual(rows[0]["error"], "input unavailable")
        with (root / "success.csv").open() as handle:
            self.assertEqual(list(csv.DictReader(handle)), [])

    def test_changed_constraint_file_requires_rerun_before_plan(self):
        import yaml
        from partiflow.workflows.validate import run, _atomic_json
        from partiflow.common.records import WorkflowError

        tied = self.task / "tied.json"
        tied.write_text(json.dumps([["L:1", "L:3"]]))
        config = self.task / "validate.yaml"
        config.write_text(
            yaml.safe_dump(
                {
                    "sequence": {
                        "sources": [
                            {"source": "gt", "count": 1},
                            {
                                "source": "proteinmpnn",
                                "count": 1,
                                "settings": {"tied_positions_path": str(tied)},
                            },
                        ]
                    }
                }
            )
        )
        self.args.config = config
        root = self.task / "validate" / "af2_ig"
        _atomic_json(root / ".state" / "config.json", self._saved_state())
        tied.write_text(json.dumps([["L:1", "L:2", "L:3"]]))
        with self.assertRaisesRegex(WorkflowError, "--rerun"):
            run(self.args)
        self.assertFalse((root / ".state" / "plan.json").exists())

    def test_unknown_and_wrong_method_criteria_fail_before_model_work(self):
        from partiflow.workflows.validate import run

        self.args.plan = False
        for name, message in [
            ("unknown_criterion", "Unknown criteria"),
            ("rosetta_bindcraft_peptide", "does not apply"),
        ]:
            self.args.criteria = [name]
            with self.assertRaisesRegex(ValueError, message):
                run(self.args)
            self.assertFalse((self.task / "validate").exists())

    def test_unknown_metric_is_rejected_before_model_work(self):
        import yaml
        from partiflow.workflows.validate import run

        presets = self.task / "invalid_presets.yaml"
        presets.write_text(
            yaml.safe_dump(
                {
                    "presets": {
                        "invalid_metric": {
                            "methods": ["af2_ig"],
                            "reduction": "all_and_mean",
                            "rules": [
                                {
                                    "metric": "invented_metric",
                                    "op": ">=",
                                    "value": 0.0,
                                    "scope": "complex",
                                }
                            ],
                        }
                    }
                }
            )
        )
        self.args.plan = False
        self.args.presets = presets
        self.args.criteria = ["invalid_metric"]
        with self.assertRaisesRegex(ValueError, "Unknown af2_ig metric"):
            run(self.args)
        self.assertFalse((self.task / "validate").exists())


if __name__ == "__main__":
    unittest.main()
