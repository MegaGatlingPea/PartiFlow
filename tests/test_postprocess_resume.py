"""Resume uses small v2 records and treats each invocation's inputs as fixed."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from partiflow.postprocess.records import WorkItem
from partiflow.postprocess.runner import process_candidates


def pdb(*, x=0.0, code="ALA", links=""):
    atoms = (("N", "N", 0.0), ("CA", "C", 1.3), ("C", "C", 2.6), ("O", "O", 3.5))
    return (
        "".join(
            f"ATOM  {serial:5d} {name:^4s} {code:3s} L   1    "
            f"{x + offset:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00          {element:>2s}  \n"
            for serial, (name, element, offset) in enumerate(atoms, 1)
        )
        + links
        + "END\n"
    )


def copy_relax(item, output, **options):
    shutil.copyfile(item.path, output)
    return output, {"method": "openmm"}


def metric_report():
    return {
        "status": "complete",
        "protocol": "complex",
        "metrics": {"shape_complementarity": 0.7},
    }


def preset(threshold):
    return {
        "test": {
            "methods": ["rosetta"],
            "reduction": "any_prediction",
            "rules": [
                {
                    "metric": "shape_complementarity",
                    "op": ">",
                    "value": threshold,
                    "scope": "complex",
                }
            ],
        }
    }


class PostprocessResumeTests(unittest.TestCase):
    def setUp(self):
        from postprocess_test_utils import inline_dispatch
        self.dispatch_patch = patch("partiflow.postprocess.dispatch.dispatch_items", side_effect=inline_dispatch)
        self.dispatch = self.dispatch_patch.start()
        self.addCleanup(self.dispatch_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "sample1.pdb"
        self.source.write_text(pdb())
        self.reports = self.root / "postprocess"
        self.item = WorkItem(
            self.source, "task", "sample1", report_dir=self.reports, replace=False
        )

    def tearDown(self):
        self.temp.cleanup()

    @property
    def report_path(self):
        return self.reports / "sample1.json"

    def complete(self, **options):
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=copy_relax
        ):
            result = process_candidates([self.item], **options)[0]
        self.assertEqual(result["status"], "success")
        return result

    def test_completed_record_keeps_settings_without_structure_payloads(self):
        self.complete()
        record = json.loads(self.report_path.read_text())
        self.assertEqual(record["version"], 2)
        self.assertEqual(set(record["resume"]), {"settings"})
        self.assertEqual(record["resume"]["settings"]["sample_id"], "sample1")

        def assert_small(value):
            if isinstance(value, dict):
                self.assertFalse(
                    set(value)
                    & {"atoms", "xyz", "coordinates", "input_state", "output_state"}
                )
                for child in value.values():
                    assert_small(child)
            elif isinstance(value, list):
                for child in value:
                    assert_small(child)

        assert_small(record)

    def test_same_path_coordinate_edits_are_not_detected_without_rerun(self):
        first = self.complete()
        output = Path(first["pdb_path"])
        old_output = output.read_bytes()
        self.source.write_text(pdb(x=0.25))
        self.dispatch.reset_mock()
        with (
            patch("partiflow.postprocess.source.PDBSource.read") as read,
            patch("partiflow.postprocess.runner.PostprocessWorker") as worker,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
        ):
            second = process_candidates([self.item])[0]
        self.dispatch.assert_not_called()
        read.assert_not_called()
        worker.assert_not_called()
        relax.assert_not_called()
        self.assertTrue(second["resumed"])
        self.assertEqual(self.source.read_text(), pdb(x=0.25))
        self.assertEqual(output.read_bytes(), old_output)

    def test_existing_output_coordinate_edits_are_not_rescanned(self):
        first = self.complete()
        output = Path(first["pdb_path"])
        changed = pdb(x=1.0)
        output.write_text(changed)
        self.dispatch.reset_mock()
        with (
            patch("partiflow.postprocess.source.PDBSource.read") as read,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
        ):
            second = process_candidates([self.item])[0]
        self.dispatch.assert_not_called()
        read.assert_not_called()
        relax.assert_not_called()
        self.assertTrue(second["resumed"])
        self.assertEqual(output.read_text(), changed)

    def test_missing_output_rejects_resume_without_replacing_report(self):
        first = self.complete()
        old_report = self.report_path.read_bytes()
        output = Path(first["pdb_path"])
        output.unlink()
        self.dispatch.reset_mock()
        with self.assertRaisesRegex(ValueError, "input/settings changed; use --rerun"):
            process_candidates([self.item])
        self.dispatch.assert_not_called()
        self.assertEqual(self.report_path.read_bytes(), old_report)
        self.assertFalse(output.exists())

    def test_old_record_version_requires_explicit_rerun(self):
        first = self.complete()
        record = json.loads(self.report_path.read_text())
        record["version"] = 1
        self.report_path.write_text(json.dumps(record))
        old_report = self.report_path.read_bytes()
        old_output = Path(first["pdb_path"]).read_bytes()
        self.dispatch.reset_mock()
        with self.assertRaisesRegex(ValueError, "input/settings changed; use --rerun"):
            process_candidates([self.item])
        self.dispatch.assert_not_called()
        self.assertEqual(self.report_path.read_bytes(), old_report)
        self.assertEqual(Path(first["pdb_path"]).read_bytes(), old_output)

    def test_changed_execution_settings_reject_resume_without_replacing_report(self):
        first = self.complete()
        old_report = self.report_path.read_bytes()
        old_output = Path(first["pdb_path"]).read_bytes()
        variants = (
            ("backend", self.item, {"method": "openmm"}),
            ("mode", self.item, {"relax_mode": "local"}),
            ("subset", self.item, {"relax_subset": "nbrs"}),
            ("iterations", self.item, {"relax_max_iter": 7}),
            ("seed", self.item, {"seed": 1227}),
            ("platform", self.item, {"platform": "CUDA"}),
            ("chain", replace(self.item, binder_chain="B"), {}),
            ("task_identity", replace(self.item, task_name="another_task"), {}),
            ("sample_identity", replace(self.item, sample_id="another_sample"), {}),
            ("candidate_identity", replace(self.item, candidate_id="candidate2"), {}),
            ("base_identity", replace(self.item, base_sample_id=2), {}),
            (
                "topology",
                replace(
                    self.item,
                    topology_entries=[{"type": "head_to_tail", "anchors": [1, 3]}],
                ),
                {},
            ),
            ("linear_policy", replace(self.item, relax_linear=False), {}),
            ("publication", replace(self.item, replace=True), {}),
        )
        for label, item, options in variants:
            with self.subTest(label=label):
                self.dispatch.reset_mock()
                with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
                    with self.assertRaisesRegex(
                        ValueError, "input/settings changed; use --rerun"
                    ):
                        process_candidates([item], **options)
                self.dispatch.assert_not_called()
                relax.assert_not_called()
                self.assertEqual(self.report_path.read_bytes(), old_report)
                self.assertEqual(Path(first["pdb_path"]).read_bytes(), old_output)

    def test_explicit_rerun_processes_changed_input(self):
        first = self.complete()
        self.source.write_text(pdb(x=0.5))
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=copy_relax
        ) as relax:
            second = process_candidates([self.item], rerun=True)[0]
        relax.assert_called_once()
        self.assertEqual(second["status"], "success")
        self.assertEqual(Path(first["pdb_path"]).read_bytes(), self.source.read_bytes())

    def test_batch_preflight_rejects_later_mismatch_before_any_report_write(self):
        other_source = self.root / "sample2.pdb"
        other_source.write_text(pdb(x=2.0))
        other = replace(self.item, path=other_source, sample_id="sample2")
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.6)),
            patch("partiflow.postprocess.metrics.compute_metrics", return_value=metric_report()),
            patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=copy_relax),
        ):
            process_candidates([self.item, other], filter=True, criteria=["test"])
        fresh_source = self.root / "sample3.pdb"
        fresh_source.write_text(pdb(x=3.0))
        fresh = replace(self.item, path=fresh_source, sample_id="sample3")
        saved = {path: path.read_bytes() for path in self.reports.rglob("*") if path.is_file()}
        self.dispatch.reset_mock()
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.8)),
            patch("partiflow.postprocess.runner._write_json") as write,
            patch("partiflow.postprocess.metrics.compute_metrics") as metrics,
            patch("partiflow.postprocess.runner.PostprocessWorker") as worker,
        ):
            with self.assertRaisesRegex(ValueError, "input/settings changed; use --rerun"):
                process_candidates(
                    [self.item, fresh, replace(other, binder_chain="B")],
                    filter=True,
                    criteria=["test"],
                )
        self.dispatch.assert_not_called()
        write.assert_not_called()
        metrics.assert_not_called()
        worker.assert_not_called()
        self.assertEqual(
            {path: path.read_bytes() for path in self.reports.rglob("*") if path.is_file()},
            saved,
        )

    def test_preset_only_refilter_reuses_geometry_and_metrics(self):
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.6)),
            patch(
                "partiflow.postprocess.metrics.compute_metrics",
                return_value=metric_report(),
            ) as metrics,
        ):
            first = self.complete(filter=True, criteria=["test"])
        self.assertEqual(first["filter_status"], "passed")
        metrics.assert_called_once()
        old_output = Path(first["pdb_path"]).read_bytes()
        self.dispatch.reset_mock()
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.8)),
            patch("partiflow.postprocess.metrics.compute_metrics") as metrics,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
            patch("partiflow.postprocess.runner.PostprocessWorker") as worker,
            patch("partiflow.postprocess.source.PDBSource.read") as read,
        ):
            second = process_candidates([self.item], filter=True, criteria=["test"])[0]
        relax.assert_not_called()
        metrics.assert_not_called()
        worker.assert_not_called()
        read.assert_not_called()
        self.dispatch.assert_not_called()
        self.assertTrue(second["resumed"])
        self.assertEqual(second["filter_status"], "failed")
        self.assertEqual(set(second["decisions"]), {"test"})
        self.assertEqual(Path(second["pdb_path"]).read_bytes(), old_output)
        self.assertEqual(
            json.loads(self.report_path.read_text())["filter_status"], "failed"
        )

    def test_enabling_filter_dispatches_only_metrics_for_saved_geometry(self):
        first = self.complete()
        old_output = Path(first["pdb_path"]).read_bytes()
        self.dispatch.reset_mock()
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.6)),
            patch(
                "partiflow.postprocess.metrics.compute_metrics",
                return_value=metric_report(),
            ) as metrics,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
            patch("partiflow.postprocess.source.PDBSource.read") as read,
        ):
            second = process_candidates([self.item], filter=True, criteria=["test"])[0]
        self.dispatch.assert_called_once()
        self.assertFalse(self.dispatch.call_args.args[0][0].needs_gpu)
        metrics.assert_called_once()
        relax.assert_not_called()
        read.assert_not_called()
        self.assertTrue(second["resumed"])
        self.assertEqual(second["filter_status"], "passed")
        self.assertEqual(Path(second["pdb_path"]).read_bytes(), old_output)

    def test_buns_setting_change_recomputes_metrics_once_without_geometry(self):
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.6)),
            patch(
                "partiflow.postprocess.metrics.compute_metrics",
                return_value=metric_report(),
            ),
        ):
            self.complete(filter=True, criteria=["test"], compute_buns=True)
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.6)),
            patch(
                "partiflow.postprocess.metrics.compute_metrics",
                return_value=metric_report(),
            ) as metrics,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
        ):
            first = process_candidates(
                [self.item], filter=True, criteria=["test"], compute_buns=False
            )[0]
            second = process_candidates(
                [self.item], filter=True, criteria=["test"], compute_buns=False
            )[0]
        relax.assert_not_called()
        metrics.assert_called_once()
        self.assertFalse(metrics.call_args.kwargs["compute_buns"])
        self.assertEqual(first["filter_status"], "passed")
        self.assertEqual(second["filter_status"], "passed")
        self.assertFalse(json.loads(self.report_path.read_text())["compute_buns"])

    def test_partial_failure_retries_only_failed_candidate(self):
        failed = self.root / "sample2.pdb"
        failed.write_text(pdb(x=2.0))
        third = self.root / "sample3.pdb"
        third.write_text(pdb(x=3.0))
        items = [
            self.item,
            replace(self.item, path=failed, sample_id="sample2"),
            replace(self.item, path=third, sample_id="sample3"),
        ]

        def partial(item, output, **options):
            if item.sample_id == "sample2":
                output.write_text("partial output")
                raise ValueError("candidate chemistry failed")
            return copy_relax(item, output, **options)

        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=partial
        ):
            first = process_candidates(items)
        self.assertEqual(
            [row["status"] for row in first], ["success", "failed", "success"]
        )
        saved = {
            row["sample_id"]: Path(row["pdb_path"]).read_bytes()
            for row in first
            if row["status"] == "success"
        }
        self.assertEqual(failed.read_text(), pdb(x=2.0))
        self.assertFalse((self.reports / "structures" / failed.name).exists())
        self.assertFalse(list((self.reports / "structures").glob(".postprocess-*")))
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=partial
        ) as relax:
            second = process_candidates(items)
        self.assertEqual(
            [call.args[0].sample_id for call in relax.call_args_list], ["sample2"]
        )
        self.assertEqual(
            [row["status"] for row in second], ["success", "failed", "success"]
        )
        self.assertIn(
            "candidate chemistry failed", second[1]["previous_postprocess_error"]
        )
        for row in second:
            if row["status"] == "success":
                self.assertTrue(row["resumed"])
                self.assertEqual(
                    Path(row["pdb_path"]).read_bytes(), saved[row["sample_id"]]
                )


if __name__ == "__main__":
    unittest.main()
