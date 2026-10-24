"""Resume checks compare real input, output, and settings values directly."""

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

    def test_changed_input_values_reject_resume_without_overwriting_old_result(self):
        first = self.complete()
        old_report = self.report_path.read_bytes()
        output = Path(first["pdb_path"])
        old_output = output.read_bytes()
        changed_inputs = {
            "coordinates": pdb(x=0.25),
            "residue_identity": pdb(code="GLY"),
            "declared_links": pdb(links="CONECT    1    3\n"),
            "atom_inventory": pdb().replace(" O   ALA", " OXT ALA"),
        }
        for label, changed in changed_inputs.items():
            with self.subTest(label=label):
                self.source.write_text(changed)
                with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
                    with self.assertRaisesRegex(
                        ValueError, "input/settings changed; use --rerun"
                    ):
                        process_candidates([self.item])
                relax.assert_not_called()
                self.assertEqual(self.source.read_text(), changed)
                self.assertEqual(self.report_path.read_bytes(), old_report)
                self.assertEqual(output.read_bytes(), old_output)
        self.source.write_text(pdb())

    def test_changed_output_rejects_resume_and_keeps_old_report(self):
        first = self.complete()
        old_report = self.report_path.read_bytes()
        output = Path(first["pdb_path"])
        changed = pdb(x=1.0)
        output.write_text(changed)
        with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
            with self.assertRaisesRegex(
                ValueError, "input/settings changed; use --rerun"
            ):
                process_candidates([self.item])
        relax.assert_not_called()
        self.assertEqual(self.report_path.read_bytes(), old_report)
        self.assertEqual(output.read_text(), changed)

    def test_changed_execution_settings_reject_resume_without_replacing_report(self):
        first = self.complete()
        old_report = self.report_path.read_bytes()
        old_output = Path(first["pdb_path"]).read_bytes()
        variants = (
            ("backend", self.item, {"method": "openmm"}),
            ("mode", self.item, {"relax_mode": "global"}),
            ("subset", self.item, {"relax_subset": "all"}),
            ("iterations", self.item, {"relax_max_iter": 7}),
            ("chain", replace(self.item, binder_chain="B"), {}),
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
                with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
                    with self.assertRaisesRegex(
                        ValueError, "input/settings changed; use --rerun"
                    ):
                        process_candidates([item], **options)
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
        with (
            patch("partiflow.common.presets.load_presets", return_value=preset(0.8)),
            patch("partiflow.postprocess.metrics.compute_metrics") as metrics,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
        ):
            second = process_candidates([self.item], filter=True, criteria=["test"])[0]
        relax.assert_not_called()
        metrics.assert_not_called()
        self.assertTrue(second["resumed"])
        self.assertEqual(second["filter_status"], "failed")
        self.assertEqual(set(second["decisions"]), {"test"})
        self.assertEqual(Path(second["pdb_path"]).read_bytes(), old_output)
        self.assertEqual(
            json.loads(self.report_path.read_text())["filter_status"], "failed"
        )

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
