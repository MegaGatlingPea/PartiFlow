from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from partiflow.common.results import (
    BASE_FIELDS,
    TAIL_FIELDS,
    read_results,
    write_results,
)
from partiflow.postprocess import process_candidates, run_postprocess_for_run
from partiflow.postprocess.records import WorkItem
from partiflow.postprocess.runner import _collect_paths, _configuration, run

PRESETS = {
    "test": {
        "methods": ["rosetta"],
        "reduction": "any_prediction",
        "rules": [
            {
                "metric": "shape_complementarity",
                "op": ">",
                "value": 0.55,
                "scope": "complex",
            }
        ],
    }
}


def pdb(chain="L", x=0.0):
    return f"ATOM      1  CA  ALA {chain}   1    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  \nEND\n"


def relaxed(item, output, **options):
    output.write_text(pdb(item.binder_chain, 1.0))
    return output, {"method": "openmm"}


class PostprocessWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.task = self.root / "task"
        self.task.mkdir()
        self.source = self.task / "sample1.pdb"
        self.source.write_text(pdb())
        self.item = WorkItem(self.source, "task", "sample1", binder_chain="L")

    def tearDown(self):
        self.temp.cleanup()

    def result(self):
        row = {key: "" for key in BASE_FIELDS + TAIL_FIELDS}
        row.update(
            filename=self.source.name,
            base_sample_id=1,
            design_chain_id="L",
            design_sequence="A",
            design_chains='[{"chain":"L","modality":"protein","sequence":"A","residue_count":1}]',
            token_length=1,
            fixed_restypes="[]",
            bond_constraints="[]",
            postprocess_status="not_run",
        )
        write_results(self.task / "result.csv", [row])
        return row

    def test_failure_keeps_original_even_after_partial_staged_write(self):
        original = self.source.read_bytes()

        def fail(item, output, **options):
            output.write_text("truncated")
            raise ValueError("chemistry failed")

        with patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=fail):
            result = process_candidates([self.item])[0]
        self.assertEqual(result["status"], "failed")
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(list(self.task.glob(".postprocess-*")))

    def test_invalid_output_never_replaces_source(self):
        original = self.source.read_bytes()

        def bad(item, output, **options):
            output.write_text("END\n")
            return output, {}

        with patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=bad):
            self.assertEqual(process_candidates([self.item])[0]["status"], "failed")
        self.assertEqual(self.source.read_bytes(), original)

    def test_filter_false_does_not_compute_metrics(self):
        with (
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
            ),
            patch("partiflow.postprocess.metrics.compute_metrics") as metrics,
        ):
            result = process_candidates([self.item], filter=False)[0]
        metrics.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filter_status"], "not_requested")

    def test_quality_failure_keeps_successful_relax_and_structure(self):
        report = {
            "status": "complete",
            "protocol": "complex",
            "metrics": {"shape_complementarity": 0.1},
        }
        with (
            patch("partiflow.common.presets.load_presets", return_value=PRESETS),
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
            ) as relax,
            patch(
                "partiflow.postprocess.metrics.compute_metrics", return_value=report
            ) as metrics,
        ):
            result = process_candidates([self.item], filter=True, criteria=["test"])[0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filter_status"], "failed")
        self.assertTrue(Path(result["pdb_path"]).is_file())
        relax.assert_called_once()
        metrics.assert_called_once()

    def test_missing_required_metric_is_unavailable(self):
        report = {"status": "complete", "protocol": "complex", "metrics": {}}
        with (
            patch("partiflow.common.presets.load_presets", return_value=PRESETS),
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
            ),
            patch("partiflow.postprocess.metrics.compute_metrics", return_value=report),
        ):
            result = process_candidates([self.item], filter=True, criteria=["test"])[0]
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["filter_status"], "unavailable")

    def test_monomer_has_no_interface_preset(self):
        report = {
            "status": "complete",
            "protocol": "monomer",
            "metrics": {"total_score": -1.0},
        }
        with (
            patch("partiflow.common.presets.load_presets", return_value=PRESETS),
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
            ),
            patch("partiflow.postprocess.metrics.compute_metrics", return_value=report),
        ):
            result = process_candidates([self.item], filter=True, criteria=["test"])[0]
        self.assertEqual(result["filter_status"], "not_applicable")

    def test_filter_after_previous_postprocess_does_not_relax_again(self):
        report = {
            "status": "complete",
            "protocol": "complex",
            "metrics": {"shape_complementarity": 0.9},
        }
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
        ):
            process_candidates([self.item], filter=False)
        with (
            patch("partiflow.common.presets.load_presets", return_value=PRESETS),
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
            patch(
                "partiflow.postprocess.metrics.compute_metrics", return_value=report
            ) as metrics,
        ):
            result = process_candidates([self.item], filter=True, criteria=["test"])[0]
            process_candidates([self.item], filter=True, criteria=["test"])
        relax.assert_not_called()
        metrics.assert_called_once()
        self.assertTrue(result["resumed"])
        self.assertEqual(result["filter_status"], "passed")

    def test_result_csv_keeps_exact_fields_and_updates_identity(self):
        self.result()
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
        ):
            run_postprocess_for_run(self.root)
        fields, rows = read_results(self.task / "result.csv")
        self.assertEqual(fields, BASE_FIELDS + TAIL_FIELDS)
        self.assertEqual(rows[0]["postprocess_status"], "succeeded")
        self.assertEqual(rows[0]["design_sequence"], "A")
        self.assertEqual(rows[0]["filename"], self.source.name)

    def test_state_chemistry_and_disabled_candidate_join_filename(self):
        self.result()
        state = self.task / ".state"
        state.mkdir()
        (state / "design.json").write_text(
            json.dumps(
                {
                    "samples": [
                        {
                            "filename": self.source.name,
                            "topology_entries": [
                                {"type": "head_to_tail", "anchors": [1, 2]}
                            ],
                            "postprocess_enabled": False,
                        }
                    ]
                }
            )
        )
        with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
            results = run_postprocess_for_run(self.root, enabled_only=True)
        relax.assert_not_called()
        self.assertEqual(results[0]["status"], "skipped")
        self.assertEqual(
            read_results(self.task / "result.csv")[1][0]["postprocess_status"],
            "not_run",
        )

    def test_candidate_record_adapter(self):
        candidate = SimpleNamespace(
            pdb_path=self.source,
            candidate_id="c1",
            task_name="task",
            base_sample_id=1,
            binder_chain="L",
            topology_entries=[],
            metadata={"postprocess_enabled": True},
        )
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
        ):
            result = process_candidates([candidate])[0]
        self.assertEqual(result["candidate_id"], "c1")

    def test_explicit_refold_writes_separate_files_and_preserves_originals(self):
        self.result()
        success = self.task / "validate" / "AF2success" / "slot1"
        success.mkdir(parents=True)
        refold = success / "sample1.pdb"
        refold.write_text(pdb("B"))
        original, csv_original = (
            refold.read_bytes(),
            (self.task / "result.csv").read_bytes(),
        )
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relaxed
        ):
            args = SimpleNamespace(
                paths=[success.parent],
                task=None,
                config=None,
                filter=False,
                criteria=None,
                presets=None,
                rerun=False,
                plan=False,
            )
            self.assertEqual(run(args), 0)
        self.assertEqual(refold.read_bytes(), original)
        self.assertEqual((self.task / "result.csv").read_bytes(), csv_original)
        self.assertTrue(
            (
                self.task
                / "postprocess"
                / "refold"
                / "slot1"
                / "structures"
                / refold.name
            ).is_file()
        )

    def test_plan_creates_no_postprocess_artifacts(self):
        self.result()
        with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
            results = run_postprocess_for_run(self.root, plan=True)
        relax.assert_not_called()
        self.assertEqual(results[0]["status"], "planned")
        self.assertFalse((self.task / "postprocess").exists())

    def test_strict_boolean_rejects_ambiguous_value(self):
        with self.assertRaises(ValueError):
            process_candidates([self.item], filter="yes")

    def test_absent_published_structure_reports_failure(self):
        self.result()
        self.source.unlink()
        results = run_postprocess_for_run(self.root)
        self.assertEqual(results[0]["status"], "failed")
        self.assertEqual(
            read_results(self.task / "result.csv")[1][0]["postprocess_status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
