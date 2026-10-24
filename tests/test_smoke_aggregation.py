import tempfile
import unittest
from unittest.mock import patch
from partiflow.design.smoke import run_smoke


class SmokeAggregationTests(unittest.TestCase):
    def test_workflow_model_results_are_combined_under_model_names(self):
        models = ["proteinmpnn", "solublempnn", "af2"]
        results = {name: {"status": "complete"} for name in models}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "partiflow.workflows.smoke.run_structure_smoke",
                return_value={
                    "status": "complete",
                    "models": results,
                    "input_path": "template.pdb",
                },
            ),
        ):
            report = run_smoke(models=models, out=directory)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["models"], results)

    def test_incomplete_workflow_cannot_be_reported_complete(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "partiflow.workflows.smoke.run_structure_smoke",
                return_value={
                    "status": "failed",
                    "models": {"af2": {"status": "failed"}},
                },
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                run_smoke(models=["af2"], out=directory)
