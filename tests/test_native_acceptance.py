"""A lost native actor retains completed geometry and retries only metrics."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from partiflow.postprocess.records import WorkItem
from partiflow.postprocess import runner


class ActorLost(BaseException):
    """Leave the sample function as a process-level native failure would."""


class NativeActorFailureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "sample1.pdb"
        self.source.write_text(
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
            "ATOM      2  CA  MAA L   1       4.000   0.000   0.000  1.00  0.00           C  \nEND\n"
        )
        self.original = self.source.read_bytes()
        self.report_dir = self.root / "postprocess"
        self.report_path = self.report_dir / "sample1.json"
        self.output_path = self.report_dir / "structures" / self.source.name
        self.entries = [
            {"type": "residue_template", "resid": 1, "resname": "MAA"}
        ]
        self.item = WorkItem(
            self.source,
            "task",
            "sample1",
            binder_chain="L",
            topology_entries=self.entries,
            report_dir=self.report_dir,
            replace=False,
            sdk_profile="native_ref2015",
        )
        self.presets = {
            "native": {
                "methods": ["rosetta"],
                "reduction": "any_prediction",
                "rules": [{
                    "metric": "surface_hydrophobicity",
                    "op": "<",
                    "value": 0.3,
                    "scope": "complex",
                }],
            }
        }
        with patch(
            "partiflow.common.presets.load_presets", return_value=self.presets
        ):
            self.options = runner._configuration(
                {"filter": True, "criteria": ["native"], "use_ray": False}
            )

    @staticmethod
    def copy_relax(candidate, output, **options):
        shutil.copyfile(candidate.path, output)
        return output, {"method": "openmm"}

    def assert_identity(self, record):
        identity = record["identity"]
        self.assertEqual(identity["design_chain_id"], "L")
        self.assertEqual(identity["design_sequence"], "[MAA]")
        self.assertEqual(identity["token_length"], 1)
        self.assertEqual(json.loads(identity["design_chains"]), [{
            "chain": "L",
            "modality": "protein",
            "sequence": "[MAA]",
            "residue_count": 1,
        }])

    def test_metric_actor_loss_retains_geometry_and_resume_reuses_it(self):
        saved_before_loss = {}

        def lose_actor(path, **options):
            saved = json.loads(self.report_path.read_text())
            saved_before_loss.update(saved)
            self.assertEqual(saved["version"], 2)
            self.assertEqual(saved["status"], "success")
            self.assertEqual(saved["geometry_status"], "complete")
            self.assertEqual(saved["pdb_path"], str(self.output_path))
            self.assertEqual(saved["output_path"], str(self.output_path))
            self.assertEqual(saved["sdk_profile"], "native_ref2015")
            self.assertEqual(set(saved["resume"]), {"settings"})
            self.assertLess(self.report_path.stat().st_size, 8192)
            self.assert_identity(saved)
            self.assertEqual(self.output_path.read_bytes(), self.original)
            self.assertEqual(self.source.read_bytes(), self.original)
            self.assertEqual(path, str(self.output_path))
            self.assertEqual(options["binder_chain"], "L")
            self.assertEqual(options["topology_entries"], self.entries)
            self.assertTrue(options["compute_buns"])
            self.assertEqual(
                options["required_metrics"], ["surface_hydrophobicity"]
            )
            self.assertEqual(
                options["context"].sdk_profile, "native_ref2015"
            )
            raise ActorLost("native actor lost during metrics")

        with (
            patch(
                "partiflow.postprocess.chemistry.relax_candidate",
                side_effect=self.copy_relax,
            ) as relax,
            patch(
                "partiflow.postprocess.metrics.compute_metrics",
                side_effect=lose_actor,
            ) as metrics,
        ):
            with self.assertRaisesRegex(ActorLost, "lost during metrics"):
                runner._process_one(self.item, self.options)
        relax.assert_called_once()
        metrics.assert_called_once()

        published_state = (
            self.output_path.read_bytes(), self.output_path.stat().st_mtime_ns
        )
        reason = "native actor exited during metrics"
        recovered = runner.recover_actor_failure(
            self.item, self.options, reason
        )
        self.assertEqual(recovered["version"], 2)
        self.assertEqual(recovered["status"], "success")
        self.assertEqual(recovered["geometry_status"], "complete")
        self.assertEqual(recovered["pdb_path"], str(self.output_path))
        self.assertEqual(recovered["output_path"], str(self.output_path))
        self.assertEqual(
            recovered["identity"], saved_before_loss["identity"]
        )
        self.assertEqual(recovered["metrics_status"], "unavailable")
        self.assertEqual(recovered["filter_status"], "unavailable")
        self.assertEqual(
            recovered["decisions"]["native"]["status"], "unavailable"
        )
        self.assertEqual(recovered["metrics_report"]["metrics"], {})
        self.assertEqual(
            recovered["metrics_report"]["metric_errors"]["native_worker"],
            reason,
        )
        self.assertEqual(recovered["error"], reason)
        self.assertEqual(
            json.loads(self.report_path.read_text()), recovered
        )
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(
            (self.output_path.read_bytes(), self.output_path.stat().st_mtime_ns),
            published_state,
        )

        complete_metrics = {
            "status": "complete",
            "protocol": "complex",
            "metrics": {"surface_hydrophobicity": 0.2},
            "raw_metrics": {"Surface_Hydrophobicity": 0.2},
            "metric_errors": {},
            "unavailable_metrics": [],
        }
        with (
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
            patch(
                "partiflow.postprocess.metrics.compute_metrics",
                return_value=complete_metrics,
            ) as metrics,
        ):
            resumed = runner._process_one(self.item, self.options)
        relax.assert_not_called()
        metrics.assert_called_once()
        self.assertTrue(resumed["resumed"])
        self.assertEqual(resumed["status"], "success")
        self.assertEqual(resumed["geometry_status"], "complete")
        self.assertEqual(resumed["metrics_status"], "complete")
        self.assertEqual(resumed["error"], "")
        self.assertEqual(resumed["filter_status"], "passed")
        self.assertEqual(
            resumed["decisions"]["native"]["status"], "passed"
        )
        self.assertEqual(resumed["metrics_report"], complete_metrics)
        self.assert_identity(resumed)
        self.assertEqual(
            resumed["identity"], saved_before_loss["identity"]
        )
        self.assertEqual(
            (self.output_path.read_bytes(), self.output_path.stat().st_mtime_ns),
            published_state,
        )
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(json.loads(self.report_path.read_text()), resumed)

    def test_actor_loss_before_geometry_publication_is_failed(self):
        with (
            patch(
                "partiflow.postprocess.chemistry.relax_candidate",
                side_effect=ActorLost("native actor lost during relax"),
            ) as relax,
            patch("partiflow.postprocess.metrics.compute_metrics") as metrics,
        ):
            with self.assertRaisesRegex(ActorLost, "lost during relax"):
                runner._process_one(self.item, self.options)
        relax.assert_called_once()
        metrics.assert_not_called()
        self.assertFalse(self.report_path.exists())
        self.assertFalse(self.output_path.exists())

        result = runner.recover_actor_failure(
            self.item, self.options, "native actor exited before publication"
        )
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["geometry_status"], "failed")
        self.assertEqual(result["metrics_status"], "unavailable")
        self.assertEqual(result["filter_status"], "unavailable")
        self.assertEqual(result["sdk_profile"], "native_ref2015")
        self.assertEqual(result["pdb_path"], str(self.source))
        self.assertIsNone(result["output_path"])
        self.assertNotIn("identity", result)
        self.assertFalse(self.output_path.exists())
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertEqual(json.loads(self.report_path.read_text()), result)


if __name__ == "__main__":
    unittest.main()
