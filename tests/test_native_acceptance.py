"""Native metric worker failures must stay local to one completed candidate."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from partiflow.postprocess.records import WorkItem
from partiflow.postprocess.runner import process_candidates


class NativeMetricWorkerFailureTests(unittest.TestCase):
    def test_native_crash_preserves_published_geometry_and_cannot_pass_required_metric(
        self,
    ):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "sample1.pdb"
            source.write_text(
                "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \n"
                "ATOM      2  CA  MAA L   1       4.000   0.000   0.000  1.00  0.00           C  \nEND\n"
            )
            original = source.read_bytes()
            report_dir = root / "postprocess"
            report_path = report_dir / "sample1.json"
            entries = [{"type": "residue_template", "resid": 1, "resname": "MAA"}]
            item = WorkItem(
                source,
                "task",
                "sample1",
                binder_chain="L",
                topology_entries=entries,
                report_dir=report_dir,
                replace=False,
            )
            presets = {
                "native": {
                    "methods": ["rosetta"],
                    "reduction": "any_prediction",
                    "rules": [
                        {
                            "metric": "surface_hydrophobicity",
                            "op": "<",
                            "value": 0.3,
                            "scope": "complex",
                        }
                    ],
                }
            }

            def relax(candidate, output, **options):
                shutil.copyfile(candidate.path, output)
                return output, {"method": "openmm"}

            def native_crash(command, **options):
                saved = json.loads(report_path.read_text())
                self.assertEqual(saved["status"], "success")
                self.assertTrue(Path(saved["pdb_path"]).is_file())
                request = json.loads(Path(command[-2]).read_text())
                self.assertEqual(
                    command[1:3], ["-m", "partiflow.postprocess.metric_worker"]
                )
                self.assertEqual(
                    request["path"],
                    str((report_dir / "structures" / source.name).resolve()),
                )
                self.assertEqual(request["binder_chain"], "L")
                self.assertEqual(request["topology_entries"], entries)
                self.assertTrue(request["compute_buns"])
                self.assertEqual(options["env"]["CUDA_VISIBLE_DEVICES"], "")
                return subprocess.CompletedProcess(
                    command, -11, stdout="", stderr="native metric crash"
                )

            with (
                patch("partiflow.common.presets.load_presets", return_value=presets),
                patch(
                    "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
                ),
                patch("subprocess.run", side_effect=native_crash) as worker,
            ):
                result = process_candidates([item], filter=True, criteria=["native"])[0]
            worker.assert_called_once()
            self.assertEqual(result["status"], "success")
            self.assertEqual(result["metrics_status"], "unavailable")
            self.assertEqual(result["filter_status"], "unavailable")
            self.assertEqual(result["decisions"]["native"]["status"], "unavailable")
            self.assertEqual(result["metrics_report"]["metrics"], {})
            self.assertEqual(result["metrics_report"]["raw_metrics"], {})
            self.assertIn(
                "exited -11", result["metrics_report"]["metric_errors"]["native_worker"]
            )
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(Path(result["pdb_path"]).read_bytes(), original)
            saved = json.loads(report_path.read_text())
            self.assertEqual(saved["status"], "success")
            self.assertEqual(saved["filter_status"], "unavailable")
            self.assertIn("resume", saved)


if __name__ == "__main__":
    unittest.main()
