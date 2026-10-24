from __future__ import annotations
import contextlib
import csv
import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from partiflow.cli import parser
from partiflow.cluster import run
from partiflow.cluster.inputs import ClusterJob
from partiflow.cluster.runner import (
    WorkerOptions,
    _write_reports,
    process_job,
    run_paths,
)
from partiflow.common.results import write_results


USALIGN = os.environ.get("PARTIFLOW_TEST_USALIGN") or shutil.which("USalign")
if not USALIGN:
    local = Path(sys.executable).parent / "USalign"
    USALIGN = str(local) if local.is_file() and os.access(local, os.X_OK) else None


def backbone(path, length=12, shift=0.0, context=100.0):
    lines = []
    for index in range(length):
        angle = index * 1.745
        x, y, z = 2.3 * math.cos(angle) + shift, 2.3 * math.sin(angle), index * 1.5
        lines.append(
            f"ATOM  {index + 1:5d}  CA  ALA L{index + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  "
        )
    lines.append("TER")
    lines.append(
        f"ATOM  {length + 1:5d}  CA  ALA A   1    {context:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  "
    )
    path.write_text("\n".join(lines) + "\nEND\n")


class ClusterWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / "task"
        self.method = self.task / "validate" / "af2_ig"
        self.method.mkdir(parents=True)
        self.job = ClusterJob(self.task, self.method)
        self.scratch = self.root / "scratch"
        self.scratch.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def inputs(self, names=("sample1", "sample2"), passed=True):
        rows, success = [], []
        for index, name in enumerate(names):
            backbone(
                self.task / (name + ".pdb"),
                shift=index * 20.0,
                context=1000.0 + index * 200.0,
            )
            rows.append(
                dict(
                    filename=name + ".pdb",
                    design_chains=json.dumps(
                        [
                            dict(
                                chain="L",
                                modality="protein",
                                sequence="A" * 12,
                                residue_count=12,
                            )
                        ]
                    ),
                )
            )
            success.append(
                dict(
                    sample=name,
                    sequence_index=0,
                    source="gt",
                    source_index=1,
                    criteria="af2_test",
                    status="passed" if passed else "failed",
                )
            )
        write_results(self.task / "result.csv", rows)
        self.success_rows(success)

    def success_rows(self, rows):
        fields = (
            "sample",
            "sequence_index",
            "source",
            "source_index",
            "criteria",
            "status",
        )
        with (self.method / "success.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def reports(self):
        with (self.method / "clusters.csv").open() as handle:
            detail = list(csv.DictReader(handle))
        with (self.method / "cluster_summary.csv").open() as handle:
            summary = list(csv.DictReader(handle))
        return detail, summary

    def test_public_plan_has_no_writes_or_asset_requirement(self):
        self.inputs()
        original = sorted(self.root.rglob("*"))
        args = parser().parse_args(
            ["cluster", str(self.root), "--plan", "--usalign", "/missing/USalign"]
        )
        output = io.StringIO()
        with (
            contextlib.redirect_stdout(output),
            patch("partiflow.cluster.runner.resolve_executable") as resolve,
        ):
            self.assertEqual(run(args), 0)
        resolve.assert_not_called()
        self.assertIn("successful_backbones=2", output.getvalue())
        self.assertIn("pairs=1", output.getvalue())
        self.assertEqual(sorted(self.root.rglob("*")), original)

    def test_empty_selected_set_publishes_empty_details_and_zero_counts_without_assets(
        self,
    ):
        self.inputs(passed=False)
        with (
            contextlib.redirect_stdout(io.StringIO()),
            patch("partiflow.cluster.runner.resolve_executable") as resolve,
            patch("partiflow.cluster.alignment.prepare_structure") as prepare,
        ):
            self.assertEqual(run_paths([self.task], executable="/missing/USalign"), 0)
        resolve.assert_not_called()
        prepare.assert_not_called()
        details, summary = self.reports()
        self.assertEqual(details, [])
        self.assertEqual(len(summary), 1)
        for field in (
            "n_success_backbones",
            "n_clusters",
            "n_singletons",
            "largest_cluster",
        ):
            self.assertEqual(summary[0][field], "0")

    def test_empty_table_without_criteria_still_has_zero_statistics(self):
        self.inputs(names=())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_paths([self.task]), 0)
        details, summary = self.reports()
        self.assertEqual(details, [])
        self.assertEqual(summary[0]["n_clusters"], "0")
        self.assertEqual(summary[0]["criterion"], "")

    def test_natural_order_ties_threshold_and_shared_pair_work(self):
        self.inputs(("sample1", "sample10", "sample2"))
        with (self.method / "success.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        rows.extend(
            [
                dict(rows[0], source="proteinmpnn", sequence_index=1),
                dict(rows[0], criteria="other"),
                dict(rows[1], criteria="other"),
            ]
        )
        self.success_rows(rows)
        scores = {(0, 1): 0.5, (0, 2): 0.1, (1, 2): 0.1}

        def score(first, second, **options):
            a, b = (int(path.stem.removeprefix("design")) for path in (first, second))
            return scores[a, b]

        options = WorkerOptions((), 0.5, "/tools/USalign", "test version", self.scratch)
        with patch(
            "partiflow.cluster.alignment.score_pair", side_effect=score
        ) as compare:
            result = process_job(self.job, options)
        self.assertEqual(result["comparisons"], 3)
        self.assertEqual(compare.call_count, 3)
        details, summary = self.reports()
        selected = [row for row in details if row["criterion"] == "af2_test"]
        self.assertEqual(
            [row["sample"] for row in selected], ["sample1", "sample2", "sample10"]
        )
        self.assertEqual(
            [row["representative"] for row in selected],
            ["sample1", "sample1", "sample10"],
        )
        self.assertEqual(selected[1]["tm_to_representative"], "0.50000000")
        self.assertEqual(
            json.loads(selected[0]["passing_branches"]), ["proteinmpnn", "self"]
        )
        self.assertEqual(
            {row["criterion"]: row["n_clusters"] for row in summary},
            {"af2_test": "2", "other": "2"},
        )
        before = (self.method / "clusters.csv").read_bytes()
        with patch("partiflow.cluster.alignment.score_pair", side_effect=score):
            process_job(self.job, options)
        self.assertEqual((self.method / "clusters.csv").read_bytes(), before)

    def test_invalid_public_thresholds_and_workers_are_errors(self):
        for threshold in (0.0, -0.1, 1.1, float("nan"), float("inf"), True):
            with (
                self.subTest(threshold=threshold),
                self.assertRaisesRegex(ValueError, "tm-threshold"),
            ):
                run_paths([], threshold=threshold)
        for workers in (0, 17, True, 1.5):
            with (
                self.subTest(workers=workers),
                self.assertRaisesRegex(ValueError, "workers"),
            ):
                run_paths([], workers=workers)

    def test_missing_executable_does_not_publish_reports(self):
        self.inputs()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(ValueError, "not executable"),
        ):
            run_paths([self.task], executable=self.root / "missing")
        self.assertFalse((self.method / "clusters.csv").exists())

    def test_report_pair_restores_existing_files_if_second_publish_fails(self):
        detail, summary = (
            self.method / "clusters.csv",
            self.method / "cluster_summary.csv",
        )
        detail.write_text("old clusters\n")
        summary.write_text("old summary\n")
        replace = os.replace

        def publish(source, destination):
            if (
                str(source).endswith(".new")
                and Path(destination).name == "cluster_summary.csv"
            ):
                raise OSError("second publication failed")
            return replace(source, destination)

        with patch("partiflow.cluster.runner.os.replace", side_effect=publish):
            with self.assertRaisesRegex(OSError, "second publication failed"):
                _write_reports(self.method, [], [], self.scratch)
        self.assertEqual(detail.read_text(), "old clusters\n")
        self.assertEqual(summary.read_text(), "old summary\n")

    def test_two_cpu_processes_complete_empty_jobs_and_restore_environment(self):
        self.inputs(names=())
        other = self.root / "other"
        method = other / "validate" / "af2_ig"
        method.mkdir(parents=True)
        write_results(other / "result.csv", [])
        shutil.copyfile(self.method / "success.csv", method / "success.csv")
        environment = dict(os.environ)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run_paths([self.root], workers=2), 0)
        self.assertEqual(dict(os.environ), environment)
        self.assertTrue((method / "cluster_summary.csv").is_file())
        self.assertEqual(self.reports()[1][0]["n_clusters"], "0")

    def test_postprocess_reports_use_separate_directory_and_preserve_records(self):
        self.inputs(names=("sample1",))
        directory = self.task / "postprocess"
        directory.mkdir()
        record = directory / "sample1.json"
        record.write_text(
            json.dumps(
                dict(
                    status="success",
                    input_path="/old/sample1.pdb",
                    metadata={},
                    decisions={"rosetta_test": {"status": "passed"}},
                )
            )
        )
        before = record.read_bytes()
        args = parser().parse_args(
            ["cluster", str(record), "--criteria", "rosetta_test"]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
        self.assertEqual(record.read_bytes(), before)
        with (directory / "clusters" / "clusters.csv").open() as handle:
            details = list(csv.DictReader(handle))
        self.assertEqual(details[0]["sample"], "sample1")
        self.assertEqual(details[0]["method"], "postprocess")
        self.assertFalse((self.method / "clusters.csv").exists())

    @unittest.skipUnless(USALIGN, "PENDING: USalign executable unavailable")
    def test_real_usalign_clusters_current_designs_without_model_imports(self):
        self.inputs()
        script = """from partiflow.cli import main
import sys
assert main(sys.argv[1:]) == 0
assert not ({'torch', 'jax', 'pyrosetta', 'openmm'} & set(sys.modules))
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                "cluster",
                str(self.task),
                "--workers",
                "1",
                "--usalign",
                USALIGN,
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        details, summary = self.reports()
        self.assertEqual(summary[0]["n_success_backbones"], "2")
        self.assertEqual(summary[0]["n_clusters"], "1")
        self.assertEqual(details[1]["tm_to_representative"], "1.00000000")
        self.assertIn("US-align", details[0]["usalign_version"])

    @unittest.skipUnless(USALIGN, "PENDING: USalign executable unavailable")
    def test_real_usalign_mean_uses_both_lengths(self):
        from partiflow.cluster.alignment import prepare_structure, score_pair

        first, second = self.root / "long.pdb", self.root / "short.pdb"
        backbone(first, length=12)
        backbone(second, length=6, shift=20.0)
        a, b = self.root / "long.cif", self.root / "short.cif"
        prepare_structure(first, ({"chain": "L", "modality": "protein"},), a)
        prepare_structure(second, ({"chain": "L", "modality": "protein"},), b)
        self.assertAlmostEqual(
            score_pair(a, b, executable=USALIGN, molecule_type="protein"),
            0.75,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
