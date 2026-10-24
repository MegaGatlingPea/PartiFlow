from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from partiflow.cluster.inputs import ClusterJob, discover_jobs, load_job
from partiflow.common.results import write_results


class ClusterInputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.task = self.root / "task"
        self.method = self.task / "validate" / "af2_ig"
        self.method.mkdir(parents=True)
        self.job = ClusterJob(self.task, self.method)

    def tearDown(self):
        self.temporary.cleanup()

    def results(self, names=("sample1",), **changes):
        rows = []
        for name in names:
            (self.task / (name + ".pdb")).write_text("END\n")
            row = dict(
                filename=name + ".pdb",
                design_chains=json.dumps(
                    [dict(chain="L", modality="protein", sequence="A", residue_count=1)]
                ),
            )
            rows.append(dict(row, **changes))
        write_results(self.task / "result.csv", rows)

    def successes(self, rows=()):
        fields = (
            "sample",
            "sequence_index",
            "source",
            "source_index",
            "criteria",
            "status",
            "reasons",
        )
        with (self.method / "success.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in rows)

    def row(
        self,
        sample="sample1",
        criteria="af2_test",
        status="passed",
        source="gt",
        index=0,
    ):
        return dict(
            sample=sample,
            sequence_index=index,
            source=source,
            source_index=1,
            criteria=criteria,
            status=status,
        )

    def postprocess(self, records):
        directory = self.task / "postprocess"
        directory.mkdir(exist_ok=True)
        for name, record in records.items():
            (directory / (name + ".json")).write_text(json.dumps(record))
        return directory

    def record(self, sample="sample1", status="success", decision="passed"):
        return dict(
            sample_id="internal_candidate",
            status=status,
            input_path="/previous/location/" + sample + ".pdb",
            pdb_path="/obsolete/location/" + sample + ".pdb",
            decisions={"rosetta_test": dict(status=decision)},
            metadata={},
        )

    def test_discovery_documented_levels_and_aliases(self):
        self.results()
        self.successes()
        success = self.method / "success"
        success.mkdir()
        for path in (
            self.root,
            self.task,
            self.task / "result.csv",
            self.task / "validate",
            self.method,
            self.method / "success.csv",
            success,
        ):
            with self.subTest(path=path):
                self.assertEqual(discover_jobs([path]), [self.job])
        alias = self.root / "alias"
        alias.symlink_to(self.task, target_is_directory=True)
        self.assertEqual(discover_jobs([self.root, self.task, alias]), [self.job])
        self.assertEqual(discover_jobs([self.root], tasks=["other"]), [])

    def test_slots_deduplicate_and_presets_stay_separate(self):
        self.results(("sample1", "sample2", "sample10"))
        self.successes(
            [
                self.row(),
                self.row(source="proteinmpnn", index=1),
                self.row("sample2", status="failed"),
                self.row("sample10", criteria="other"),
            ]
        )
        data = load_job(self.job)
        self.assertEqual(set(data.samples), {"sample1", "sample10"})
        self.assertEqual(
            data.groups, {"af2_test": ("sample1",), "other": ("sample10",)}
        )
        self.assertEqual(data.sources["af2_test"]["sample1"], ("proteinmpnn", "self"))
        self.assertEqual(data.samples["sample1"].structure, self.task / "sample1.pdb")
        self.assertEqual(
            load_job(self.job, criteria=["other"]).groups, {"other": ("sample10",)}
        )

    def test_default_discovery_ignores_postprocess(self):
        self.results()
        self.successes([self.row()])
        directory = self.postprocess({"sample1": self.record()})
        self.assertEqual(discover_jobs([self.task]), [self.job])
        self.assertEqual(discover_jobs([directory])[0].method, "postprocess")
        self.assertEqual(
            discover_jobs([directory])[0].report_dir, directory / "clusters"
        )

    def test_failed_only_and_empty_cohorts_do_not_read_result_inputs(self):
        self.successes([self.row(status="failed")])
        self.assertEqual(load_job(self.job).groups, {"af2_test": ()})
        self.successes()
        self.assertEqual(load_job(self.job).groups, {"": ()})
        self.assertEqual(
            load_job(self.job, criteria=["af2_test"]).groups, {"af2_test": ()}
        )
        (self.method / "summary.json").write_text(
            json.dumps({"criteria": ["af2_test"]})
        )
        self.assertEqual(load_job(self.job).groups, {"af2_test": ()})
        with self.assertRaisesRegex(ValueError, "unknown criteria"):
            load_job(self.job, criteria=["other"])

    def test_conflicting_slot_decisions_are_errors(self):
        self.results()
        self.successes([self.row(), self.row(status="failed")])
        with self.assertRaisesRegex(ValueError, "conflicting decisions"):
            load_job(self.job)

    def test_unselected_result_rows_are_not_inspected(self):
        self.results(("sample1", "sample2"))
        (self.task / "sample2.pdb").unlink()
        self.successes([self.row(), self.row("sample2", status="failed")])
        self.assertEqual(tuple(load_job(self.job).samples), ("sample1",))
        (self.task / "sample1.pdb").unlink()
        with self.assertRaisesRegex(ValueError, "structure is missing"):
            load_job(self.job)

    def test_result_paths_cannot_select_external_or_derived_structures(self):
        self.results()
        self.successes([self.row()])
        outside = self.root / "sample1.pdb"
        outside.write_text("END\n")
        (self.task / "validate" / "sample1.pdb").write_text("END\n")
        for filename in ("../sample1.pdb", str(outside), "validate/sample1.pdb"):
            with self.subTest(filename=filename):
                self.results(filename=filename)
                with self.assertRaisesRegex(ValueError, "generated task structure"):
                    load_job(self.job)
        self.results()
        (self.task / "sample1.pdb").unlink()
        (self.task / "sample1.pdb").symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "generated task structure"):
            load_job(self.job)

    def test_selected_design_metadata_and_duplicate_rows_are_checked(self):
        self.results(design_chains="[]")
        self.successes([self.row()])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            load_job(self.job)
        self.results()
        with (self.task / "result.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        write_results(
            self.task / "result.csv", [rows[0], dict(rows[0], error="different")]
        )
        with self.assertRaisesRegex(ValueError, "conflicting result rows"):
            load_job(self.job)

    def test_explicit_postprocess_record_directory_and_summary(self):
        self.results(("sample1", "sample2", "sample3"))
        records = {
            "sample1": self.record(),
            "sample2": self.record("sample2", decision="failed"),
            "sample3": self.record("sample3", status="failed"),
        }
        directory = self.postprocess(records)
        summary = directory / "summary.json"
        summary.write_text(json.dumps({"samples": list(records.values())}))
        for paths in ([directory], [summary], [directory / "sample1.json", summary]):
            with self.subTest(paths=paths):
                job = discover_jobs(paths)[0]
                self.assertEqual(
                    load_job(job).groups,
                    {"postprocess_success": ("sample1", "sample2")},
                )
                chosen = load_job(job, criteria=["rosetta_test"])
                self.assertEqual(chosen.groups, {"rosetta_test": ("sample1",)})
                self.assertEqual(
                    chosen.samples["sample1"].structure, self.task / "sample1.pdb"
                )

    def test_postprocess_refold_records_are_rejected(self):
        self.results()
        record = self.record()
        record["metadata"] = {"source_kind": "refold"}
        directory = self.postprocess({"sample1": record})
        with self.assertRaisesRegex(ValueError, "refold postprocess"):
            load_job(discover_jobs([directory])[0])
        record["metadata"] = {}
        record["input_path"] = "/old/task/validate/af2_ig/refolds/sample1.pdb"
        self.postprocess({"sample1": record})
        with self.assertRaisesRegex(ValueError, "derived structure"):
            load_job(discover_jobs([directory])[0])

    def test_empty_postprocess_summary_selects_an_empty_criterion(self):
        self.results()
        directory = self.postprocess({"summary": {"samples": []}})
        data = load_job(discover_jobs([directory])[0], criteria=["rosetta_test"])
        self.assertEqual(data.groups, {"rosetta_test": ()})
        self.assertFalse(data.samples)


if __name__ == "__main__":
    unittest.main()
