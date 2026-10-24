from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from partiflow.common.results import (
    BASE_FIELDS,
    TAIL_FIELDS,
    read_results,
    write_results,
)
from partiflow.postprocess.runner import run, run_postprocess_for_run


def pdb(*, x=0.0, code="ALA", chain="L"):
    return (
        f"ATOM      1  CA  {code:3s} {chain}   1    "
        f"{x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  \nEND\n"
    )


def relax(item, output, **options):
    output.write_text(pdb(x=5.0, code="GLY", chain=item.binder_chain))
    return output, {"method": "openmm"}


class PostprocessPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_task(self, name="task", count=1):
        task = self.root / name
        task.mkdir()
        rows, sources = [], []
        for index in range(1, count + 1):
            source = task / f"sample{index}.pdb"
            source.write_text(pdb(x=float(index)))
            row = {key: "" for key in BASE_FIELDS + TAIL_FIELDS}
            row.update(
                filename=source.name,
                base_sample_id=index,
                design_chain_id="L",
                design_sequence="A",
                token_length=1,
                fixed_restypes="[]",
                bond_constraints="[]",
                postprocess_status="not_run",
                design_chains='[{"chain":"L","modality":"protein","sequence":"A","residue_count":1}]',
            )
            sources.append(source)
            rows.append(row)
        write_results(task / "result.csv", rows)
        return task, sources

    def files(self, task):
        return {
            path.relative_to(task): path.read_bytes()
            for path in task.rglob("*")
            if path.is_file()
        }

    def fail_csv_replace(self, result_path, *, after_replace=False):
        replace = os.replace

        def fail(source, destination):
            if Path(destination) == result_path:
                if after_replace:
                    replace(source, destination)
                raise OSError("CSV publication failed")
            return replace(source, destination)

        return fail

    def assert_no_backups(self, task):
        self.assertFalse(list(task.glob(".postprocess-backup-*")))
        self.assertFalse(list(task.rglob("*.tmp")))

    def test_cli_csv_replace_failure_restores_original_files(self):
        task, _ = self.make_task(count=2)
        original = self.files(task)
        args = SimpleNamespace(
            paths=[self.root],
            task=None,
            config=None,
            filter=False,
            criteria=None,
            presets=None,
            rerun=False,
            plan=False,
        )
        with (
            patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=relax),
            patch(
                "partiflow.common.results.os.replace",
                side_effect=self.fail_csv_replace(task / "result.csv"),
            ),
            redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(run(args), 1)
        self.assertEqual(self.files(task), original)
        self.assertFalse(list((task / "postprocess").glob("*.json")))
        self.assert_no_backups(task)

    def test_csv_replace_failure_restores_files_with_copy_backups(self):
        task, _ = self.make_task()
        original = self.files(task)
        with (
            patch(
                "partiflow.postprocess.runner.os.link",
                side_effect=OSError("links unavailable"),
            ) as link,
            patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=relax),
            patch(
                "partiflow.common.results.os.replace",
                side_effect=self.fail_csv_replace(task / "result.csv"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "CSV publication failed"):
                run_postprocess_for_run(self.root)
        self.assertTrue(link.called)
        self.assertEqual(self.files(task), original)
        self.assert_no_backups(task)

    def test_rerun_failure_restores_previous_structure_csv_and_reports(self):
        task, _ = self.make_task(count=2)
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
        ):
            run_postprocess_for_run(self.root)
        original = self.files(task)

        def rerelax(item, output, **options):
            output.write_text(pdb(x=9.0, code="VAL"))
            return output, {"method": "openmm"}

        with (
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=rerelax
            ) as backend,
            patch(
                "partiflow.common.results.os.replace",
                side_effect=self.fail_csv_replace(
                    task / "result.csv", after_replace=True
                ),
            ),
        ):
            with self.assertRaisesRegex(OSError, "CSV publication failed"):
                run_postprocess_for_run(self.root, rerun=True)
        self.assertEqual(backend.call_count, 2)
        self.assertEqual(self.files(task), original)
        self.assert_no_backups(task)

    def test_resumed_report_update_is_rolled_back_on_csv_failure(self):
        task, _ = self.make_task()
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
        ):
            run_postprocess_for_run(self.root)
        original = self.files(task)
        with (
            patch("partiflow.postprocess.chemistry.relax_candidate") as backend,
            patch(
                "partiflow.common.results.os.replace",
                side_effect=self.fail_csv_replace(task / "result.csv"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "CSV publication failed"):
                run_postprocess_for_run(self.root)
        backend.assert_not_called()
        self.assertEqual(self.files(task), original)
        self.assert_no_backups(task)

    def test_candidate_failure_commits_other_candidates_and_retries_only_failure(self):
        task, sources = self.make_task(count=3)
        failed_original = sources[1].read_bytes()

        def partial(item, output, **options):
            if item.sample_id == "sample2":
                output.write_text("partial structure")
                raise ValueError("candidate chemistry failed")
            return relax(item, output, **options)

        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=partial
        ):
            results = run_postprocess_for_run(self.root)
        self.assertEqual(
            [result["sample_id"] for result in results],
            ["sample1", "sample2", "sample3"],
        )
        self.assertEqual(
            [result["status"] for result in results], ["success", "failed", "success"]
        )
        self.assertEqual(sources[1].read_bytes(), failed_original)
        fields, rows = read_results(task / "result.csv")
        self.assertEqual(fields, BASE_FIELDS + TAIL_FIELDS)
        self.assertEqual(
            [row["postprocess_status"] for row in rows],
            ["succeeded", "failed", "succeeded"],
        )
        self.assertEqual([row["design_sequence"] for row in rows], ["G", "A", "G"])
        self.assertIn("candidate chemistry failed", rows[1]["error"])
        summary = json.loads((task / "postprocess" / "summary.json").read_text())
        self.assertEqual(summary["relaxation_successes"], 2)
        successful = [sources[index].read_bytes() for index in (0, 2)]
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
        ) as backend:
            retried = run_postprocess_for_run(self.root)
        self.assertEqual(
            [call.args[0].sample_id for call in backend.call_args_list], ["sample2"]
        )
        self.assertEqual([result["status"] for result in retried], ["success"] * 3)
        self.assertEqual([sources[index].read_bytes() for index in (0, 2)], successful)
        self.assertEqual(read_results(task / "result.csv")[1][1]["error"], "")
        self.assert_no_backups(task)

    def test_later_task_csv_failure_keeps_committed_task_and_leaves_next_task(self):
        first, _ = self.make_task("a_task")
        second, _ = self.make_task("b_task", count=2)
        third, _ = self.make_task("c_task")
        first_original = self.files(first)
        second_original = self.files(second)
        third_original = self.files(third)
        with (
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
            ) as backend,
            patch(
                "partiflow.common.results.os.replace",
                side_effect=self.fail_csv_replace(second / "result.csv"),
            ),
        ):
            with self.assertRaisesRegex(OSError, "CSV publication failed"):
                run_postprocess_for_run(self.root)
        self.assertEqual(
            [call.args[0].task_name for call in backend.call_args_list],
            ["a_task", "b_task", "b_task"],
        )
        self.assertNotEqual(self.files(first), first_original)
        self.assertEqual(
            read_results(first / "result.csv")[1][0]["postprocess_status"], "succeeded"
        )
        self.assertEqual(
            json.loads((first / "postprocess" / "sample1.json").read_text())["status"],
            "success",
        )
        self.assertEqual(self.files(second), second_original)
        self.assertEqual(self.files(third), third_original)
        for task in (first, second, third):
            self.assert_no_backups(task)

    def test_successful_tasks_return_collected_order(self):
        self.make_task("b_task", count=2)
        self.make_task("a_task", count=2)
        with patch(
            "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
        ):
            results = run_postprocess_for_run(self.root)
        self.assertEqual(
            [(result["task_name"], result["sample_id"]) for result in results],
            [
                ("a_task", "sample1"),
                ("a_task", "sample2"),
                ("b_task", "sample1"),
                ("b_task", "sample2"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
