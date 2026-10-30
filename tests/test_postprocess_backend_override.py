"""Current explicit backend requests take precedence over saved task requests."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from partiflow.common.records import Candidate
from partiflow.common.results import BASE_FIELDS, TAIL_FIELDS, write_results
from partiflow.postprocess.chemistry import relax_candidate
from partiflow.postprocess.records import WorkItem
from partiflow.postprocess.runner import process_candidates, run_postprocess_for_run


def pdb():
    return "ATOM      1  CA  ALA L   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n"


def copy_backend(*, pdb_path, output_path, **options):
    shutil.copyfile(pdb_path, output_path)
    return output_path, {}


@contextmanager
def backends(*, openmm_error=None, rosetta_error=None):
    with (
        patch(
            "partiflow.postprocess.chemistry.relax_with_topology_openmm",
            side_effect=openmm_error or copy_backend,
        ) as openmm,
        patch(
            "partiflow.postprocess.chemistry.relax_with_rosetta",
            side_effect=rosetta_error or copy_backend,
        ) as rosetta,
        patch(
            "partiflow.postprocess.chemistry.relax_candidate", wraps=relax_candidate
        ) as relax,
    ):
        yield SimpleNamespace(openmm=openmm, rosetta=rosetta, relax=relax)


class PostprocessBackendOverrideTests(unittest.TestCase):
    def setUp(self):
        from postprocess_test_utils import inline_dispatch
        self.dispatch_patch = patch("partiflow.postprocess.dispatch.dispatch_items", side_effect=inline_dispatch)
        self.dispatch_patch.start()
        self.addCleanup(self.dispatch_patch.stop)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.counter = 0

    def tearDown(self):
        self.temp.cleanup()

    def input_case(self, kind, saved_method):
        self.counter += 1
        task = self.root / f"task{self.counter}"
        task.mkdir()
        source = task / "sample1.pdb"
        source.write_text(pdb())
        metadata = {"postprocess_method": saved_method, "postprocess_enabled": True}
        state = None
        if kind == "candidate":
            caller = Candidate(
                "candidate1",
                task.name,
                1,
                1,
                1,
                source,
                "L",
                1,
                "A",
                metadata=metadata,
            )
        elif kind == "work_item":
            caller = WorkItem(
                source,
                task.name,
                "sample1",
                method=saved_method,
                metadata=metadata,
                report_dir=task / "postprocess",
            )
        else:
            caller = None
            row = {name: "" for name in BASE_FIELDS + TAIL_FIELDS}
            row.update(
                filename=source.name,
                base_sample_id=1,
                design_chain_id="L",
                design_sequence="A",
                token_length=1,
                fixed_restypes="[]",
                bond_constraints="[]",
                postprocess_status="not_run",
            )
            write_results(task / "result.csv", [row])
            state = task / ".state" / "design.json"
            state.parent.mkdir()
            state.write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                **metadata,
                                "filename": source.name,
                                "binder_chain": "L",
                            }
                        ]
                    }
                )
            )
        invoke = (
            (lambda **options: run_postprocess_for_run(task, **options))
            if caller is None
            else (lambda **options: process_candidates([caller], **options))
        )
        return SimpleNamespace(
            run=invoke,
            caller=caller,
            source=source,
            state=state,
        )

    def assert_route(self, case, expected, **options):
        before = (
            asdict(case.caller) if case.caller is not None else case.state.read_bytes()
        )
        with backends() as backend:
            result = case.run(**options)[0]
        self.assertEqual(result["status"], "success", result.get("error"))
        self.assertEqual(result["postprocess"]["method"], expected)
        selected = backend.openmm if expected == "openmm" else backend.rosetta
        other = backend.rosetta if expected == "openmm" else backend.openmm
        selected.assert_called_once()
        other.assert_not_called()
        if case.caller is not None:
            self.assertEqual(asdict(case.caller), before)
        else:
            self.assertEqual(case.state.read_bytes(), before)
        return backend.relax.call_args.args[0]

    def test_current_config_and_kwargs_override_saved_methods_for_all_inputs(self):
        for kind in ("candidate", "work_item", "standalone"):
            for current, saved in (
                ("openmm", "auto"),
                ("openmm", "rosetta"),
                ("rosetta_topology", "auto"),
                ("rosetta_topology", "openmm"),
                ("openmm", "obsolete_backend"),
            ):
                for via in ("config", "kwargs"):
                    with self.subTest(kind=kind, current=current, saved=saved, via=via):
                        options = (
                            {"config": {"postprocess": {"method": current}}}
                            if via == "config"
                            else {"method": current}
                        )
                        item = self.assert_route(
                            self.input_case(kind, saved), current, **options
                        )
                        self.assertEqual(item.method, current)

    def test_kwargs_override_current_config_and_saved_request(self):
        for kind in ("candidate", "work_item", "standalone"):
            with self.subTest(kind=kind):
                self.assert_route(
                    self.input_case(kind, "rosetta"),
                    "openmm",
                    config={"method": "rosetta_topology"},
                    method="openmm",
                )

    def test_explicit_openmm_accepts_native_headtail_chemistry(self):
        source = self.root / "native.pdb"
        source.write_text(pdb().replace("ALA", "BAL"))
        item = WorkItem(source, "task", "native", method="openmm",
            topology_entries=[{"type": "head_to_tail", "anchors": [1, 2]}])
        with backends() as backend:
            output, info = relax_candidate(item, self.root / "output.pdb")
        backend.openmm.assert_called_once()
        backend.rosetta.assert_not_called()
        self.assertEqual(info["method"], "openmm")
        self.assertEqual(output.read_bytes(), source.read_bytes())

    def test_auto_preserves_each_saved_request_for_all_inputs(self):
        for kind in ("candidate", "work_item", "standalone"):
            for saved in ("auto", "openmm", "rosetta"):
                for options in ({}, {"config": {"method": "auto"}}):
                    with self.subTest(kind=kind, saved=saved, options=options):
                        expected = "openmm" if saved == "openmm" else "rosetta_topology"
                        item = self.assert_route(
                            self.input_case(kind, saved), expected, **options
                        )
                        self.assertEqual(
                            item.method, "auto" if saved == "auto" else expected
                        )

    def test_plan_normalizes_existing_aliases_without_changing_callers(self):
        for alias in (
            "rosetta",
            "pyrosetta",
            "pyrosetta_disulfide",
            "rosetta_disulfide",
            "rosetta_topology",
            "rosetta_cyclic",
            " ROSETTA ",
        ):
            for kind in ("candidate", "work_item", "standalone"):
                with self.subTest(alias=alias, kind=kind):
                    case = self.input_case(kind, alias)
                    before = (
                        asdict(case.caller)
                        if case.caller is not None
                        else case.state.read_bytes()
                    )
                    with patch("partiflow.postprocess.runner._process_one") as process:
                        result = case.run(plan=True)[0]
                    process.assert_not_called()
                    self.assertEqual(result["method"], "rosetta_topology")
                    self.assertFalse((case.source.parent / "postprocess").exists())
                    after = (
                        asdict(case.caller)
                        if case.caller is not None
                        else case.state.read_bytes()
                    )
                    self.assertEqual(after, before)
        result = process_candidates([], method=" OPENMM ", plan=True)
        self.assertEqual(result, [])

    def test_unknown_global_method_is_rejected_before_collecting_inputs(self):
        for plan in (False, True):
            for options in ({"method": "invented"}, {"config": {"method": "invented"}}):
                with self.subTest(plan=plan, options=options):
                    with patch(
                        "partiflow.postprocess.runner._collect_paths"
                    ) as collect:
                        with self.assertRaisesRegex(
                            ValueError, "Unsupported postprocess method"
                        ):
                            run_postprocess_for_run(
                                self.root / "missing", plan=plan, **options
                            )
                    collect.assert_not_called()
        with self.assertRaisesRegex(ValueError, "Unsupported postprocess method"):
            process_candidates([], method="invented", plan=True)

    def test_unknown_saved_request_is_rejected_before_plan_or_processing(self):
        for kind in ("candidate", "work_item", "standalone"):
            for plan in (False, True):
                with self.subTest(kind=kind, plan=plan):
                    case = self.input_case(kind, "invented")
                    original = case.source.read_bytes()
                    with patch("partiflow.postprocess.runner._process_one") as process:
                        with self.assertRaisesRegex(
                            ValueError, "Unsupported postprocess method"
                        ):
                            case.run(plan=plan)
                    process.assert_not_called()
                    self.assertEqual(case.source.read_bytes(), original)
                    self.assertFalse((case.source.parent / "postprocess").exists())
                    self.assertFalse(
                        list(case.source.parent.glob(".postprocess-backup-*"))
                    )

    def test_explicit_backend_failure_never_falls_back(self):
        for kind in ("candidate", "work_item", "standalone"):
            for method in ("openmm", "rosetta_topology"):
                with self.subTest(kind=kind, method=method):
                    case = self.input_case(kind, "auto")
                    original = case.source.read_bytes()
                    errors = {
                        method.replace("rosetta_topology", "rosetta")
                        + "_error": RuntimeError("backend failed")
                    }
                    with backends(**errors) as backend:
                        result = case.run(method=method)[0]
                    self.assertEqual(result["status"], "failed")
                    self.assertIn("backend failed", result["error"])
                    selected = backend.openmm if method == "openmm" else backend.rosetta
                    other = backend.rosetta if method == "openmm" else backend.openmm
                    selected.assert_called_once()
                    other.assert_not_called()
                    self.assertEqual(case.source.read_bytes(), original)

    def test_plan_and_invalid_method_do_not_load_models(self):
        script = """
from pathlib import Path
import sys
from partiflow.postprocess.records import WorkItem
from partiflow.postprocess.runner import process_candidates
item = WorkItem(Path('/missing/input.pdb'), 'task', 'sample1', method='rosetta')
assert process_candidates([item], method='openmm', plan=True)[0]['method'] == 'openmm'
try:
    process_candidates([item], method='invented', plan=True)
except ValueError:
    pass
else:
    raise AssertionError('unknown method accepted')
assert not ({'torch', 'jax', 'pyrosetta', 'openmm', 'ray'} & set(sys.modules))
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
