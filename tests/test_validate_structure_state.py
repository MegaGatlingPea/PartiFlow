"""Saved structure values and model-free filtering contracts."""

import contextlib
import io
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdbx
import yaml

from partiflow.common.records import WorkflowError
from partiflow.common.results import write_results
from partiflow.paths import config_dir
from partiflow.workflows import af2, sequence, validate
from partiflow.workflows.structure import read_structure, write_structure_atomic


def atoms_fixture():
    atoms = struc.AtomArray(20)
    atoms.chain_id = np.repeat(["T", "L"], 10)
    atoms.res_id = np.tile(np.repeat([1, 2], 5), 2)
    atoms.res_name = np.full(20, "ALA")
    atoms.atom_name = np.tile(["N", "CA", "C", "O", "CB"], 4)
    atoms.element = np.tile(["N", "C", "C", "O", "C"], 4)
    offsets = np.asarray(
        [[0, 0, 0], [1.2, 0.2, 0], [2.4, 0, 0], [3.2, 0.6, 0], [1.2, -1, 0]]
    )
    atoms.coord = np.concatenate(
        [
            offsets + np.asarray([3.7 * residue, 4 * chain, 0])
            for chain in range(2)
            for residue in range(2)
        ]
    )
    atoms.set_annotation("b_factor", np.zeros(20))
    atoms.set_annotation("occupancy", np.ones(20))
    atoms.bonds = struc.BondList(
        20,
        np.asarray(
            [
                [0, 1, 1],
                [1, 2, 1],
                [2, 5, 1],
                [10, 11, 1],
                [11, 12, 1],
                [12, 15, 1],
            ],
            dtype=np.uint32,
        ),
    )
    return atoms


def cif_atom_change(text, field, replacement):
    fields = []
    lines = text.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("_atom_site."):
            fields.append(line.split()[0].removeprefix("_atom_site."))
        if line.startswith("ATOM "):
            span = list(re.finditer(r"\S+", line))[fields.index(field)].span()
            assert span[1] - span[0] == len(replacement)
            lines[index] = line[: span[0]] + replacement + line[span[1] :]
            return "".join(lines)
    raise AssertionError("fixture has no atom_site row")


class StructureStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.task = Path(self.temporary.name) / "denovo"
        self.task.mkdir()
        self.config = Path(self.temporary.name) / "validate.yaml"
        self.config.write_text(
            yaml.safe_dump(
                {
                    "sequence": {
                        "seed": 101,
                        "sources": [{"source": "gt", "count": 1}],
                    },
                    "criteria": [],
                }
            )
        )
        (self.task / "resolved_config.yaml").write_text(
            yaml.safe_dump(
                {
                    "_runtime": {
                        "planned_base_samples": 1,
                        "replicas_per_base": 1,
                        "planned_candidates": 1,
                    }
                }
            )
        )
        self.model_dir = Path(self.temporary.name) / "models"

    def tearDown(self):
        self.temporary.cleanup()

    def write_inputs(self, suffix=".pdb", rows=None):
        path = self.task / ("sample1" + suffix)
        write_structure_atomic(atoms_fixture(), path)
        rows = rows or [
            {
                "filename": path.name,
                "design_chain_id": "L",
                "design_sequence": "AA",
                "token_length": 2,
            }
        ]
        write_results(self.task / "result.csv", rows)
        return path

    def args(self, mode="run", criteria=None, presets=None):
        return SimpleNamespace(
            paths=[self.task],
            config=self.config,
            task=None,
            source=None,
            criteria=criteria,
            presets=presets,
            limit=None,
            rerun=False,
            plan=mode == "plan",
            refilter=mode == "refilter",
            recompute_metrics=mode == "recompute",
            model_dir=self.model_dir,
        )

    def state(self):
        loaded = validate._load(self.args())
        _, rows = validate._tasks([self.task], None, loaded)[0]
        defaults = yaml.safe_load((config_dir() / "sequence.yaml").read_text())
        return {
            "config": {
                key: value for key, value in loaded.items() if key != "criteria"
            },
            "sequence_defaults": defaults,
            "sequence_input_files": validate._sequence_input_files(
                loaded, defaults, self.model_dir
            ),
            "budget": validate._budget(self.task),
            "model_dir": str(self.model_dir.resolve()),
            "inputs": [
                validate._input_snapshot(
                    validate._candidate(self.task, row, index), row
                )
                for index, row in enumerate(rows, 1)
            ],
        }

    @contextlib.contextmanager
    def no_models(self):
        with (
            patch.object(
                af2.AF2Backend,
                "__init__",
                side_effect=AssertionError("AF2 model called"),
            ) as fold,
            patch.object(
                sequence.MPNNBackend,
                "__init__",
                side_effect=AssertionError("MPNN model called"),
            ) as mpnn,
        ):
            yield
            fold.assert_not_called()
            mpnn.assert_not_called()

    def run_quiet(self, args):
        with contextlib.redirect_stdout(io.StringIO()):
            return validate.run(args)

    def reject_same_stat_change(self, path, change, expected_field):
        old_state = self.state()
        root = self.task / "validate/af2_ig"
        state_path = root / ".state/config.json"
        plan_path = root / ".state/plan.json"
        validate._atomic_json(state_path, old_state)
        validate._atomic_json(plan_path, {"old_plan": "retained"})
        before_files = {
            p: (p.read_bytes(), p.stat().st_mtime_ns) for p in (state_path, plan_path)
        }
        stat = path.stat()
        before = path.read_text()
        after = change(before)
        self.assertNotEqual(before, after)
        self.assertEqual(len(before.encode()), len(after.encode()))
        path.write_text(after)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(path.stat().st_size, stat.st_size)
        self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
        new_state = self.state()
        self.assertNotEqual(
            old_state["inputs"][0]["structure"][expected_field],
            new_state["inputs"][0]["structure"][expected_field],
        )
        with self.no_models():
            for mode in ("run", "plan", "refilter", "recompute"):
                with (
                    self.subTest(mode=mode),
                    self.assertRaisesRegex(WorkflowError, "changed.*--rerun"),
                ):
                    self.run_quiet(self.args(mode))
                for saved, (content, mtime) in before_files.items():
                    self.assertEqual(saved.read_bytes(), content)
                    self.assertEqual(saved.stat().st_mtime_ns, mtime)

    def test_pdb_coordinate_change_with_restored_stat_rejects_before_plan_write(self):
        path = self.write_inputs()
        self.reject_same_stat_change(
            path, lambda text: text.replace("   0.000", "   1.000", 1), "coordinates"
        )

    def test_pdb_identity_change_with_restored_stat_rejects_before_plan_write(self):
        path = self.write_inputs()
        self.reject_same_stat_change(
            path, lambda text: text.replace("ALA T", "GLY T", 1), "annotations"
        )

    def test_pdb_connection_change_with_restored_stat_rejects_before_plan_write(self):
        path = self.write_inputs()
        self.reject_same_stat_change(
            path,
            lambda text: text.replace("CONECT    1    2", "CONECT    1    4", 1),
            "bonds",
        )

    def test_cif_coordinate_change_with_restored_stat_rejects_before_plan_write(self):
        path = self.write_inputs(".cif")
        self.reject_same_stat_change(
            path, lambda text: cif_atom_change(text, "Cartn_x", "1.0"), "coordinates"
        )

    def test_cif_identity_change_with_restored_stat_rejects_before_plan_write(self):
        path = self.write_inputs(".cif")
        self.reject_same_stat_change(
            path,
            lambda text: cif_atom_change(text, "label_comp_id", "GLY"),
            "annotations",
        )

    def test_cif_connection_change_with_restored_stat_rejects_before_plan_write(self):
        path = self.write_inputs(".cif")
        self.reject_same_stat_change(
            path, lambda text: text.replace("SING", "DOUB", 1), "bonds"
        )

    def test_pdb_unresolved_connection_record_is_compared_directly(self):
        path = self.write_inputs()
        with path.open("a") as handle:
            handle.write("CONECT   98   99\n")
        self.reject_same_stat_change(
            path,
            lambda text: text.replace("CONECT   98   99", "CONECT   97   99"),
            "records",
        )

    def test_cif_relevant_tables_are_stored_directly(self):
        path = self.write_inputs(".cif")
        file = pdbx.CIFFile.read(path)
        block = next(iter(file.values()))
        block["chem_comp_atom"] = pdbx.CIFCategory(
            {"comp_id": ["ALA"], "atom_id": ["N"], "charge": ["0"]}
        )
        file.write(path)
        value = validate._structure_state(path)
        self.assertNotIn("error", value)
        self.assertEqual(value["tables"]["chem_comp_atom"]["charge"], ["0"])
        self.assertIn("atom_site", value["tables"])
        self.assertIn("chem_comp_bond", value["tables"])
        self.assertEqual(value["coordinates"], read_structure(path).coord.tolist())
        self.assertEqual(value["bonds"], read_structure(path).bonds.as_array().tolist())
        self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
        self.reject_same_stat_change(
            path,
            lambda text: text.replace(
                "_chem_comp_atom.charge    0", "_chem_comp_atom.charge    1", 1
            ),
            "tables",
        )

    def test_cif_missing_optional_b_factor_remains_json_comparable(self):
        path = self.write_inputs(".cif")
        file = pdbx.CIFFile.read(path)
        del next(iter(file.values()))["atom_site"]["B_iso_or_equiv"]
        file.write(path)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            value = validate._structure_state(path)
            self.assertNotIn("error", value)
            self.assertEqual(value["annotations"]["b_factor"], ["nan"] * 20)
            self.assertEqual(json.loads(json.dumps(value, allow_nan=False)), value)
            self.assertEqual(self.run_quiet(self.args("refilter")), 2)

    def test_nonstructural_pdb_comment_does_not_change_structure_values(self):
        path = self.write_inputs()
        with path.open("a") as handle:
            handle.write("REMARK previous\n")
        before = self.state()
        stat = path.stat()
        path.write_text(path.read_text().replace("REMARK previous", "REMARK replaced"))
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(before, self.state())

    def test_malformed_pdb_and_cif_fail_only_their_candidates(self):
        for suffix, content in (
            (".pdb", "not a protein structure\n"),
            (".cif", "data_bad\n#\n"),
        ):
            with self.subTest(suffix=suffix):
                task = self.task / suffix.removeprefix(".")
                task.mkdir()
                self.task, old_task = task, self.task
                try:
                    path = self.write_inputs(
                        suffix,
                        [
                            {
                                "filename": "sample1" + suffix,
                                "design_chain_id": "L",
                                "design_sequence": "AA",
                                "token_length": 2,
                            },
                            {
                                "filename": "sample2.pdb",
                                "design_chain_id": "L",
                                "design_sequence": "AA",
                                "token_length": 2,
                            },
                        ],
                    )
                    path.write_text(content)
                    write_structure_atomic(atoms_fixture(), self.task / "sample2.pdb")
                    self.assertIn("error", validate._structure_state(path))
                    with self.no_models():
                        self.assertEqual(self.run_quiet(self.args("refilter")), 2)
                    output = self.task / "validate/af2_ig/refolds"
                    failed = json.loads((output / "sample1/seq0.json").read_text())
                    other = json.loads((output / "sample2/seq0.json").read_text())
                    self.assertEqual(failed["status"], "failed")
                    self.assertTrue(failed["error"])
                    self.assertEqual(other["status"], "incomplete")
                finally:
                    self.task = old_task

    def test_preset_only_refilter_rejudges_saved_metrics_without_models(self):
        path = self.write_inputs()
        root = self.task / "validate/af2_ig"
        sample = root / "refolds/sample1"
        sample.mkdir(parents=True)
        expected = ["complex_model1_sample0", "monomer_model1_sample0"]
        predictions = {}
        for protocol, key in zip(("complex", "monomer"), expected):
            output = sample / (protocol + ".pdb")
            atoms = read_structure(path)
            write_structure_atomic(
                atoms if protocol == "complex" else atoms[atoms.chain_id == "L"], output
            )
            arrays = output.with_suffix(".npz")
            np.savez_compressed(
                arrays, pae_matrix=np.ones((4, 4) if protocol == "complex" else (2, 2))
            )
            predictions[key] = {
                "key": key,
                "protocol": protocol,
                "status": "complete",
                "metrics_status": "complete",
                "structure_path": str(output),
                "confidence_path": str(arrays),
                "metrics": {
                    "plddt": 0.9,
                    "ptm": 0.8,
                    "iptm": 0.8,
                    "ipae_normalized": 0.1,
                },
            }
        record = {
            "method": "af2_ig",
            "candidate_id": "sample1",
            "base_sample_id": 1,
            "seq_index": 0,
            "source": "gt",
            "source_label": "self",
            "source_index": 1,
            "sequence": "AA",
            "fixed_positions": [],
            "status": "complete",
            "expected_predictions": expected,
            "predictions": predictions,
            "paired_metrics": {"model1_sample0": {"bound_unbound_rmsd": 1.0}},
        }
        validate._atomic_json(root / ".state/config.json", self.state())
        validate._atomic_json(sample / "seq0.json", record)
        before = {
            Path(pred[field]): (
                Path(pred[field]).read_bytes(),
                Path(pred[field]).stat().st_mtime_ns,
            )
            for pred in predictions.values()
            for field in ("structure_path", "confidence_path")
        }
        with self.no_models():
            self.assertEqual(
                self.run_quiet(self.args("refilter", ["af2_ig_peptide"])), 0
            )
            passed = json.loads((sample / "seq0.json").read_text())
            self.assertEqual(passed["decisions"]["af2_ig_peptide"]["status"], "passed")
            presets = yaml.safe_load((config_dir() / "preset.yaml").read_text())
            presets["presets"]["af2_ig_peptide"]["rules"][0]["value"] = 0.95
            stricter = Path(self.temporary.name) / "stricter.yaml"
            stricter.write_text(yaml.safe_dump(presets))
            self.assertEqual(
                self.run_quiet(self.args("refilter", ["af2_ig_peptide"], stricter)), 0
            )
        failed = json.loads((sample / "seq0.json").read_text())
        self.assertEqual(failed["decisions"]["af2_ig_peptide"]["status"], "failed")
        self.assertEqual(failed["predictions"], record["predictions"])
        self.assertEqual(failed["paired_metrics"], record["paired_metrics"])
        for artifact, value in before.items():
            self.assertEqual(
                (artifact.read_bytes(), artifact.stat().st_mtime_ns), value
            )


if __name__ == "__main__":
    unittest.main()
