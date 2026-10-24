"""CPU checks of final design tables, structures, and deferred publication."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from partiflow.common.records import Candidate, Evaluation
from partiflow.common.results import (
    BASE_FIELDS,
    TAIL_FIELDS,
    read_results,
    write_results,
)
from partiflow.design.runtime import _prepare, _publish


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "inference" / "acceptance_raw_1" / "raw" / "sample1.pdb"
MATRIX = ROOT / "inference" / "acceptance_matrix_raw_1"
REFERENCE = ROOT / "pdbs" / "design" / "mdm2" / "mdm2_motif.pdb"
PUBLIC_BASE = (
    "filename",
    "design_chains",
    "base_sample_id",
    "design_modality",
    "design_chain_id",
    "token_length",
    "reward_method",
    "global_rank",
    "design_sequence",
    "contig_string",
    "tip_atoms",
    "fixed_restypes",
    "bond_constraints",
)
PUBLIC_TAIL = ("postprocess_status", "error")
STANDARD = dict(
    zip(
        "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)
NO_STRUCTURE = (
    "filename",
    "design_chains",
    "design_modality",
    "design_chain_id",
    "token_length",
    "global_rank",
    "design_sequence",
    "contig_string",
    "tip_atoms",
    "fixed_restypes",
    "bond_constraints",
)


def atoms(path):
    """Read actual PDB fields independently of the production constraint reader."""
    result = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            result.append(
                {
                    "serial": int(line[6:11]),
                    "atom": line[12:16].strip(),
                    "restype": line[17:20].strip(),
                    "chain": line[21:22],
                    "res_id": int(line[22:26]),
                    "ins_code": line[26:27].strip(),
                    "xyz": tuple(
                        float(line[start : start + 8]) for start in (30, 38, 46)
                    ),
                }
            )
    return result


def sequence(path, chain="L"):
    return "".join(
        STANDARD.get(atom["restype"], "[" + atom["restype"] + "]")
        for atom in atoms(path)
        if atom["chain"] == chain and atom["atom"] == "CA"
    )


def small_pdb(path, length=3, offset=0.0):
    """Select a few complete real residues and one context residue."""
    lines = []
    for line in RAW.read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain, resid = line[21:22], int(line[22:26])
        if not (chain == "L" and 1 <= resid <= length or chain == "A" and resid == 1):
            continue
        if chain == "L" and offset:
            xyz = [float(line[start : start + 8]) for start in (30, 38, 46)]
            line = (
                line[:30]
                + "".join(f"{value + offset:8.3f}" for value in xyz)
                + line[54:]
            )
        lines.append(line)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines + ["END"]) + "\n")


class ResultTableContractTests(unittest.TestCase):
    def test_exact_public_base_and_tail(self):
        self.assertEqual(BASE_FIELDS, PUBLIC_BASE)
        self.assertEqual(TAIL_FIELDS, PUBLIC_TAIL)

    def test_actual_metrics_are_sorted_and_missing_values_stay_blank(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            written = write_results(
                path,
                [
                    {
                        "filename": "sample1_L_rank1.pdb",
                        "base_sample_id": 1,
                        "zeta_metric": 0.8,
                        "alpha_metric": 80.0,
                    },
                    {
                        "filename": "sample1_L_rank2.pdb",
                        "base_sample_id": 1,
                        "alpha_metric": 75.0,
                    },
                ],
            )
            self.assertEqual(
                written, PUBLIC_BASE + ("alpha_metric", "zeta_metric") + PUBLIC_TAIL
            )
            fields, rows = read_results(path)
            self.assertEqual(fields, written)
            self.assertEqual(rows[0]["zeta_metric"], "0.8")
            self.assertEqual(rows[1]["zeta_metric"], "")
            self.assertEqual(rows[1]["alpha_metric"], "75.0")

    def test_raw_tables_do_not_gain_empty_reward_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            self.assertEqual(
                write_results(
                    path,
                    [
                        {
                            "filename": "sample1.pdb",
                            "base_sample_id": 1,
                            "postprocess_status": "disabled",
                        },
                    ],
                ),
                PUBLIC_BASE + PUBLIC_TAIL,
            )
            self.assertEqual(read_results(path)[0], PUBLIC_BASE + PUBLIC_TAIL)

    def test_explicit_unsorted_metrics_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_results(
                    Path(directory) / "result.csv",
                    [],
                    fields=PUBLIC_BASE + ("zeta_metric", "alpha_metric") + PUBLIC_TAIL,
                )

    def test_invalid_metrics_preserve_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            write_results(path, [{"base_sample_id": 1, "filename": "sample1.pdb"}])
            previous = path.read_bytes()
            for value in (float("nan"), float("inf"), True, "not numeric"):
                with self.subTest(value=value):
                    with self.assertRaises((ValueError, TypeError)):
                        write_results(path, [{"base_sample_id": 2, "metric": value}])
                    self.assertEqual(path.read_bytes(), previous)
                    self.assertFalse(path.with_name("result.csv.tmp").exists())

    def test_atomic_write_failures_preserve_previous_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.csv"
            write_results(path, [{"base_sample_id": 1, "filename": "sample1.pdb"}])
            previous = path.read_bytes()
            for operation in ("fsync", "replace"):
                with self.subTest(operation=operation):
                    with patch(
                        "partiflow.common.results.os." + operation,
                        side_effect=OSError("simulated publication failure"),
                    ):
                        with self.assertRaisesRegex(
                            OSError, "simulated publication failure"
                        ):
                            write_results(path, [{"base_sample_id": 2, "metric": 0.5}])
                    self.assertEqual(path.read_bytes(), previous)
                    self.assertFalse(path.with_name("result.csv.tmp").exists())


@unittest.skipUnless(
    RAW.is_file(), "Run the raw acceptance fixture before these CPU checks"
)
class DesignPublicationContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def prepare(self, counts, *, replicas=1, reward=False):
        tasks = {name: {"task_type": "denovo"} for name in counts}
        requests = [
            {"task_name": name} for name, count in counts.items() for _ in range(count)
        ]
        cfg = OmegaConf.create(
            {
                "output": {"root_dir": str(self.directory), "run_name": "published"},
                "search": {
                    "algorithm": "best-of-n" if reward else None,
                    "best_of_n": {"replicas": replicas},
                },
                "reward": {"method": "af2_ig"} if reward else None,
                "postprocess_runtime": {
                    "platform": "CPU",
                    "use_ray": False,
                    "filter": False,
                },
                "_model_dir": str(self.directory),
            }
        )
        root, actual_replicas = _prepare(cfg, requests, tasks)
        self.assertEqual(actual_replicas, replicas if reward else 1)
        return cfg, root, requests, tasks

    def candidate(
        self,
        root,
        task,
        base,
        ordinal,
        *,
        replica=1,
        length=3,
        enabled=False,
        metadata=None,
        topology=None,
    ):
        path = (
            root / task / ".state" / "candidates" / f"base{base}_replica{replica}.pdb"
        )
        small_pdb(path, length, offset=ordinal / 100)
        settings = {"postprocess_enabled": enabled, **(metadata or {})}
        return Candidate(
            f"{task}/{base}/{replica}",
            task,
            base,
            ordinal,
            replica,
            path,
            "L",
            length,
            sequence(path),
            metadata=settings,
            topology_entries=topology or [],
        )

    def assert_no_structure(self, row):
        for field in NO_STRUCTURE:
            self.assertEqual(row[field], "", field)

    def test_raw_publication_uses_real_final_identity_and_no_reward_fields(self):
        cfg, root, requests, tasks = self.prepare({"raw": 1})
        item = self.candidate(root, "raw", 1, 0)
        previous = item.pdb_path.read_bytes()
        with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
            self.assertEqual(_publish(cfg, root, requests, tasks, [item], []), 0)
        relax.assert_not_called()
        fields, rows = read_results(root / "raw" / "result.csv")
        self.assertEqual(fields, PUBLIC_BASE + PUBLIC_TAIL)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["filename"], "sample1.pdb")
        self.assertEqual(row["postprocess_status"], "disabled")
        self.assertEqual(row["error"], "")
        self.assertEqual(row["reward_method"], "")
        self.assertEqual(row["global_rank"], "")
        self.assertEqual(
            row["design_sequence"], sequence(root / "raw" / row["filename"])
        )
        self.assertEqual(row["token_length"], "3")
        self.assertEqual(json.loads(row["fixed_restypes"]), [])
        self.assertEqual(json.loads(row["bond_constraints"]), [])
        self.assertEqual((root / "raw" / row["filename"]).read_bytes(), previous)
        self.assertEqual({entry.name for entry in root.iterdir()}, {"raw"})

    def test_csv_publication_failure_leaves_no_final_structures_or_completed_state(
        self,
    ):
        cfg, root, requests, tasks = self.prepare({"interrupted": 2})
        candidates = [
            self.candidate(root, "interrupted", base, base - 1) for base in (1, 2)
        ]
        inputs = {item.pdb_path: item.pdb_path.read_bytes() for item in candidates}
        task = root / "interrupted"
        replace = os.replace
        attempted = []

        def fail_csv(source, destination, *args, **kwargs):
            if Path(destination) == task / "result.csv":
                attempted.append(Path(destination))
                raise OSError("simulated final CSV publication failure")
            return replace(source, destination, *args, **kwargs)

        with patch("partiflow.design.runtime.os.replace", side_effect=fail_csv):
            with self.assertRaisesRegex(
                OSError, "simulated final CSV publication failure"
            ):
                _publish(cfg, root, requests, tasks, candidates, [])
        self.assertEqual(attempted, [task / "result.csv"])
        self.assertEqual(list(task.glob("sample*.pdb")), [])
        self.assertFalse((task / "result.csv").exists())
        self.assertFalse((task / ".state" / "design.json").exists())
        for path, original in inputs.items():
            self.assertEqual(path.read_bytes(), original)

    def test_later_csv_publication_failure_preserves_previously_committed_task(self):
        cfg, root, requests, tasks = self.prepare({"committed": 1, "interrupted": 1})
        first = self.candidate(root, "committed", 1, 0)
        second = self.candidate(root, "interrupted", 1, 1)
        original = first.pdb_path.read_bytes()
        committed_paths = [
            root / "committed" / "sample1.pdb",
            root / "committed" / "result.csv",
            root / "committed" / ".state" / "design.json",
        ]
        committed = {}
        replace = os.replace

        def fail_later_csv(source, destination, *args, **kwargs):
            if Path(destination) == root / "interrupted" / "result.csv":
                committed.update({path: path.read_bytes() for path in committed_paths})
                raise OSError("simulated second-task CSV publication failure")
            return replace(source, destination, *args, **kwargs)

        with patch("partiflow.design.runtime.os.replace", side_effect=fail_later_csv):
            with self.assertRaisesRegex(
                OSError, "simulated second-task CSV publication failure"
            ):
                _publish(cfg, root, requests, tasks, [second, first], [])
        self.assertEqual(len(committed), 3)
        for path, contents in committed.items():
            self.assertEqual(path.read_bytes(), contents)
        self.assertEqual(committed_paths[0].read_bytes(), original)
        fields, rows = read_results(committed_paths[1])
        self.assertEqual(fields, PUBLIC_BASE + PUBLIC_TAIL)
        self.assertEqual([row["filename"] for row in rows], ["sample1.pdb"])
        state = json.loads(committed_paths[2].read_text())
        self.assertEqual(len(state["samples"]), 1)
        failed_task = root / "interrupted"
        self.assertEqual(list(failed_task.glob("sample*.pdb")), [])
        self.assertFalse((failed_task / "result.csv").exists())
        self.assertFalse((failed_task / ".state" / "design.json").exists())

    def test_best_of_n_keeps_every_valid_replica_with_stable_task_length_ranks(self):
        cfg, root, requests, tasks = self.prepare(
            {"alpha": 4, "beta": 1}, replicas=2, reward=True
        )
        specs = [
            ("alpha", 1, 20, 1, 3, 0.8),
            ("alpha", 1, 10, 2, 3, 0.8),
            ("alpha", 2, 30, 1, 4, 0.1),
            ("alpha", 2, 40, 2, 4, 0.6),
            ("alpha", 4, 50, 1, 3, 0.3),
            ("alpha", 4, 60, 2, 3, None),
            ("beta", 1, 70, 1, 3, 0.99),
            ("beta", 1, 80, 2, 3, 0.01),
        ]
        candidates, evaluations, originals = [], [], {}
        for task, base, ordinal, replica, length, score in specs:
            item = self.candidate(
                root, task, base, ordinal, replica=replica, length=length
            )
            candidates.append(item)
            originals[(task, base, replica)] = item.pdb_path.read_bytes()
            metrics = {"af2_plddt": 80.0}
            if replica == 1:
                metrics["af2_iptm"] = 0.8
            evaluations.append(
                Evaluation(
                    item.candidate_id,
                    score,
                    item.pdb_path,
                    item.sequence,
                    metrics,
                    "refold unavailable" if score is None else "",
                )
            )
        failures = [
            {
                "task_name": "alpha",
                "base_sample_id": 3,
                "error": "generation: decode unavailable",
            },
            {
                "task_name": "alpha",
                "base_sample_id": 3,
                "error": "generation: decode unavailable",
            },
        ]
        with (
            patch(
                "partiflow.workflows.reward.evaluate_candidates",
                side_effect=lambda group, *_: [
                    item
                    for item in reversed(evaluations)
                    if item.candidate_id
                    in {candidate.candidate_id for candidate in group}
                ],
            ) as evaluate,
            patch("partiflow.workflows.reward.close") as close,
            patch("partiflow.postprocess.chemistry.relax_candidate") as relax,
        ):
            self.assertEqual(
                _publish(
                    cfg, root, requests, tasks, list(reversed(candidates)), failures
                ),
                1,
            )
        self.assertEqual(sum(len(call.args[0]) for call in evaluate.call_args_list), 8)
        for call in evaluate.call_args_list:
            self.assertEqual(
                {item.task_name for item in call.args[0]},
                {Path(call.args[3]).parents[1].name},
            )
        close.assert_called_once_with()
        relax.assert_not_called()
        fields, alpha = read_results(root / "alpha" / "result.csv")
        self.assertEqual(fields, PUBLIC_BASE + ("af2_iptm", "af2_plddt") + PUBLIC_TAIL)
        self.assertEqual(
            [(row["base_sample_id"], row["global_rank"]) for row in alpha],
            [("1", "1"), ("1", "2"), ("2", "1"), ("2", "2"), ("3", ""), ("4", "3")],
        )
        expected = {
            "sample1_L_rank1.pdb": (1, 2),
            "sample1_L_rank2.pdb": (1, 1),
            "sample2_L_rank1.pdb": (2, 2),
            "sample2_L_rank2.pdb": (2, 1),
            "sample4_L_rank3.pdb": (4, 1),
        }
        for filename, (base, replica) in expected.items():
            self.assertEqual(
                (root / "alpha" / filename).read_bytes(),
                originals[("alpha", base, replica)],
            )
        self.assertEqual(len(list((root / "alpha").glob("sample*.pdb"))), 5)
        self.assertEqual(alpha[0]["af2_iptm"], "")
        self.assertEqual(alpha[1]["af2_iptm"], "0.8")
        failed = alpha[4]
        self.assert_no_structure(failed)
        self.assertEqual(failed["postprocess_status"], "not_run")
        self.assertEqual(failed["error"], "generation: decode unavailable")
        beta = read_results(root / "beta" / "result.csv")[1]
        self.assertEqual([row["global_rank"] for row in beta], ["1", "2"])
        self.assertEqual(len(list((root / "beta").glob("sample*.pdb"))), 2)
        from partiflow.workflows.validate import _budget

        expected_budget = {
            "planned_base_samples": 4,
            "replicas_per_base": 2,
            "planned_candidates": 8,
        }
        self.assertEqual(_budget(root / "alpha"), expected_budget)
        resolved = OmegaConf.load(root / "alpha" / "resolved_config.yaml")
        self.assertEqual(OmegaConf.to_container(resolved._runtime), expected_budget)
        saved = json.loads((root / "alpha" / ".state" / "design.json").read_text())
        self.assertEqual(saved["generation"], expected_budget)
        self.assertEqual(len(saved["samples"]), 5)
        self.assertEqual(len(alpha), 6)

    def test_reward_ranks_follow_actual_final_structure_lengths(self):
        cfg, root, requests, tasks = self.prepare({"ranked": 2}, reward=True)
        short = self.candidate(root, "ranked", 1, 0, length=3)
        longer = self.candidate(root, "ranked", 2, 1, length=4)
        folded = self.directory / "refold.pdb"
        small_pdb(folded, 4, offset=0.5)
        evaluations = [
            Evaluation(
                short.candidate_id, 0.9, folded, sequence(folded), {"af2_iptm": 0.9}
            ),
            Evaluation(
                longer.candidate_id,
                0.8,
                longer.pdb_path,
                longer.sequence,
                {"af2_iptm": 0.8},
            ),
        ]
        with (
            patch(
                "partiflow.workflows.reward.evaluate_candidates",
                return_value=evaluations,
            ),
            patch("partiflow.workflows.reward.close"),
        ):
            self.assertEqual(
                _publish(cfg, root, requests, tasks, [short, longer], []), 0
            )
        rows = read_results(root / "ranked" / "result.csv")[1]
        self.assertEqual([row["token_length"] for row in rows], ["4", "4"])
        self.assertEqual([row["global_rank"] for row in rows], ["1", "2"])

    def test_success_refreshes_constraints_to_final_atom_serials_and_generated_positions(
        self,
    ):
        cfg, root, requests, tasks = self.prepare({"fixed": 1}, reward=True)
        item = self.candidate(
            root,
            "fixed",
            1,
            0,
            enabled=True,
            metadata={"fixed_positions": [1]},
            topology=[
                {"type": "residue_template", "resid": 1, "resname": "GLU"},
                {"type": "head_to_tail", "anchors": [1, 3]},
            ],
        )
        before = {
            (atom["chain"], atom["res_id"], atom["ins_code"], atom["atom"]): atom
            for atom in atoms(item.pdb_path)
        }

        def relax(work, output, **options):
            self.assertEqual(options["platform"], "CPU")
            lines = []
            for index, line in enumerate(work.path.read_text().splitlines()):
                if not line.startswith(("ATOM  ", "HETATM")):
                    continue
                line = line[:6] + f"{500 + index * 3:5d}" + line[11:]
                if line[21:22] == "L":
                    xyz = [float(line[start : start + 8]) for start in (30, 38, 46)]
                    xyz = [
                        value + delta for value, delta in zip(xyz, (1.25, -0.5, 0.75))
                    ]
                    line = (
                        line[:30]
                        + "".join(f"{value:8.3f}" for value in xyz)
                        + line[54:]
                    )
                lines.append(line)
            output.write_text("\n".join(lines + ["END"]) + "\n")
            return output, {"method": "openmm"}

        evaluation = Evaluation(
            item.candidate_id,
            0.7,
            item.pdb_path,
            item.sequence,
            {"af2_iptm": 0.7, "af2_plddt": 82.0},
        )
        with (
            patch(
                "partiflow.workflows.reward.evaluate_candidates",
                return_value=[evaluation],
            ),
            patch("partiflow.workflows.reward.close"),
            patch(
                "partiflow.postprocess.chemistry.relax_candidate", side_effect=relax
            ) as backend,
        ):
            self.assertEqual(_publish(cfg, root, requests, tasks, [item], []), 0)
        backend.assert_called_once()
        row = read_results(root / "fixed" / "result.csv")[1][0]
        self.assertEqual(row["postprocess_status"], "succeeded")
        self.assertEqual(row["filename"], "sample1_L_rank1.pdb")
        self.assertEqual(row["reward_method"], "af2multimer")
        self.assertEqual(row["af2_iptm"], "0.7")
        self.assertEqual(row["af2_plddt"], "82.0")
        self.assertEqual(
            json.loads(row["fixed_restypes"]),
            [{"chain": "L", "res_id": 1, "restype": "GLU"}],
        )
        final = root / "fixed" / row["filename"]
        final_atoms = atoms(final)
        by_serial = {atom["serial"]: atom for atom in final_atoms}
        self.assertEqual(len(by_serial), len(final_atoms))
        for atom in final_atoms:
            key = (atom["chain"], atom["res_id"], atom["ins_code"], atom["atom"])
            old = before[key]
            expected = (
                tuple(
                    value + delta
                    for value, delta in zip(old["xyz"], (1.25, -0.5, 0.75))
                )
                if atom["chain"] == "L"
                else old["xyz"]
            )
            for actual, target in zip(atom["xyz"], expected):
                self.assertAlmostEqual(actual, target, places=3)
            self.assertNotEqual(atom["serial"], old["serial"])
        bonds = json.loads(row["bond_constraints"])
        self.assertEqual(len(bonds), 1)
        bond = bonds[0]
        self.assertEqual(bond["order"], "single")
        self.assertIs(bond["required_output"], True)
        endpoints = []
        for name in ("atom1", "atom2"):
            locator = bond[name]
            atom = by_serial[locator["atom_id"]]
            self.assertEqual(
                (
                    locator["chain"],
                    locator["res_id"],
                    locator.get("ins_code", ""),
                    locator["atom"],
                ),
                (atom["chain"], atom["res_id"], atom["ins_code"], atom["atom"]),
            )
            endpoints.append((atom["res_id"], atom["atom"]))
        self.assertEqual(set(endpoints), {(1, "N"), (3, "C")})
        connections = set()
        for line in final.read_text().splitlines():
            if line.startswith("CONECT"):
                serials = [
                    int(line[start : start + 5])
                    for start in range(6, len(line), 5)
                    if line[start : start + 5].strip()
                ]
                connections.update(
                    tuple(sorted((serials[0], other))) for other in serials[1:]
                )
        self.assertIn(
            tuple(sorted((bond["atom1"]["atom_id"], bond["atom2"]["atom_id"]))),
            connections,
        )

    def test_postprocess_failure_preserves_input_and_publishes_valid_base_filename(
        self,
    ):
        cfg, root, requests, tasks = self.prepare({"failure": 1})
        item = self.candidate(
            root, "failure", 1, 0, enabled=True, metadata={"fixed_positions": [1]}
        )
        original = item.pdb_path.read_bytes()

        def fail(work, output, **options):
            output.write_text("partial staged output")
            raise ValueError("simulated minimization failure")

        with patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=fail):
            self.assertEqual(_publish(cfg, root, requests, tasks, [item], []), 1)
        fields, rows = read_results(root / "failure" / "result.csv")
        self.assertEqual(fields, PUBLIC_BASE + PUBLIC_TAIL)
        row = rows[0]
        self.assertEqual(row["filename"], "sample1.pdb")
        self.assertEqual(row["postprocess_status"], "failed")
        self.assertIn("simulated minimization failure", row["error"])
        self.assertEqual(item.pdb_path.read_bytes(), original)
        self.assertEqual((root / "failure" / row["filename"]).read_bytes(), original)
        self.assertEqual(
            json.loads(row["fixed_restypes"]),
            [{"chain": "L", "res_id": 1, "restype": "GLU"}],
        )
        self.assertFalse(list((root / "failure").rglob(".postprocess-*")))

    def test_unresolved_fixed_position_has_no_invented_output_locators(self):
        cfg, root, requests, tasks = self.prepare({"unresolved": 1})
        item = self.candidate(
            root, "unresolved", 1, 0, metadata={"fixed_positions": [99]}
        )
        with patch("partiflow.postprocess.chemistry.relax_candidate") as relax:
            self.assertEqual(_publish(cfg, root, requests, tasks, [item], []), 1)
        relax.assert_not_called()
        row = read_results(root / "unresolved" / "result.csv")[1][0]
        self.assert_no_structure(row)
        self.assertEqual(row["postprocess_status"], "not_run")
        self.assertIn("Fixed residue 99 missing", row["error"])
        self.assertEqual(list((root / "unresolved").glob("sample*.pdb")), [])

    def test_refold_without_design_chain_has_no_invented_output_identity(self):
        cfg, root, requests, tasks = self.prepare({"unresolved": 1}, reward=True)
        item = self.candidate(root, "unresolved", 1, 0)
        folded = self.directory / "context_only_refold.pdb"
        small_pdb(folded)
        folded.write_text(
            "\n".join(
                line
                for line in folded.read_text().splitlines()
                if not (line.startswith(("ATOM  ", "HETATM")) and line[21:22] == "L")
            )
            + "\n"
        )
        evaluation = Evaluation(item.candidate_id, 0.8, folded, "", {"af2_iptm": 0.8})
        with (
            patch(
                "partiflow.workflows.reward.evaluate_candidates",
                return_value=[evaluation],
            ),
            patch("partiflow.workflows.reward.close"),
        ):
            self.assertEqual(_publish(cfg, root, requests, tasks, [item], []), 1)
        row = read_results(root / "unresolved" / "result.csv")[1][0]
        self.assert_no_structure(row)
        self.assertEqual(row["postprocess_status"], "not_run")
        self.assertTrue(row["error"])
        self.assertEqual(list((root / "unresolved").glob("sample*.pdb")), [])

    def test_unresolved_refold_bond_has_no_invented_structure_or_constraint_locators(
        self,
    ):
        cfg, root, requests, tasks = self.prepare({"unresolved": 1}, reward=True)
        item = self.candidate(
            root,
            "unresolved",
            1,
            0,
            topology=[{"type": "head_to_tail", "anchors": [1, 3]}],
        )
        folded = self.directory / "incomplete_refold.pdb"
        small_pdb(folded)
        folded.write_text(
            "\n".join(
                line
                for line in folded.read_text().splitlines()
                if not (
                    line.startswith("ATOM  ")
                    and line[21:22] == "L"
                    and int(line[22:26]) == 1
                    and line[12:16].strip() == "N"
                )
            )
            + "\n"
        )
        original = item.pdb_path.read_bytes()
        evaluation = Evaluation(
            item.candidate_id, 0.8, folded, item.sequence, {"af2_iptm": 0.8}
        )
        with (
            patch(
                "partiflow.workflows.reward.evaluate_candidates",
                return_value=[evaluation],
            ),
            patch("partiflow.workflows.reward.close"),
        ):
            self.assertEqual(_publish(cfg, root, requests, tasks, [item], []), 1)
        row = read_results(root / "unresolved" / "result.csv")[1][0]
        self.assert_no_structure(row)
        self.assertEqual(row["postprocess_status"], "not_run")
        self.assertIn("Declared bond atom missing", row["error"])
        self.assertEqual(list((root / "unresolved").glob("sample*.pdb")), [])
        self.assertEqual(
            (root / "unresolved" / "original" / "sample1_candidate1.pdb").read_bytes(),
            original,
        )


@unittest.skipUnless(
    all(
        (MATRIX / name / "result.csv").is_file()
        for name in ("denovo", "docking", "inv_folding", "motif", "packing")
    ),
    "Run all five raw task families before these CPU checks",
)
class RealDesignTaskFamilyContractTests(unittest.TestCase):
    def test_real_task_results_match_final_structures_and_fixed_identity_positions(
        self,
    ):
        source_residues = {
            atom["res_id"]: atom["restype"]
            for atom in atoms(REFERENCE)
            if atom["chain"] == "L" and atom["atom"] == "CA"
        }
        expected = {
            "denovo": (12, {}),
            "docking": (11, {position: position for position in range(1, 12)}),
            "inv_folding": (11, {}),
            "motif": (11, {position: position + 1 for position in range(3, 8)}),
            "packing": (11, {position: position for position in range(1, 12)}),
        }
        for name, (length, fixed_positions) in expected.items():
            with self.subTest(task=name):
                task = MATRIX / name
                with (task / "result.csv").open(newline="") as handle:
                    table = csv.DictReader(handle)
                    self.assertEqual(tuple(table.fieldnames), PUBLIC_BASE + PUBLIC_TAIL)
                    rows = list(table)
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["filename"], "sample1.pdb")
                self.assertEqual(row["postprocess_status"], "disabled")
                self.assertEqual(row["error"], "")
                self.assertEqual(row["reward_method"], "")
                self.assertEqual(row["global_rank"], "")
                final = task / row["filename"]
                final_atoms = atoms(final)
                self.assertTrue(final_atoms)
                self.assertTrue(
                    all(
                        math.isfinite(value)
                        for atom in final_atoms
                        for value in atom["xyz"]
                    )
                )
                residues = {
                    atom["res_id"]: atom
                    for atom in final_atoms
                    if atom["chain"] == "L" and atom["atom"] == "CA"
                }
                self.assertEqual(len(residues), length)
                self.assertEqual(row["token_length"], str(length))
                self.assertEqual(row["design_sequence"], sequence(final))
                chains = json.loads(row["design_chains"])
                self.assertEqual(len(chains), 1)
                self.assertEqual(chains[0]["chain"], "L")
                self.assertEqual(chains[0]["modality"], "protein")
                self.assertEqual(chains[0]["sub_modality"], "peptide")
                self.assertEqual(chains[0]["sequence"], sequence(final))
                self.assertEqual(chains[0]["residue_count"], length)
                fixed = json.loads(row["fixed_restypes"])
                self.assertEqual(
                    {entry["res_id"] for entry in fixed}, set(fixed_positions)
                )
                for entry in fixed:
                    self.assertEqual(entry["chain"], "L")
                    self.assertEqual(
                        entry["restype"], residues[entry["res_id"]]["restype"]
                    )
                    self.assertEqual(
                        entry["restype"],
                        source_residues[fixed_positions[entry["res_id"]]],
                    )
                self.assertEqual(json.loads(row["bond_constraints"]), [])


if __name__ == "__main__":
    unittest.main()
