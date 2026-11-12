"""CPU-only fixtures and public CLI mocks for the paired-validation helper."""

import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import validation as harness
from partiflow.cli import main as public_main
from partiflow.workflows import af2, sequence as sequence_module


def write_pdb(path):
    lines = []
    for chain_index, chain in enumerate(("T", "L")):
        for residue in range(1, 5):
            for atom, offset, element in (
                ("N", (0, 0, 0), "N"), ("CA", (1.2, 0.3, 0), "C"),
                ("C", (2.4, 0, 0), "C"), ("O", (3.2, 0.6, 0), "O"),
                ("CB", (1.2, -1, 0.7), "C"),
            ):
                xyz = (3.8 * residue + offset[0], 8 * chain_index + residue % 2 + offset[1],
                       0.2 * residue * residue + offset[2])
                lines.append(f"ATOM  {len(lines) + 1:5d} {atom:^4s} ALA {chain}{residue:4d}    "
                             f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}{1:6.2f}{90:6.2f}          {element:>2s}  \n")
    path.write_text("".join(lines) + "END\n")


class FakeAF2:
    def __init__(self, model_root, config):
        self.config = config
        self.cache_hit_count = 0
        self.cache_miss_count = 2

    def predict(self, data, variant, output, completed_keys=()):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        from partiflow.workflows.structure import write_structure_atomic

        for protocol in ("complex", "monomer"):
            for model in self.config["models"]:
                key = f"{protocol}_model{model}_sample0"
                if key in completed_keys:
                    continue
                atoms = data["atoms"].copy()
                if protocol == "monomer":
                    atoms = atoms[atoms.chain_id == "L"]
                atoms.b_factor[:] = 90
                target = output / f"seq{variant['seq_index']}_{key}.pdb"
                write_structure_atomic(atoms, target)
                length = len(set(zip(atoms.chain_id, atoms.res_id)))
                confidence = {"protocol": protocol, "plddt": 90.0, "plddt_normalized": 0.9,
                              "plddt_total": 0.9, "ptm": 0.8, "pae": 1 / 31,
                              "pae_angstrom": 1.0, "pae_matrix_mean_angstrom": 1.0}
                if protocol == "complex":
                    confidence.update(iptm=0.8, ipae=1 / 31, ipae_angstrom=1.0, pae_interaction=1.0)
                yield {"key": key, "protocol": protocol, "model": model, "sample": 0,
                       "status": "complete", "structure_path": str(target), "seed": 101,
                       "confidence": confidence, "source": variant["source"],
                       "source_index": variant["source_index"], "cyclic": False,
                       "_confidence_arrays": {"pae_matrix": np.ones((length, length))}}

    def close(self):
        pass


class FakeMPNN:
    hit_count = 0
    load_count = 0

    def generate(self, candidate, data, settings, positions, seed):
        return candidate.sequence

    def close(self):
        pass


class ValidationHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.source = self.base / "raw"
        for task in ("denovo", "motif"):
            directory = self.source / task
            directory.mkdir(parents=True)
            write_pdb(directory / "sample1.pdb")
            fixed = [{"chain": "L", "res_id": index, "ins_code": "", "restype": "ALA"}
                     for index in (2, 3)] if task == "motif" else []
            row = {"filename": "sample1.pdb", "design_chains": json.dumps([
                {"chain": "L", "modality": "protein", "sequence": "AAAA", "residue_count": 4}]),
                "base_sample_id": "1", "design_modality": "protein", "design_chain_id": "L",
                "token_length": "4", "reward_method": "", "global_rank": "", "design_sequence": "AAAA",
                "contig_string": "1/L2-3/1" if task == "motif" else "", "tip_atoms": "",
                "fixed_restypes": json.dumps(fixed), "bond_constraints": "[]", "postprocess_status": "skipped", "error": ""}
            with (directory / "result.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            (directory / "resolved_config.yaml").write_text(yaml.safe_dump({"tasks": {task: {"task_type": task}},
                "_runtime": {"planned_base_samples": 1, "replicas_per_base": 1, "planned_candidates": 1}}))
        self.output = self.base / "validation"
        self.manifest = harness.prepare(self.source, self.base / "models", self.output)

    def tearDown(self):
        self.temporary.cleanup()

    def fake_run(self, case="public", mode=None):
        with (patch.object(af2, "AF2Backend", FakeAF2), patch.object(sequence_module, "MPNNBackend", FakeMPNN),
              contextlib.redirect_stdout(io.StringIO())):
            if mode is not None:
                return harness.run_child(self.output, mode, mode)
            return public_main(harness.public_argv(self.manifest, case))

    def command_record(self):
        harness.save(self.output / "commands/public.json", {
            "returncode": 0, "status": "completed", "public_argv": harness.public_argv(self.manifest),
            "argv": [sys.executable, "-m", "partiflow", *harness.public_argv(self.manifest)], "unit_fixture": True})

    def test_prepare_creates_only_fresh_external_case_inputs_and_exact_protocol(self):
        self.assertEqual(self.manifest["cases"]["public"]["prediction_count"], 24)
        self.assertEqual(self.manifest["cases"]["public"]["sequence_slots"], 6)
        self.assertEqual(harness.yaml_read(self.output / "configs/public.yaml")["af2"]["models"], [2, 1])
        for case in ("public", "interruption"):
            self.assertFalse(any((self.output / case).rglob("validate")))
        with self.assertRaises(harness.AcceptanceError):
            harness.prepare(self.source, self.base / "models", self.output)
        with self.assertRaises(harness.AcceptanceError):
            harness.external_root(harness.CHECKOUT / "generated", harness.CHECKOUT)

    def test_full_cpu_mock_cli_has_24_predictions_and_six_matched_sequence_slots(self):
        self.assertEqual(self.fake_run(), 0)
        result = harness.audit_case(self.manifest, check_command=False)
        self.assertEqual((result["predictions"], result["sequence_slots"], result["matched_pairs"]), (24, 6, 12))
        self.assertTrue(result["software_complete"])
        self.assertEqual(result["scientific_decisions"]["passed"], 6)
        self.assertTrue(all(len(slot["fixed_positions"]) == 2 for slot in result["slots"] if slot["task"] == "motif"))

    def test_audit_rejects_confidence_pair_and_fixed_identity_corruption(self):
        self.assertEqual(self.fake_run(), 0)
        path = self.output / "public/motif/validate/af2_ig/refolds/sample1/seq1.json"
        baseline = path.read_bytes()
        for kind in ("metric", "pair", "identity", "array"):
            value = json.loads(baseline)
            first = next(iter(value["predictions"].values()))
            if kind == "metric":
                first["metrics"]["pae_angstrom"] = 5.0
            elif kind == "pair":
                value["paired_metrics"].pop("model1_sample0")
            elif kind == "identity":
                value["sequence"] = "ACAA"
            else:
                first["confidence"]["pae_matrix_mean_angstrom"] = 3.0
            harness.save(path, value)
            with self.subTest(kind=kind), self.assertRaises(harness.AcceptanceError):
                harness.audit_case(self.manifest, check_command=False)
            path.write_bytes(baseline)

    def test_scientific_failure_is_separate_from_software_completion(self):
        record = {"predictions": {"a": {"protocol": "complex", "metrics": {"plddt": 0.9, "ptm": 0.8,
            "iptm": 0.8, "ipae_normalized": 0.1}}, "b": {"protocol": "monomer", "metrics": {"plddt": 0.9}}},
            "paired_metrics": {"model1_sample0": {"bound_unbound_rmsd": 8.0}}}
        self.assertEqual(harness.scientific_decision(record), "failed")
        record["paired_metrics"]["model1_sample0"]["bound_unbound_rmsd"] = 1.0
        self.assertEqual(harness.scientific_decision(record), "passed")
        record["predictions"]["a"]["metrics"].pop("iptm")
        self.assertEqual(harness.scientific_decision(record), "unavailable")

    def test_actual_backend_factory_uses_ordered_ptm_and_distinct_complex_monomer_settings(self):
        backend = object.__new__(af2.AF2Backend)
        backend.models_numbers = [2, 1]
        backend.recycles = 3
        backend.params_dir = self.base / "models/af2"
        with patch.object(backend, "_import_factory") as importer:
            factory = importer.return_value
            for protocol in ("complex", "monomer"):
                backend._build_model(protocol)
            complex_settings, monomer_settings = [call.kwargs for call in factory.call_args_list]
        self.assertEqual(complex_settings["model_names"], ["model_2_ptm", "model_1_ptm"])
        self.assertEqual(monomer_settings["model_names"], ["model_2_ptm", "model_1_ptm"])
        self.assertFalse(complex_settings["use_multimer"])
        self.assertFalse(monomer_settings["use_multimer"])
        self.assertEqual(complex_settings["protocol"], "binder")
        self.assertTrue(complex_settings["use_initial_guess"])
        self.assertEqual(monomer_settings["protocol"], "hallucination")
        self.assertFalse(monomer_settings["initial_guess"])

    def test_interruption_hook_runs_after_complex_files_and_resume_only_yields_monomer(self):
        class Stopped(BaseException):
            pass
        with patch.object(harness.os, "_exit", side_effect=Stopped) as stop:
            with self.assertRaises(Stopped):
                self.fake_run("interruption", "interrupt")
        stop.assert_called_once_with(86)
        record = harness.read(self.output / "interruption/denovo/validate/af2_ig/refolds/sample1/seq0.json")
        self.assertEqual(set(record["predictions"]), {"complex_model1_sample0"})
        saved = harness.keep_files(harness.artifact_paths(self.manifest, "interruption"), self.output, "unit_complex")
        self.assertEqual(len(saved), 2)
        self.assertEqual(self.fake_run("interruption", "resume"), 0)
        observed = harness.read(self.output / "commands/resume.hook.json")
        self.assertEqual(observed["yielded_keys"], ["monomer_model1_sample0"])
        self.assertEqual(observed["predict_calls"][0]["completed_keys"], ["complex_model1_sample0"])
        harness.unchanged(saved)
        self.assertEqual(harness.audit_case(self.manifest, "interruption", check_command=False)["predictions"], 2)

    def test_real_model_free_subprocesses_block_constructors_and_reject_same_stat_changes(self):
        self.assertEqual(self.fake_run(), 0)
        self.command_record()
        result = harness.model_free(self.output)
        self.assertEqual(result["af2_constructors"], 0)
        self.assertEqual(result["mpnn_constructors"], 0)
        self.assertEqual(result["unchanged_prediction_files"], 48)
        self.assertEqual(len(result["same_size_mtime_mutations_rejected"]), 3)
        self.assertTrue(result["software_complete"])

    def test_execute_records_full_public_command_and_never_audits_failure(self):
        argv = [sys.executable, "-m", "partiflow", *harness.public_argv(self.manifest)]
        with patch.object(harness.subprocess, "run", return_value=type("Result", (), {"returncode": 7})()):
            with self.assertRaises(harness.AcceptanceError):
                harness.execute(self.manifest, "unit_failed", argv, public=harness.public_argv(self.manifest))
        result = harness.read(self.output / "commands/unit_failed.json")
        self.assertEqual((result["status"], result["returncode"]), ("failed", 7))
        self.assertEqual(result["argv"], argv)
        self.assertIn("models:", result["configs"]["public"])

    def test_import_and_prepare_do_not_import_a_model_or_gpu_runtime(self):
        script = ("import importlib.util, sys; "
                  f"spec = importlib.util.spec_from_file_location('validation_helper', {str(Path(harness.__file__))!r}); "
                  "module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
                  "assert not ({'torch','jax','pyrosetta','openmm'} & set(sys.modules))")
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
