"""CPU checks against the actual packaged MPNN wrappers when assets exist."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = all(
    (
        importlib.util.find_spec(name) is not None
        for name in ("numpy", "torch", "biotite")
    )
)
ASSETS = all(
    (
        (ROOT / "proteinmpnn" / family / "v_48_020.pt").is_file()
        for family in ("vanilla", "soluble")
    )
)


NUMERIC = all(
    importlib.util.find_spec(name) is not None for name in ("numpy", "biotite")
)


@unittest.skipUnless(NUMERIC, "requires NumPy and Biotite")
class FixedPositionPolicyTests(unittest.TestCase):
    def setUp(self):
        from test_af2_contract import make_atoms
        from partiflow.common.records import Candidate

        self.atoms = make_atoms()
        self.candidate = Candidate(
            "sample", "task", 1, 1, 1, Path("input.pdb"), "L", 3, "ACA",
            contig_string="L7/L9A", tip_atoms="L9A:CA",
        )
        self.data = {"atoms": self.atoms, "context_chains": ["T"]}

    def test_coordinate_only_contig_and_tip_do_not_fix_identity(self):
        import numpy as np
        from partiflow.workflows.sequence import fixed_positions

        coordinates = self.atoms.coord.copy()
        names = self.atoms.res_name.copy()
        self.assertEqual(fixed_positions(self.candidate, self.data, {}), [])
        np.testing.assert_array_equal(self.atoms.coord, coordinates)
        np.testing.assert_array_equal(self.atoms.res_name, names)

    def test_identity_fixed_occurrences_include_insertion_codes(self):
        from partiflow.workflows.sequence import fixed_positions

        self.candidate.fixed_restypes = [
            {"chain": "L", "res_id": 9, "restype": "CYS"},
            {"chain": "L", "res_id": 9, "ins_code": "A", "restype": "ALA"},
        ]
        self.assertEqual(fixed_positions(self.candidate, self.data, {}), [2, 3])

    def test_five_identity_motif_positions_remain_fixed(self):
        import numpy as np
        from biotite import structure as struc
        from partiflow.common.records import Candidate
        from partiflow.workflows.sequence import fixed_positions
        from partiflow.workflows.structure import read_structure, write_structure_atomic

        atoms = struc.AtomArray(5)
        atoms.chain_id = np.full(5, "L")
        atoms.res_id = np.arange(1, 6)
        atoms.res_name = np.full(5, "ALA")
        atoms.atom_name = np.full(5, "CA")
        atoms.element = np.full(5, "C")
        atoms.coord = np.asarray([[3.8 * i, 0, 0] for i in range(5)])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "motif.pdb"
            write_structure_atomic(atoms, path)
            observed = read_structure(path)
        candidate = Candidate(
            "motif", "task", 1, 1, 1, Path("input.pdb"), "L", 5, "AAAAA",
            contig_string="L1-5", tip_atoms="L1:CA",
            fixed_restypes=[
                {"chain": "L", "res_id": i, "restype": "ALA"} for i in range(1, 6)
            ],
        )
        self.assertEqual(
            fixed_positions(candidate, {"atoms": observed, "context_chains": []}, {}),
            [1, 2, 3, 4, 5],
        )

    def test_known_context_positions_validate_without_fixing_binder(self):
        from partiflow.workflows.sequence import fixed_positions

        self.assertEqual(
            fixed_positions(self.candidate, self.data, {"fixed_positions": ["T:01"]}),
            [],
        )
        self.assertEqual(
            fixed_positions(self.candidate, self.data, {"fixed_positions": ["T:3", "L:03"]}),
            [3],
        )

    def test_unknown_chains_and_out_of_range_positions_are_candidate_errors(self):
        from partiflow.common.records import CandidateError
        from partiflow.workflows.sequence import fixed_positions

        for item in ("U:1", "L:4", "T:4", "L:9"):
            with self.subTest(item=item), self.assertRaises(CandidateError):
                fixed_positions(self.candidate, self.data, {"fixed_positions": [item]})

    def test_fixed_selector_duplicates_and_malformed_values_are_config_errors(self):
        from partiflow.common.records import CandidateError
        from partiflow.workflows.sequence import fixed_positions

        for values in (["L:1", "L:01"], ["L:1:2"], [":1"], ["L:0"], ["L:-1"]):
            with self.subTest(values=values), self.assertRaises(ValueError) as failure:
                fixed_positions(self.candidate, self.data, {"fixed_positions": values})
            self.assertNotIsInstance(failure.exception, CandidateError)

    def test_existing_disulfide_identity_dependency_stays_fixed(self):
        import numpy as np
        from biotite import structure as struc
        from partiflow.workflows.sequence import fixed_positions

        self.atoms.res_name[:10] = "CYS"
        self.atoms.atom_name[[4, 9]] = "SG"
        self.atoms.bonds = struc.BondList(
            len(self.atoms), np.asarray([[4, 9, 1]], dtype=np.uint32)
        )
        self.candidate.sequence = "CCA"
        self.assertEqual(fixed_positions(self.candidate, self.data, {}), [1, 2])

    def test_mismatched_fixed_identity_is_rejected(self):
        from partiflow.common.records import CandidateError
        from partiflow.workflows.sequence import fixed_positions

        self.candidate.fixed_restypes = [{"chain": "L", "res_id": 9, "restype": "ALA"}]
        with self.assertRaises(CandidateError):
            fixed_positions(self.candidate, self.data, {})


@unittest.skipUnless(
    DEPENDENCIES and ASSETS,
    "requires task environment and local ProteinMPNN/SolubleMPNN assets",
)
class NativeMPNNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        import torch

        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        from test_af2_contract import make_atoms
        from partiflow.common.records import Candidate
        from partiflow.workflows.structure import (
            write_structure_atomic,
            candidate_input,
        )
        from partiflow.workflows.sequence import MPNNBackend

        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        path = self.directory / "input.pdb"
        write_structure_atomic(make_atoms(), path)
        self.candidate = Candidate(
            "sample1",
            "task",
            1,
            1,
            1,
            path,
            "L",
            3,
            "ACA",
            fixed_restypes=[{"chain": "L", "res_id": 9, "restype": "CYS"}],
        )
        self.data = candidate_input(self.candidate)
        self.backend = MPNNBackend(device="cpu")

    def tearDown(self):
        self.backend.close()
        self.temporary.cleanup()

    def test_both_pretrained_families_preserve_fixed_cys_and_reuse_models(self):
        import numpy as np
        from partiflow.workflows.sequence import backend_settings, fixed_positions

        coordinates = self.data["atoms"].coord.copy()
        identities = self.data["atoms"].res_name.copy()
        positions = fixed_positions(self.candidate, self.data, {})
        self.assertEqual(positions, [2])
        for source in ("proteinmpnn", "solublempnn"):
            settings = backend_settings(source, {}, ROOT)
            first = self.backend.generate(
                self.candidate, self.data, settings, positions, 101
            )
            second = self.backend.generate(
                self.candidate, self.data, settings, positions, 101
            )
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)
            self.assertEqual(first[1], "C")
            self.assertNotIn("C", first[0] + first[2])
        self.assertEqual(self.backend.load_count, 2)
        self.assertEqual(self.backend.hit_count, 2)
        np.testing.assert_array_equal(self.data["atoms"].coord, coordinates)
        np.testing.assert_array_equal(self.data["atoms"].res_name, identities)

    def test_tied_positions_use_native_tied_sampling(self):
        from partiflow.workflows.sequence import backend_settings

        path = self.directory / "tied.json"
        path.write_text(json.dumps([["L:1", "L:3"]]))
        settings = backend_settings(
            "proteinmpnn", {}, ROOT, {"tied_positions_path": str(path)}
        )
        sequence = self.backend.generate(self.candidate, self.data, settings, [2], 103)
        self.assertEqual(sequence[0], sequence[2])
        self.assertEqual(sequence[1], "C")

    def test_pssm_applies_only_to_free_positions(self):
        from partiflow.workflows.sequence import backend_settings, ALPHABET

        path = self.directory / "pssm.json"
        glycine = [float(aa == "G") for aa in ALPHABET]
        path.write_text(
            json.dumps({"L": {"pssm_coef": [1.0] * 3, "pssm_bias": [glycine] * 3}})
        )
        settings = backend_settings(
            "proteinmpnn",
            {},
            ROOT,
            {"pssm_path": str(path), "pssm_bias": True, "pssm_multi": 1.0},
        )
        self.assertEqual(
            self.backend.generate(self.candidate, self.data, settings, [2], 107), "GCG"
        )

    def test_complete_fixed_interface_needs_no_model_load(self):
        from partiflow.workflows.sequence import backend_settings

        settings = backend_settings(
            "solublempnn", {}, ROOT, {"checkpoint": str(self.directory / "absent.pt")}
        )
        self.assertEqual(
            self.backend.generate(self.candidate, self.data, settings, [1, 2, 3], 109),
            "ACA",
        )
        self.assertEqual(self.backend.load_count, 0)

    def test_missing_checkpoint_is_fatal(self):
        from partiflow.workflows.sequence import backend_settings
        from partiflow.common.records import WorkflowError

        settings = backend_settings(
            "proteinmpnn", {}, ROOT, {"checkpoint": str(self.directory / "absent.pt")}
        )
        with self.assertRaises(WorkflowError):
            self.backend.generate(self.candidate, self.data, settings, [2], 113)

    def test_public_structure_smoke_executes_both_selected_mpnn_models(self):
        from partiflow.workflows.smoke import run_structure_smoke

        output = self.directory / "smoke"
        report = run_structure_smoke(
            self.candidate.pdb_path, "L", ROOT, ["proteinmpnn", "solublempnn"], output
        )
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["mpnn_model_cache_misses"], 2)
        for name in ("proteinmpnn", "solublempnn"):
            model = report["models"][name]
            self.assertEqual(model["status"], "complete")
            self.assertEqual(len(model["sequence"]), 3)
            self.assertNotIn("C", model["sequence"])
            self.assertIn(model["sequence"], Path(model["sequence_path"]).read_text())
        self.assertEqual(
            json.loads((output / "smoke.json").read_text())["status"], "complete"
        )

    def test_public_structure_smoke_persists_fatal_model_failure(self):
        from partiflow.workflows.smoke import run_structure_smoke
        from partiflow.common.records import WorkflowError

        output = self.directory / "failed_smoke"
        with self.assertRaises(WorkflowError):
            run_structure_smoke(
                self.candidate.pdb_path,
                "L",
                self.directory / "missing_models",
                ["proteinmpnn"],
                output,
            )
        self.assertEqual(
            json.loads((output / "smoke.json").read_text())["status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
