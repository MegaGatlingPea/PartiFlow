"""CPU numerical contracts; no successful model inference is simulated."""

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

NUMERIC = all(
    (importlib.util.find_spec(name) is not None for name in ("numpy", "biotite"))
)


def make_atoms():
    import numpy as np
    from biotite import structure as struc
    from partiflow.workflows.structure import read_structure, write_structure_atomic

    names = ("N", "CA", "C", "O", "CB")
    atoms = struc.AtomArray(30)
    atoms.chain_id = np.asarray(["L"] * 15 + ["T"] * 15)
    atoms.res_id = np.asarray([7] * 5 + [9] * 5 + [9] * 5 + [3] * 5 + [4] * 5 + [5] * 5)
    atoms.ins_code = np.asarray([""] * 10 + ["A"] * 5 + [""] * 15)
    atoms.res_name = np.asarray(["ALA"] * 5 + ["CYS"] * 5 + ["ALA"] * 5 + ["ALA"] * 15)
    atoms.atom_name = np.asarray(list(names) * 6)
    atoms.element = np.asarray(["N", "C", "C", "O", "C"] * 6)
    base = np.asarray([[0, 0, 0], [3.8, 1, 0], [7.2, 0.4, 1]])
    offsets = np.asarray(
        [[-1.2, 0.1, 0], [0, 0, 0], [1.2, 0.1, 0], [1.4, 1, 0], [0, -1, 0]]
    )
    atoms.coord = np.concatenate(
        [
            point + offsets + shift
            for shift in (np.zeros(3), np.asarray([0, 0, 5]))
            for point in base
        ]
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "input.pdb"
        write_structure_atomic(atoms, path)
        return read_structure(path)


@unittest.skipUnless(NUMERIC, "requires NumPy and Biotite in the task environment")
class AF2NumericTests(unittest.TestCase):
    def test_confidence_uses_native_units_and_binder_slice(self):
        import numpy as np
        from partiflow.workflows.af2 import extract_confidence

        model = SimpleNamespace(
            aux={
                "plddt": np.asarray([0.2, 0.9, 0.7]),
                "pae": np.full((3, 3), 6.2),
                "log": {"ptm": 0.6, "pae": 0.2, "i_ptm": 0.5, "i_pae": 0.3},
            }
        )
        confidence, pae = extract_confidence(
            model, "complex", design_start=1, design_length=2
        )
        self.assertAlmostEqual(confidence["plddt"], 80.0)
        self.assertAlmostEqual(confidence["plddt_normalized"], 0.8)
        self.assertAlmostEqual(confidence["pae_angstrom"], 6.2)
        self.assertAlmostEqual(confidence["ipae_angstrom"], 9.3)
        self.assertEqual(pae.shape, (3, 3))

    def test_missing_confidence_is_protocol_failure(self):
        from partiflow.workflows.af2 import extract_confidence, AF2SystemicError

        with self.assertRaises(AF2SystemicError):
            extract_confidence(
                SimpleNamespace(aux={}), "complex", design_start=0, design_length=1
            )

    def test_cyclic_offset_preserves_even_ties_and_context(self):
        import numpy as np
        from partiflow.workflows.af2 import signed_circular_offset, apply_cyclic_offset

        expected = np.asarray(
            [[0, -1, -2, 1], [1, 0, -1, -2], [2, 1, 0, -1], [-1, 2, 1, 0]]
        )
        np.testing.assert_array_equal(signed_circular_offset(4), expected)
        model = SimpleNamespace(_inputs={"residue_index": np.arange(6)})
        apply_cyclic_offset(model, design_start=2, design_length=4, enabled=True)
        np.testing.assert_array_equal(model._inputs["offset"][2:, 2:], expected)
        self.assertEqual(model._inputs["offset"][0, 5], -5)
        apply_cyclic_offset(model, design_start=2, design_length=4, enabled=False)
        self.assertNotIn("offset", model._inputs)

    def test_reward_ipae_weights_interfaces_equally(self):
        import numpy as np
        from partiflow.workflows.reward_backend import _interface_ipae

        pae = np.zeros((4, 4))
        pae[0, 3] = 1
        pae[3, 0] = 3
        pae[1:3, 3] = 9
        pae[3, 1:3] = 11
        self.assertAlmostEqual(
            _interface_ipae(
                pae, np.asarray([1, 2, 2, 3]), design_start=3, design_length=1
            ),
            6.0,
        )

    def test_private_mapping_restores_insertions_and_public_ids(self):
        import numpy as np
        from partiflow.workflows.af2_mapping import (
            af2_source_ledger,
            restore_af2_residue_identity,
            write_private_af2_pdb,
            validate_prepared_af2,
        )
        from partiflow.workflows.structure import read_structure, polymer_residue_groups

        source = make_atoms()
        source.set_annotation("is_design", source.chain_id == "L")
        ledger = af2_source_ledger(source, ["T", "L"], "L")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.pdb"
            write_private_af2_pdb(source, path, ledger=ledger)
            private = read_structure(path)
            rows = np.concatenate(
                (
                    np.flatnonzero(private.chain_id == "T"),
                    np.flatnonzero(private.chain_id == "L"),
                )
            )
            native = private[rows]
            restored = restore_af2_residue_identity(native, ledger, ["T", "L"])
            self.assertEqual(
                [
                    group["locator"]["res_id"]
                    for group in polymer_residue_groups(restored, "L")
                ],
                [7, 9, 9],
            )
            self.assertEqual(
                [
                    group["locator"]["ins_code"]
                    for group in polymer_residue_groups(restored, "L")
                ],
                ["", "", "A"],
            )
        selected = [
            record for chain in ledger["chains"] for record in chain["residues"]
        ]
        chains = np.asarray(["T"] * 3 + ["L"] * 3)
        ids = np.asarray([record["private_res_id"] for record in selected])
        prepared = SimpleNamespace(
            _pdb={"idx": {"chain": chains, "residue": ids}},
            _inputs={"residue_index": ids.copy()},
        )
        validate_prepared_af2(prepared, ledger, ["T", "L"])
        prepared._pdb["idx"]["residue"][0] += 1
        with self.assertRaises(ValueError):
            validate_prepared_af2(prepared, ledger, ["T", "L"])

    def test_fixed_identity_motifs_and_interface_use_occurrences(self):
        from partiflow.common.records import Candidate
        from partiflow.workflows.sequence import fixed_positions

        atoms = make_atoms()
        candidate = Candidate(
            "sample",
            "task",
            1,
            1,
            1,
            Path("input.pdb"),
            "L",
            3,
            "ACA",
            contig_string="L7",
            tip_atoms="L9A:CA",
            fixed_restypes=[{"chain": "L", "res_id": 9, "restype": "CYS"}],
        )
        data = {"atoms": atoms, "context_chains": ["T"]}
        self.assertEqual(fixed_positions(candidate, data, {}), [2])
        candidate.contig_string = ""
        candidate.tip_atoms = ""
        candidate.fixed_restypes = []
        self.assertEqual(
            fixed_positions(
                candidate, data, {"fixed_interface": {"enabled": True, "cutoff": 5.0}}
            ),
            [1, 2, 3],
        )

    def test_nonstandard_and_cross_chain_inputs_are_unsupported(self):
        import numpy as np
        from biotite import structure as struc
        from partiflow.workflows.af2 import (
            validate_capability,
            AF2UnsupportedChemistryError,
        )

        atoms = make_atoms()
        data = {
            "atoms": atoms,
            "design_chain": "L",
            "context_chains": ["T"],
            "bond_constraints": [],
        }
        validate_capability(data)
        atoms.res_name[0:5] = "MSE"
        atoms.mol_type[0:5] = "other"
        with self.assertRaises(AF2UnsupportedChemistryError):
            validate_capability(data)
        atoms = make_atoms()
        atoms.bonds = struc.BondList(
            len(atoms), np.asarray([[0, 15, 1]], dtype=np.uint32)
        )
        data["atoms"] = atoms
        with self.assertRaises(AF2UnsupportedChemistryError):
            validate_capability(data)

    def test_pair_rmsd_fits_ca_and_measures_all_heavy(self):
        import numpy as np
        from partiflow.workflows.metrics import _paired, _measure

        reference = make_atoms()
        predicted = reference.copy()
        predicted.coord += np.asarray([3, 7, -2])
        ca_ref, ca_pred = _paired(reference, predicted, ["L"], ("CA",))
        a, b = _paired(reference, predicted, ["L"])
        self.assertAlmostEqual(_measure(a, b, ca_ref, ca_pred), 0.0, places=5)
        predicted.coord[4] += np.asarray([0, 0, 1])
        a, b = _paired(reference, predicted, ["L"])
        self.assertAlmostEqual(
            _measure(a, b, ca_ref, ca_pred), np.sqrt(1 / 15), places=5
        )

    def test_pdb_roundtrip_preserves_terminal_and_insertion_bonds(self):
        import numpy as np
        from biotite import structure as struc
        from partiflow.workflows.structure import read_structure, write_structure_atomic

        source = make_atoms()
        bonds = source.bonds.as_array().tolist() + [[12, 0, 1]]
        source.bonds = struc.BondList(len(source), np.asarray(bonds, dtype=np.uint32))
        expected = {
            tuple(sorted((int(a), int(b)))) for a, b, _ in source.bonds.as_array()
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "roundtrip.pdb"
            write_structure_atomic(source, path)
            restored = read_structure(path)
            observed = {
                tuple(sorted((int(a), int(b)))) for a, b, _ in restored.bonds.as_array()
            }
            self.assertEqual(observed, expected)
            self.assertIn((7, 10), observed)
            self.assertIn((0, 12), observed)
            np.testing.assert_allclose(restored.coord, source.coord, atol=0.0006)

    def test_restoration_does_not_truncate_long_public_chain_ids(self):
        import numpy as np
        from partiflow.workflows.af2_mapping import (
            af2_source_ledger,
            write_private_af2_pdb,
            restore_af2_residue_identity,
            af2_output_suffix,
        )
        from partiflow.workflows.structure import read_structure, write_structure_atomic

        source = make_atoms()
        public = np.where(source.chain_id == "L", "PEPTIDE_LONG", "TARGET_LONG")
        for field in ("chain_id", "label_asym_id", "auth_asym_id"):
            source.set_annotation(field, public)
        source.set_annotation("is_design", public == "PEPTIDE_LONG")
        ledger = af2_source_ledger(
            source, ["TARGET_LONG", "PEPTIDE_LONG"], "PEPTIDE_LONG"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "private.pdb"
            write_private_af2_pdb(source, path, ledger=ledger)
            native = read_structure(path)
            context = ledger["chains"][0]["private_chain"]
            design = ledger["chains"][1]["private_chain"]
            native = native[
                np.concatenate(
                    (
                        np.flatnonzero(native.chain_id == context),
                        np.flatnonzero(native.chain_id == design),
                    )
                )
            ]
            restored = restore_af2_residue_identity(
                native, ledger, ["TARGET_LONG", "PEPTIDE_LONG"]
            )
            self.assertEqual(set(restored.chain_id), {"PEPTIDE_LONG", "TARGET_LONG"})
            self.assertEqual(
                af2_output_suffix(ledger, ["TARGET_LONG", "PEPTIDE_LONG"]), ".cif"
            )
            output = Path(directory) / "restored.cif"
            write_structure_atomic(restored, output)
            self.assertEqual(
                set(read_structure(output).chain_id), {"PEPTIDE_LONG", "TARGET_LONG"}
            )

    def test_candidate_modified_residue_is_explicitly_unsupported(self):
        from partiflow.common.records import Candidate
        from partiflow.workflows.structure import (
            candidate_input,
            write_structure_atomic,
            UnsupportedProteinChemistryError,
        )

        atoms = make_atoms()
        atoms.res_name[:5] = "MSE"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "modified.pdb"
            write_structure_atomic(atoms, path)
            candidate = Candidate("sample", "task", 1, 1, 1, path, "L", 3, "[MSE]CA")
            with self.assertRaises(UnsupportedProteinChemistryError) as caught:
                candidate_input(candidate)
            self.assertEqual(caught.exception.code, "unsupported_chemistry")
            self.assertIn("MSE", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
