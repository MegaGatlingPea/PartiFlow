"""Focused CPU checks for the read-only PepInter graph/data auditor."""

import copy
import csv
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import data as harness
from partiflow.training.prepare_data import build_indexes
from partiflow.utils.dense_padding_data_loader import DensePaddingCollater


def graph_fixture(path):
    coords = torch.full((2, 37, 3), 1e-5)
    mask = torch.zeros((2, 37), dtype=torch.bool)
    lines = []
    for index, chain in enumerate(("A", "L")):
        for name, offset in (("N", (0, 0, 0)), ("CA", (1.2, 0.3, 0)),
                             ("C", (2.4, 0, 0)), ("O", (3.2, 0.6, 0)), ("CB", (1.2, -1, 0.7))):
            point = (offset[0], offset[1] + index * 8, offset[2])
            atom_index = harness.ATOM37.index(name)
            coords[index, atom_index] = torch.tensor(point)
            mask[index, atom_index] = True
            element = "N" if name == "N" else "O" if name == "O" else "C"
            lines.append(f"ATOM  {len(lines) + 1:5d} {name:^4s} ALA {chain}{1:4d}    "
                         f"{point[0]:8.3f}{point[1]:8.3f}{point[2]:8.3f}{1:6.2f}{90:6.2f}          {element:>2s}  \n")
    path.write_text("".join(lines) + "END\n")
    return Data(id="fixture", coords=coords, coord_mask=mask, chains=torch.tensor([0, 1]),
                residues=["ALA", "ALA"], residue_id=["A:ALA:1:", "L:ALA:1:"],
                residue_type=torch.tensor([0, 0]), residue_pdb_idx=torch.tensor([1, 1]),
                orig_chain_id=["A", "L"], is_peptide=torch.tensor([0, 1]), seq_pos=torch.arange(2).unsqueeze(-1))


def transformed(graph):
    sample = graph.clone()
    sample.coords_unmodified = sample.coords.clone()
    sample.coords_nm = sample.coords / 10
    sample.ca_cond_mask = torch.tensor([True, False])
    sample.latent_cond_mask = torch.tensor([True, False])
    sample.motif_mask = sample.coord_mask & sample.latent_cond_mask[:, None]
    sample.x_motif = sample.coords_nm * sample.motif_mask[..., None]
    sample.seq_motif_mask = sample.motif_mask.any(-1)
    sample.seq_motif = sample.residue_type * sample.seq_motif_mask
    sample.binder_ss_token = torch.tensor([3, 2])
    sample.binder_block_adj = torch.full((2, 2), 2, dtype=torch.long)
    sample.chemical_bond_type = torch.zeros((2, 2), dtype=torch.long)
    return sample


class DataAuditHarnessTests(unittest.TestCase):
    def test_fixed_sorted_subset_normalizes_valid_and_rejects_cross_split_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, size in (("train", 3), ("valid", 2), ("test", 1)):
                target = root / "raw" / split
                target.mkdir(parents=True)
                for index in reversed(range(size)):
                    (target / f"{split}_{index}.pdb").write_text("ATOM\n")
            build_indexes(root, verify_counts=False)
            full, _, _ = harness.index_audit(root, expected_counts={"train": 3, "val": 2, "test": 1},
                                             subset_counts={"train": 3, "val": 2})
            self.assertEqual([row["split"] for row in full], ["train"] * 3 + ["val"] * 2 + ["test"])
            index = root / "indexes/split.csv"
            full[-1]["pdbid"] = full[0]["pdbid"]
            with index.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(full[0]))
                writer.writeheader()
                writer.writerows(full)
            with self.assertRaisesRegex(harness.AuditError, "leakage"):
                harness.index_audit(root, expected_counts={"train": 3, "val": 2, "test": 1},
                                    subset_counts={"train": 3, "val": 2})

    def test_exact_comparison_rejects_every_value_dtype_or_metadata_change(self):
        expected = {"coords": torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
                    "chains": torch.tensor([0, 1]), "residues": ["ALA", "GLY"], "source_path": "same.pdb"}
        result = harness.compare(copy.deepcopy(expected), expected)
        self.assertEqual(result["tensor_elements"], 14)
        for field in expected:
            actual = copy.deepcopy(expected)
            if isinstance(actual[field], torch.Tensor):
                actual[field].reshape(-1)[0] += 1
            elif field == "residues":
                actual[field][1] = "CYS"
            else:
                actual[field] = "different.pdb"
            with self.subTest(field=field), self.assertRaises(harness.AuditError):
                harness.compare(actual, expected)
        actual = copy.deepcopy(expected)
        actual["coords"] = actual["coords"].double()
        with self.assertRaisesRegex(harness.AuditError, "dtype"):
            harness.compare(actual, expected)

    def test_atom37_coordinates_masks_residue_types_and_chain_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "fixture.pdb"
            graph = graph_fixture(source)
            result = harness.graph_audit(graph, {"pdbid": "fixture"}, source)
            self.assertEqual((result["residues"], result["present_atoms"]), (2, 10))
            for change in ("coordinate", "mask", "chain"):
                altered = graph.clone()
                if change == "coordinate":
                    altered.coords[0, 1, 0] += 0.1
                elif change == "mask":
                    altered.coord_mask[0, 1] = False
                else:
                    altered.is_peptide[0] = 1
                with self.subTest(change=change), self.assertRaises(harness.AuditError):
                    harness.graph_audit(altered, {"pdbid": "fixture"}, source)

    def test_transformed_units_motif_masks_and_native_collation_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            graph = graph_fixture(Path(directory) / "fixture.pdb")
            sample = transformed(graph)
            result = harness.transformed_audit(sample, graph)
            self.assertEqual(result["length"], 2)
            collater = DensePaddingCollater([sample], max_size=8)
            batch = collater([sample])
            result = harness.collated_audit(batch, sample, 8)
            self.assertEqual(result["coords_shape"], [1, 8, 37, 3])
            self.assertEqual(result["padded_residues"], 6)
            sample.coords_nm *= 2
            with self.assertRaisesRegex(harness.AuditError, "scale"):
                harness.transformed_audit(sample, graph)
            batch.mask_dict["coords"][0, 7] = True
            with self.assertRaisesRegex(harness.AuditError, "Padding mask"):
                harness.collated_audit(batch, transformed(graph), 8)

    def test_nonfinite_values_and_outputs_inside_inputs_are_rejected(self):
        with self.assertRaises(harness.AuditError):
            harness.finite({"coords": torch.tensor([float("nan")])})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "input"
            with self.assertRaises(harness.AuditError):
                harness.external_root(root / "reports", root)


if __name__ == "__main__":
    unittest.main()
