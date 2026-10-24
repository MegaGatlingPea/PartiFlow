from __future__ import annotations
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from partiflow.cluster.alignment import greedy_clusters, prepare_structure, score_pair


TM_OUTPUT = """TM-score= 0.40000 (normalized by length of Structure_1: L=20, d0=0.5)
TM-score= 0.80000 (normalized by length of Structure_2: L=10, d0=0.5)
"""


class ClusterScoreTests(unittest.TestCase):
    def test_mean_normalization_and_exact_cpu_command(self):
        completed = subprocess.CompletedProcess([], 0, TM_OUTPUT, "")
        with patch(
            "partiflow.cluster.alignment.subprocess.run", return_value=completed
        ) as execute:
            value = score_pair(
                Path("first.cif"),
                Path("second.cif"),
                executable="/tools/USalign",
                multimer=True,
                molecule_type="dna",
            )
        self.assertAlmostEqual(value, 0.6)
        args, options = execute.call_args
        self.assertEqual(
            args[0],
            [
                "/tools/USalign",
                str(Path("first.cif").resolve()),
                str(Path("second.cif").resolve()),
                "-mol",
                "RNA",
                "-mm",
                "1",
                "-ter",
                "0",
                "-split",
                "2",
            ],
        )
        self.assertFalse(options["shell"])
        self.assertTrue(options["check"])
        self.assertEqual(options["timeout"], 300)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        ):
            self.assertEqual(options["env"][name], "1")

    def test_bad_or_missing_scores_fail_explicitly(self):
        for output in (
            TM_OUTPUT.splitlines()[0],
            TM_OUTPUT + TM_OUTPUT.splitlines()[0],
            TM_OUTPUT.replace("0.40000", "nan"),
            TM_OUTPUT.replace("0.40000", "1.2"),
            TM_OUTPUT.replace("0.40000", "bad"),
        ):
            with (
                self.subTest(output=output),
                patch(
                    "partiflow.cluster.alignment.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 0, output, ""),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "TM-score"):
                    score_pair(Path("a"), Path("b"), executable="USalign")

    def test_subprocess_failures_keep_the_real_error(self):
        error = subprocess.CalledProcessError(7, ["USalign"], stderr="invalid trace")
        with patch("partiflow.cluster.alignment.subprocess.run", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "exit 7: invalid trace"):
                score_pair(Path("a"), Path("b"), executable="USalign")
        with patch(
            "partiflow.cluster.alignment.subprocess.run",
            side_effect=subprocess.TimeoutExpired("USalign", 300),
        ):
            with self.assertRaisesRegex(RuntimeError, "US-align failed"):
                score_pair(Path("a"), Path("b"), executable="USalign")

    def test_greedy_largest_neighborhood_is_inclusive_and_stable(self):
        scores = np.array(
            [
                [1.0, 0.5, 0.1, 0.1],
                [0.5, 1.0, 0.6, 0.1],
                [0.1, 0.6, 1.0, 0.1],
                [0.1, 0.1, 0.1, 1.0],
            ]
        )
        self.assertEqual(
            greedy_clusters(["a", "b", "c", "d"], scores, 0.5), [[1, 0, 2], [3]]
        )
        tied = np.array(
            [
                [0.0, 0.5, 0.1, 0.1],
                [0.5, 0.0, 0.1, 0.1],
                [0.1, 0.1, 0.0, 0.5],
                [0.1, 0.1, 0.5, 0.0],
            ]
        )
        self.assertEqual(
            greedy_clusters(["a", "b", "c", "d"], tied, 0.5), [[0, 1], [2, 3]]
        )
        self.assertEqual(greedy_clusters([], np.empty((0, 0)), 0.5), [])
        self.assertEqual(
            greedy_clusters(
                ["a", "b"], np.array([[1.0, 0.5], [0.5, 1.0]]), np.nextafter(0.5, 1.0)
            ),
            [[0], [1]],
        )

    def test_greedy_rejects_invalid_matrices_and_thresholds(self):
        for matrix in (
            np.ones((1, 2)),
            np.array([[1.0, 0.2], [0.3, 1.0]]),
            np.array([[1.0, np.nan], [np.nan, 1.0]]),
            np.full((2, 2), 1.1),
        ):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                greedy_clusters(["a", "b"], matrix, 0.5)
        for threshold in (True, float("nan"), -0.1, 1.1, "0.5"):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                greedy_clusters(["a"], np.ones((1, 1)), threshold)


@unittest.skipUnless(
    importlib.util.find_spec("biotite"), "Biotite structural reader is unavailable"
)
class ClusterTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def structure(self):
        from biotite.structure import AtomArray
        from biotite.structure.io import pdbx

        atoms = AtomArray(6)
        atoms.set_annotation(
            "chain_id", np.array(["LONG_LABEL"] * 3 + ["R"] * 2 + ["context"])
        )
        atoms.res_id = [10, 10, 20, 2, 2, 1]
        atoms.res_name = ["MSE", "MSE", "ALA", "A", "A", "ALA"]
        atoms.atom_name = ["CA", "CB", "CA", "C3'", "P", "CA"]
        atoms.element = ["C", "C", "C", "C", "P", "C"]
        atoms.hetero = [True, True, False, False, False, False]
        atoms.coord = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [3.8, 0.0, 0.0],
                [1.0, 2.0, 0.0],
                [1.0, 3.0, 0.0],
                [99.0, 99.0, 99.0],
            ]
        )
        file = pdbx.CIFFile()
        pdbx.set_structure(file, atoms)
        file.block["atom_site"]["auth_asym_id"] = pdbx.CIFColumn(["X"] * 6)
        path = self.root / "source.cif"
        file.write(path)
        return path

    def test_cif_final_labels_modified_anchors_and_context_exclusion(self):
        from biotite.structure.io import pdbx

        source = self.structure()
        output = self.root / "trace.cif"
        chains = (
            {"chain": "LONG_LABEL", "modality": "protein", "residue_count": 2},
            {"chain": "R", "modality": "rna", "residue_count": 1},
        )
        self.assertEqual(prepare_structure(source, chains, output), ("protein", "rna"))
        atoms = pdbx.get_structure(
            pdbx.CIFFile.read(output), model=1, use_author_fields=False
        )
        self.assertEqual(atoms.atom_name.tolist(), ["CA", "CA", "C3'"])
        self.assertEqual(atoms.res_name.tolist(), ["MSE", "ALA", "A"])
        self.assertFalse(atoms.hetero.any())
        np.testing.assert_allclose(
            atoms.coord, [[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [1.0, 2.0, 0.0]]
        )

    def test_temporary_pdb_keeps_design_chains_separate_with_dense_ids(self):
        from biotite.structure.io import pdb

        output = self.root / "trace.pdb"
        prepare_structure(
            self.structure(),
            (
                {"chain": "LONG_LABEL", "modality": "protein"},
                {"chain": "R", "modality": "dna"},
            ),
            output,
        )
        atoms = pdb.get_structure(pdb.PDBFile.read(output), model=1)
        self.assertEqual(atoms.chain_id.tolist(), ["A", "A", "B"])
        self.assertEqual(atoms.res_id.tolist(), [1, 2, 1])

    def test_missing_anchors_counts_and_unsupported_chains_are_errors(self):
        source = self.structure()
        output = self.root / "trace.cif"
        cases = (
            ({"chain": "missing", "modality": "protein"}, "absent"),
            ({"chain": "R", "modality": "protein"}, "exactly one CA"),
            (
                {"chain": "LONG_LABEL", "modality": "protein", "residue_count": 3},
                "count mismatch",
            ),
            ({"chain": "LONG_LABEL", "modality": "ligand"}, "unsupported"),
        )
        for chain, message in cases:
            with self.subTest(chain=chain), self.assertRaisesRegex(ValueError, message):
                prepare_structure(source, (chain,), output)
        with self.assertRaisesRegex(ValueError, "differ"):
            prepare_structure(
                source, ({"chain": "LONG_LABEL", "modality": "protein"},), source
            )


if __name__ == "__main__":
    unittest.main()
