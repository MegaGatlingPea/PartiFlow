from __future__ import annotations
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import partiflow.postprocess


class NativeChemicalSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(partiflow.postprocess.__file__).parent / "native.py"
        spec = importlib.util.spec_from_file_location("native_source_checks", path)
        cls.module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"pyrosetta": types.ModuleType("pyrosetta")}):
            spec.loader.exec_module(cls.module)

    def test_all_45_rows_are_score_only(self):
        matrix = self.module.support_matrix()["residues"]
        self.assertEqual(len(matrix), 45)
        self.assertTrue(
            all(
                item["score_supported"] and not item["relax_supported"]
                for item in matrix.values()
            )
        )
        self.assertEqual(
            sum(item["params"] is not None for item in matrix.values()), 27
        )

    def test_atom_graph_charge_and_connections_match_sources(self):
        report = self.module.validate_generated_params_sources()
        self.assertEqual(len(report), 45)
        for code, item in report.items():
            self.assertGreater(item["source_atom_count"], 0, code)
            self.assertGreater(item["source_bond_count"], 0, code)
            self.assertTrue({"N", "C"}.issubset(item["source_external_bonds"]), code)
            self.assertTrue(Path(item["passive_pdb"]).is_file(), code)
            if item["params"]:
                self.assertEqual(
                    item["params_net_formal_charge"],
                    item["runtime_formal_charge"],
                    code,
                )

    def test_native_alias_patch_mappings_are_retained(self):
        rows = self.module.support_matrix()["residues"]
        self.assertIn("N_Methylation", rows["SAR"]["runtime_name"])
        self.assertEqual(rows["MAA"]["handling"], "custom_alias")
        self.assertEqual(rows["SAR"]["handling"], "native_patch_alias")
        self.assertFalse(rows["MAA"]["strict_exact"])


if __name__ == "__main__":
    unittest.main()
