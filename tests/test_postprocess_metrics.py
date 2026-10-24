from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import partiflow.postprocess as package
from partiflow.postprocess.metrics import (
    canonical_metrics,
    _compute_metrics as compute_metrics,
)
from partiflow.postprocess.records import refresh_locators


class PostprocessMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "input.pdb"
        self.path.write_text(
            "ATOM      1  CA  SAR L   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n"
        )

    def tearDown(self):
        self.temp.cleanup()

    def native(self, pose):
        return SimpleNamespace(
            load_score_only_pose=Mock(return_value=(pose, {})),
            snapshot_coordinates=lambda pose: ((1.0, 2.0, 3.0),),
            _score_function_for_pose=lambda pose: (lambda pose: -1.0, []),
            _score_components=lambda function, pose: ({"total": -1.0}, {}, [], []),
        )

    def test_full_metrics_receive_exact_strict_native_pose_without_packing_or_relax(
        self,
    ):
        pose = object()
        native = self.native(pose)
        rosetta = SimpleNamespace(
            _chain_ids_in_pose=lambda pose: ["A", "L"],
            score_interface=Mock(
                return_value={
                    "ShapeComplementarity": 0.7,
                    "n_InterfaceHbonds": 3.0,
                    "n_InterfaceUnsatHbonds": 1.0,
                    "Surface_Hydrophobicity": 0.2,
                }
            ),
        )
        with (
            patch.object(package, "native", native, create=True),
            patch.object(package, "rosetta", rosetta, create=True),
        ):
            report = compute_metrics(self.path)
        call = rosetta.score_interface.call_args.kwargs
        self.assertIs(call["native_pose"], pose)
        self.assertTrue(call["native_no_pack"])
        self.assertFalse(call["relax_before_scoring"])
        self.assertFalse(report["packing_enabled"])
        self.assertEqual(report["metrics"]["interface_unsat_hbonds"], 1.0)
        self.assertEqual(report["unavailable_metrics"], [])

    def test_monomer_computes_energy_without_interface_analyzer(self):
        pose = object()
        rosetta = SimpleNamespace(
            _chain_ids_in_pose=lambda pose: ["L"], score_interface=Mock()
        )
        with (
            patch.object(package, "native", self.native(pose), create=True),
            patch.object(package, "rosetta", rosetta, create=True),
        ):
            report = compute_metrics(self.path)
        rosetta.score_interface.assert_not_called()
        self.assertEqual(report["protocol"], "monomer")
        self.assertEqual(report["metrics"]["total_score"], -1.0)

    def test_nonfinite_required_metrics_are_omitted(self):
        result = canonical_metrics(
            {
                "ShapeComplementarity": float("nan"),
                "n_InterfaceHbonds": 2.0,
                "n_InterfaceUnsatHbonds": None,
                "Surface_Hydrophobicity": float("inf"),
            }
        )
        self.assertNotIn("shape_complementarity", result)
        self.assertNotIn("surface_hydrophobicity", result)
        self.assertNotIn("interface_unsat_hbonds", result)
        self.assertEqual(result["interface_hbonds"], 2.0)

    def test_atom_locators_follow_final_serial_numbers(self):
        output = self.path.with_name("output.pdb")
        output.write_text(self.path.read_text().replace("ATOM      1", "ATOM      9"))
        row = {
            "fixed_restypes": '[{"chain":"L","res_id":1,"restype":"SAR"}]',
            "bond_constraints": '[{"left":{"chain":"L","res_id":1,"atom":"CA","atom_id":1}}]',
        }
        refreshed = refresh_locators(self.path, output, row)
        import json

        self.assertEqual(
            json.loads(refreshed["bond_constraints"])[0]["left"]["atom_id"], 9
        )
        self.assertEqual(json.loads(refreshed["fixed_restypes"])[0]["restype"], "SAR")

    def test_missing_declared_atom_is_explicit_error(self):
        output = self.path.with_name("output.pdb")
        output.write_text(self.path.read_text().replace(" CA ", " CB "))
        with self.assertRaises(ValueError):
            refresh_locators(
                self.path,
                output,
                {
                    "bond_constraints": '[{"chain":"L","res_id":1,"atom":"CA","atom_id":1}]'
                },
            )


if __name__ == "__main__":
    unittest.main()
