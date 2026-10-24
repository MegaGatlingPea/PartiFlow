from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from partiflow.postprocess.chemistry import relax_candidate
from partiflow.postprocess.records import WorkItem


def pdb(code="ALA"):
    return f"ATOM      1  CA  {code:3} L   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n"


class PostprocessRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "input.pdb"
        self.path.write_text(pdb())
        self.output = self.path.with_name("output.pdb")

    def tearDown(self):
        self.temp.cleanup()

    def item(self, **kwargs):
        return WorkItem(self.path, "task", "sample1", **kwargs)

    def test_nonstandard_residues_use_openmm(self):
        self.path.write_text(pdb("SAR"))
        with (
            patch(
                "partiflow.postprocess.chemistry.relax_with_topology_openmm",
                return_value=(str(self.output), {}),
            ) as mm,
            patch("partiflow.postprocess.chemistry.relax_with_rosetta") as ros,
        ):
            _, info = relax_candidate(self.item(), self.output)
        mm.assert_called_once()
        ros.assert_not_called()
        self.assertEqual(info["method"], "openmm")

    def test_explicit_rosetta_rejects_nonstandard(self):
        self.path.write_text(pdb("MAA"))
        with patch("partiflow.postprocess.chemistry.relax_with_rosetta") as ros:
            with self.assertRaises(ValueError):
                relax_candidate(self.item(method="rosetta"), self.output)
        ros.assert_not_called()

    def test_auto_rosetta_failure_retries_openmm(self):
        with (
            patch(
                "partiflow.postprocess.chemistry.relax_with_rosetta",
                side_effect=RuntimeError("backend unavailable"),
            ),
            patch(
                "partiflow.postprocess.chemistry.relax_with_topology_openmm",
                return_value=(str(self.output), {}),
            ) as mm,
        ):
            _, info = relax_candidate(self.item(), self.output)
        mm.assert_called_once()
        self.assertEqual(info["method"], "openmm")
        self.assertIn("backend unavailable", info["rosetta_error"])

    def test_explicit_backend_does_not_fallback(self):
        with (
            patch(
                "partiflow.postprocess.chemistry.relax_with_rosetta",
                side_effect=RuntimeError("backend unavailable"),
            ),
            patch("partiflow.postprocess.chemistry.relax_with_topology_openmm") as mm,
        ):
            with self.assertRaises(RuntimeError):
                relax_candidate(self.item(method="rosetta"), self.output)
        mm.assert_not_called()

    def test_complex_covalent_topology_uses_openmm(self):
        with patch(
            "partiflow.postprocess.chemistry.relax_with_topology_openmm",
            return_value=(str(self.output), {}),
        ) as mm:
            relax_candidate(
                self.item(topology_entries=[{"type": "bicycle", "anchors": [1, 3, 5]}]),
                self.output,
            )
        mm.assert_called_once()

    def test_unknown_topology_is_explicit_failure(self):
        with self.assertRaises(ValueError):
            relax_candidate(
                self.item(topology_entries=[{"type": "invented"}]), self.output
            )

    def test_linear_flag_preserves_openmm_route(self):
        with patch(
            "partiflow.postprocess.chemistry.relax_with_topology_openmm",
            return_value=(str(self.output), {}),
        ) as mm:
            relax_candidate(self.item(relax_linear=False), self.output)
        mm.assert_called_once()

    def test_explicit_openmm_head_tail_is_strict(self):
        with self.assertRaises(ValueError):
            relax_candidate(
                self.item(
                    method="openmm",
                    topology_entries=[{"type": "head_to_tail", "anchors": [1, 2]}],
                ),
                self.output,
            )


if __name__ == "__main__":
    unittest.main()
