"""Failure-sensitive checks for the explicit CPU chemistry acceptance harness."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from acceptance import chemistry


class ChemistryAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def native_run(self):
        cases = []
        for index in range(45):
            cases.append(dict(residue_code=f"X{index:02}", path=f"/fixtures/{index}.pdb", status="PASS",
                identity=dict(residues=[dict(type="B3G", formal_charge=0, atom_charges=[["CM", -0.18]],
                    heavy_atoms=[["CM", "C", -0.18, 1.0, 2.0, 3.0]],
                    heavy_bonds=[["CA", "CM", "SingleBond"]], connections=[[1, "N", 2, 2]])], heavy_atom_count=1),
                load=dict(runtime_residue_audit=[dict(runtime_name="B3G")], native_runtime=dict(status="success"),
                    residue_connection_audit=dict(status="success"), validated_bonds=[dict(status="present")],
                    input_heavy_atom_audit=dict(status="success", max_delta_angstrom=0)),
                energies={key: -2.0 for key in chemistry.ENERGIES}, energy_components=dict(fa_atr=-3.0),
                score_component_weights=dict(fa_atr=1.0), scorefxn_adjustments=[]))
        return dict(status="PASS", sdk="same-sdk", seed=2026, threads=1,
                    sdk_profile="native_ref2015", fixture_index="/fixtures/cases.json", cases=cases)

    def compare(self, baseline, current):
        first, second = self.root / "first.json", self.root / "second.json"
        chemistry.save(first, baseline)
        chemistry.save(second, current)
        return chemistry.compare_native(first, second)

    def test_native_comparison_accepts_same_actual_data(self):
        result = self.compare(self.native_run(), self.native_run())
        self.assertEqual(result["count"], 45)
        self.assertEqual(result["maximum_energy_delta"], 0)

    def test_native_comparison_rejects_chemical_changes(self):
        baseline = self.native_run()
        changes = {
            "native_type": lambda r: r["identity"]["residues"][0].update(type="ALA"),
            "heavy_coordinates": lambda r: r["identity"]["residues"][0]["heavy_atoms"][0].__setitem__(3, 1.001),
            "heavy_element": lambda r: r["identity"]["residues"][0]["heavy_atoms"][0].__setitem__(1, "N"),
            "partial_charge": lambda r: r["identity"]["residues"][0]["atom_charges"][0].__setitem__(1, 0.0),
            "formal_charge": lambda r: r["identity"]["residues"][0].update(formal_charge=1),
            "bond_graph": lambda r: r["identity"]["residues"][0].update(heavy_bonds=[]),
            "connection": lambda r: r["identity"]["residues"][0].update(connections=[]),
            "runtime": lambda r: r["load"].update(native_runtime=dict(status="failed")),
        }
        for name, change in changes.items():
            with self.subTest(name=name):
                changed = deepcopy(baseline)
                change(changed["cases"][0])
                with self.assertRaises(AssertionError):
                    self.compare(baseline, changed)

    def test_native_comparison_requires_coverage_controls_and_components(self):
        baseline = self.native_run()
        changes = (
            lambda r: r["cases"].pop(),
            lambda r: r["cases"][0].update(status="FAIL"),
            lambda r: r.update(seed=2027),
            lambda r: r.update(sdk="other-sdk"),
            lambda r: r["cases"][0]["energy_components"].update(fa_atr=-3.001),
            lambda r: r["cases"][0]["energy_components"].clear(),
        )
        for change in changes:
            changed = deepcopy(baseline)
            change(changed)
            with self.assertRaises(AssertionError):
                self.compare(baseline, changed)

    def write_pair(self, kind, distance):
        names = {"head_to_tail": ("N", "C"), "disulfide": ("SG", "SG"),
                 "mk8_staple": ("CE", "CE"), "nle_staple": ("CE", "CE")}[kind]
        path = self.root / "pair.pdb"
        lines = []
        for serial, (name, x) in enumerate(zip(names, (0.0, distance)), 1):
            lines.append(f"ATOM  {serial:5d} {name:^4s} ALA L{serial:4d}    {x:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00  0.00           C  \n")
        path.write_text("".join(lines) + "END\n")
        return path

    def test_each_native_connection_window_is_closed_and_enforced(self):
        for kind, (lower, upper) in chemistry.WINDOWS.items():
            for distance in (lower, upper):
                with self.subTest(kind=kind, distance=distance):
                    observed = chemistry.physical_connections(self.write_pair(kind, distance), "L",
                        [dict(type=kind, anchors=[1, 2])])
                    self.assertEqual(observed[0]["distance"], distance)
            for distance in (lower - 0.001, upper + 0.001):
                with self.subTest(kind=kind, distance=distance):
                    with self.assertRaises(AssertionError):
                        chemistry.physical_connections(self.write_pair(kind, distance), "L",
                            [dict(type=kind, anchors=[1, 2])])

    def test_independent_rules_keep_unavailable_separate_from_failure(self):
        rules = [dict(metric="shape", op=">", value=0.55), dict(metric="surface", op="<", value=0.5)]
        self.assertEqual(chemistry.rule_status(dict(shape=0.56, surface=0.49), rules)[0], "passed")
        self.assertEqual(chemistry.rule_status(dict(shape=0.55, surface=0.49), rules), ("failed", ["shape"], []))
        self.assertEqual(chemistry.rule_status(dict(shape=0.55), rules), ("unavailable", ["shape"], ["surface"]))
        self.assertEqual(chemistry.rule_status(dict(shape=True, surface=0.1), rules)[0], "unavailable")

    def test_csv_locators_must_match_final_atom_identity(self):
        path = self.root / "final.pdb"
        path.write_text("ATOM      8  CA  ALA L   1       1.000   2.000   3.000  1.00  0.00           C  \nEND\n")
        fixed = dict(chain="L", res_id=1, ins_code="", atom="CA", atom_id=8, restype="ALA")
        self.assertEqual(chemistry.locator_meaning(fixed), chemistry.locator_meaning({**fixed, "atom_id": 99}))
        self.assertNotEqual(chemistry.locator_meaning(fixed), chemistry.locator_meaning({**fixed, "restype": "GLY"}))
        row = dict(design_chain_id="L", design_sequence="A", token_length="1",
            design_chains=json.dumps([dict(chain="L", sequence="A", residue_count=1)]),
            fixed_restypes=json.dumps([fixed]), bond_constraints="[]")
        chemistry.locator_check(row, path)
        for field, value in (("chain", "A"), ("res_id", 2), ("restype", "GLY"), ("atom", "N")):
            with self.subTest(field=field):
                row["fixed_restypes"] = json.dumps([{**fixed, field: value}])
                with self.assertRaises(AssertionError):
                    chemistry.locator_check(row, path)
        row["fixed_restypes"] = json.dumps([dict(chain="L", res_id=1, restype="ALA")])
        chemistry.locator_check(row, path)
        row["fixed_restypes"] = json.dumps([dict(chain="L", res_id=1, restype="GLY")])
        with self.assertRaises(AssertionError):
            chemistry.locator_check(row, path)


if __name__ == "__main__":
    unittest.main()
