"""Closed sequence configuration checks that never load model runtimes."""

import copy
from pathlib import Path
import subprocess
import sys
import unittest

from partiflow.workflows.sequence import (
    ALIASES,
    backend_settings,
    normalize_sources,
    validate_sequence_config,
)


class SequenceSchemaTests(unittest.TestCase):
    def test_defaults_keep_validation_self_and_soluble_and_reward_design(self):
        sequence = validate_sequence_config({})
        self.assertEqual(
            [
                (item["source"], item["label"], item["count"])
                for item in sequence["sources"]
            ],
            [("gt", "self", 1), ("solublempnn", "soluble", 1)],
        )
        self.assertEqual(sequence["sources"][1]["settings"], {"temperature": 0.1})
        self.assertEqual(validate_sequence_config({}, reward=True)["source"], "design")

    def test_schema_and_backend_validation_leave_model_libraries_unloaded(self):
        code = """
import sys
from pathlib import Path
from partiflow.workflows.sequence import validate_sequence_config, backend_settings
for reward in (False, True):
    validate_sequence_config({}, reward=reward)
backend_settings("proteinmpnn", {}, Path("/missing"))
assert not any(name in sys.modules for name in (
    "numpy", "biotite", "torch", "jax", "pyro", "colabdesign",
    "partiflow._vendor.proteinmpnn.protein_mpnn_utils",
))
"""
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_root_and_every_backend_block_have_closed_keys(self):
        for bad in (
            {"sead": 101},
            {"fixed_postions": []},
            {"ligandmpnn": {}},
            {"cyclicmpnn": {}},
            {"na_mpnn": {}},
            *(
                {name: {"temperatur": 0.1}}
                for name in (
                    "proteinmpnn",
                    "auto_proteinmpnn",
                    "solublempnn",
                    "soluble_mpnn",
                    "solublempnn_defaults",
                )
            ),
            {"auto": {"protien": "soluble_mpnn"}},
            {"auto": {"protein": {"temperatur": 0.1}}},
            {"settings": {"temperatur": 0.1}},
            {"sources": [{"source": "proteinmpnn", "settings": {"temperatur": 0.1}}]},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_sequence_config(bad, reward=True)

    def test_sequence_shapes_seeds_and_backend_shapes_are_strict(self):
        for value in (None, [], 1, "design"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_sequence_config(value)
        for seed in (True, -1, 1.0, "1", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                validate_sequence_config({"seed": seed})
        for name in ("proteinmpnn", "solublempnn", "auto", "settings"):
            for value in (None, [], "soluble_mpnn"):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    validate_sequence_config({name: value})

    def test_existing_aliases_and_fixed_interface_suffix_remain_supported(self):
        for alias, expected in ALIASES.items():
            sequence = validate_sequence_config({"sources": [{"source": alias}]})
            self.assertEqual(sequence["sources"][0]["source"], expected)
            self.assertEqual(
                validate_sequence_config({"source": alias}, reward=True)["source"],
                alias,
            )
        branch = normalize_sources(
            [{"source": "protein_mpnn-fi", "settings": {"fixed_positions": ["B:01"]}}]
        )[0]
        self.assertEqual(branch["settings"]["fixed_positions"], ["B:1"])
        self.assertTrue(branch["settings"]["fixed_interface"]["enabled"])
        with self.assertRaises(ValueError):
            normalize_sources(
                [
                    {
                        "source": "proteinmpnn-fi",
                        "settings": {"fixed_interface": {"enabled": False}},
                    }
                ]
            )
        for source in ("ligandmpnn", "cyclicmpnn", "na_mpnn", "auto"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                normalize_sources([{"source": source}])

        with self.assertRaises(ValueError):
            validate_sequence_config({"source": "proteinmpnn-fi"}, reward=True)

    def test_fixed_positions_are_canonical_detached_and_duplicate_aware(self):
        value = {
            "fixed_positions": ["B:01", "T:002"],
            "settings": {"fixed_positions": ["B:003"]},
            "sources": [
                {"source": "soluble_mpnn", "settings": {"fixed_positions": ["B:004"]}}
            ],
        }
        original = copy.deepcopy(value)
        sequence = validate_sequence_config(value)
        self.assertEqual(sequence["fixed_positions"], ["B:1", "T:2"])
        self.assertEqual(sequence["settings"]["fixed_positions"], ["B:3"])
        self.assertEqual(sequence["sources"][0]["settings"]["fixed_positions"], ["B:4"])
        self.assertEqual(value, original)
        for value in (
            {"fixed_positions": ["B:1", "B:01"]},
            {"settings": {"fixed_positions": ["B:1", "B:01"]}},
            {
                "sources": [
                    {
                        "source": "proteinmpnn",
                        "settings": {"fixed_positions": ["B:1", "B:01"]},
                    }
                ]
            },
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "Duplicate fixed"),
            ):
                validate_sequence_config(value)

    def test_fixed_positions_require_one_chain_and_positive_integer(self):
        for item in (
            "",
            "B",
            ":1",
            "B:",
            "B:0",
            "B:-1",
            "B:+1",
            "B:1.0",
            "B: 1",
            "B:1:2",
            1,
            None,
        ):
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_sequence_config({"fixed_positions": [item]}, reward=True)
        for value in (None, (), "B:1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_sequence_config({"fixed_positions": value})

    def test_fixed_interface_schema_is_shared_and_detached(self):
        value = {"fixed_interface": {"enabled": True, "target_chains": ["T"]}}
        sequence = validate_sequence_config(value, reward=True)
        self.assertEqual(sequence["fixed_interface"]["cutoff"], 6.0)
        sequence["fixed_interface"]["target_chains"].append("U")
        self.assertEqual(value["fixed_interface"]["target_chains"], ["T"])
        for value in (
            {"enabled": 1},
            {"enabled": True, "cutof": 6},
            {"enabled": True, "cutoff": 0},
            {"enabled": True, "target_chains": ["T", "T"]},
            {"enabled": True, "target_chains": ["T:1"]},
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_sequence_config({"fixed_interface": value})

    def test_mpnn_numeric_flags_bias_and_amino_acids_match_native_contract(self):
        for options in (
            {"temperature": True},
            {"temperature": 0},
            {"temperature": float("nan")},
            {"backbone_noise": -0.1},
            {"pssm_multi": 1.1},
            {"pssm_threshold": float("inf")},
            {"ca_only": 1},
            {"pssm_bias": "true"},
            {"pssm_log_odds": 1},
            {"model_type": "ligand_mpnn"},
            {"bias_aa": []},
            {"bias_aa": {"X": 1}},
            {"bias_aa": {"AA": 1}},
            {"bias_aa": {"A": True}},
            {"bias_aa": {"A": float("nan")}},
            {"rm_aa": ["C"]},
            {"rm_aa": "Z"},
            {"rm_aa": "ACDEFGHIKLMNPQRSTVWY"},
            {"rm_aa": None, "omit_aas": "Z"},
            {"rm_aa": None, "omit_aas": 1},
            {"checkpoint": ""},
            {"checkpoint": 123},
            {"pssm_path": "https://example.org/pssm.json"},
            {"tied_positions_path": []},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                validate_sequence_config({"proteinmpnn": options}, reward=True)

    def test_pssm_dependencies_are_checked_before_assets_are_read(self):
        for options in (
            {"pssm_bias": True},
            {"pssm_log_odds": True},
            {"pssm_multi": 0.5},
            {"pssm_threshold": 1.0},
            {"pssm_path": "/missing/pssm.json"},
            {
                "pssm_path": "/missing/pssm.json",
                "pssm_bias": True,
                "pssm_threshold": 1.0,
            },
            {
                "pssm_path": "/missing/pssm.json",
                "pssm_log_odds": True,
                "pssm_multi": 0.5,
            },
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                backend_settings("proteinmpnn", {}, Path("/missing"), options)
        settings = backend_settings(
            "proteinmpnn",
            {},
            Path("/missing"),
            {
                "pssm_path": "/missing/pssm.json",
                "pssm_bias": True,
                "pssm_log_odds": True,
                "pssm_multi": 0.5,
                "pssm_threshold": 1.0,
                "tied_positions_path": "/missing/tied.json",
                "bias_aa": {"G": 0.5},
            },
        )
        self.assertEqual(settings["pssm_multi"], 0.5)

    def test_legacy_omission_priority_and_ca_model_remain_supported(self):
        self.assertEqual(
            backend_settings(
                "proteinmpnn", {}, Path("/missing"), {"rm_aa": "", "omit_aas": "C"}
            )["omit_aas"],
            "",
        )
        self.assertEqual(
            backend_settings(
                "proteinmpnn", {}, Path("/missing"), {"rm_aa": None, "omit_aas": "A"}
            )["omit_aas"],
            "A",
        )
        config = backend_settings(
            "proteinmpnn",
            {},
            Path("/missing"),
            {"model_type": "ca_protein_mpnn", "ca_only": True},
        )
        self.assertIn("/ca/", config["checkpoint"])
        with self.assertRaises(ValueError):
            backend_settings(
                "solublempnn",
                {},
                Path("/missing"),
                {"model_type": "protein_mpnn", "use_soluble_model": False},
            )

    def test_default_auto_uses_soluble_and_keeps_caller_values_detached(self):
        raw = {"source": "auto"}
        sequence = validate_sequence_config(raw, reward=True)
        self.assertEqual(raw, {"source": "auto"})
        self.assertEqual(sequence["proteinmpnn"]["model_type"], "soluble_mpnn")
        config = backend_settings("auto", sequence, Path("/missing"))
        self.assertEqual(config["temperature"], 0.1)
        self.assertIn("/soluble/", config["checkpoint"])

    def test_auto_keeps_explicit_backend_contract_and_rejects_unknown_policy(self):
        explicit = validate_sequence_config(
            {
                "source": "auto",
                "auto_proteinmpnn": {
                    "model_type": "protein_mpnn",
                    "temperature": 0.3,
                    "ca_only": False,
                    "use_soluble_model": False,
                },
            },
            reward=True,
        )
        self.assertEqual(
            backend_settings("auto", explicit, Path("/missing"))["temperature"], 0.3
        )
        for automatic in (
            {"protein": "ligand_mpnn"},
            {"protein": "protein_mpnn"},
            {"protein": True},
            {"protein": {"use_soluble_model": False}},
        ):
            with self.subTest(automatic=automatic), self.assertRaises(ValueError):
                validate_sequence_config({"auto": automatic})

    def test_source_and_global_settings_are_validated_after_overlay(self):
        with self.assertRaisesRegex(ValueError, "pssm_path"):
            validate_sequence_config(
                {
                    "settings": {"pssm_bias": True},
                    "sources": [{"source": "proteinmpnn"}],
                }
            )
        sequence = validate_sequence_config(
            {
                "settings": {"pssm_bias": True},
                "sources": [
                    {
                        "source": "proteinmpnn",
                        "settings": {"pssm_path": "/missing/pssm.json"},
                    }
                ],
            }
        )
        self.assertTrue(sequence["settings"]["pssm_bias"])


if __name__ == "__main__":
    unittest.main()
