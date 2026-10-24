from pathlib import Path
import unittest
from partiflow.workflows.sequence import (
    backend_settings,
    sequence_settings,
    validate_sequence_config,
)


class SequenceSettingsTests(unittest.TestCase):
    def test_fixed_conditions_combine_while_local_sampling_settings_override(self):
        value = sequence_settings(
            {
                "fixed_positions": ["L:1"],
                "fixed_interface": {"enabled": True},
                "settings": {"fixed_positions": ["L:2"], "temperature": 0.1},
            },
            {
                "fixed_positions": ["L:2", "T:1"],
                "temperature": 0.3,
                "fixed_interface": {"enabled": False},
            },
        )
        self.assertEqual(value["fixed_positions"], ["L:1", "L:2", "T:1"])
        self.assertEqual(value["temperature"], 0.3)
        self.assertEqual(value["fixed_interface"], {"enabled": False})

    def test_local_omit_resets_inherited_alias_but_explicit_alias_wins(self):
        base = {"proteinmpnn": {"rm_aa": "C", "omit_aas": "C"}}
        config = backend_settings(
            "proteinmpnn", base, Path("/models"), {"omit_aas": "W"}
        )
        self.assertIsNone(config["rm_aa"])
        self.assertEqual(config["omit_aas"], "W")
        config = backend_settings(
            "proteinmpnn", base, Path("/models"), {"omit_aas": "W", "rm_aa": "C"}
        )
        self.assertEqual(config["omit_aas"], "C")

    def test_validation_backend_layer_omit_matches_reference_precedence(self):
        sequence = validate_sequence_config({"proteinmpnn": {"omit_aas": "W"}})
        config = backend_settings("proteinmpnn", sequence, Path("/models"))
        self.assertEqual(config["omit_aas"], "W")
        self.assertIsNone(config["rm_aa"])
