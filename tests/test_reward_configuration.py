import unittest
from partiflow.workflows.reward import _config


class RewardConfigurationTests(unittest.TestCase):
    def test_nonnegative_nonzero_complex_weights(self):
        for weights in ((-0.1, 1), (1, -0.1), (0, 0), (True, 1), (float("nan"), 1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                _config(
                    {
                        "score": dict(
                            zip(
                                ("complex_plddt_weight", "complex_iptm_weight"), weights
                            )
                        )
                    }
                )
        for weights in ((0, 1), (1, 0), (0.25, 0.75)):
            config = _config(
                {
                    "score": dict(
                        zip(("complex_plddt_weight", "complex_iptm_weight"), weights)
                    )
                }
            )
            self.assertEqual(tuple(config["score"].values()), weights)

    def test_design_structure_requires_design_sequence_before_models(self):
        for source in ("proteinmpnn", "solublempnn", "auto"):
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(ValueError, "requires sequence.source=design"),
            ):
                _config({"structure_source": "design", "sequence": {"source": source}})
        self.assertEqual(
            _config({"structure_source": "design"})["sequence"]["source"], "design"
        )

    def test_zero_recycles_matches_reference(self):
        self.assertEqual(
            _config({"af2": {"num_recycles": 0}})["af2"]["num_recycles"], 0
        )
        for value in (-1, True, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _config({"af2": {"num_recycles": value}})
