"""Focused checks for the native CPU core acceptance observer and comparator."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import core

import torch


class CoreAcceptanceHarnessTests(unittest.TestCase):
    def test_tolerances_accept_fp32_roundoff_and_reject_large_errors(self):
        with tempfile.TemporaryDirectory() as root:
            compare = core.Comparisons(Path(root) / "results.jsonl")
            reference = torch.tensor([1.0, 0.0], dtype=torch.float32)
            compare.tree("close", reference + torch.tensor([1e-6, 1e-7]), reference)
            compare.tree("different", reference + 0.01, reference)
            self.assertEqual((compare.count, compare.tensor_count, compare.failures), (2, 2, 1))
            records = [json.loads(line) for line in compare.path.read_text().splitlines()]
            self.assertEqual([record["result"] for record in records], ["PASS", "FAIL"])
            self.assertGreater(records[1]["max_abs_delta"], 0.009)
            self.assertEqual(records[0]["rtol"], 1e-5)
            self.assertEqual(records[0]["atol"], 1e-6)

    def test_shapes_dtypes_nonfinite_and_mapping_keys_fail(self):
        with tempfile.TemporaryDirectory() as root:
            compare = core.Comparisons(Path(root) / "results.jsonl")
            reference = torch.ones(2)
            compare.tree("shape", torch.ones(3), reference)
            compare.tree("dtype", reference.double(), reference)
            compare.tree("nan", torch.tensor([1.0, float("nan")]), reference)
            compare.tree("inf", torch.tensor([1.0, float("inf")]), reference)
            compare.tree("keys", {"other": reference}, {"wanted": reference})
            self.assertEqual((compare.count, compare.failures), (5, 5))

    def test_integer_and_boolean_values_require_exact_equality(self):
        with tempfile.TemporaryDirectory() as root:
            compare = core.Comparisons(Path(root) / "results.jsonl")
            compare.tree("bool", torch.tensor([True]), torch.tensor([False]))
            compare.tree("int", torch.tensor([2]), torch.tensor([1]))
            self.assertEqual(compare.failures, 2)

    def test_observer_calls_native_once_and_retains_pre_step_input(self):
        with tempfile.TemporaryDirectory() as root:
            batch = {"x_sc": {"bb_ca": torch.tensor([3.0])}}
            calls = []
            original_input = torch.tensor([2.0])
            result = torch.tensor([5.0])

            def native(**kwargs):
                calls.append(kwargs)
                kwargs["x_t"]["bb_ca"].add_(1.0)
                return result

            observer = core.StepObserver(native, batch, root)
            actual = observer(x_t={"bb_ca": original_input})
            self.assertIs(actual, result)
            self.assertEqual(len(calls), 1)
            self.assertEqual((observer.steps, observer.self_conditioned_steps), (1, 1))
            stored = torch.load(Path(root) / "0000.pt", weights_only=False)
            self.assertTrue(torch.equal(stored["x_t"]["bb_ca"], torch.tensor([2.0])))
            self.assertTrue(torch.equal(stored["updated"], result))
            self.assertTrue(torch.equal(stored["self_conditioning"]["bb_ca"], batch["x_sc"]["bb_ca"]))

    def test_observer_compares_current_against_saved_original(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            native = lambda **kwargs: {"bb_ca": kwargs["x_t"]["bb_ca"] + 1}
            original = core.StepObserver(native, {}, root / "original")
            original(x_t={"bb_ca": torch.tensor([1.0])})
            compare = core.Comparisons(root / "results.jsonl")
            current = core.StepObserver(native, {}, root / "current", reference=root / "original", comparisons=compare)
            current(x_t={"bb_ca": torch.tensor([1.1])})
            self.assertEqual(compare.failures, 2)
            self.assertEqual(current.self_conditioned_steps, 0)

    def test_runtime_outputs_reject_checkouts_and_their_ancestors(self):
        with tempfile.TemporaryDirectory() as root:
            checkout = Path(root) / "checkout"
            for candidate in (checkout, checkout / "runs", Path(root), Path("/")):
                with self.subTest(candidate=candidate), self.assertRaises(core.AcceptanceError):
                    core.external_root(candidate, checkout)
            self.assertEqual(core.external_root(Path(root) / "outside", checkout), Path(root) / "outside")

    def test_cpu_seeding_repeats_all_three_rng_streams(self):
        import numpy as np
        import random

        core.seed_cpu(2026)
        first = (random.random(), np.random.random(), torch.rand(3))
        core.seed_cpu(2026)
        second = (random.random(), np.random.random(), torch.rand(3))
        self.assertEqual(first[:2], second[:2])
        self.assertTrue(torch.equal(first[2], second[2]))


if __name__ == "__main__":
    unittest.main()
