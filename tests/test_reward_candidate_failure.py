from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
from partiflow.workflows import reward_backend as backend


class RewardCandidateFailureTests(unittest.TestCase):
    def model(self):
        return SimpleNamespace(
            aux={
                "plddt": np.array([0.8, 0.8, 0.9, 0.9]),
                "pae": np.ones((4, 4)),
                "log": {"ptm": 0.75, "i_ptm": 0.6},
            },
            _inputs={"asym_id": np.array([1, 1, 2, 2])},
            predict=Mock(),
            save_pdb=Mock(),
        )

    def predict(self, model, directory):
        with patch.object(backend, "_prepare", return_value=(2, 2)):
            return backend._predict_one(
                model,
                {
                    "candidate_id": "candidate",
                    "design_sequence": "AA",
                    "output_pdb": str(Path(directory) / "output.pdb"),
                },
                {"num_recycles": 3},
            )

    def test_bad_confidence_is_local_while_valid_metrics_keep_reference_units(self):
        with tempfile.TemporaryDirectory() as directory:
            valid = self.predict(self.model(), directory)
            self.assertAlmostEqual(valid["design_plddt"], 90)
            self.assertEqual(valid["interface_iptm"], 0.6)
            self.assertEqual(valid["interface_ipae"], 1)
            for condition in (
                "nonfinite_plddt",
                "negative_pae",
                "missing_iptm",
                "invalid_ptm",
                "bad_shape",
            ):
                model = self.model()
                if condition == "nonfinite_plddt":
                    model.aux["plddt"][2] = np.nan
                elif condition == "negative_pae":
                    model.aux["pae"][0, 0] = -1
                elif condition == "missing_iptm":
                    del model.aux["log"]["i_ptm"]
                elif condition == "invalid_ptm":
                    model.aux["log"]["ptm"] = 2
                else:
                    model.aux["pae"] = np.ones((3, 3))
                with (
                    self.subTest(condition=condition),
                    self.assertRaises(backend.AF2CandidateError),
                ):
                    self.predict(model, directory)

    def test_prediction_device_fault_is_not_reclassified_as_candidate_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            model = self.model()
            model.predict.side_effect = RuntimeError("device execution failed")
            with self.assertRaisesRegex(
                RuntimeError, "device execution failed"
            ) as failure:
                self.predict(model, directory)
            self.assertNotIsInstance(failure.exception, backend.AF2CandidateError)


class RewardBatchFailureTests(unittest.TestCase):
    def evaluate(self, failure):
        from partiflow.common.records import Candidate
        from partiflow.workflows.reward import evaluate_candidates

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weights = root / "af2/params_model_1_multimer_v3.npz"
            weights.parent.mkdir()
            weights.write_bytes(b"unit preflight")
            fixture = (
                Path(__file__).resolve().parents[1] / "pdbs/design/mdm2/mdm2_motif.pdb"
            )
            candidates = [
                Candidate(
                    name, "task", index, index, 1, fixture, "L", 11, "SPMDEFEEMVL"
                )
                for index, name in enumerate(("bad", "good"), 1)
            ]
            loaded = []
            key = ("test",)

            def model(_configuration):
                if key not in backend._MODEL_CACHE:
                    backend._MODEL_CACHE[key] = SimpleNamespace(
                        _inputs={}, aux={}, _tmp={}
                    )
                    loaded.append(backend._MODEL_CACHE[key])
                return backend._MODEL_CACHE[key], len(loaded) == 1

            def predict(_model, request, _configuration):
                if request["candidate_id"] == "bad" and failure == "confidence":
                    raise backend.AF2CandidateError("invalid candidate confidence")
                if request["candidate_id"] == "bad" and failure == "device":
                    raise RuntimeError("device execution failed")
                output = request["input_pdb"]
                if request["candidate_id"] == "bad" and failure == "restoration":
                    output = str(root / "absent.pdb")
                return dict(
                    fold_pdb=output,
                    design_plddt=90.0,
                    ptm=0.8,
                    interface_iptm=0.7,
                    interface_ipae=2.0,
                    pae=2.0,
                )

            backend._MODEL_CACHE.clear()
            try:
                with (
                    patch.object(backend, "_model", side_effect=model),
                    patch.object(backend, "_cache_key", return_value=key),
                    patch.object(backend, "_predict_one", side_effect=predict),
                ):
                    result = evaluate_candidates(candidates, {}, root, root / "work")
                return result, len(loaded)
            finally:
                backend._MODEL_CACHE.clear()

    def test_bad_result_remains_local_and_later_candidate_is_published(self):
        for failure in ("confidence", "restoration"):
            with self.subTest(failure=failure):
                results, loads = self.evaluate(failure)
                self.assertEqual(
                    [item.candidate_id for item in results], ["bad", "good"]
                )
                self.assertTrue(results[0].error)
                self.assertEqual(results[1].error, "")
                self.assertAlmostEqual(results[1].score, 0.75, places=6)
                if failure == "confidence":
                    self.assertEqual(loads, 2)

    def test_device_failure_remains_run_level(self):
        from partiflow.common.records import WorkflowError

        with self.assertRaisesRegex(WorkflowError, "device execution failed"):
            self.evaluate("device")
