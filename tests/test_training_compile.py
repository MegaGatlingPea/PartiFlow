"""CPU checks for independent training and evaluation compile caches."""

from __future__ import annotations

import copy
import importlib.util
import io
import unittest
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_RUNNER = (
    HAS_TORCH
    and importlib.util.find_spec("hydra") is not None
    and importlib.util.find_spec("lightning") is not None
)


@unittest.skipUnless(HAS_TORCH, "PyTorch is required")
class TrainingCompileTests(unittest.TestCase):
    def setUp(self):
        import torch._dynamo

        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

    def _model(self):
        import torch

        class Trunk(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layer = torch.nn.Linear(2, 2, dtype=torch.float64)
                self.register_buffer("shift", torch.tensor(0.25, dtype=torch.float64))
                with torch.no_grad():
                    self.layer.weight.copy_(
                        torch.tensor([[0.5, -0.25], [0.75, 1.0]], dtype=torch.float64)
                    )
                    self.layer.bias.copy_(
                        torch.tensor([0.1, -0.2], dtype=torch.float64)
                    )

            def forward(self, batch, *, offset=0):
                value = self.layer(batch["x"]) + self.shift + offset
                return value * 2 if self.training else value

        model = torch.nn.Module()
        model.nn = Trunk()
        return model

    def _compile(self, model):
        from partiflow.training.compilation import compile_forward_by_mode

        self.assertIs(
            compile_forward_by_mode(
                model.nn, backend="eager", fullgraph=True, dynamic=False
            ),
            model.nn,
        )
        compiled = model.nn.forward.compiled_forwards
        self.assertEqual(len(compiled), 2)
        self.assertFalse(hasattr(model.nn.forward, "_torchdynamo_orig_callable"))
        self.assertIsNot(
            compiled[0]._torchdynamo_orig_callable.__code__,
            compiled[1]._torchdynamo_orig_callable.__code__,
        )
        return compiled

    def test_mode_dispatch_preserves_outputs_gradients_and_contexts(self):
        import torch
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        model = self._model()
        expected = copy.deepcopy(model)
        compiled = self._compile(model)
        for training, context, grad_enabled, inference_enabled in (
            (True, torch.enable_grad, True, False),
            (True, torch.no_grad, False, False),
            (False, torch.inference_mode, False, True),
            (False, torch.no_grad, False, False),
            (False, torch.enable_grad, True, False),
            (True, torch.enable_grad, True, False),
        ):
            with self.subTest(
                training=training, grad=grad_enabled, inference=inference_enabled
            ):
                model.train(training)
                expected.train(training)
                model.zero_grad(set_to_none=True)
                expected.zero_grad(set_to_none=True)
                value = torch.tensor(
                    [[0.2, 0.4], [-0.3, 0.7]],
                    dtype=torch.float64,
                    requires_grad=True,
                )
                eager_value = value.detach().clone().requires_grad_(True)
                with context():
                    self.assertEqual(torch.is_grad_enabled(), grad_enabled)
                    self.assertEqual(
                        torch.is_inference_mode_enabled(), inference_enabled
                    )
                    actual = model.nn({"x": value}, offset=1)
                    wanted = expected.nn({"x": eager_value}, offset=1)
                    torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
                    self.assertEqual(actual.requires_grad, grad_enabled)
                    self.assertEqual(actual.is_inference(), inference_enabled)
                    if grad_enabled:
                        actual.square().sum().backward()
                        wanted.square().sum().backward()
                        torch.testing.assert_close(
                            value.grad, eager_value.grad, rtol=0, atol=0
                        )
                        for actual_parameter, eager_parameter in zip(
                            model.parameters(), expected.parameters()
                        ):
                            torch.testing.assert_close(
                                actual_parameter.grad,
                                eager_parameter.grad,
                                rtol=0,
                                atol=0,
                            )
                if not grad_enabled:
                    self.assertTrue(
                        all(parameter.grad is None for parameter in model.parameters())
                    )
        self.assertEqual(
            tuple(
                len(_debug_get_cache_entry_list(entry._torchdynamo_orig_callable))
                for entry in compiled
            ),
            (2, 3),
        )

    def test_evaluation_has_capacity_after_training_cache_is_full(self):
        import torch
        from torch._dynamo.eval_frame import _debug_get_cache_entry_list

        model = self._model()
        original_forward = model.nn.forward
        training_forward, evaluation_forward = self._compile(model)
        limit = torch._dynamo.config.cache_size_limit
        suppress_errors = torch._dynamo.config.suppress_errors
        value = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
        for offset in range(limit):
            model.nn({"x": value}, offset=offset).sum().backward()
        self.assertEqual(
            len(
                _debug_get_cache_entry_list(training_forward._torchdynamo_orig_callable)
            ),
            limit,
        )
        self.assertEqual(
            len(
                _debug_get_cache_entry_list(
                    evaluation_forward._torchdynamo_orig_callable
                )
            ),
            0,
        )
        model.eval()
        with torch.inference_mode():
            actual = model.nn({"x": value}, offset=0)
            wanted = original_forward({"x": value}, offset=0)
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        model.train()
        model.nn({"x": value}, offset=0).sum().backward()
        self.assertEqual(
            len(
                _debug_get_cache_entry_list(training_forward._torchdynamo_orig_callable)
            ),
            limit,
        )
        self.assertEqual(
            len(
                _debug_get_cache_entry_list(
                    evaluation_forward._torchdynamo_orig_callable
                )
            ),
            1,
        )
        self.assertEqual(len(_debug_get_cache_entry_list(original_forward)), 0)
        self.assertEqual(torch._dynamo.config.cache_size_limit, limit)
        self.assertEqual(torch._dynamo.config.suppress_errors, suppress_errors)

    def test_module_identity_and_strict_checkpoint_keys_are_preserved(self):
        import torch

        model = self._model()
        state = copy.deepcopy(model.state_dict())
        parameters = tuple(
            (name, id(value)) for name, value in model.named_parameters()
        )
        buffers = tuple((name, id(value)) for name, value in model.named_buffers())
        modules = tuple((name, id(value)) for name, value in model.named_modules())
        self._compile(model)
        self.assertEqual(
            tuple((name, id(value)) for name, value in model.named_parameters()),
            parameters,
        )
        self.assertEqual(
            tuple((name, id(value)) for name, value in model.named_buffers()), buffers
        )
        self.assertEqual(
            tuple((name, id(value)) for name, value in model.named_modules()), modules
        )
        self.assertEqual(tuple(model.state_dict()), tuple(state))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, state[name], rtol=0, atol=0)
        checkpoint = io.BytesIO()
        torch.save({"state_dict": model.state_dict()}, checkpoint)
        checkpoint.seek(0)
        eager = self._model()
        restored = eager.load_state_dict(
            torch.load(checkpoint, weights_only=True)["state_dict"], strict=True
        )
        self.assertEqual((restored.missing_keys, restored.unexpected_keys), ([], []))
        restored = model.load_state_dict(eager.state_dict(), strict=True)
        self.assertEqual((restored.missing_keys, restored.unexpected_keys), ([], []))

    @unittest.skipUnless(HAS_RUNNER, "Training runner dependencies are required")
    def test_runner_installs_both_entry_points_with_configured_options(self):
        import torch
        from partiflow.training.runner import maybe_compile_model

        model = self._model()
        with patch("partiflow.training.runner.compile_forward_by_mode") as install:
            for config in ({}, {"compile": {"enabled": False}}):
                self.assertIs(maybe_compile_model(config, model), model)
            install.assert_not_called()
        actual_compile = torch.compile

        def compile_with_eager(entry, **kwargs):
            return actual_compile(entry, backend="eager", **kwargs)

        with patch(
            "partiflow.training.compilation.torch.compile",
            side_effect=compile_with_eager,
        ) as compiler:
            result = maybe_compile_model(
                {"compile": {"enabled": True, "dynamic": False, "fullgraph": True}},
                model,
            )
        self.assertIs(result, model)
        self.assertEqual(compiler.call_count, 2)
        for call in compiler.call_args_list:
            self.assertEqual(
                call.kwargs, {"mode": "default", "dynamic": False, "fullgraph": True}
            )
        value = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
        model.nn({"x": value}).sum().backward()
        model.eval()
        with torch.inference_mode():
            output = model.nn({"x": value})
            self.assertTrue(output.is_inference())


if __name__ == "__main__":
    unittest.main()
