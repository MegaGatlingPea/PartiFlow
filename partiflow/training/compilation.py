"""Compile forward calls with separate training and evaluation caches."""

from __future__ import annotations

import torch


def compile_forward_by_mode(
    module: torch.nn.Module, **compile_kwargs
) -> torch.nn.Module:
    """Install two compiled entry points without changing the module hierarchy.

    Dynamo caches specializations by code object. These separate function
    definitions give training and evaluation independent specialization budgets,
    while both call the original bound forward with the caller's gradient and
    inference contexts. The dispatcher itself remains outside torch.compile.

    ``module.forward.compiled_forwards`` contains the actual compiled callables
    in (training, evaluation) order for runtime inspection.
    """
    original_forward = module.forward

    def training_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    def evaluation_forward(*args, **kwargs):
        return original_forward(*args, **kwargs)

    compiled_training_forward = torch.compile(training_forward, **compile_kwargs)
    compiled_evaluation_forward = torch.compile(evaluation_forward, **compile_kwargs)

    def forward(*args, **kwargs):
        if module.training:
            return compiled_training_forward(*args, **kwargs)
        return compiled_evaluation_forward(*args, **kwargs)

    forward.compiled_forwards = (
        compiled_training_forward,
        compiled_evaluation_forward,
    )
    module.forward = forward
    return module
