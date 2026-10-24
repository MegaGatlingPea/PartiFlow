from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
from loguru import logger


@dataclass
class PartialLoadReport:
    loaded_keys: List[str]
    partial_loaded_keys: List[str]
    missing_keys: List[str]
    unexpected_keys: List[str]
    shape_mismatch_keys: List[str]


def _extract_state_dict(checkpoint_or_state_dict: Dict) -> Dict[str, torch.Tensor]:
    if "state_dict" in checkpoint_or_state_dict:
        return checkpoint_or_state_dict["state_dict"]
    return checkpoint_or_state_dict


def _format_preview(keys: Iterable[str], limit: int = 8) -> str:
    keys = list(keys)
    if not keys:
        return "[]"
    preview = keys[:limit]
    suffix = "" if len(keys) <= limit else f", ... (+{len(keys) - limit} more)"
    return "[" + ", ".join(preview) + suffix + "]"


def load_state_dict_partial(
    model: torch.nn.Module,
    checkpoint_or_state_dict: Dict,
    *,
    strict: bool = False,
    log_prefix: str = "checkpoint",
    allow_shape_prefix: bool = True,
) -> PartialLoadReport:
    """
    Load a checkpoint into ``model`` while skipping keys whose tensor shapes no longer
    match the current module. This is useful when warm-starting a modified architecture
    from an older checkpoint.

    If ``allow_shape_prefix`` is true, tensors with the same rank but different
    sizes are partially loaded by copying the overlapping prefix slice in every
    dimension and leaving the rest of the target tensor at its initialization.
    """
    source_state_dict = _extract_state_dict(checkpoint_or_state_dict)
    model_state_dict = model.state_dict()

    filtered_state_dict: Dict[str, torch.Tensor] = {}
    unexpected_keys: List[str] = []
    shape_mismatch_keys: List[str] = []
    partial_loaded_keys: List[str] = []

    for key, value in source_state_dict.items():
        if key not in model_state_dict:
            unexpected_keys.append(key)
            continue
        if model_state_dict[key].shape != value.shape:
            if allow_shape_prefix and value.ndim == model_state_dict[key].ndim:
                target_value = model_state_dict[key].clone()
                slices = tuple(
                    slice(0, min(src_dim, dst_dim))
                    for src_dim, dst_dim in zip(value.shape, target_value.shape)
                )
                target_value[slices] = value[slices].to(device=target_value.device, dtype=target_value.dtype)
                filtered_state_dict[key] = target_value
                partial_loaded_keys.append(key)
            else:
                shape_mismatch_keys.append(key)
            continue
        filtered_state_dict[key] = value

    load_ret = model.load_state_dict(filtered_state_dict, strict=strict)
    missing_keys = list(load_ret.missing_keys)
    loaded_keys = sorted(filtered_state_dict.keys())

    logger.info(
        f"{log_prefix}: loaded {len(loaded_keys)} tensors, "
        f"partially loaded {len(partial_loaded_keys)} shape-changed tensors, "
        f"skipped {len(shape_mismatch_keys)} shape-mismatched tensors, "
        f"ignored {len(unexpected_keys)} unexpected tensors, "
        f"left {len(missing_keys)} tensors missing in the target model."
    )
    if partial_loaded_keys:
        logger.info(
            f"{log_prefix}: partially loaded keys {_format_preview(partial_loaded_keys)}"
        )
        for key in partial_loaded_keys[:8]:
            src_shape = tuple(source_state_dict[key].shape)
            dst_shape = tuple(model_state_dict[key].shape)
            copied_shape = tuple(
                min(src_dim, dst_dim)
                for src_dim, dst_dim in zip(source_state_dict[key].shape, model_state_dict[key].shape)
            )
            logger.info(
                f"{log_prefix}: partial load for '{key}': "
                f"ckpt {src_shape} -> model {dst_shape}, copied prefix {copied_shape}"
            )
    if shape_mismatch_keys:
        logger.info(
            f"{log_prefix}: shape-mismatched keys {_format_preview(shape_mismatch_keys)}"
        )
        for key in shape_mismatch_keys[:8]:
            logger.info(
                f"{log_prefix}: shape mismatch for '{key}': "
                f"ckpt {tuple(source_state_dict[key].shape)} vs model {tuple(model_state_dict[key].shape)}"
            )
    if unexpected_keys:
        logger.info(
            f"{log_prefix}: unexpected checkpoint keys {_format_preview(unexpected_keys)}"
        )
    if missing_keys:
        logger.info(
            f"{log_prefix}: missing model keys after partial load {_format_preview(missing_keys)}"
        )

    return PartialLoadReport(
        loaded_keys=loaded_keys,
        partial_loaded_keys=partial_loaded_keys,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatch_keys=shape_mismatch_keys,
    )
