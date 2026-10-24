import copy
from typing import Any, List
import torch


def _batch_items(batch: Any):
    return (
        list(batch.items()) if hasattr(batch, "items") else list(batch.__dict__.items())
    )


def _batch_copy(batch: Any) -> Any:
    return copy.copy(batch)


def _slice_like(value: Any, start: int, end: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value[start:end]
    if isinstance(value, list):
        return value[start:end]
    if isinstance(value, tuple):
        return value[start:end]
    if isinstance(value, dict):
        return {k: _slice_like(v, start, end) for k, v in value.items()}
    return value


def _repeat_interleave_like(value: Any, repeats: int) -> Any:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    if isinstance(value, list):
        return [item for item in value for _ in range(repeats)]
    if isinstance(value, tuple):
        return tuple(item for item in value for _ in range(repeats))
    if isinstance(value, dict):
        return {k: _repeat_interleave_like(v, repeats) for k, v in value.items()}
    return value


def batch_size_from_any(batch: Any) -> int:
    for _, value in _batch_items(batch):
        if isinstance(value, torch.Tensor) and value.dim() > 0:
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Could not infer batch size.")


def slice_batch(batch: Any, start: int, end: int) -> Any:
    out = _batch_copy(batch)
    for key, value in _batch_items(batch):
        out[key] = _slice_like(value, start, end)
    if hasattr(out, "_num_graphs"):
        out._num_graphs = end - start
    return out


def repeat_interleave_batch(batch: Any, repeats: int) -> Any:
    if repeats <= 1:
        return batch
    out = _batch_copy(batch)
    for key, value in _batch_items(batch):
        out[key] = _repeat_interleave_like(value, repeats)
    if hasattr(out, "_num_graphs"):
        out._num_graphs = batch_size_from_any(out)
    return out


def replicate_batch(batch: Any, replicas: int) -> Any:
    """Repeat-interleave a batch so replicas of the same sample stay adjacent."""
    return repeat_interleave_batch(batch, replicas)


def chunk_batch(batch: Any, max_batch_size: int) -> list[Any]:
    total = batch_size_from_any(batch)
    if max_batch_size is None or total <= max_batch_size:
        return [batch]
    return [
        slice_batch(batch, start, min(start + max_batch_size, total))
        for start in range(0, total, max_batch_size)
    ]


def concat_tensor_dicts(parts: List[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {}
    result: dict[str, Any] = {}
    keys = parts[0].keys()
    for key in keys:
        values = [part[key] for part in parts]
        first = values[0]
        if isinstance(first, torch.Tensor):
            result[key] = torch.cat(values, dim=0)
        elif isinstance(first, list):
            merged = []
            for value in values:
                merged.extend(value)
            result[key] = merged
        else:
            result[key] = values[-1]
    return result
