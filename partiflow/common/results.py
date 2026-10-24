"""The public normal-design CSV contract."""

from __future__ import annotations
import csv
import math
import os
from pathlib import Path

BASE_FIELDS = (
    "filename",
    "design_chains",
    "base_sample_id",
    "design_modality",
    "design_chain_id",
    "token_length",
    "reward_method",
    "global_rank",
    "design_sequence",
    "contig_string",
    "tip_atoms",
    "fixed_restypes",
    "bond_constraints",
)
TAIL_FIELDS = ("postprocess_status", "error")


def read_results(path):
    path = Path(path)
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        if fields[: len(BASE_FIELDS)] != BASE_FIELDS or fields[-2:] != TAIL_FIELDS:
            raise ValueError(f"Invalid result.csv columns: {path}")
        return fields, list(reader)


def write_results(path, rows, *, fields=None):
    path = Path(path)
    rows = list(rows)
    if fields is None:
        extra = (
            set().union(*(set(row) for row in rows)) - set(BASE_FIELDS + TAIL_FIELDS)
            if rows
            else set()
        )
        for row in rows:
            for key in extra:
                value = row.get(key, "")
                if value not in ("", None) and (
                    isinstance(value, bool) or not math.isfinite(float(value))
                ):
                    raise ValueError(f"Non-finite result metric: {key}")
        fields = BASE_FIELDS + tuple(sorted(extra)) + TAIL_FIELDS
    fields = tuple(fields)
    if len(fields) != len(set(fields)) or fields[len(BASE_FIELDS) : -2] != tuple(
        sorted(fields[len(BASE_FIELDS) : -2])
    ):
        raise ValueError("Duplicate or unsorted public metric columns")
    unknown = set().union(*(set(row) for row in rows)) - set(fields) if rows else set()
    if unknown:
        raise ValueError(f"Unexpected public fields: {sorted(unknown)}")
    for row in rows:
        for name in fields[len(BASE_FIELDS) : -2]:
            value = row.get(name, "")
            if value not in ("", None) and (
                isinstance(value, bool) or not math.isfinite(float(value))
            ):
                raise ValueError(f"Non-finite result metric: {name}")
    if fields[: len(BASE_FIELDS)] != BASE_FIELDS or fields[-2:] != TAIL_FIELDS:
        raise ValueError("Invalid public result field order")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows({key: row.get(key, "") for key in fields} for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return fields
