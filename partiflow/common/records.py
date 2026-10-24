"""Primitive records shared by generation and downstream workflows."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    task_name: str
    base_sample_id: int
    candidate_ordinal: int
    replica_id: int
    pdb_path: Path
    binder_chain: str
    length: int
    sequence: str
    contig_string: str = ""
    tip_atoms: str = ""
    context_chains: tuple[str, ...] = ()
    fixed_restypes: list[dict[str, Any]] = field(default_factory=list)
    bond_constraints: list[dict[str, Any]] = field(default_factory=list)
    topology_entries: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Evaluation:
    candidate_id: str
    score: float | None = None
    pdb_path: Path | None = None
    sequence: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    error: str = ""


class CandidateError(ValueError):
    """One input cannot be evaluated without changing its requested meaning."""


class WorkflowError(RuntimeError):
    """A model, device or workflow protocol failed."""
