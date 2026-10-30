"""Worker-local PDB text and parsed coordinate records."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Callable, Hashable, TypeVar


CANONICAL_RESIDUE_NAMES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
STANDARD_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS"}
_Value = TypeVar("_Value")


@dataclass
class PDBSource:
    """One input read, with reusable views kept only in the owning worker.

    Ordered records retain TER boundaries. Residue occurrences are distinct
    even when a receptor reuses an author residue locator.
    """

    path: Path
    text: str
    lines: tuple[str, ...] = field(init=False)
    atoms: list[dict[str, Any]] = field(init=False)
    links: list[str] = field(init=False)
    records: tuple[tuple[str, dict[str, Any] | None], ...] = field(init=False)
    residue_occurrences: tuple[dict[str, Any], ...] = field(init=False)
    _views: dict[Hashable, Any] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.lines = tuple(self.text.splitlines())
        self.atoms = []
        records = []
        residues = []
        counts = {}
        current_signature = None
        current_residue = None
        segment = 0
        for line in self.lines:
            atom = None
            if line.startswith("TER"):
                current_signature = None
                current_residue = None
                segment += 1
            elif line.startswith(("ATOM  ", "HETATM")):
                if len(line) < 54:
                    raise ValueError(f"Truncated coordinate record in {self.path}")
                xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
                if not all(math.isfinite(value) for value in xyz):
                    raise ValueError(f"Non-finite coordinates in {self.path}")
                atom = {
                    "chain": line[21:22].strip() or " ",
                    "res_id": line[22:26].strip(),
                    "ins_code": line[26:27].strip(),
                    "resname": line[17:20].strip(),
                    "atom": line[12:16].strip(),
                    "atom_id": int(line[6:11]),
                    "xyz": xyz,
                    "element": line[76:78].strip().upper() if len(line) > 77 else "",
                }
                base_key = (atom["chain"], atom["res_id"], atom["ins_code"])
                signature = (*base_key, atom["resname"])
                if signature != current_signature:
                    occurrence = counts.get(base_key, 0)
                    counts[base_key] = occurrence + 1
                    current_residue = {
                        "chain": atom["chain"], "res_id": atom["res_id"],
                        "ins_code": atom["ins_code"], "resname": atom["resname"],
                        "occurrence": occurrence, "segment": segment, "atom_indices": [],
                    }
                    residues.append(current_residue)
                    current_signature = signature
                current_residue["atom_indices"].append(len(self.atoms))
                self.atoms.append(atom)
            records.append((line, atom))
        if not self.atoms:
            raise ValueError(f"No coordinate atoms in {self.path}")
        self.links = [line for line in self.lines if line.startswith(("CONECT", "LINK  ", "SSBOND"))]
        self.records = tuple(records)
        self.residue_occurrences = tuple(residues)

    @classmethod
    def read(cls, path: str | Path) -> "PDBSource":
        path = Path(path)
        return cls(path, path.read_text())

    @classmethod
    def from_text(cls, text: str, path: str | Path | None = None) -> "PDBSource":
        return cls(Path(path) if path is not None else Path("<memory>"), text)

    def cached(self, name: Hashable, factory: Callable[[], _Value]) -> _Value:
        if name not in self._views:
            self._views[name] = factory()
        return self._views[name]

    @property
    def nonstandard_codes(self) -> list[str]:
        return self.cached(
            "nonstandard_codes",
            lambda: sorted({atom["resname"] for atom in self.atoms}
                           - CANONICAL_RESIDUE_NAMES - set(STANDARD_ALIASES)),
        )
