"""Lightweight input records and final PDB identity extraction."""

from __future__ import annotations
import ast
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

STANDARD = dict(
    zip(
        (
            "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL"
        ).split(),
        "ARNDCQEGHILKMFPSTWYV",
    )
)
STANDARD_ALIASES = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS"}
SUPPORTED_TOPOLOGIES = {
    "none",
    "residue_template",
    "head_to_tail",
    "disulfide",
    "stapled",
    "bicycle",
    "staple_rty",
    "mk8_staple",
    "nle_staple",
}


def normalize_type(value):
    value = str(value or "none").strip().lower().replace("-", "_").replace(" ", "_")
    return {
        "head_tail": "head_to_tail",
        "bicyclic": "bicycle",
        "staple": "stapled",
        "isopep": "stapled",
        "isopeptide": "stapled",
        "rty": "staple_rty",
        "mk8": "mk8_staple",
        "staple_mk8": "mk8_staple",
        "alkane_staple": "mk8_staple",
        "nle": "nle_staple",
        "staple_nle": "nle_staple",
        "alkene_staple": "nle_staple",
        "residue_replacement": "residue_template",
        "linear_residue": "residue_template",
        "nonstandard_residue": "residue_template",
    }.get(value, value)


def decode_topology(value):
    if value is None or value == "":
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = ast.literal_eval(value)
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, dict) for item in value
    ):
        raise ValueError("topology_entries must be a list of mappings")
    return [dict(item) for item in value]


def strict_bool(value, *, field_name="boolean"):
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.lower() in {"true", "false"}:
        return value.lower() == "true"
    raise ValueError(f"{field_name} must be true or false")


def pdb_atoms(path):
    atoms = []
    for line in Path(path).read_text().splitlines():
        if line.startswith(("ATOM  ", "HETATM")):
            if len(line) < 54:
                raise ValueError(f"Truncated coordinate record in {path}")
            xyz = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
            if not all(math.isfinite(value) for value in xyz):
                raise ValueError(f"Non-finite coordinates in {path}")
            atoms.append(
                {
                    "chain": line[21:22].strip() or " ",
                    "res_id": line[22:26].strip(),
                    "ins_code": line[26:27].strip(),
                    "resname": line[17:20].strip(),
                    "atom": line[12:16].strip(),
                    "atom_id": int(line[6:11]),
                    "xyz": xyz,
                    "element": line[76:78].strip().upper() if len(line) > 77 else "",
                }
            )
    if not atoms:
        raise ValueError(f"No coordinate atoms in {path}")
    return atoms


def nonstandard_codes(path):
    return sorted(
        {a["resname"] for a in pdb_atoms(path)} - set(STANDARD) - set(STANDARD_ALIASES)
    )


def final_identity(path, binder_chain, original=None):
    atoms = pdb_atoms(path)
    by_chain = {}
    for atom in atoms:
        if atom["atom"] != "CA":
            continue
        key = (atom["res_id"], atom["ins_code"])
        by_chain.setdefault(atom["chain"], {})[key] = atom["resname"]
    if binder_chain not in by_chain:
        raise ValueError(f"Binder chain {binder_chain!r} absent from final structure")
    chains = []
    extras = {}
    selected = {binder_chain}
    if original:
        try:
            declared = json.loads(original.get("design_chains", "[]"))
            selected = {str(item["chain"]) for item in declared} or selected
            extras = {
                str(item["chain"]): {
                    key: item[key]
                    for key in ("sub_modality", "cyclic_offset")
                    if key in item
                }
                for item in declared
            }
        except (ValueError, TypeError, KeyError):
            pass
    for chain, residues in by_chain.items():
        if chain in selected:
            sequence = "".join(
                STANDARD.get(STANDARD_ALIASES.get(name, name), f"[{name}]")
                for name in residues.values()
            )
            chains.append(
                {
                    "chain": chain,
                    "modality": "protein",
                    "sequence": sequence,
                    "residue_count": len(residues),
                    **extras.get(chain, {}),
                }
            )
    binder = next(item for item in chains if item["chain"] == binder_chain)
    return {
        "design_chains": json.dumps(chains, separators=(",", ":")),
        "design_chain_id": binder_chain,
        "design_sequence": binder["sequence"],
        "token_length": binder["residue_count"],
    }


@dataclass
class WorkItem:
    path: Path
    task_name: str
    sample_id: str
    base_sample_id: Any = ""
    binder_chain: str = "L"
    topology_entries: list = field(default_factory=list)
    method: str = "auto"
    relax_linear: bool = True
    row: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    result_path: Path | None = None
    report_dir: Path | None = None
    replace: bool = True
    candidate_id: str = ""


def refresh_locators(original_path, output_path, row):
    """Refresh declared atom locators after the backend writes its final atom order."""
    if not row:
        return {}
    source_atoms = pdb_atoms(original_path)
    final_atoms = pdb_atoms(output_path)
    source_by_serial = {atom["atom_id"]: atom for atom in source_atoms}
    final_by_key = {
        (atom["chain"], str(atom["res_id"]), atom["ins_code"], atom["atom"]): atom
        for atom in final_atoms
    }
    final_residues = {
        (atom["chain"], str(atom["res_id"]), atom["ins_code"]) for atom in final_atoms
    }

    def visit(value):
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        item = {key: visit(child) for key, child in value.items()}
        if "chain" not in item:
            return item
        chain = str(item["chain"])
        resid = item.get("res_id")
        ins = str(item.get("ins_code") or "")
        name = item.get("atom", item.get("atom_name"))
        if name is not None or item.get("atom_id") is not None:
            old = None
            try:
                old = source_by_serial.get(int(item.get("atom_id")))
            except (TypeError, ValueError):
                pass
            if resid is None and old is not None:
                resid = old["res_id"]
            if name is None and old is not None:
                name = old["atom"]
            key = (chain, str(resid), ins, str(name))
            if key not in final_by_key:
                raise ValueError(
                    f"Declared atom locator is absent after postprocess: {key}"
                )
            atom = final_by_key[key]
            item["atom_id"] = atom["atom_id"]
            if "atom" in item:
                item["atom"] = atom["atom"]
            if "atom_name" in item:
                item["atom_name"] = atom["atom"]
        elif resid is not None and (chain, str(resid), ins) not in final_residues:
            raise ValueError(
                f"Declared residue locator is absent after postprocess: {chain}:{resid}{ins}"
            )
        return item

    refreshed = {}
    for key in ("fixed_restypes", "bond_constraints"):
        if row.get(key) not in (None, ""):
            value = row[key]
            value = json.loads(value) if isinstance(value, str) else value
            refreshed[key] = json.dumps(visit(value), separators=(",", ":"))
    return refreshed
