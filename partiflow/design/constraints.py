"""Resolve public constraints against final PDB atom locators."""

from __future__ import annotations
import csv
import json
from pathlib import Path
from partiflow.postprocess.records import (
    pdb_atoms,
    normalize_type,
    decode_topology,
    STANDARD,
    STANDARD_ALIASES,
)


def _locator(atom):
    value = {
        "chain": atom["chain"],
        "res_id": int(atom["res_id"]),
        "atom": atom["atom"],
        "atom_id": atom["atom_id"],
    }
    if atom["ins_code"]:
        value["ins_code"] = atom["ins_code"]
    return value


def populate(candidate, *, final=False):
    atoms = pdb_atoms(candidate.pdb_path)
    lookup = {(a["chain"], int(a["res_id"]), a["atom"]): a for a in atoms}
    residues = {
        int(a["res_id"]): a
        for a in atoms
        if a["chain"] == candidate.binder_chain and a["atom"] == "CA"
    }
    entries = decode_topology(candidate.topology_entries)
    expected = {
        int(e.get("resid", e.get("position", 0))): str(e.get("resname", "")).upper()
        for e in entries
        if normalize_type(e.get("type")) == "residue_template"
    }
    indices = dict(enumerate(STANDARD))
    for position, aa in candidate.metadata.get("residue_type_constraints", {}).items():
        expected.setdefault(int(position), indices[int(aa)])
    fixed = []
    positions = set(map(int, candidate.metadata.get("fixed_positions", []))) | set(
        expected
    )
    for position in sorted(positions):
        if position not in residues:
            raise ValueError(f"Fixed residue {position} missing from candidate")
        atom = residues[position]
        restype = expected.get(position) or atom["resname"]
        actual = STANDARD_ALIASES.get(atom["resname"], atom["resname"])
        if (
            expected.get(position)
            and (final or restype in STANDARD)
            and actual != restype
        ):
            raise ValueError(f"Fixed identity missing: {restype} at {position}")
        item = {"chain": candidate.binder_chain, "res_id": position, "restype": restype}
        if atom["ins_code"]:
            item["ins_code"] = atom["ins_code"]
        fixed.append(item)
    candidate.fixed_restypes = fixed
    connections = []
    for entry in entries:
        kind = normalize_type(entry.get("type"))
        if kind in {"none", "residue_template"}:
            continue
        anchors = list(map(int, entry.get("anchors") or []))
        if kind == "head_to_tail":
            left, right = anchors if len(anchors) == 2 else (1, candidate.length)
            pairs = [(left, "N", right, "C", "single")]
        elif kind == "disulfide" and len(anchors) == 2:
            pairs = [(anchors[0], "SG", anchors[1], "SG", "single")]
        elif kind in {"mk8_staple", "nle_staple"} and len(anchors) == 2:
            code = "MK8" if kind == "mk8_staple" else "NLE"
            table = (
                Path(__file__).parents[1]
                / "postprocess"
                / "rosetta_ncaa"
                / "support_matrix.csv"
            )
            with table.open() as handle:
                row = next(
                    row for row in csv.DictReader(handle) if row["residue_code"] == code
                )
            order = {"1": "single", "2": "double", "3": "triple"}[
                row.get("bond_order") or "1"
            ]
            name = row.get("inter_residue_bond") or "CE"
            pairs = [(anchors[0], name, anchors[1], name, order)]
        elif kind == "stapled" and len(anchors) == 2:
            acceptor = (
                "CD" if (candidate.binder_chain, anchors[1], "CD") in lookup else "CG"
            )
            pairs = [(anchors[0], "NZ", anchors[1], acceptor, "single")]
        elif kind == "staple_rty" and len(anchors) == 3:
            a, b, c = anchors
            pairs = [
                (a, "N", b, "C2", "single"),
                (a, "C", b, "N", "single"),
                (c, "N", b, "C6", "single"),
            ]
        elif kind == "bicycle" and len(anchors) == 3:
            pairs = []
            for resid, name in zip(anchors, ("CM1", "CM2", "CM3")):
                links = [
                    a
                    for a in atoms
                    if a["chain"] == candidate.binder_chain and a["atom"] == name
                ]
                if len(links) != 1:
                    if final:
                        raise ValueError(
                            f"Bicycle linker atom missing or ambiguous: {name}"
                        )
                    continue
                pairs.append((resid, "SG", int(links[0]["res_id"]), name, "single"))
        else:
            raise ValueError(f"Invalid topology anchors: {kind}: {anchors}")
        for left, name1, right, name2, order in pairs:
            a = lookup.get((candidate.binder_chain, left, name1))
            b = lookup.get((candidate.binder_chain, right, name2))
            if a is None or b is None:
                if final:
                    raise ValueError(
                        f"Declared bond atom missing: {left}:{name1}-{right}:{name2}"
                    )
                continue
            aa, bb = sorted((a, b), key=lambda atom: atom["atom_id"])
            connections.append(
                {
                    "atom1": _locator(aa),
                    "atom2": _locator(bb),
                    "order": order,
                    "required_output": True,
                }
            )
    candidate.bond_constraints = connections
    return atoms


def public_identity(candidate):
    atoms = populate(candidate, final=True)
    residues = []
    seen = set()
    for atom in atoms:
        key = (atom["chain"], atom["res_id"], atom["ins_code"])
        if (
            atom["chain"] == candidate.binder_chain
            and atom["atom"] == "CA"
            and key not in seen
        ):
            seen.add(key)
            residues.append(atom)
    if not residues:
        raise ValueError(
            f"Final structure has no peptide residues on chain {candidate.binder_chain}"
        )
    from partiflow.postprocess.records import STANDARD, STANDARD_ALIASES

    sequence = "".join(
        STANDARD.get(
            STANDARD_ALIASES.get(a["resname"], a["resname"]), f"[{a['resname']}]"
        )
        for a in residues
    )
    chain = {
        "chain": candidate.binder_chain,
        "modality": "protein",
        "sequence": sequence,
        "residue_count": len(residues),
        "sub_modality": "peptide",
    }
    if candidate.metadata.get("cyclic_offset"):
        chain["cyclic_offset"] = True
    return {
        "design_chains": json.dumps([chain], separators=(",", ":")),
        "design_chain_id": candidate.binder_chain,
        "design_sequence": sequence,
        "token_length": len(residues),
        "fixed_restypes": json.dumps(candidate.fixed_restypes, separators=(",", ":")),
        "bond_constraints": json.dumps(
            candidate.bond_constraints, separators=(",", ":")
        ),
    }


def serialize_connections(candidate):
    populate(candidate, final=True)
    if not candidate.bond_constraints:
        return
    path = Path(candidate.pdb_path)
    lines = path.read_text().splitlines()
    existing = set()
    for line in lines:
        if line.startswith("CONECT"):
            serials = [
                int(line[i : i + 5])
                for i in range(6, len(line), 5)
                if line[i : i + 5].strip()
            ]
            if serials:
                existing.update(
                    tuple(sorted((serials[0], other))) for other in serials[1:]
                )
    extra = []
    for item in candidate.bond_constraints:
        left, right = item["atom1"]["atom_id"], item["atom2"]["atom_id"]
        if tuple(sorted((left, right))) not in existing:
            count = {"single": 1, "double": 2, "triple": 3}.get(item["order"], 1)
            extra.append(f"CONECT{left:5d}" + f"{right:5d}" * count)
    if extra:
        lines = [line for line in lines if line.strip() != "END"] + extra + ["END"]
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text("\n".join(lines) + "\n")
        temporary.replace(path)
