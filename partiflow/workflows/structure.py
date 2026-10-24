"""Protein structure IO with reversible public residue occurrences."""

from __future__ import annotations
import os
from pathlib import Path
from partiflow.common.records import CandidateError
import numpy as np
from biotite import structure as struc
from biotite.structure.io import pdb, pdbx

AA3_TO_1 = dict(
    zip(
        (
            "ALA",
            "ARG",
            "ASN",
            "ASP",
            "CYS",
            "GLN",
            "GLU",
            "GLY",
            "HIS",
            "ILE",
            "LEU",
            "LYS",
            "MET",
            "PHE",
            "PRO",
            "SER",
            "THR",
            "TRP",
            "TYR",
            "VAL",
        ),
        "ARNDCQEGHILKMFPSTWYV",
    )
)
STANDARD_ATOMS = {
    "ALA": frozenset(("C", "CA", "CB", "N", "O", "OXT")),
    "ARG": frozenset(
        ("C", "CA", "CB", "CD", "CG", "CZ", "N", "NE", "NH1", "NH2", "O", "OXT")
    ),
    "ASN": frozenset(("C", "CA", "CB", "CG", "N", "ND2", "O", "OD1", "OXT")),
    "ASP": frozenset(("C", "CA", "CB", "CG", "N", "O", "OD1", "OD2", "OXT")),
    "CYS": frozenset(("C", "CA", "CB", "N", "O", "OXT", "SG")),
    "GLN": frozenset(("C", "CA", "CB", "CD", "CG", "N", "NE2", "O", "OE1", "OXT")),
    "GLU": frozenset(("C", "CA", "CB", "CD", "CG", "N", "O", "OE1", "OE2", "OXT")),
    "GLY": frozenset(("C", "CA", "N", "O", "OXT")),
    "HIS": frozenset(
        ("C", "CA", "CB", "CD2", "CE1", "CG", "N", "ND1", "NE2", "O", "OXT")
    ),
    "ILE": frozenset(("C", "CA", "CB", "CD1", "CG1", "CG2", "N", "O", "OXT")),
    "LEU": frozenset(("C", "CA", "CB", "CD1", "CD2", "CG", "N", "O", "OXT")),
    "LYS": frozenset(("C", "CA", "CB", "CD", "CE", "CG", "N", "NZ", "O", "OXT")),
    "MET": frozenset(("C", "CA", "CB", "CE", "CG", "N", "O", "OXT", "SD")),
    "PHE": frozenset(
        ("C", "CA", "CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "N", "O", "OXT")
    ),
    "PRO": frozenset(("C", "CA", "CB", "CD", "CG", "N", "O", "OXT")),
    "SER": frozenset(("C", "CA", "CB", "N", "O", "OG", "OXT")),
    "THR": frozenset(("C", "CA", "CB", "CG2", "N", "O", "OG1", "OXT")),
    "TRP": frozenset(
        (
            "C",
            "CA",
            "CB",
            "CD1",
            "CD2",
            "CE2",
            "CE3",
            "CG",
            "CH2",
            "CZ2",
            "CZ3",
            "N",
            "NE1",
            "O",
            "OXT",
        )
    ),
    "TYR": frozenset(
        ("C", "CA", "CB", "CD1", "CD2", "CE1", "CE2", "CG", "CZ", "N", "O", "OH", "OXT")
    ),
    "VAL": frozenset(("C", "CA", "CB", "CG1", "CG2", "N", "O", "OXT")),
}
SOURCE_ANNOTATIONS = (
    "source_format",
    "source_graph_complete",
    "source_break_before",
    "source_polymer_sequence",
    "source_polymer_position",
)


class UnsupportedProteinChemistryError(CandidateError):
    code = "unsupported_chemistry"

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = dict(details or {})


def read_structure(path):
    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        file = pdbx.CIFFile.read(path)
        atoms = pdbx.get_structure(
            file,
            model=1,
            use_author_fields=False,
            include_bonds=True,
            extra_fields=["atom_id", "b_factor", "occupancy"],
        )
        site = next(iter(file.values()))["atom_site"]
        source_ids = np.asarray(site["id"].as_array(str))
        rows = {str(value): index for index, value in enumerate(source_ids)}
        order = [rows[str(value)] for value in atoms.atom_id]
        for column, annotation in [
            ("auth_asym_id", "auth_asym_id"),
            ("auth_seq_id", "auth_seq_id"),
            ("pdbx_PDB_ins_code", "auth_ins_code"),
            ("label_seq_id", "label_seq_id"),
        ]:
            if column in site:
                values = np.asarray(site[column].as_array(str))[order]
                if column == "pdbx_PDB_ins_code":
                    values = np.asarray(
                        ["" if value in (".", "?") else value for value in values]
                    )
                atoms.set_annotation(annotation, values)
    else:
        file = pdb.PDBFile.read(path)
        atoms = pdb.get_structure(
            file,
            model=1,
            include_bonds=False,
            extra_fields=["atom_id", "b_factor", "occupancy"],
        )
    if not len(atoms):
        raise ValueError(f"Empty protein structure: {path}")
    if not np.isfinite(atoms.coord).all():
        raise ValueError("Protein structure has non-finite coordinates")
    categories = set(atoms.get_annotation_categories())
    groups = {}
    uid = np.empty(len(atoms), dtype=np.int64)
    for row in range(len(atoms)):
        key = (
            str(atoms.chain_id[row]),
            int(atoms.res_id[row]),
            str(atoms.ins_code[row]),
        )
        uid[row] = groups.setdefault(key, len(groups))
    for name, values in {
        "label_asym_id": np.asarray(atoms.chain_id).astype(str),
        "label_seq_id": np.asarray(atoms.res_id).astype(str),
        "auth_asym_id": np.asarray(atoms.chain_id).astype(str),
        "auth_seq_id": np.asarray(atoms.res_id).astype(str),
        "auth_ins_code": np.asarray(atoms.ins_code).astype(str),
        "residue_uid": uid,
        "token_uid": uid,
        "atom_uid": np.arange(len(atoms), dtype=np.int64),
        "asym_uid": np.asarray(
            [
                list(dict.fromkeys(atoms.chain_id.tolist())).index(chain)
                for chain in atoms.chain_id
            ]
        ),
        "coord_valid": np.ones(len(atoms), dtype=bool),
        "source_format": np.full(
            len(atoms), "cif" if path.suffix.lower() in (".cif", ".mmcif") else "pdb"
        ),
        "mol_type": np.asarray(
            [
                "protein" if str(name) in AA3_TO_1 or str(name) == "UNK" else "other"
                for name in atoms.res_name
            ]
        ),
    }.items():
        if name not in categories:
            atoms.set_annotation(name, values)
    if path.suffix.lower() not in (".cif", ".mmcif"):
        bonds = (
            {}
            if atoms.bonds is None
            else {
                tuple(sorted((int(a), int(b)))): int(o)
                for a, b, o in atoms.bonds.as_array()
            }
        )
        lines = path.read_text().splitlines()
        ter_before = set()
        after_ter = False
        previous = None
        for line in lines:
            if line.startswith("TER"):
                after_ter = True
            elif line.startswith(("ATOM  ", "HETATM")):
                key = (line[21:22].strip(), int(line[22:26]), line[26:27].strip())
                if after_ter and key != previous:
                    ter_before.add(key)
                previous = key
                after_ter = False
        atoms.set_annotation(
            "source_break_before",
            np.asarray(
                [
                    (
                        str(atoms.chain_id[row]),
                        int(atoms.res_id[row]),
                        str(atoms.ins_code[row]),
                    )
                    in ter_before
                    for row in range(len(atoms))
                ]
            ),
        )
        serial_rows = {int(serial): row for row, serial in enumerate(atoms.atom_id)}
        if len(serial_rows) != len(atoms):
            raise ValueError("PDB atom serials must be unique")
        for line in lines:
            if line.startswith("CONECT"):
                serials = [
                    int(line[start : start + 5])
                    for start in range(6, min(len(line), 31), 5)
                    if line[start : start + 5].strip()
                ]
                if serials and serials[0] in serial_rows:
                    left = serial_rows[serials[0]]
                    for serial in serials[1:]:
                        if serial in serial_rows and serial_rows[serial] != left:
                            bonds.setdefault(
                                tuple(sorted((left, serial_rows[serial]))), 0
                            )
            if line.startswith("LINK  "):
                locs = [
                    (
                        line[21:22].strip(),
                        int(line[22:26]),
                        line[26:27].strip(),
                        line[12:16].strip(),
                    ),
                    (
                        line[51:52].strip(),
                        int(line[52:56]),
                        line[56:57].strip(),
                        line[42:46].strip(),
                    ),
                ]
                endpoints = []
                for chain, res_id, ins, name in locs:
                    found = np.flatnonzero(
                        (atoms.chain_id == chain)
                        & (atoms.res_id == res_id)
                        & (atoms.ins_code == ins)
                        & (atoms.atom_name == name)
                    )
                    if len(found) != 1:
                        raise ValueError("LINK endpoint is not unique")
                    endpoints.append(int(found[0]))
                bonds.setdefault(tuple(sorted(endpoints)), 1)
            if line.startswith("SSBOND"):
                locs = [
                    (line[15:16].strip(), int(line[17:21]), line[21:22].strip()),
                    (line[29:30].strip(), int(line[31:35]), line[35:36].strip()),
                ]
                rows = []
                for chain, res_id, ins in locs:
                    found = np.flatnonzero(
                        (atoms.chain_id == chain)
                        & (atoms.res_id == res_id)
                        & (atoms.ins_code == ins)
                        & (atoms.atom_name == "SG")
                    )
                    if len(found) != 1:
                        raise ValueError("SSBOND endpoint is not unique")
                    rows.append(int(found[0]))
                bonds[tuple(sorted(rows))] = 1
        for (left, right), order in list(bonds.items()):
            if (
                order != 0
                or atoms.chain_id[left] != atoms.chain_id[right]
                or atoms.residue_uid[left] == atoms.residue_uid[right]
            ):
                continue
            names = {str(atoms.atom_name[left]), str(atoms.atom_name[right])}
            standard = (
                str(atoms.res_name[left]) in AA3_TO_1
                and str(atoms.res_name[right]) in AA3_TO_1
            )
            if standard and (
                names == {"C", "N"}
                or (
                    names == {"SG"}
                    and atoms.res_name[left] == atoms.res_name[right] == "CYS"
                )
            ):
                bonds[left, right] = 1
        for chain in dict.fromkeys(atoms.chain_id.tolist()):
            residues = polymer_residue_groups(atoms, chain)
            for left, right in zip(residues, residues[1:]):
                a = [row for row in left["row_indices"] if atoms.atom_name[row] == "C"]
                b = [row for row in right["row_indices"] if atoms.atom_name[row] == "N"]
                if (
                    str(atoms.chain_id[right["row_indices"][0]]),
                    int(atoms.res_id[right["row_indices"][0]]),
                    str(atoms.ins_code[right["row_indices"][0]]),
                ) in ter_before:
                    continue
                if (
                    len(a) == len(b) == 1
                    and np.linalg.norm(atoms.coord[a[0]] - atoms.coord[b[0]]) <= 2.0
                ):
                    bonds.setdefault(tuple(sorted((a[0], b[0]))), 1)
        atoms.bonds = struc.BondList(
            len(atoms),
            np.asarray(
                [(a, b, o) for (a, b), o in bonds.items()], dtype=np.uint32
            ).reshape(-1, 3),
        )
    return atoms


def write_structure_atomic(atoms, path, *, entry_id="prediction"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        if path.suffix.lower() in (".cif", ".mmcif"):
            file = pdbx.CIFFile()
            pdbx.set_structure(file, atoms, data_block=entry_id, include_bonds=True)
            site = file[entry_id]["atom_site"]
            for annotation, column in [
                ("label_asym_id", "label_asym_id"),
                ("label_seq_id", "label_seq_id"),
                ("auth_asym_id", "auth_asym_id"),
                ("auth_seq_id", "auth_seq_id"),
                ("auth_ins_code", "pdbx_PDB_ins_code"),
            ]:
                if hasattr(atoms, annotation):
                    site[column] = pdbx.CIFColumn(
                        np.asarray(getattr(atoms, annotation)).astype(str)
                    )
        else:
            file = _pdb_file(atoms)
        file.write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _pdb_file(atoms):
    """Serialize every actual bond, including equal-number insertion residues."""
    if not len(atoms) or not np.isfinite(atoms.coord).all():
        raise ValueError("PDB requires nonempty finite atom coordinates")
    if any((len(str(value)) > 1 for value in atoms.chain_id)) or any(
        (len(str(value)) > 1 for value in atoms.ins_code)
    ):
        raise ValueError(
            "PDB cannot preserve multicharacter chain or insertion identifiers"
        )
    if np.any((atoms.res_id < -999) | (atoms.res_id > 9999)):
        raise ValueError("PDB cannot preserve residue numbers outside -999..9999")
    for annotation, expected in [
        ("label_asym_id", atoms.chain_id),
        ("label_seq_id", atoms.res_id.astype(str)),
        ("auth_asym_id", atoms.chain_id),
        ("auth_seq_id", atoms.res_id.astype(str)),
        ("auth_ins_code", atoms.ins_code),
    ]:
        if hasattr(atoms, annotation) and (
            not np.array_equal(
                np.asarray(getattr(atoms, annotation)).astype(str),
                np.asarray(expected).astype(str),
            )
        ):
            raise ValueError(f"PDB cannot preserve distinct {annotation} identifiers")
    private = atoms.copy()
    if not hasattr(private, "atom_id"):
        private.set_annotation(
            "atom_id", np.arange(1, len(private) + 1, dtype=np.int64)
        )
    serials = np.asarray(private.atom_id, dtype=np.int64)
    if len(set(serials.tolist())) != len(serials) or np.any(
        (serials < 1) | (serials > 99999)
    ):
        raise ValueError("PDB requires unique atom serials in 1..99999")
    file = pdb.PDBFile()
    pdb.set_structure(file, private, hybrid36=False)
    lines = []
    row_by_serial = {int(serial): row for row, serial in enumerate(serials)}
    previous_occurrence = None
    breaks = getattr(private, "source_break_before", np.zeros(len(private), dtype=bool))
    for line in file.lines:
        if line.startswith("CONECT"):
            continue
        if line.startswith(("ATOM  ", "HETATM")):
            row = row_by_serial[int(line[6:11])]
            occurrence = (
                str(private.chain_id[row]),
                int(private.res_id[row]),
                str(private.ins_code[row]),
            )
            if (
                previous_occurrence is not None
                and occurrence != previous_occurrence
                and bool(breaks[row])
            ):
                lines.append("TER")
            previous_occurrence = occurrence
        lines.append(line)
    file.lines = lines
    neighbors = {}
    if private.bonds is not None:
        for left, right, _order in private.bonds.as_array():
            neighbors.setdefault(int(serials[left]), set()).add(int(serials[right]))
            neighbors.setdefault(int(serials[right]), set()).add(int(serials[left]))
    for center, connected in sorted(neighbors.items()):
        connected = sorted(connected)
        for start in range(0, len(connected), 4):
            file.lines.append(
                f"CONECT{center:5d}"
                + "".join((f"{serial:5d}" for serial in connected[start : start + 4]))
            )
    return file


def export_pdb(path, out_path):
    """Export a restored prediction to representable PDB with every bond."""
    atoms = read_structure(path)
    write_structure_atomic(atoms, Path(out_path))
    return Path(out_path)


def canonical_chain_ids(atoms):
    return np.asarray(getattr(atoms, "label_asym_id", atoms.chain_id)).astype(str)


def source_locator(atoms, row):
    def text(name, default):
        return str(getattr(atoms, name, np.full(len(atoms), default))[row])

    chain = text("label_asym_id", str(atoms.chain_id[row]))
    position = int(atoms.res_id[row])
    ins = str(atoms.ins_code[row])
    return dict(
        chain_id=chain,
        res_id=position,
        ins_code=ins,
        label_asym_id=chain,
        label_seq_id=text("label_seq_id", str(position)),
        auth_asym_id=text("auth_asym_id", chain),
        auth_seq_id=text("auth_seq_id", str(position)),
        auth_ins_code=text("auth_ins_code", ins),
        source_format=text("source_format", "pdb"),
    )


def polymer_residue_groups(atoms, chain_id):
    rows = np.flatnonzero(canonical_chain_ids(atoms) == str(chain_id))
    by_uid = {}
    for row in rows:
        by_uid.setdefault(int(atoms.residue_uid[row]), []).append(int(row))
    result = []
    for uid, members in by_uid.items():
        first = members[0]
        record = dict(
            row_indices=members,
            residue_uid=uid,
            token_uid=int(atoms.token_uid[first]),
            locator=source_locator(atoms, first),
            res_name=str(atoms.res_name[first]),
            modality=str(atoms.mol_type[first]),
        )
        for name in SOURCE_ANNOTATIONS:
            if hasattr(atoms, name):
                record[name] = getattr(atoms, name)[first]
        result.append(record)
    return result


def polymer_break_before(atoms, groups):
    pairs = (
        set()
        if atoms.bonds is None
        else {tuple(sorted((int(a), int(b)))) for a, b, _ in atoms.bonds.as_array()}
    )
    result = set()
    complete = getattr(atoms, "source_graph_complete", None)
    recorded = getattr(atoms, "source_break_before", None)
    for index, (left, right) in enumerate(zip(groups, groups[1:]), 1):
        lr, rr = (left["row_indices"][0], right["row_indices"][0])
        ends = [
            (a, b)
            for a in left["row_indices"]
            if atoms.atom_name[a] == "C"
            for b in right["row_indices"]
            if atoms.atom_name[b] == "N"
        ]
        old_break = recorded is not None and bool(recorded[rr])
        graph_complete = (
            complete is not None and bool(complete[lr]) and bool(complete[rr])
        )
        if any((tuple(sorted(edge)) in pairs for edge in ends)) and (
            graph_complete or not old_break
        ):
            continue
        if (
            old_break
            or graph_complete
            or left["modality"] != right["modality"]
            or (atoms.asym_uid[lr] != atoms.asym_uid[rr])
            or (ends and pairs)
        ):
            result.add(index)
            continue
        la = [r for r in left["row_indices"] if atoms.atom_name[r] == "CA"]
        ra = [r for r in right["row_indices"] if atoms.atom_name[r] == "CA"]
        if (
            len(la) == len(ra) == 1
            and np.linalg.norm(atoms.coord[la[0]] - atoms.coord[ra[0]]) > 4.8
        ):
            result.add(index)
    return result


def candidate_input(candidate):
    atoms = read_structure(candidate.pdb_path)
    atoms.set_annotation(
        "is_design", np.asarray(atoms.chain_id) == candidate.binder_chain
    )
    chains = list(dict.fromkeys(atoms.chain_id.tolist()))
    if candidate.binder_chain not in chains:
        raise ValueError("Design chain is absent from structure")
    observed_context = [chain for chain in chains if chain != candidate.binder_chain]
    if candidate.context_chains and (
        len(set(candidate.context_chains)) != len(candidate.context_chains)
        or set(candidate.context_chains) != set(observed_context)
    ):
        raise ValueError("Explicit context chains must preserve every source chain")
    unsupported = sorted(set((str(name) for name in atoms.res_name)) - set(AA3_TO_1))
    if unsupported:
        raise UnsupportedProteinChemistryError(
            f"Protein sequence/fold workflows do not support these residue identities: {unsupported}",
            details={"residues": unsupported},
        )
    groups = polymer_residue_groups(atoms, candidate.binder_chain)
    sequence = "".join((AA3_TO_1.get(group["res_name"], "X") for group in groups))
    if len(groups) != candidate.length or sequence != candidate.sequence:
        raise ValueError("Design sequence/length differs from input structure")
    for entry in candidate.fixed_restypes:
        if not isinstance(entry, dict):
            raise ValueError("Malformed fixed identity entry")
        chain = str(entry.get("chain", entry.get("chain_id", "")))
        res_id = entry.get("res_id")
        ins = str(entry.get("ins_code", ""))
        matches = [
            group
            for group in groups
            if chain == candidate.binder_chain
            and str(group["locator"]["res_id"]) == str(res_id)
            and (group["locator"].get("ins_code", "") == ins)
        ]
        if len(matches) != 1 or entry.get("restype") != matches[0]["res_name"]:
            raise ValueError("Fixed identity differs from its input residue occurrence")
    metadata = candidate.metadata or {}
    return dict(
        atoms=atoms,
        design_chain=candidate.binder_chain,
        context_chains=list(
            candidate.context_chains
            or tuple((chain for chain in chains if chain != candidate.binder_chain))
        ),
        path=str(candidate.pdb_path),
        sample_id=candidate.candidate_id,
        bond_constraints=candidate.bond_constraints,
        cyclic_offset_chains=metadata.get(
            "cyclic_offset_chains",
            [candidate.binder_chain] if metadata.get("cyclic_offset", False) else [],
        ),
    )
