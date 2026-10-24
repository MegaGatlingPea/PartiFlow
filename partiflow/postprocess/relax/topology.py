from __future__ import annotations

import io
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np
import openmm
import pdbfixer
from openmm import Vec3
from openmm import app as openmm_app
from openmm import unit
from openmm.app import Modeller, PDBFile

from partiflow.openfold.np.residue_constants import restype_1to3
from partiflow.utils.conditioning_utils import design_condition_mode_to_name
from partiflow.utils.nonstandard_residue_registry import get_nonstandard_residue_spec

ASSET_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "utils", "relax_assets")
CUSTOM_RESIDUE_XML_PATH = os.path.join(ASSET_DIR, "topology.xml")
TMB_PDB_PATH = os.path.join(ASSET_DIR, "1_3_5_TMB_with_H.pdb")
RTY_PDB_PATH = os.path.join(ASSET_DIR, "ideal_RTY_with_H.pdb")

STAPLED_RESNAME_MAP = {"LYS": "LYZ", "ASP": "ADZ", "GLU": "GLZ"}
STAPLED_DELETE_ATOMS = {"LYS": {"HZ2", "HZ3"}, "ASP": {"OD2"}, "GLU": {"OE2"}}
STAPLED_CONNECT_ATOMS = {"LYZ": "NZ", "ADZ": "CG", "GLZ": "CD"}

BICYCLE_RESNAME_NTERM = "CYN"
BICYCLE_RESNAME_INTERNAL = "CYZ"
BICYCLE_RESNAME_CTERM = "CYC"
BICYCLE_CUSTOM_RESIDUES = {"CYN", "CYZ", "CYC", "TMB"}
STAPLED_CUSTOM_RESIDUES = {"LYZ", "ADZ", "GLZ", "NLYZ", "CLYZ", "NADZ", "CADZ", "NGLZ", "CGLZ"}
STAPLE_RTY_RESNAME = "RTY"
STAPLE_RTY_DELETE_ATOMS = {"H2", "O2", "H20", "BR1"}
STAPLE_RTY_CUSTOM_RESIDUES = {"RTY"}
CC_STAPLE_SPECS = {
    "mk8_staple": {
        "resname": "MK8",
        "connect_atom": "CE",
        "custom_residue_names": {"MK8"},
    },
    "nle_staple": {
        "resname": "NLE",
        "connect_atom": "CE",
        "custom_residue_names": {"NLE"},
    },
}


def _terminal_variant_resname(base: str, pos: int, chain_len: int) -> str:
    if pos == 0:
        return f"N{base}"
    if pos == chain_len - 1:
        return f"C{base}"
    return base


def _resolve_terminal_variant_spec(base_spec, pos: int, chain_len: int):
    """Return the registry spec for the position-appropriate terminal variant
    of `base_spec` (e.g. NMK8/CMK8 for MK8). Falls back to `base_spec` if no
    such variant is registered."""
    variant_name = _terminal_variant_resname(base_spec.resname, pos, chain_len)
    if variant_name == base_spec.resname:
        return base_spec
    return get_nonstandard_residue_spec(variant_name) or base_spec

# Fixed application order: lighter/backbone first, heavier/sidechain last.
APPLY_ORDER = ["residue_template", "head_to_tail", "disulfide", "stapled", "mk8_staple", "nle_staple", "bicycle", "staple_rty"]
SUPPORTED_TYPES = set(APPLY_ORDER)

# Receptor-side non-standard residues that PDBFixer's built-in `substitutions`
# table does not cover and that do not yet have a custom passive template.
_EXTRA_NONSTANDARD_SUBSTITUTIONS: dict[str, str] = {
    "NVA": "VAL",
    "ORN": "LYS",
    "DAB": "LYS",
    "DAP": "ALA",
    "AIB": "ALA",
    "MLE": "LEU",
    "MVA": "VAL",
    "MLY": "LYS",
    "M3L": "LYS",
    "SEC": "CYS",
    "PYL": "LYS",
    "HYP": "PRO",
    "PCA": "GLN",
    "TPO": "THR",
    "SEP": "SER",
    "PTR": "TYR",
    "CSO": "CYS",
    "CME": "CYS",
    "OCS": "CYS",
    "KCX": "LYS",
    "FME": "MET",
}
# Design-side custom residue names that must never be mapped to a standard one
# — they have their own templates in the custom topology XML.
_DESIGN_CUSTOM_RESIDUE_NAMES: set[str] = (
    BICYCLE_CUSTOM_RESIDUES
    | STAPLED_CUSTOM_RESIDUES
    | STAPLE_RTY_CUSTOM_RESIDUES
    | {spec["resname"] for spec in CC_STAPLE_SPECS.values()}
    | {name for spec in CC_STAPLE_SPECS.values() for name in spec["custom_residue_names"]}
)


# ---------------------------------------------------------------------------
# Data objects
# ---------------------------------------------------------------------------

@dataclass
class TopologyEntry:
    type: str
    anchors: list[int]
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyFixResult:
    pdb_str: str
    topology: Any
    positions: Any
    connects: list[str]
    custom_residue_names: set[str]
    residue_template_overrides: dict[tuple[str, str], str]
    applied_entries: list[dict[str, Any]]
    skipped_entries: list[dict[str, Any]]
    warnings: list[str]
    notes: list[str]
    force_fixed_residues: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_type(value: Any) -> str:
    if value is None:
        return "none"
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"staple", "stapled", "isopep", "isopeptide"}:
        normalized = "stapled"
    if normalized == "head_tail":
        normalized = "head_to_tail"
    if normalized == "bicyclic":
        normalized = "bicycle"
    if normalized in {"residue_template", "residue_replacement", "linear_residue", "nonstandard_residue"}:
        return "residue_template"
    if normalized in {"staple_rty", "rty"}:
        return "staple_rty"
    if normalized in {"mk8_staple", "staple_mk8", "mk8"}:
        return "mk8_staple"
    if normalized in {"nle_staple", "staple_nle", "nle"}:
        return "nle_staple"
    if normalized in {"alkane_staple"}:
        return "mk8_staple"
    if normalized in {"alkene_staple"}:
        return "nle_staple"
    return design_condition_mode_to_name(normalized)


def _write_pdb(topology, positions) -> str:
    buf = io.StringIO()
    openmm_app.PDBFile.writeFile(topology, positions, buf, keepIds=True)
    return buf.getvalue()


def _written_atom_serials(topology, pdb_str: str) -> dict[Any, int]:
    """Map OpenMM atoms to the serials in the PDB just written.

    PDBFile.writeFile numbers records in its output stream, not by
    atom.index + 1.  In particular, every TER record consumes a serial,
    so atoms after an earlier chain can be offset from their OpenMM
    indices.  Pairing the writer's ATOM/HETATM records with the same ordered
    topology atoms keeps explicit CONECT records tied to the actual output.
    """
    atoms = list(topology.atoms())
    serials: list[int] = []
    for line in pdb_str.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        raw_serial = line[6:11].strip()
        if not raw_serial:
            raise ValueError("OpenMM PDB output contains an atom record without a serial")
        try:
            serials.append(int(raw_serial))
        except ValueError as exc:
            raise ValueError(
                f"OpenMM PDB output contains a non-integer atom serial: {raw_serial!r}"
            ) from exc
    if len(serials) != len(atoms):
        raise ValueError(
            "OpenMM PDB atom record count does not match topology atom count: "
            f"records={len(serials)} topology={len(atoms)}"
        )
    return dict(zip(atoms, serials))



def _add_connects(pdb_str: str, connects: list[str]) -> str:
    pdb_str = pdb_str.strip().rstrip("END").strip()
    lines = pdb_str.splitlines()
    lines.extend(connects)
    lines.append("END")
    return "\n".join(lines) + "\n"


def _kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    C = np.dot(P.T, Q)
    V, S, W = np.linalg.svd(C)
    if (np.linalg.det(V) * np.linalg.det(W)) < 0:
        S[-1] = -S[-1]
        V[:, -1] = -V[:, -1]
    return np.dot(V, W)


def _kabsch(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_mean, b_mean = a.mean(0), b.mean(0)
    R = _kabsch_rotation(a - a_mean, b - b_mean)
    t = b_mean - np.dot(a_mean, R)
    return np.dot(a, R) + t, R, t


def _get_tmb_h3_atoms(topology) -> list:
    order = ["H13", "H23", "H33"]
    atoms: dict[str, Any] = {}
    for chain in topology.chains():
        for res in chain.residues():
            for atom in res.atoms():
                if atom.name in order:
                    atoms[atom.name] = atom
    return [atoms[n] for n in order]


def _make_tmb(sg_coords_nm: np.ndarray, chain_id: str, resid: int):
    """Build an aligned TMB molecule from three SG coordinates (nanometers)."""
    tmb = PDBFile(open(TMB_PDB_PATH, "r"))
    pos = np.array([[c.x, c.y, c.z] for c in tmb.getPositions()], dtype=np.float64)
    h3_atoms = _get_tmb_h3_atoms(tmb.topology)
    h3_coords = np.array(
        [[tmb.positions[a.index].x, tmb.positions[a.index].y, tmb.positions[a.index].z]
         for a in h3_atoms],
        dtype=np.float64,
    )
    _, R, t = _kabsch(h3_coords, sg_coords_nm)
    pos = np.dot(pos, R) + t
    tmb.positions = openmm.unit.Quantity(
        [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in pos],
        unit=openmm.unit.nanometer,
    )
    modeller = Modeller(tmb.topology, tmb.positions)
    for chain in modeller.topology.chains():
        chain.id = chain_id
        for res in chain.residues():
            res.id = str(resid)
    modeller.delete(_get_tmb_h3_atoms(modeller.topology))
    return modeller.topology, modeller.positions


def _make_rty(anchor_coords_nm: np.ndarray, chain_id: str, resid: int):
    """Build an aligned RTY precursor from three product-state anchor coordinates."""
    rty = PDBFile(open(RTY_PDB_PATH, "r"))
    pos = np.array([[c.x, c.y, c.z] for c in rty.getPositions()], dtype=np.float64)
    atom_by_name = {atom.name: atom for atom in rty.topology.atoms()}
    ref_atom_names = ["C2", "CA", "C6"]
    ref_coords = np.array(
        [[rty.positions[atom_by_name[name].index].x,
          rty.positions[atom_by_name[name].index].y,
          rty.positions[atom_by_name[name].index].z]
         for name in ref_atom_names],
        dtype=np.float64,
    )
    _, R, t = _kabsch(ref_coords, anchor_coords_nm)
    pos = np.dot(pos, R) + t
    rty.positions = openmm.unit.Quantity(
        [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in pos],
        unit=openmm.unit.nanometer,
    )
    modeller = Modeller(rty.topology, rty.positions)
    for chain in modeller.topology.chains():
        chain.id = chain_id
        for res in chain.residues():
            res.id = str(resid)
    delete_atoms = [atom for atom in modeller.topology.atoms() if atom.name in STAPLE_RTY_DELETE_ATOMS]
    modeller.delete(delete_atoms)
    return modeller.topology, modeller.positions


def _make_nonstandard_residue(
    residue_spec,
    anchor_coords_nm: np.ndarray,
    chain_id: str,
    resid: int,
    *,
    role: str,
):
    """Build an aligned non-standard residue from generated proxy coordinates."""
    residue_pdb = PDBFile(open(residue_spec.pdb_path_for_role(role), "r"))
    pos = np.array([[c.x, c.y, c.z] for c in residue_pdb.getPositions()], dtype=np.float64)
    atom_by_name = {atom.name: atom for atom in residue_pdb.topology.atoms()}
    ref_atom_names = _spec_alignment_atom_names(residue_spec)
    ref_coords = np.array(
        [[residue_pdb.positions[atom_by_name[name].index].x,
          residue_pdb.positions[atom_by_name[name].index].y,
          residue_pdb.positions[atom_by_name[name].index].z]
         for name in ref_atom_names],
        dtype=np.float64,
    )
    _, R, t = _kabsch(ref_coords, anchor_coords_nm)
    pos = np.dot(pos, R) + t
    residue_pdb.positions = openmm.unit.Quantity(
        [Vec3(float(p[0]), float(p[1]), float(p[2])) for p in pos],
        unit=openmm.unit.nanometer,
    )
    modeller = Modeller(residue_pdb.topology, residue_pdb.positions)
    for chain in modeller.topology.chains():
        chain.id = chain_id
        for res in chain.residues():
            res.id = str(resid)
    return modeller.topology, modeller.positions


def _make_cc_staple_residue(spec: dict[str, Any], anchor_coords_nm: np.ndarray, chain_id: str, resid: int):
    residue_spec = get_nonstandard_residue_spec(spec["resname"])
    if residue_spec is None:
        raise ValueError(f"Unsupported C-C staple residue '{spec['resname']}'.")
    return _make_nonstandard_residue(residue_spec, anchor_coords_nm, chain_id, resid, role="active")


def _make_template_residue(resname: str, anchor_coords_nm: np.ndarray, chain_id: str, resid: int):
    """Build an aligned non-standard residue from a generated proxy residue."""
    spec = get_nonstandard_residue_spec(resname)
    if spec is None:
        raise ValueError(f"Unsupported non-standard residue template '{resname}'.")
    return _make_nonstandard_residue(spec, anchor_coords_nm, chain_id, resid, role="passive")


def _proxy_residue_name(proxy_aa: str) -> str:
    proxy = str(proxy_aa).strip().upper()
    return restype_1to3.get(proxy, proxy)


def _alignment_atom_names(proxy_aa: str) -> list[str]:
    return ["N", "CA", "C"] if str(proxy_aa).strip().upper() == "G" else ["N", "CA", "C", "CB"]


def _spec_alignment_atom_names(residue_spec) -> list[str]:
    if getattr(residue_spec, "alignment_atoms", None):
        return list(residue_spec.alignment_atoms)
    return _alignment_atom_names(residue_spec.proxy_aa)


@lru_cache(maxsize=None)
def _custom_residue_template_graphs() -> dict[str, dict[str, set[frozenset[str]] | set[str]]]:
    tree = ET.parse(CUSTOM_RESIDUE_XML_PATH)
    residues_node = tree.getroot().find("Residues")
    if residues_node is None:
        return {}
    graphs: dict[str, dict[str, set[frozenset[str]] | set[str]]] = {}
    for residue_node in residues_node:
        if residue_node.tag != "Residue":
            continue
        name = str(residue_node.get("name", "")).strip().upper()
        if not name:
            continue
        atoms: set[str] = set()
        bonds: set[frozenset[str]] = set()
        external: set[str] = set()
        for child in residue_node:
            if child.tag == "Atom":
                atom_name = child.get("name")
                if atom_name:
                    atoms.add(str(atom_name))
            elif child.tag == "Bond":
                atom_a = child.get("atomName1")
                atom_b = child.get("atomName2")
                if atom_a and atom_b:
                    bonds.add(frozenset((str(atom_a), str(atom_b))))
            elif child.tag == "ExternalBond":
                atom_name = child.get("atomName")
                if atom_name:
                    external.add(str(atom_name))
        graphs[name] = {"atoms": atoms, "bonds": bonds, "external": external}
    return graphs


def _choose_existing_nonstandard_template(residue, spec) -> str:
    atom_names = {atom.name for atom in residue.atoms()}
    base_template = str(spec.passive_template).strip().upper()
    candidate_names = [base_template]
    nterm_template = f"N{base_template}"
    cterm_template = f"C{base_template}"
    graphs = _custom_residue_template_graphs()
    if ("H2" in atom_names or "H3" in atom_names) and nterm_template in graphs:
        candidate_names.insert(0, nterm_template)
    if "OXT" in atom_names and cterm_template in graphs:
        candidate_names.insert(0, cterm_template)
    for template_name in candidate_names:
        graph = graphs.get(template_name)
        if graph is None:
            continue
        if atom_names == graph["atoms"]:
            return template_name
    return base_template


def _as_nm_vec3(position) -> Vec3:
    if hasattr(position, "value_in_unit"):
        position = position.value_in_unit(unit.nanometer)
    return Vec3(float(position.x), float(position.y), float(position.z))


def _copy_residue_to_topology(old_residue, old_positions, new_topology, new_chain, atom_map, new_positions):
    new_residue = new_topology.addResidue(
        old_residue.name,
        new_chain,
        id=old_residue.id,
        insertionCode=getattr(old_residue, "insertionCode", ""),
    )
    for old_atom in old_residue.atoms():
        new_atom = new_topology.addAtom(
            old_atom.name,
            old_atom.element,
            new_residue,
            id=getattr(old_atom, "id", None),
            formalCharge=getattr(old_atom, "formalCharge", None),
        )
        atom_map[old_atom] = new_atom
        new_positions.append(_as_nm_vec3(old_positions[old_atom.index]))


def _insert_rty_residues_into_binder_chain(topology, positions, binder_chain: str, rty_insertions: list[dict]):
    """Rebuild topology so RTY is a real binder-chain residue, not a separate molecule."""
    if not rty_insertions:
        return topology, positions

    insertions_by_upstream_resid = {str(spec["upstream_resid"]): spec for spec in rty_insertions}
    new_topology = openmm_app.Topology()
    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        new_topology.setPeriodicBoxVectors(box_vectors)

    atom_map: dict[Any, Any] = {}
    new_positions: list[Any] = []

    # The binder chain is the protein chain, not same-id small molecule chains such as TMB.
    binder_chain_obj = None
    for chain in topology.chains():
        if chain.id != binder_chain:
            continue
        residues = list(chain.residues())
        if residues and residues[0].name != "TMB":
            binder_chain_obj = chain
            break

    for old_chain in topology.chains():
        new_chain = new_topology.addChain(old_chain.id)
        is_binder_chain = old_chain is binder_chain_obj
        for old_residue in old_chain.residues():
            _copy_residue_to_topology(old_residue, positions, new_topology, new_chain, atom_map, new_positions)
            if not is_binder_chain or old_residue.id not in insertions_by_upstream_resid:
                continue

            spec = insertions_by_upstream_resid[old_residue.id]
            rty_topology = spec["topology"]
            rty_positions = spec["positions"]
            for rty_chain in rty_topology.chains():
                for rty_residue in rty_chain.residues():
                    _copy_residue_to_topology(
                        rty_residue,
                        rty_positions,
                        new_topology,
                        new_chain,
                        atom_map,
                        new_positions,
                    )
            for atom_a, atom_b in rty_topology.bonds():
                new_topology.addBond(atom_map[atom_a], atom_map[atom_b])

    for atom_a, atom_b in topology.bonds():
        if atom_a in atom_map and atom_b in atom_map:
            new_topology.addBond(atom_map[atom_a], atom_map[atom_b])

    return new_topology, openmm.unit.Quantity(new_positions, unit=openmm.unit.nanometer)


def _replace_residues_in_binder_chain(topology, positions, binder_chain: str, replacements: list[dict]):
    """Rebuild topology while replacing selected binder residues with custom residues."""
    if not replacements:
        return topology, positions

    replacements_by_resid = {str(spec["target_resid"]): spec for spec in replacements}
    new_topology = openmm_app.Topology()
    box_vectors = topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        new_topology.setPeriodicBoxVectors(box_vectors)

    atom_map: dict[Any, Any] = {}
    replaced_atom_map: dict[Any, Any] = {}
    replacement_atom_map: dict[tuple[str, str], Any] = {}
    new_positions: list[Any] = []

    binder_chain_obj = None
    for chain in topology.chains():
        if chain.id != binder_chain:
            continue
        residues = list(chain.residues())
        if residues and residues[0].name != "TMB":
            binder_chain_obj = chain
            break

    for old_chain in topology.chains():
        new_chain = new_topology.addChain(old_chain.id)
        is_binder_chain = old_chain is binder_chain_obj
        for old_residue in old_chain.residues():
            spec = replacements_by_resid.get(str(old_residue.id)) if is_binder_chain else None
            if spec is None:
                _copy_residue_to_topology(old_residue, positions, new_topology, new_chain, atom_map, new_positions)
                continue

            repl_topology = spec["topology"]
            repl_positions = spec["positions"]
            for repl_chain in repl_topology.chains():
                for repl_residue in repl_chain.residues():
                    _copy_residue_to_topology(
                        repl_residue,
                        repl_positions,
                        new_topology,
                        new_chain,
                        atom_map,
                        new_positions,
                    )
                    for old_atom in repl_residue.atoms():
                        replacement_atom_map[(str(old_residue.id), old_atom.name)] = atom_map[old_atom]
                    new_atoms_by_name = {old_atom.name: atom_map[old_atom] for old_atom in repl_residue.atoms()}
                    for old_atom in old_residue.atoms():
                        mapped_atom = new_atoms_by_name.get(old_atom.name)
                        if mapped_atom is not None:
                            replaced_atom_map[old_atom] = mapped_atom
            for atom_a, atom_b in repl_topology.bonds():
                new_topology.addBond(atom_map[atom_a], atom_map[atom_b])

    for atom_a, atom_b in topology.bonds():
        new_atom_a = replaced_atom_map.get(atom_a, atom_map.get(atom_a))
        new_atom_b = replaced_atom_map.get(atom_b, atom_map.get(atom_b))
        if new_atom_a is not None and new_atom_b is not None:
            if (
                new_atom_a.residue is new_atom_b.residue
                and (atom_a in replaced_atom_map or atom_b in replaced_atom_map)
            ):
                continue
            new_topology.addBond(new_atom_a, new_atom_b)

    return (
        new_topology,
        openmm.unit.Quantity(new_positions, unit=openmm.unit.nanometer),
        replacement_atom_map,
    )


def _extend_nonstandard_residues(fixer) -> None:
    already = {r for r, _ in fixer.nonstandardResidues}
    for residue in fixer.topology.residues():
        if residue in already:
            continue
        name = residue.name
        if name in _DESIGN_CUSTOM_RESIDUE_NAMES:
            continue
        if get_nonstandard_residue_spec(name) is not None:
            continue
        replacement = _EXTRA_NONSTANDARD_SUBSTITUTIONS.get(name)
        if replacement is None:
            continue
        if fixer._getTemplate(replacement) is None:
            continue
        fixer.nonstandardResidues.append((residue, replacement))
    fixer.nonstandardResidues.sort(key=lambda item: item[0].index)


def _preserve_supported_nonstandard_residues(fixer) -> None:
    fixer.nonstandardResidues = [
        (residue, replacement)
        for residue, replacement in fixer.nonstandardResidues
        if get_nonstandard_residue_spec(residue.name) is None
    ]


def _existing_nonstandard_template_overrides(
    topology, binder_chain: str | None = None
) -> tuple[set[str], dict[tuple[str, str], str]]:
    custom_residue_names: set[str] = set()
    template_overrides: dict[tuple[str, str], str] = {}
    for chain in topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        is_binder = binder_chain is not None and str(chain.id) == str(binder_chain)
        for residue in residues:
            name = str(residue.name).strip().upper()
            if is_binder and name in _DESIGN_CUSTOM_RESIDUE_NAMES:
                continue
            spec = get_nonstandard_residue_spec(name)
            if spec is None:
                continue
            template_name = _choose_existing_nonstandard_template(residue, spec)
            custom_residue_names.add(template_name)
            template_overrides[(str(chain.id), str(residue.id))] = template_name
    return custom_residue_names, template_overrides


def _passive_template_path(resname: str, *, is_n_term: bool, is_c_term: bool) -> str | None:
    candidates: list[str] = []
    if is_n_term:
        candidates.append(f"N{resname}")
    if is_c_term:
        candidates.append(f"C{resname}")
    candidates.append(resname)
    for cand in candidates:
        path = os.path.join(ASSET_DIR, f"passive_{cand}_with_H.pdb")
        if os.path.exists(path):
            return path
    return None


@lru_cache(maxsize=None)
def _load_passive_template(path: str) -> tuple[dict[str, openmm.app.topology.Atom],
                                                np.ndarray,
                                                list[tuple[str, str]]]:
    pdb = PDBFile(path)
    atoms_by_name: dict[str, Any] = {a.name: a for a in pdb.topology.atoms()}
    coords_nm = np.array(
        [pdb.positions[a.index].value_in_unit(unit.nanometer) for a in pdb.topology.atoms()],
        dtype=np.float64,
    )
    bond_pairs = [(b[0].name, b[1].name) for b in pdb.topology.bonds()]
    return atoms_by_name, coords_nm, bond_pairs


def _graft_passive_hydrogens(fixer, binder_chain: str | None = None) -> int:
    old_topology = fixer.topology
    old_positions = fixer.positions
    positions_nm = np.array(
        [v.value_in_unit(unit.nanometer) for v in old_positions],
        dtype=np.float64,
    )

    # Plan per-residue actions: which atoms to drop, which to graft (with positions).
    plans: dict[Any, dict[str, Any]] = {}
    any_change = False
    for chain in old_topology.chains():
        chain_residues = list(chain.residues())
        n = len(chain_residues)
        is_binder = binder_chain is not None and str(chain.id) == str(binder_chain)
        for i, residue in enumerate(chain_residues):
            name = str(residue.name).strip().upper()
            if is_binder and name in _DESIGN_CUSTOM_RESIDUE_NAMES:
                continue
            if get_nonstandard_residue_spec(name) is None:
                continue
            tmpl_path = _passive_template_path(name, is_n_term=(i == 0), is_c_term=(i == n - 1))
            if tmpl_path is None:
                continue
            tmpl_atoms, tmpl_coords, tmpl_bonds = _load_passive_template(tmpl_path)

            drop_atoms: set[Any] = set()
            for atom in residue.atoms():
                if atom.element is not None and atom.element.symbol == "H":
                    drop_atoms.add(atom)

            kept_atom_names = {a.name for a in residue.atoms() if a not in drop_atoms}
            missing = [a_name for a_name in tmpl_atoms if a_name not in kept_atom_names]
            if not drop_atoms and not missing:
                continue

            common_heavy = [
                a_name for a_name in kept_atom_names
                if a_name in tmpl_atoms
            ]

            common_heavy = [
                a_name for a_name in common_heavy
                if next((a for a in residue.atoms() if a.name == a_name), None) is not None
                and next(a for a in residue.atoms() if a.name == a_name).element is not None
                and next(a for a in residue.atoms() if a.name == a_name).element.symbol != "H"
            ]

            grafts: list[tuple[str, Any, np.ndarray]] = []  # (name, element, pos_nm)
            graft_bond_names: list[tuple[str, str]] = []
            if missing:
                if len(common_heavy) < 3:
                    if not drop_atoms:
                        continue
                else:
                    cur_coords = np.array([
                        positions_nm[next(a for a in residue.atoms() if a.name == n_).index]
                        for n_ in common_heavy
                    ], dtype=np.float64)
                    tmpl_anchor = np.array(
                        [tmpl_coords[tmpl_atoms[n_].index] for n_ in common_heavy],
                        dtype=np.float64,
                    )
                    _, R, t = _kabsch(tmpl_anchor, cur_coords)
                    for a_name in missing:
                        tmpl_atom = tmpl_atoms[a_name]
                        tmpl_pos = tmpl_coords[tmpl_atom.index]
                        new_pos = np.dot(tmpl_pos.reshape(1, 3), R)[0] + t
                        grafts.append((a_name, tmpl_atom.element, new_pos))
                    new_names = set(missing)
                    for a_name, b_name in tmpl_bonds:
                        if a_name in new_names or b_name in new_names:
                            graft_bond_names.append((a_name, b_name))

            if drop_atoms or grafts:
                plans[residue] = {
                    "drop": drop_atoms,
                    "grafts": grafts,
                    "graft_bond_names": graft_bond_names,
                }
                any_change = True

    if not any_change:
        return 0

    new_topology = openmm_app.Topology()
    box_vectors = old_topology.getPeriodicBoxVectors()
    if box_vectors is not None:
        new_topology.setPeriodicBoxVectors(box_vectors)

    atom_map: dict[Any, Any] = {}
    residue_grafted_atoms: dict[Any, dict[str, Any]] = {}
    new_positions_list: list[Any] = []
    added_total = 0

    for old_chain in old_topology.chains():
        new_chain = new_topology.addChain(old_chain.id)
        for old_residue in old_chain.residues():
            new_residue = new_topology.addResidue(
                old_residue.name,
                new_chain,
                id=old_residue.id,
                insertionCode=getattr(old_residue, "insertionCode", ""),
            )
            plan = plans.get(old_residue)
            drop = plan["drop"] if plan else set()
            for old_atom in old_residue.atoms():
                if old_atom in drop:
                    continue
                new_atom = new_topology.addAtom(
                    old_atom.name,
                    old_atom.element,
                    new_residue,
                    id=getattr(old_atom, "id", None),
                    formalCharge=getattr(old_atom, "formalCharge", None),
                )
                atom_map[old_atom] = new_atom
                new_positions_list.append(_as_nm_vec3(old_positions[old_atom.index]))
            if plan and plan["grafts"]:
                graft_atoms: dict[str, Any] = {}
                for a_name, element, pos_nm in plan["grafts"]:
                    new_atom = new_topology.addAtom(a_name, element, new_residue)
                    new_positions_list.append(Vec3(float(pos_nm[0]), float(pos_nm[1]), float(pos_nm[2])))
                    graft_atoms[a_name] = new_atom
                    added_total += 1
                residue_grafted_atoms[new_residue] = graft_atoms

    for atom_a, atom_b in old_topology.bonds():
        new_a = atom_map.get(atom_a)
        new_b = atom_map.get(atom_b)
        if new_a is None or new_b is None:
            continue
        new_topology.addBond(new_a, new_b)

    existing_bond_pairs = {frozenset((a.index, b.index)) for a, b in new_topology.bonds()}
    new_residues_by_old: dict[Any, Any] = {}
    for old_chain, new_chain in zip(old_topology.chains(), new_topology.chains()):
        for old_res, new_res in zip(old_chain.residues(), new_chain.residues()):
            new_residues_by_old[old_res] = new_res

    for old_residue, plan in plans.items():
        if not plan["graft_bond_names"]:
            continue
        new_residue = new_residues_by_old[old_residue]
        atoms_by_name = {a.name: a for a in new_residue.atoms()}
        for a_name, b_name in plan["graft_bond_names"]:
            a = atoms_by_name.get(a_name)
            b = atoms_by_name.get(b_name)
            if a is None or b is None:
                continue
            key = frozenset((a.index, b.index))
            if key in existing_bond_pairs:
                continue
            new_topology.addBond(a, b)
            existing_bond_pairs.add(key)

    fixer.topology = new_topology
    fixer.positions = unit.Quantity(new_positions_list, unit=unit.nanometer)
    return added_total


def _patch_pdb_name_replacements_for_custom_residues() -> None:
    """OpenMM's pdbNames.xml aliases some PDB residue names (e.g. DCY -> DC,
    treating it as DNA cytosine). For registered custom residues we want the
    name to survive PDB parsing untouched. Override the replacement table to
    map each registered name to itself."""
    openmm_app.PDBFile._loadNameReplacementTables()
    table = openmm_app.PDBFile._residueNameReplacements
    from partiflow.utils.nonstandard_residue_registry import NONSTANDARD_RESIDUE_SPECS as _SPECS
    for name in _SPECS:
        if table.get(name) != name:
            table[name] = name


def _prepare_fixer(pdb_str: str, binder_chain: str | None = None):
    _patch_pdb_name_replacements_for_custom_residues()
    fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
    fixer.findNonstandardResidues()
    _preserve_supported_nonstandard_residues(fixer)
    _extend_nonstandard_residues(fixer)
    fixer.replaceNonstandardResidues()
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    _graft_passive_hydrogens(fixer, binder_chain)
    return fixer

_TOPOLOGY_REQUIRED_AA: dict[str, dict[int, str]] = {
    "disulfide": {0: "CYS", 1: "CYS"},
    "bicycle": {0: "CYS", 1: "CYS", 2: "CYS"},
    "stapled": {0: "LYS", 1: "ASP"},
    "staple_rty": {2: "PRO"},
}

_REQUIREMENT_SATISFIED_BY: dict[str, set[str]] = {
    "CYS": {"CYS", "CYN", "CYZ", "CYC"},
    "PRO": {"PRO"},
    "LYS": {"LYS", "LYZ"},
    "ASP": {"ASP", "GLU", "ADZ", "GLZ"},
}

# Fallback residue used to rescue cc_staple anchors that are GLY (the planner
# requires CB). ALA is the smallest non-GLY residue with a CB.
_CC_STAPLE_GLY_RESCUE = "ALA"


def _derive_force_fix_mutations(
    fixer,
    binder_chain: str,
    entries: list["TopologyEntry"],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Return (PDBFixer-style mutation strings, per-fix metadata) for anchors
    whose current identity disagrees with what the planner requires."""

    binder_residues: list[Any] = []
    for chain in fixer.topology.chains():
        if chain.id == binder_chain:
            binder_residues = list(chain.residues())
            break
    if not binder_residues:
        return [], []
    by_resid: dict[int, Any] = {int(res.id): res for res in binder_residues}

    requested: dict[int, str] = {}
    sources: dict[int, dict[str, Any]] = {}

    def _request(resid: int, target: str, source: dict[str, Any]) -> None:
        prior = requested.get(resid)
        if prior is not None and prior != target:
            # Conflicting demands (e.g. two topologies want different AAs at the
            # same position). Keep the first request and skip the second; the
            # planner for the losing entry will surface its own skip reason.
            return
        requested[resid] = target
        sources[resid] = source

    for entry in entries:
        required_map = _TOPOLOGY_REQUIRED_AA.get(entry.type)
        if required_map is not None:
            for anchor_idx, target_resname in required_map.items():
                if anchor_idx >= len(entry.anchors):
                    continue
                resid = int(entry.anchors[anchor_idx])
                _request(resid, target_resname, {"topology": entry.type, "anchor_resid": resid})
            continue
        if entry.type in CC_STAPLE_SPECS:
            for resid in entry.anchors[:2]:
                resid = int(resid)
                res = by_resid.get(resid)
                if res is not None and res.name == "GLY":
                    _request(resid, _CC_STAPLE_GLY_RESCUE,
                             {"topology": entry.type, "anchor_resid": resid, "reason": "cc_staple_gly_rescue"})

    mutations: list[str] = []
    fix_log: list[dict[str, Any]] = []
    for resid, target in requested.items():
        res = by_resid.get(resid)
        if res is None:
            continue
        satisfied = _REQUIREMENT_SATISFIED_BY.get(target, {target})
        if res.name in satisfied:
            continue
        mutations.append(f"{res.name}-{resid}-{target}")
        fix_log.append({
            "resid": resid,
            "from": res.name,
            "to": target,
            **sources[resid],
        })
    return mutations, fix_log


def _apply_force_fix_mutations(
    fixer,
    binder_chain: str,
    entries: list["TopologyEntry"],
) -> list[dict[str, Any]]:
    """Mutate anchor residues whose identity does not satisfy the topology
    planner, then rebuild missing sidechain atoms in place."""
    mutations, fix_log = _derive_force_fix_mutations(fixer, binder_chain, entries)
    if not mutations:
        return []
    fixer.applyMutations(mutations, binder_chain)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms(seed=0)
    fixer.addMissingHydrogens()
    return fix_log


def _has_sg_sg_bond(topology, chain_id: str, idx_a: int, idx_b: int) -> bool:
    """Return True if an SG-SG bond already exists between residues at idx_a and idx_b."""
    sg_a, sg_b = None, None
    for chain in topology.chains():
        if chain.id != chain_id:
            continue
        for idx, res in enumerate(chain.residues()):
            sg = next((a for a in res.atoms() if a.name == "SG"), None)
            if idx == idx_a:
                sg_a = sg
            elif idx == idx_b:
                sg_b = sg
    if sg_a is None or sg_b is None:
        return False
    for bond_a, bond_b in topology.bonds():
        if {bond_a, bond_b} == {sg_a, sg_b}:
            return True
    return False


def _remove_sg_sg_bonds_for_residues(
    topology,
    residues: set[Any],
    *,
    keep_residue_pairs: set[frozenset[Any]] | None = None,
) -> list[tuple[Any, Any]]:
    """Remove SG-SG bonds involving any residue in ``residues``, with optional exclusions."""
    removed: list[tuple[Any, Any]] = []
    keep_residue_pairs = keep_residue_pairs or set()
    kept_bonds = []
    for bond in list(topology._bonds):
        atom_a, atom_b = bond
        is_sg_sg = atom_a.name == "SG" and atom_b.name == "SG"
        touches_target = atom_a.residue in residues or atom_b.residue in residues
        residue_pair = frozenset((atom_a.residue, atom_b.residue))
        if is_sg_sg and touches_target and residue_pair not in keep_residue_pairs:
            removed.append((atom_a.residue, atom_b.residue))
            continue
        kept_bonds.append(bond)
    topology._bonds = kept_bonds
    return removed


def _deduplicate_bonds(topology) -> int:
    seen: set[tuple[int, int]] = set()
    kept_bonds = []
    removed = 0
    for bond in list(topology._bonds):
        atom_a, atom_b = bond
        key = tuple(sorted((atom_a.index, atom_b.index)))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        kept_bonds.append(bond)
    if removed:
        topology._bonds = kept_bonds
    return removed


def _ensure_passive_nonstandard_internal_bonds(topology) -> int:
    existing = {frozenset((atom_a, atom_b)) for atom_a, atom_b in topology.bonds()}
    added = 0
    template_graphs = _custom_residue_template_graphs()
    for residue in topology.residues():
        name = str(residue.name).strip().upper()
        spec = get_nonstandard_residue_spec(name)
        if spec is None:
            continue
        template_name = _choose_existing_nonstandard_template(residue, spec)
        graph = template_graphs.get(template_name)
        if graph is None:
            continue
        atoms = {atom.name: atom for atom in residue.atoms()}
        for bond_names in graph["bonds"]:
            atom_name_a, atom_name_b = tuple(bond_names)
            atom_a = atoms.get(atom_name_a)
            atom_b = atoms.get(atom_name_b)
            if atom_a is None or atom_b is None:
                continue
            key = frozenset((atom_a, atom_b))
            if key in existing:
                continue
            topology.addBond(atom_a, atom_b)
            existing.add(key)
            added += 1
    return added


def _restore_missing_cys_thiols(topology, positions, binder_chain: str, head_tail_specs: list[dict]) -> tuple[Any, Any, list[str]]:
    """Restore HG on standard CYS residues that are no longer disulfide-bonded."""
    needs_restore: list[tuple[str, str]] = []
    sg_sg_bonded: set[Any] = set()
    for atom_a, atom_b in topology.bonds():
        if atom_a.name == "SG" and atom_b.name == "SG":
            sg_sg_bonded.add(atom_a.residue)
            sg_sg_bonded.add(atom_b.residue)

    for residue in topology.residues():
        if residue.name != "CYS":
            continue
        atom_names = {atom.name for atom in residue.atoms()}
        if "SG" in atom_names and "HG" not in atom_names and residue not in sg_sg_bonded:
            needs_restore.append((residue.chain.id, residue.id))

    if not needs_restore:
        return topology, positions, []

    modeller = Modeller(topology, positions)
    modeller.addHydrogens()

    delete_atoms: list[Any] = []
    binder_residues: list[Any] = []
    for chain in modeller.topology.chains():
        if chain.id == binder_chain:
            binder_residues = list(chain.residues())
            break
    for spec in head_tail_specs:
        res_idx = spec["res_a_idx"]
        if 0 <= res_idx < len(binder_residues):
            residue = binder_residues[res_idx]
            for atom in residue.atoms():
                if atom.name in {"H2", "H3"}:
                    delete_atoms.append(atom)
    if delete_atoms:
        modeller.delete(delete_atoms)

    restored_tags = [f"{chain}:{resid}" for chain, resid in needs_restore]
    warnings = [f"[topology] restored missing cysteine HG after SG-SG cleanup: {restored_tags}"]
    return modeller.topology, modeller.positions, warnings


# ---------------------------------------------------------------------------
# TopologyFixer
# ---------------------------------------------------------------------------

class TopologyFixer:
    """
    Accepts a raw PDB string and a list of TopologyEntry objects.
    Returns a TopologyFixResult ready to be passed to a relaxer.

    All topology rules are applied in a single Modeller pass. CONECT records
    are generated once from the final atom IDs after a single re-parse,
    avoiding the stale-ID problem that occurs when re-parsing between rules.
    """

    def __call__(
        self,
        pdb_str: str,
        binder_chain: str,
        entries: list[TopologyEntry],
    ) -> TopologyFixResult:
        applied: list[dict] = []
        skipped: list[dict] = []
        warnings: list[str] = []
        notes: list[str] = []
        custom_residue_names: set[str] = set()

        # Normalize and sort entries by fixed apply order.
        order_map = {t: i for i, t in enumerate(APPLY_ORDER)}
        norm_entries = []
        for e in entries:
            t = _normalize_type(e.type)
            if t not in SUPPORTED_TYPES:
                skipped.append({"entry": {"type": e.type, "anchors": e.anchors}, "reason": f"unsupported type '{t}'"})
                continue
            norm_entries.append(TopologyEntry(type=t, anchors=e.anchors, options=e.options))
        norm_entries.sort(key=lambda e: order_map[e.type])

        fixer = _prepare_fixer(pdb_str, binder_chain)
        force_fixed_residues = _apply_force_fix_mutations(fixer, binder_chain, norm_entries)
        if force_fixed_residues:
            descriptions = ", ".join(
                f"{fix['from']}{fix['resid']}->{fix['to']}({fix['topology']})"
                for fix in force_fixed_residues
            )
            notes.append(f"force_fixed_anchor_residues:{descriptions}")
        duplicate_bonds_removed = _deduplicate_bonds(fixer.topology)
        if duplicate_bonds_removed:
            warnings.append(f"[topology] removed duplicate bonds after PDBFixer cleanup: {duplicate_bonds_removed}")
        passive_bonds_added = _ensure_passive_nonstandard_internal_bonds(fixer.topology)
        if passive_bonds_added:
            warnings.append(f"[topology] restored passive nonstandard internal bonds: {passive_bonds_added}")
        existing_custom_residue_names, existing_template_overrides = _existing_nonstandard_template_overrides(
            fixer.topology, binder_chain
        )
        custom_residue_names.update(existing_custom_residue_names)

        modeller = Modeller(fixer.topology, fixer.positions)
        binder_residues: list[Any] = []
        for chain in modeller.topology.chains():
            if chain.id == binder_chain:
                binder_residues = list(chain.residues())
                break
        if not binder_residues:
            raise ValueError(f"Binder chain '{binder_chain}' not found.")
        chain_len = len(binder_residues)
        last_resid = max(int(r.id) for r in binder_residues)

        atoms_to_delete: list[Any] = []
        # Each spec records how to find the two bonding atoms after re-parse.
        # Format: {"kind": ..., "chain": ..., "res_a_idx": int, "atom_a": str, ...}
        bond_specs: list[dict] = []
        tmb_insertions: list[tuple] = []  # (topo, positions)
        rty_insertions: list[dict] = []  # {"topology": topo, "positions": pos, "upstream_resid": ...}
        residue_replacements: list[dict] = []  # {"topology": topo, "positions": pos, "target_resid": ...}

        for entry in norm_entries:
            try:
                if entry.type == "head_to_tail":
                    self._plan_head_to_tail(
                        binder_residues, binder_chain, chain_len,
                        atoms_to_delete, bond_specs, applied, entry,
                    )
                elif entry.type == "residue_template":
                    self._plan_residue_template(
                        binder_residues, binder_chain, chain_len,
                        modeller, residue_replacements, custom_residue_names,
                        applied, skipped, entry,
                    )
                elif entry.type == "disulfide":
                    self._plan_disulfide(
                        binder_residues, binder_chain, chain_len,
                        modeller.topology, atoms_to_delete, bond_specs,
                        applied, skipped, notes, warnings, entry,
                    )
                elif entry.type == "stapled":
                    self._plan_stapled(
                        binder_residues, binder_chain, chain_len,
                        atoms_to_delete, bond_specs, custom_residue_names,
                        applied, skipped, entry,
                    )
                elif entry.type in CC_STAPLE_SPECS:
                    self._plan_cc_staple(
                        binder_residues, binder_chain, chain_len,
                        modeller, atoms_to_delete, bond_specs,
                        residue_replacements, custom_residue_names,
                        applied, skipped, entry,
                    )
                elif entry.type == "bicycle":
                    self._plan_bicycle(
                        binder_residues, binder_chain, chain_len,
                        modeller, atoms_to_delete, bond_specs,
                        tmb_insertions, custom_residue_names,
                        applied, skipped, warnings, entry, last_resid,
                    )
                    if tmb_insertions:
                        last_resid += len(tmb_insertions)
                elif entry.type == "staple_rty":
                    self._plan_staple_rty(
                        binder_residues, binder_chain, chain_len,
                        modeller, atoms_to_delete, bond_specs,
                        rty_insertions, custom_residue_names,
                        applied, skipped, warnings, entry, last_resid,
                    )
                    if rty_insertions:
                        last_resid += len(rty_insertions)
            except Exception as exc:
                skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": str(exc)})
                warnings.append(f"[{entry.type}] apply failed: {exc}")

        # Step 3: Apply all atom deletions in one shot.
        for chain in modeller.topology.chains():
            residues = list(chain.residues())
            if not residues:
                continue
            cterm = residues[-1]
            atom_names = {atom.name for atom in cterm.atoms()}
            if "HXT" in atom_names and "OXT" in atom_names:
                atoms_to_delete.extend(atom for atom in cterm.atoms() if atom.name == "HXT")
        if atoms_to_delete:
            modeller.delete(atoms_to_delete)

        # Step 4: Insert TMB molecules (bicycle only).
        for tmb_topo, tmb_pos in tmb_insertions:
            modeller.add(tmb_topo, tmb_pos)
        if rty_insertions:
            rebuilt_topology, rebuilt_positions = _insert_rty_residues_into_binder_chain(
                modeller.topology,
                modeller.positions,
                binder_chain,
                rty_insertions,
            )
            modeller = Modeller(rebuilt_topology, rebuilt_positions)
        if residue_replacements:
            rebuilt_topology, rebuilt_positions, _replacement_atom_map = _replace_residues_in_binder_chain(
                modeller.topology,
                modeller.positions,
                binder_chain,
                residue_replacements,
            )
            modeller = Modeller(rebuilt_topology, rebuilt_positions)

        # Step 5: Build atom lookup tables directly from modeller topology
        # (avoids write+reparse which corrupts CA atom element → Calcium bug in OpenMM PDB writer).
        final_topology = modeller.topology
        final_positions = modeller.positions

        # Identify the binder chain object (first chain with matching id that has protein residues).
        # Multiple chains may share the same chain.id (e.g. TMB is added with the binder chain id)
        # so we must use the chain OBJECT reference, not just chain.id, to avoid key collisions.
        binder_chain_obj = None
        for chain in final_topology.chains():
            if chain.id == binder_chain:
                residues_in_chain = list(chain.residues())
                if residues_in_chain and residues_in_chain[0].name != "TMB":
                    binder_chain_obj = chain
                    break

        chain_res_atom_map: dict[tuple, dict[str, Any]] = {}
        tmb_residues: list[Any] = []
        rty_residues: list[Any] = []
        cc_staple_residues: dict[str, list[Any]] = {spec["resname"]: [] for spec in CC_STAPLE_SPECS.values()}
        for chain in final_topology.chains():
            for idx, res in enumerate(chain.residues()):
                if res.name == "TMB":
                    tmb_residues.append(res)
                elif res.name == "RTY":
                    rty_residues.append(res)
                elif res.name in cc_staple_residues:
                    cc_staple_residues[res.name].append(res)
                if chain is binder_chain_obj:
                    chain_res_atom_map[(binder_chain, idx)] = {a.name: a for a in res.atoms()}

        # Step 6: Resolve bond specs → add bonds.
        # PDBFile.writeFile auto-emits CONECT for non-standard bonds (disulfide
        # SG-SG, stapled NZ-CD/CG, bicycle SG-CM) using correct written serials.
        # head_to_tail N-C looks like a standard peptide bond, so we track those
        # atoms separately and add CONECT after writing, using write-order serials.
        head_tail_specs: list[dict] = []
        tmb_cm_order = ["CM1", "CM2", "CM3"]

        for spec in bond_specs:
            kind = spec["kind"]
            chain = spec["chain"]

            if kind in ("head_to_tail", "disulfide", "stapled"):
                atom_a = chain_res_atom_map.get((chain, spec["res_a_idx"]), {}).get(spec["atom_a"])
                atom_b = chain_res_atom_map.get((chain, spec["res_b_idx"]), {}).get(spec["atom_b"])
                if atom_a is None or atom_b is None:
                    warnings.append(f"[{kind}] could not resolve atoms for CONECT")
                    continue
                final_topology.addBond(atom_a, atom_b)
                if kind == "head_to_tail":
                    head_tail_specs.append({
                        "chain": chain,
                        "res_a_idx": spec["res_a_idx"],
                        "atom_a": spec["atom_a"],
                        "res_b_idx": spec["res_b_idx"],
                        "atom_b": spec["atom_b"],
                    })

            elif kind == "disulfide_existing":
                pass  # bond already in topology; PDBFile.writeFile emits CONECT

            elif kind == "bicycle":
                positions = spec["positions"]
                tmb_idx = spec["tmb_idx"]
                if tmb_idx >= len(tmb_residues):
                    warnings.append("[bicycle] TMB residue not found after re-parse")
                    continue
                tmb_res = tmb_residues[tmb_idx]
                tmb_atoms = {a.name: a for a in tmb_res.atoms()}
                for i, pos in enumerate(positions):
                    sg = chain_res_atom_map.get((chain, pos), {}).get("SG")
                    cm = tmb_atoms.get(tmb_cm_order[i])
                    if sg is None or cm is None:
                        warnings.append(f"[bicycle] SG or CM missing at pos {pos}")
                        continue
                    final_topology.addBond(sg, cm)

            elif kind == "staple_rty":
                rty_idx = spec["rty_idx"]
                if rty_idx >= len(rty_residues):
                    warnings.append("[staple_rty] RTY residue not found after insertion")
                    continue
                rty_atoms = {a.name: a for a in rty_residues[rty_idx].atoms()}
                binder_res_by_id = {
                    res.id: {a.name: a for a in res.atoms()}
                    for chain_obj in final_topology.chains()
                    if chain_obj.id == binder_chain
                    for res in chain_obj.residues()
                    if res.name != "TMB"
                }
                external_specs = [
                    (spec["upstream_resid"], "N", "C2"),
                    (spec["upstream_resid"], "C", "N"),
                    (spec["downstream_resid"], "N", "C6"),
                ]
                for resid, binder_atom_name, rty_atom_name in external_specs:
                    binder_atom = binder_res_by_id.get(str(resid), {}).get(binder_atom_name)
                    rty_atom = rty_atoms.get(rty_atom_name)
                    if binder_atom is None or rty_atom is None:
                        warnings.append(
                            f"[staple_rty] missing atom for {resid}:{binder_atom_name}-RTY:{rty_atom_name}"
                        )
                        continue
                    final_topology.addBond(binder_atom, rty_atom)

            elif kind in CC_STAPLE_SPECS:
                staple_spec = CC_STAPLE_SPECS[kind]
                connect_atom = staple_spec["connect_atom"]
                atom_a = chain_res_atom_map.get((chain, spec["res_a_idx"]), {}).get(connect_atom)
                atom_b = chain_res_atom_map.get((chain, spec["res_b_idx"]), {}).get(connect_atom)
                if atom_a is None or atom_b is None:
                    warnings.append(
                        f"[{kind}] could not resolve {staple_spec['resname']} {connect_atom} atoms"
                    )
                    continue
                final_topology.addBond(atom_a, atom_b)

        final_topology, final_positions, thiol_warnings = _restore_missing_cys_thiols(
            final_topology, final_positions, binder_chain, head_tail_specs
        )
        warnings.extend(thiol_warnings)

        # Step 7: Write final PDB. Build write-order serial map for head_tail CONECT.
        binder_chain_obj = None
        for chain in final_topology.chains():
            if chain.id == binder_chain:
                residues_in_chain = list(chain.residues())
                if residues_in_chain and residues_in_chain[0].name != "TMB":
                    binder_chain_obj = chain
                    break
        chain_res_atom_map = {}
        for chain in final_topology.chains():
            if chain is binder_chain_obj:
                for idx, res in enumerate(chain.residues()):
                    chain_res_atom_map[(binder_chain, idx)] = {a.name: a for a in res.atoms()}
        final_pdb = _write_pdb(final_topology, final_positions)
        atom_write_serial = _written_atom_serials(final_topology, final_pdb)
        connects: list[str] = []
        for spec in head_tail_specs:
            atom_a = chain_res_atom_map.get((spec["chain"], spec["res_a_idx"]), {}).get(spec["atom_a"])
            atom_b = chain_res_atom_map.get((spec["chain"], spec["res_b_idx"]), {}).get(spec["atom_b"])
            if atom_a is None or atom_b is None:
                warnings.append("[head_to_tail] could not resolve atoms for final CONECT after hydrogen repair")
                continue
            sa, sb = atom_write_serial[atom_a], atom_write_serial[atom_b]
            connects.append(f"CONECT{sa:>5}{sb:>5}")
            connects.append(f"CONECT{sb:>5}{sa:>5}")
        if connects:
            final_pdb = _add_connects(final_pdb, connects)

        return TopologyFixResult(
            pdb_str=final_pdb,
            topology=final_topology,
            positions=final_positions,
            connects=connects,
            custom_residue_names=custom_residue_names,
            residue_template_overrides={
                **existing_template_overrides,
                **{
                    (str(spec["chain_id"]), str(spec["target_resid"])): str(spec["template_name"])
                    for spec in residue_replacements
                    if spec.get("template_name")
                },
            },
            applied_entries=applied,
            skipped_entries=skipped,
            warnings=warnings,
            notes=notes,
            force_fixed_residues=force_fixed_residues,
        )

    def _plan_head_to_tail(
        self, binder_residues, binder_chain, chain_len,
        atoms_to_delete, bond_specs, applied, entry,
    ):
        first_res = binder_residues[0]
        last_res = binder_residues[-1]
        for atom in first_res.atoms():
            if atom.name in {"H2", "H3"}:
                atoms_to_delete.append(atom)
        for atom in last_res.atoms():
            if atom.name == "OXT":
                atoms_to_delete.append(atom)
        bond_specs.append({
            "kind": "head_to_tail",
            "chain": binder_chain,
            "res_a_idx": 0,
            "atom_a": "N",
            "res_b_idx": chain_len - 1,
            "atom_b": "C",
        })
        applied.append({"type": "head_to_tail", "anchors": entry.anchors})

    def _plan_disulfide(
        self, binder_residues, binder_chain, chain_len,
        topology, atoms_to_delete, bond_specs,
        applied, skipped, notes, warnings, entry,
    ):
        if len(entry.anchors) < 2:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "need 2 anchors"})
            return
        pos_a, pos_b = entry.anchors[0] - 1, entry.anchors[1] - 1
        if not (0 <= pos_a < chain_len and 0 <= pos_b < chain_len):
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "anchor out of range"})
            return
        res_a = binder_residues[pos_a]
        res_b = binder_residues[pos_b]
        if res_a.name != "CYS" or res_b.name != "CYS":
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors},
                "reason": f"disulfide requires CYS/CYS, got {res_a.name}/{res_b.name}",
            })
            return
        anchor_residues = {res_a, res_b}
        original_anchor_sg_sg = []
        for bond_a, bond_b in topology.bonds():
            if bond_a.name == "SG" and bond_b.name == "SG":
                if bond_a.residue in anchor_residues or bond_b.residue in anchor_residues:
                    original_anchor_sg_sg.append((bond_a.residue, bond_b.residue))
        if original_anchor_sg_sg:
            sg_degree: dict[int, int] = {}
            for bond_res_a, bond_res_b in original_anchor_sg_sg:
                sg_degree[int(bond_res_a.id)] = sg_degree.get(int(bond_res_a.id), 0) + 1
                sg_degree[int(bond_res_b.id)] = sg_degree.get(int(bond_res_b.id), 0) + 1
            multi_bonded = sorted(resid for resid, degree in sg_degree.items() if degree > 1)
            if multi_bonded:
                warnings.append(
                    f"[disulfide] anchor SG atoms participated in multiple SG-SG bonds before cleanup: {multi_bonded}"
                )
        removed_sg_sg = _remove_sg_sg_bonds_for_residues(
            topology,
            anchor_residues,
            keep_residue_pairs={frozenset((res_a, res_b))},
        )
        if removed_sg_sg:
            removed_pairs = sorted(
                {
                    tuple(sorted((int(bond_res_a.id), int(bond_res_b.id))))
                    for bond_res_a, bond_res_b in removed_sg_sg
                }
            )
            warnings.append(
                f"[disulfide] removed pre-existing SG-SG bonds conflicting with requested anchors {sorted([pos_a + 1, pos_b + 1])}: {removed_pairs}"
            )
        if _has_sg_sg_bond(topology, binder_chain, pos_a, pos_b):
            notes.append(f"disulfide_already_present:{sorted([pos_a + 1, pos_b + 1])}")
            bond_specs.append({"kind": "disulfide_existing", "chain": binder_chain,
                                "res_a_idx": pos_a, "res_b_idx": pos_b})
        else:
            for res in (res_a, res_b):
                for atom in res.atoms():
                    if atom.name == "HG":
                        atoms_to_delete.append(atom)
            bond_specs.append({
                "kind": "disulfide",
                "chain": binder_chain,
                "res_a_idx": pos_a, "atom_a": "SG",
                "res_b_idx": pos_b, "atom_b": "SG",
            })
        applied.append({"type": "disulfide", "anchors": entry.anchors})

    def _plan_stapled(
        self, binder_residues, binder_chain, chain_len,
        atoms_to_delete, bond_specs, custom_residue_names,
        applied, skipped, entry,
    ):
        if len(entry.anchors) < 2:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "need 2 anchors"})
            return
        pos_a, pos_b = entry.anchors[0] - 1, entry.anchors[1] - 1
        if not (0 <= pos_a < chain_len and 0 <= pos_b < chain_len):
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "anchor out of range"})
            return
        res_a = binder_residues[pos_a]
        res_b = binder_residues[pos_b]
        for res, pos_idx in [(res_a, pos_a), (res_b, pos_b)]:
            if res.name not in STAPLED_RESNAME_MAP:
                skipped.append({
                    "entry": {"type": entry.type, "anchors": entry.anchors},
                    "reason": f"stapled requires LYS/ASP/GLU at pos {pos_idx + 1}, got {res.name}",
                })
                return

        connect_atoms = {}
        for res, pos_idx, label in [(res_a, pos_a, "a"), (res_b, pos_b, "b")]:
            old_name = res.name
            new_name = STAPLED_RESNAME_MAP[old_name]
            for atom in res.atoms():
                if atom.name in STAPLED_DELETE_ATOMS[old_name]:
                    atoms_to_delete.append(atom)
            res.name = new_name
            custom_residue_names.add(new_name)
            connect_atoms[label] = STAPLED_CONNECT_ATOMS[new_name]

        bond_specs.append({
            "kind": "stapled",
            "chain": binder_chain,
            "res_a_idx": pos_a, "atom_a": connect_atoms["a"],
            "res_b_idx": pos_b, "atom_b": connect_atoms["b"],
        })
        applied.append({"type": "stapled", "anchors": entry.anchors})

    def _plan_residue_template(
        self, binder_residues, binder_chain, chain_len,
        modeller, residue_replacements, custom_residue_names,
        applied, skipped, entry,
    ):
        resid = entry.options.get("resid", entry.anchors[0] if entry.anchors else None)
        resname = entry.options.get("resname", None)
        if resid is None or resname is None:
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors, **entry.options},
                "reason": "residue_template requires resid and resname",
            })
            return
        pos = int(resid) - 1
        if not (0 <= pos < chain_len):
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "resid out of range"})
            return
        base_spec = get_nonstandard_residue_spec(resname)
        if base_spec is None:
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors, **entry.options},
                "reason": f"unsupported non-standard residue template '{resname}'",
            })
            return
        spec = _resolve_terminal_variant_spec(base_spec, pos, chain_len)
        res = binder_residues[pos]
        alignment_atoms = _spec_alignment_atom_names(spec)
        if "CB" in alignment_atoms and res.name == "GLY":
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors, **entry.options},
                "reason": f"residue_template {resname} requires CB at pos {pos + 1}, but residue is GLY",
            })
            return
        coords: dict[str, np.ndarray] = {}
        for atom_name in alignment_atoms:
            atom = next((a for a in res.atoms() if a.name == atom_name), None)
            if atom is None:
                skipped.append({
                    "entry": {"type": entry.type, "anchors": entry.anchors, **entry.options},
                    "reason": f"residue_template could not find {atom_name} at anchor residue {res.id}",
                })
                return
            coord = modeller.positions[atom.index]
            coords[atom_name] = np.array([coord.x, coord.y, coord.z], dtype=np.float64)
        repl_topo, repl_pos = _make_nonstandard_residue(
            spec,
            np.array([coords[name] for name in alignment_atoms]),
            binder_chain,
            int(res.id),
            role="passive",
        )
        residue_replacements.append({
            "topology": repl_topo,
            "positions": repl_pos,
            "target_resid": res.id,
            "chain_id": binder_chain,
            "template_name": spec.passive_template,
        })
        custom_residue_names.update(spec.custom_residue_names_for_role("passive"))
        applied.append({
            "type": "residue_template",
            "resid": pos + 1,
            "resname": spec.resname,
            "template": spec.passive_template,
        })

    def _plan_cc_staple(
        self, binder_residues, binder_chain, chain_len,
        modeller, atoms_to_delete, bond_specs,
        residue_replacements, custom_residue_names,
        applied, skipped, entry,
    ):
        spec = CC_STAPLE_SPECS[entry.type]
        resname = spec["resname"]
        residue_spec = get_nonstandard_residue_spec(resname)
        if residue_spec is None:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": f"unsupported residue {resname}"})
            return
        if len(entry.anchors) < 2:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "need 2 anchors"})
            return
        pos_a, pos_b = entry.anchors[0] - 1, entry.anchors[1] - 1
        if not (0 <= pos_a < chain_len and 0 <= pos_b < chain_len):
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "anchor out of range"})
            return

        for res, pos_idx in [(binder_residues[pos_a], pos_a), (binder_residues[pos_b], pos_b)]:
            if res.name == "GLY":
                skipped.append({
                    "entry": {"type": entry.type, "anchors": entry.anchors},
                    "reason": f"{entry.type} requires CB at pos {pos_idx + 1}, but residue is GLY",
                })
                return

        for res, pos_idx in [(binder_residues[pos_a], pos_a), (binder_residues[pos_b], pos_b)]:
            variant_spec = _resolve_terminal_variant_spec(residue_spec, pos_idx, chain_len)
            coords: dict[str, np.ndarray] = {}
            for atom_name in ["N", "CA", "C", "CB"]:
                atom = next((a for a in res.atoms() if a.name == atom_name), None)
                if atom is None:
                    skipped.append({
                        "entry": {"type": entry.type, "anchors": entry.anchors},
                        "reason": f"{entry.type} could not find {atom_name} at anchor residue {res.id}",
                    })
                    return
                coord = modeller.positions[atom.index]
                coords[atom_name] = np.array([coord.x, coord.y, coord.z], dtype=np.float64)
            staple_topo, staple_pos = _make_nonstandard_residue(
                variant_spec,
                np.array([coords["N"], coords["CA"], coords["C"], coords["CB"]]),
                binder_chain,
                int(res.id),
                role="active",
            )
            residue_replacements.append({
                "topology": staple_topo,
                "positions": staple_pos,
                "target_resid": res.id,
                "chain_id": binder_chain,
                "template_name": variant_spec.active_template,
            })
            custom_residue_names.update(variant_spec.custom_residue_names_for_role("active"))
        bond_specs.append({
            "kind": entry.type,
            "chain": binder_chain,
            "res_a_idx": pos_a,
            "res_b_idx": pos_b,
            "res_a_id": binder_residues[pos_a].id,
            "res_b_id": binder_residues[pos_b].id,
        })
        applied.append({"type": entry.type, "anchors": entry.anchors, "resname": resname})

    def _plan_bicycle(
        self, binder_residues, binder_chain, chain_len,
        modeller, atoms_to_delete, bond_specs,
        tmb_insertions, custom_residue_names,
        applied, skipped, warnings, entry, last_resid,
    ):
        if len(entry.anchors) < 3:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "need 3 anchors"})
            return
        positions = [a - 1 for a in entry.anchors[:3]]
        for pos in positions:
            if not (0 <= pos < chain_len):
                skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": f"anchor {pos + 1} out of range"})
                return

        bicycle_residues = {binder_residues[pos] for pos in positions}
        removed_sg_sg = _remove_sg_sg_bonds_for_residues(modeller.topology, bicycle_residues)
        if removed_sg_sg:
            removed_pairs = sorted(
                {
                    tuple(sorted((int(res_a.id), int(res_b.id))))
                    for res_a, res_b in removed_sg_sg
                }
            )
            warnings.append(
                f"[bicycle] removed pre-existing SG-SG bonds involving bicycle anchors: {removed_pairs}"
            )
            sg_degree: dict[int, int] = {}
            for res_a, res_b in removed_sg_sg:
                sg_degree[int(res_a.id)] = sg_degree.get(int(res_a.id), 0) + 1
                sg_degree[int(res_b.id)] = sg_degree.get(int(res_b.id), 0) + 1
            multi_bonded = sorted(resid for resid, degree in sg_degree.items() if degree > 1)
            if multi_bonded:
                warnings.append(
                    f"[bicycle] anchor SG atoms participated in multiple SG-SG bonds before cleanup: {multi_bonded}"
                )

        sg_coords: list[np.ndarray] = []
        for pos in positions:
            res = binder_residues[pos]
            if res.name != "CYS":
                skipped.append({
                    "entry": {"type": entry.type, "anchors": entry.anchors},
                    "reason": f"bicycle requires CYS at pos {pos + 1}, got {res.name}",
                })
                return
            if pos == 0:
                new_name = BICYCLE_RESNAME_NTERM
            elif pos == chain_len - 1:
                new_name = BICYCLE_RESNAME_CTERM
            else:
                new_name = BICYCLE_RESNAME_INTERNAL
            res.name = new_name
            custom_residue_names.add(new_name)

            for atom in res.atoms():
                if atom.name == "HG":
                    atoms_to_delete.append(atom)
                elif atom.name == "SG":
                    coord = modeller.positions[atom.index]
                    sg_coords.append(np.array([coord.x, coord.y, coord.z], dtype=np.float64))

        if len(sg_coords) != 3:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "could not find 3 SG atoms"})
            return

        tmb_topo, tmb_pos = _make_tmb(np.array(sg_coords), binder_chain, last_resid + 1)
        tmb_idx = len(tmb_insertions)
        tmb_insertions.append((tmb_topo, tmb_pos))
        custom_residue_names.add("TMB")

        bond_specs.append({
            "kind": "bicycle",
            "chain": binder_chain,
            "positions": positions,
            "tmb_idx": tmb_idx,
        })
        applied.append({"type": "bicycle", "anchors": entry.anchors})

    def _plan_staple_rty(
        self, binder_residues, binder_chain, chain_len,
        modeller, atoms_to_delete, bond_specs,
        rty_insertions, custom_residue_names,
        applied, skipped, warnings, entry, last_resid,
    ):
        if len(entry.anchors) < 3:
            skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": "need 3 anchors"})
            return
        positions = [a - 1 for a in entry.anchors[:3]]
        for pos in positions:
            if not (0 <= pos < chain_len):
                skipped.append({"entry": {"type": entry.type, "anchors": entry.anchors}, "reason": f"anchor {pos + 1} out of range"})
                return
        if positions[1] != positions[0] + 1 or positions[2] != positions[1] + 1:
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors},
                "reason": "staple_rty currently requires consecutive anchors [i, i+1, i+2]",
            })
            return
        if positions[0] != 0:
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors},
                "reason": "staple_rty currently supports only N-terminal insertion anchors [1,2,3]",
            })
            return

        upstream = binder_residues[positions[0]]
        placeholder = binder_residues[positions[1]]
        downstream = binder_residues[positions[2]]
        if downstream.name != "PRO":
            skipped.append({
                "entry": {"type": entry.type, "anchors": entry.anchors},
                "reason": f"staple_rty currently requires PRO at anchor 3 so the substituted backbone N matches an OpenMM template, got {downstream.name}",
            })
            return

        target_coords: dict[str, np.ndarray] = {}
        for label, res, atom_name in [
            ("C2", upstream, "N"),
            ("CA", placeholder, "CA"),
            ("C6", downstream, "N"),
        ]:
            atom = next((a for a in res.atoms() if a.name == atom_name), None)
            if atom is None:
                skipped.append({
                    "entry": {"type": entry.type, "anchors": entry.anchors},
                    "reason": f"staple_rty could not find {atom_name} at anchor residue {res.id}",
                })
                return
            coord = modeller.positions[atom.index]
            target_coords[label] = np.array([coord.x, coord.y, coord.z], dtype=np.float64)

        for atom in upstream.atoms():
            if atom.name in {"H2", "H3"}:
                atoms_to_delete.append(atom)
        for atom in downstream.atoms():
            if atom.name == "H":
                atoms_to_delete.append(atom)
        atoms_to_delete.extend(list(placeholder.atoms()))

        rty_topo, rty_pos = _make_rty(
            np.array([target_coords["C2"], target_coords["CA"], target_coords["C6"]]),
            binder_chain,
            int(placeholder.id),
        )
        rty_idx = len(rty_insertions)
        rty_insertions.append({
            "topology": rty_topo,
            "positions": rty_pos,
            "upstream_resid": upstream.id,
        })
        custom_residue_names.add(STAPLE_RTY_RESNAME)

        bond_specs.append({
            "kind": "staple_rty",
            "chain": binder_chain,
            "positions": positions,
            "upstream_resid": upstream.id,
            "downstream_resid": downstream.id,
            "rty_idx": rty_idx,
        })
        applied.append({"type": "staple_rty", "anchors": entry.anchors})


# ---------------------------------------------------------------------------
# Public helper: parse raw entry dicts into TopologyEntry objects
# ---------------------------------------------------------------------------

def parse_topology_entries(raw: Any) -> list[TopologyEntry]:
    """Convert raw dicts (from task YAML or private sample records) to TopologyEntry list."""
    import ast
    import json

    if raw is None:
        return []
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            entries = json.loads(text)
        except Exception:
            try:
                entries = ast.literal_eval(text)
            except Exception:
                return []
    else:
        return []

    result = []
    for item in entries:
        if not isinstance(item, dict):
            continue
        t = _normalize_type(item.get("type", "none"))
        if t not in SUPPORTED_TYPES:
            continue
        anchors_raw = item.get("anchors", [])
        if t == "residue_template" and not anchors_raw:
            anchors_raw = [item.get("resid", item.get("pos", 0))]
        if hasattr(anchors_raw, "tolist"):
            anchors_raw = anchors_raw.tolist()
        anchors = [int(v) for v in anchors_raw if int(v) > 0]
        options = {k: v for k, v in item.items() if k not in {"type", "anchors"}}
        result.append(TopologyEntry(type=t, anchors=anchors, options=options))
    return result
