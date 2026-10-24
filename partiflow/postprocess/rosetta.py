from dataclasses import dataclass
import json
import os
import traceback
from typing import Any, Iterable, Literal, Optional

import numpy as np
from scipy.spatial import cKDTree

import pyrosetta as pr
from pyrosetta.rosetta.core.chemical import CUTPOINT_LOWER, CUTPOINT_UPPER
from pyrosetta.rosetta.core.conformation import form_disulfide, is_disulfide_bond
from pyrosetta.rosetta.core.id import AtomID
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.core.pose import (
    add_variant_type_to_pose_residue,
    remove_lower_terminus_type_from_pose_residue,
    remove_upper_terminus_type_from_pose_residue,
)
try:
    from pyrosetta.rosetta.core.pose import DockingPartners
except ImportError:  # Newer PyRosetta accepts the interface string directly.
    DockingPartners = None
from pyrosetta.rosetta.core.scoring.constraints import AtomPairConstraint
from pyrosetta.rosetta.core.scoring.func import HarmonicFunc
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.rosetta_scripts import XmlObjects
from pyrosetta.rosetta.core.pack.task import TaskFactory, operation
from pyrosetta.rosetta.core.select import residue_selector as selections
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action



_HEAD_TAIL_MIN_DISTANCE = 1.1
_HEAD_TAIL_MAX_DISTANCE = 1.7
_DISULFIDE_MIN_DISTANCE = 1.7
_DISULFIDE_MAX_DISTANCE = 2.3

def _set_interface_analyzer_interface(iam: InterfaceAnalyzerMover, interface: str) -> None:
    """Set InterfaceAnalyzerMover interface across old and new PyRosetta APIs."""
    try:
        iam.set_interface(interface)
        return
    except TypeError:
        if DockingPartners is None:
            raise
        partners = DockingPartners().docking_partners_from_string(interface)
        iam.set_interface(partners)

def _parse_chain_ids(chains: str | Iterable[str]) -> list[str]:
    """
    Accepts:
      - "A" / "ABCD" / "A,B,C,D"
      - ["A","B","C"]
    Returns: unique 1-char chain IDs in stable order.
    """
    if isinstance(chains, str):
        raw = chains.replace(" ", "").strip()
        if not raw:
            return []
        # Support both "A,B" and "AB"
        raw = raw.replace(",", "")
        chain_ids = list(raw)
    else:
        parts = [str(c).strip() for c in chains if str(c).strip()]
        if len(parts) == 1 and len(parts[0]) > 1:
            chain_ids = list(parts[0].replace(",", "").replace(" ", ""))
        else:
            chain_ids = parts

    out: list[str] = []
    seen: set[str] = set()
    for c in chain_ids:
        if len(c) != 1:
            raise ValueError(f"Chain IDs must be 1 character for PDBs; got {c!r}")
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def _chain_ids_in_pose(pose: Any) -> list[str]:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose has no PDBInfo; cannot infer chain IDs.")

    out: list[str] = []
    seen: set[str] = set()
    for i in range(1, pose.total_residue() + 1):
        chain_id = str(pdb_info.chain(i))
        if chain_id and chain_id not in seen:
            seen.add(chain_id)
            out.append(chain_id)
    return out


def _infer_target_chains_from_pose(pose: Any, *, binder_chain: str) -> str:
    chain_ids = _chain_ids_in_pose(pose)
    if binder_chain not in chain_ids:
        raise ValueError(f"binder_chain {binder_chain!r} not found in PDB chains {chain_ids}")
    target = [c for c in chain_ids if c != binder_chain]
    if not target:
        raise ValueError("No target chains inferred (PDB only contains binder chain).")
    return "".join(target)


def infer_target_chains(
    pdb_path: str,
    *,
    binder_chain: str = "L",
) -> str:
    """
    Default rule: after reading the PDB, every non-binder chain is considered target.

    Returns a concatenated chain-id string usable by InterfaceAnalyzerMover, e.g. "ABCD".
    """
    pose = pr.pose_from_pdb(pdb_path)
    return _infer_target_chains_from_pose(pose, binder_chain=binder_chain)


def init_pyrosetta(
    *,
    dalphaball_path: str | None = None,
    mute: str = "all",
    relax_default_repeats: int | None = 1,
    beta_nov16: bool = True,
    ignore_unrecognized_res: bool = False,
    ignore_zero_occupancy: bool = True,
    use_input_sc: bool = False,
    extra_options: list[str] | None = None,
    silent: bool = True,
) -> None:

    from partiflow.postprocess.native import initialize_pyrosetta_score_only
    initialize_pyrosetta_score_only(silent=silent)


def clean_pdb(pdb_file):
    with open(pdb_file, 'r') as f_in:
        relevant_lines = [
            line
            for line in f_in
            if line.startswith(('ATOM', 'HETATM', 'MODEL', 'TER', 'END', 'LINK', 'CONECT', 'SSBOND'))
        ]
    with open(pdb_file, 'w') as f_out:
        f_out.writelines(relevant_lines)


def pose_from_pdb_no_link(pdb_path: str) -> Any:
    """Load a pose while stripping LINK/CONECT records first.

    LINK records in cyclic-peptide PDBs cause PyRosetta to merge all chains
    into a single conformation chain, which breaks InterfaceAnalyzerMover.
    We strip those records so each chain loads independently, then let
    cyclize_head_tail_binder_pose re-declare the chemical bond properly.
    """
    with open(pdb_path) as fh:
        lines = [l for l in fh if not l.startswith(("LINK", "CONECT"))]
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    try:
        pose = pr.pose_from_pdb(tmp_path)
    finally:
        os.unlink(tmp_path)
    return pose


def _parse_boolish(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return None


def _decode_topology_entries(raw: Any) -> list[dict[str, Any]]:
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
            return []
    else:
        return []
    return [entry for entry in entries if isinstance(entry, dict)]


def _normalize_topology_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    if text == "head_tail":
        return "head_to_tail"
    return text


def _binder_pose_indices(pose: Any, *, binder_chain: str) -> list[int]:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose has no PDBInfo; cannot resolve binder chain indices.")
    indices = [i for i in range(1, pose.total_residue() + 1) if str(pdb_info.chain(i)) == binder_chain]
    if not indices:
        chain_ids = _chain_ids_in_pose(pose)
        raise ValueError(f"binder_chain {binder_chain!r} not found in PDB chains {chain_ids}")
    return indices


def _head_tail_topology_requested(
    topology_entries: Any,
    *,
    binder_len: int | None = None,
) -> bool:
    for entry in _decode_topology_entries(topology_entries):
        if _normalize_topology_type(entry.get("type", "")) != "head_to_tail":
            continue
        anchors = entry.get("anchors", []) or []
        try:
            anchors = [int(v) for v in anchors]
        except Exception:
            anchors = []
        if binder_len is None or not anchors:
            return True
        if len(anchors) >= 2 and sorted(anchors[:2]) == [1, int(binder_len)]:
            return True
    return False


def _disulfide_anchor_pairs(topology_entries: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for entry in _decode_topology_entries(topology_entries):
        if _normalize_topology_type(entry.get("type", "")) != "disulfide":
            continue
        anchors = entry.get("anchors")
        if not isinstance(anchors, (list, tuple)):
            anchors = [entry.get("begin_resid"), entry.get("end_resid")]
        parsed: list[int] = []
        for value in anchors:
            try:
                position = int(value)
            except (TypeError, ValueError):
                continue
            if position > 0:
                parsed.append(position)
        if len(parsed) >= 2:
            pairs.append((parsed[0], parsed[1]))
    return pairs


def apply_disulfide_topology_to_pose(
    pose: Any,
    *,
    binder_chain: str,
    topology_entries: Any,
) -> list[dict[str, Any]]:
    pairs = _disulfide_anchor_pairs(topology_entries)
    if not pairs:
        return []

    binder_indices = _binder_pose_indices(pose, binder_chain=binder_chain)
    applied: list[dict[str, Any]] = []
    for pos_a, pos_b in pairs:
        if pos_a > len(binder_indices) or pos_b > len(binder_indices):
            raise ValueError(
                f"Disulfide anchors {pos_a}-{pos_b} exceed binder length {len(binder_indices)}."
            )
        pose_a = int(binder_indices[pos_a - 1])
        pose_b = int(binder_indices[pos_b - 1])
        res_a = pose.residue(pose_a)
        res_b = pose.residue(pose_b)
        if not res_a.has("SG") or not res_b.has("SG"):
            raise ValueError(
                f"Disulfide anchors {pos_a}-{pos_b} require SG atoms, got "
                f"{res_a.name3()} and {res_b.name3()}."
            )
        pre_distance = float(res_a.xyz("SG").distance(res_b.xyz("SG")))
        lower, upper = sorted((pose_a, pose_b))
        form_disulfide(pose.conformation(), lower, upper, False, False)
        pose.update_residue_neighbors()
        if not bool(is_disulfide_bond(pose.conformation(), lower, upper)):
            raise ValueError(f"Rosetta did not register disulfide bond for anchors {pos_a}-{pos_b}.")
        atom_sg_a = AtomID(pose.residue(lower).atom_index("SG"), lower)
        atom_sg_b = AtomID(pose.residue(upper).atom_index("SG"), upper)
        pose.add_constraint(AtomPairConstraint(atom_sg_a, atom_sg_b, HarmonicFunc(2.038, 0.05)))
        applied.append(
            {
                "anchors": f"{pos_a}-{pos_b}",
                "pose_indices": f"{lower}-{upper}",
                "pre_sg_distance": pre_distance,
            }
        )
    return applied


def validate_declared_disulfides(
    pose: Any,
    disulfide_info: list[dict[str, Any]],
    *,
    min_sg_distance: float = _DISULFIDE_MIN_DISTANCE,
    max_sg_distance: float = 2.3,
) -> list[dict[str, Any]]:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose has no PDBInfo; cannot serialize disulfide records.")
    validated: list[dict[str, Any]] = []
    for item in disulfide_info:
        raw_pose_indices = str(item.get("pose_indices", ""))
        try:
            pose_a, pose_b = (int(value) for value in raw_pose_indices.split("-", 1))
        except Exception as exc:
            raise ValueError(f"Invalid disulfide pose_indices in Rosetta metadata: {raw_pose_indices!r}") from exc
        if not bool(is_disulfide_bond(pose.conformation(), pose_a, pose_b)):
            raise ValueError(f"Rosetta disulfide was lost after relax for anchors {item.get('anchors')}.")
        res_a = pose.residue(pose_a)
        res_b = pose.residue(pose_b)
        post_distance = float(res_a.xyz("SG").distance(res_b.xyz("SG")))
        if not np.isfinite(post_distance) or not (
            float(min_sg_distance) <= post_distance <= float(max_sg_distance)
        ):
            raise ValueError(
                f"Rosetta disulfide SG-SG distance is outside the accepted range for anchors "
                f"{item.get('anchors')}: {post_distance:.3f} A "
                f"(required {float(min_sg_distance):.1f}-{float(max_sg_distance):.1f} A)"
            )
        updated = dict(item)
        updated["post_sg_distance"] = post_distance
        updated.update(
            {
                "first_chain_id": str(pdb_info.chain(pose_a)),
                "first_pdb_resnum": int(pdb_info.number(pose_a)),
                "first_icode": str(pdb_info.icode(pose_a)).strip(),
                "first_resname": str(res_a.name3()).strip(),
                "second_chain_id": str(pdb_info.chain(pose_b)),
                "second_pdb_resnum": int(pdb_info.number(pose_b)),
                "second_icode": str(pdb_info.icode(pose_b)).strip(),
                "second_resname": str(res_b.name3()).strip(),
            }
        )
        validated.append(updated)
    return validated


def _refresh_head_tail_geometry_after_relax(
    pose: Any,
    cyclization_info: dict[str, Any],
    *,
    min_distance: float = _HEAD_TAIL_MIN_DISTANCE,
    max_distance: float = _HEAD_TAIL_MAX_DISTANCE,
) -> float:
    """Refresh and validate the terminal peptide-bond distance after relaxation.

    ``cyclize_head_tail_binder_pose`` records the input distance before
    FastRelax.  The post-relax value must be read from the mutated pose rather
    than reused from that initial snapshot; otherwise the audit reports the
    unclosed input even when the dumped PDB is correctly cyclized.
    """

    first_idx = int(cyclization_info["first_pose_idx"])
    last_idx = int(cyclization_info["last_pose_idx"])
    distance = float(pose.residue(first_idx).xyz("N").distance(pose.residue(last_idx).xyz("C")))
    cyclization_info["post_n_c_distance"] = distance
    if not np.isfinite(distance) or not (float(min_distance) <= distance <= float(max_distance)):
        raise ValueError(
            "Rosetta head-tail C(last)-N(first) distance is outside the accepted range "
            f"after relaxation: {distance:.3f} A "
            f"(required {float(min_distance):.1f}-{float(max_distance):.1f} A)"
        )
    return distance


def should_cyclize_binder(
    pose: Any,
    *,
    binder_chain: str = "L",
    binder_is_cyclic: bool = False,
    topology_entries: Any = None,
) -> bool:
    boolish = _parse_boolish(binder_is_cyclic)
    if boolish is True:
        return True
    binder_indices = _binder_pose_indices(pose, binder_chain=binder_chain)
    binder_len = len(binder_indices)
    return _head_tail_topology_requested(topology_entries, binder_len=binder_len)


def cyclize_head_tail_binder_pose(
    pose: Any,
    *,
    binder_chain: str = "L",
) -> dict[str, Any]:
    binder_indices = _binder_pose_indices(pose, binder_chain=binder_chain)
    if len(binder_indices) < 2:
        raise ValueError(f"Need at least 2 residues on binder chain {binder_chain!r} to cyclize.")

    first_idx = binder_indices[0]
    last_idx = binder_indices[-1]
    first_res = pose.residue(first_idx)
    last_res = pose.residue(last_idx)
    pre_distance = float(first_res.xyz("N").distance(last_res.xyz("C")))

    try:
        remove_lower_terminus_type_from_pose_residue(pose, first_idx)
    except RuntimeError:
        pass
    try:
        remove_upper_terminus_type_from_pose_residue(pose, last_idx)
    except RuntimeError:
        pass

    conf = pose.conformation()
    atom_n = AtomID(pose.residue(first_idx).atom_index("N"), first_idx)
    atom_c = AtomID(pose.residue(last_idx).atom_index("C"), last_idx)
    if not conf.is_bonded(atom_n, atom_c):
        conf.declare_chemical_bond(first_idx, "N", last_idx, "C")
    pose.update_residue_neighbors()

    pdb_info = pose.pdb_info()
    return {
        "binder_chain": binder_chain,
        "first_pose_idx": int(first_idx),
        "last_pose_idx": int(last_idx),
        "first_pdb_resnum": int(pdb_info.number(first_idx)),
        "last_pdb_resnum": int(pdb_info.number(last_idx)),
        "first_chain_id": str(pdb_info.chain(first_idx)),
        "last_chain_id": str(pdb_info.chain(last_idx)),
        "first_icode": str(pdb_info.icode(first_idx)).strip(),
        "last_icode": str(pdb_info.icode(last_idx)).strip(),
        "pre_n_c_distance": pre_distance,
        "post_n_c_distance": float(pose.residue(first_idx).xyz("N").distance(pose.residue(last_idx).xyz("C"))),
    }


def _format_pdb_atom_name(atom_name: str) -> str:
    text = str(atom_name).strip()
    if len(text) >= 4:
        return text[:4]
    if len(text) == 1:
        return f" {text}  "
    if len(text) == 2:
        return f" {text} "
    return f"{text} "


def _pdb_atom_serials_for_residue_atoms(
    pdb_path: str,
    *,
    chain_id: str,
    residue_number: int,
    insertion_code: str = "",
    atom_names: Iterable[str],
) -> dict[str, int]:
    wanted = {_format_pdb_atom_name(name): str(name).strip() for name in atom_names}
    out: dict[str, int] = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21].strip() != str(chain_id).strip():
                continue
            if int(line[22:26]) != int(residue_number):
                continue
            if str(line[26]).strip() != str(insertion_code).strip():
                continue
            atom_key = line[12:16]
            if atom_key not in wanted:
                continue
            out[wanted[atom_key]] = int(line[6:11])
    return out


def _pdb_atom_coordinates_for_residue_atoms(
    pdb_path: str,
    *,
    chain_id: str,
    residue_number: int,
    insertion_code: str = "",
    atom_names: Iterable[str],
) -> dict[str, np.ndarray]:
    wanted = {_format_pdb_atom_name(name): str(name).strip() for name in atom_names}
    out: dict[str, np.ndarray] = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            if line[21].strip() != str(chain_id).strip():
                continue
            if int(line[22:26]) != int(residue_number):
                continue
            if str(line[26]).strip() != str(insertion_code).strip():
                continue
            atom_key = line[12:16]
            if atom_key not in wanted:
                continue
            out[wanted[atom_key]] = np.asarray(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                dtype=float,
            )
    return out


def append_head_tail_conect(
    pdb_path: str,
    *,
    cyclization_info: dict[str, Any],
) -> None:
    serials_first = _pdb_atom_serials_for_residue_atoms(
        pdb_path,
        chain_id=str(cyclization_info["first_chain_id"]),
        residue_number=int(cyclization_info["first_pdb_resnum"]),
        insertion_code=str(cyclization_info.get("first_icode", "")),
        atom_names=["N"],
    )
    serials_last = _pdb_atom_serials_for_residue_atoms(
        pdb_path,
        chain_id=str(cyclization_info["last_chain_id"]),
        residue_number=int(cyclization_info["last_pdb_resnum"]),
        insertion_code=str(cyclization_info.get("last_icode", "")),
        atom_names=["C"],
    )
    serial_n = serials_first.get("N")
    serial_c = serials_last.get("C")
    if serial_n is None or serial_c is None:
        raise ValueError(
            f"Failed to resolve head-tail atom serials in {pdb_path} for "
            f"{cyclization_info['first_chain_id']}{cyclization_info['first_pdb_resnum']}:N and "
            f"{cyclization_info['last_chain_id']}{cyclization_info['last_pdb_resnum']}:C."
        )

    new_connects = {
        f"CONECT{serial_n:>5}{serial_c:>5}",
        f"CONECT{serial_c:>5}{serial_n:>5}",
    }
    with open(pdb_path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle]

    existing_connects = {line for line in lines if line.startswith("CONECT")}
    for connect in sorted(new_connects):
        if connect not in existing_connects:
            insert_idx = len(lines)
            for i, line in enumerate(lines):
                if line.startswith(("ENDMDL", "END")):
                    insert_idx = i
                    break
            lines.insert(insert_idx, connect)
            existing_connects.add(connect)

    with open(pdb_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


def append_disulfide_records(
    pdb_path: str,
    *,
    disulfide_info: list[dict[str, Any]],
) -> None:
    """Serialize validated Rosetta disulfides as explicit PDB records.

    PyRosetta's PDB dump does not reliably emit ``SSBOND``/``CONECT`` for a
    bond created with ``form_disulfide``.  Keep both a standard ``SSBOND``
    record and reciprocal SG ``CONECT`` records so downstream readers can
    audit the same endpoints independently of Rosetta's in-memory topology.
    """

    if not disulfide_info:
        return

    with open(pdb_path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle]

    existing_ssbond_pairs: set[tuple[tuple[str, int, str], tuple[str, int, str]]] = set()
    existing_ssbond_indices: dict[tuple[tuple[str, int, str], tuple[str, int, str]], int] = {}
    existing_ssbond_serials: dict[tuple[tuple[str, int, str], tuple[str, int, str]], int | None] = {}
    existing_conect_pairs: set[tuple[int, int]] = set()
    max_ssbond_serial = 0
    for line_index, line in enumerate(lines):
        if line.startswith("SSBOND"):
            try:
                first = (line[15].strip(), int(line[17:21]), line[21].strip())
                second = (line[29].strip(), int(line[31:35]), line[35].strip())
                pair = tuple(sorted((first, second)))
                existing_ssbond_pairs.add(pair)
                existing_ssbond_indices[pair] = line_index
                try:
                    serial = int(line[7:10])
                except (TypeError, ValueError):
                    serial = None
                existing_ssbond_serials[pair] = serial
                if serial is not None:
                    max_ssbond_serial = max(max_ssbond_serial, serial)
            except (IndexError, TypeError, ValueError):
                continue
        elif line.startswith("CONECT"):
            try:
                source = int(line[6:11].strip())
            except (IndexError, TypeError, ValueError):
                continue
            for offset in range(11, len(line), 5):
                try:
                    target = int(line[offset : offset + 5].strip())
                except (IndexError, TypeError, ValueError):
                    continue
                existing_conect_pairs.add(tuple(sorted((source, target))))

    new_lines: list[str] = []
    for item in disulfide_info:
        post_distance = float(item.get("post_sg_distance", float("nan")))
        if not np.isfinite(post_distance) or not (
            _DISULFIDE_MIN_DISTANCE <= post_distance <= _DISULFIDE_MAX_DISTANCE
        ):
            raise ValueError(
                "Cannot serialize disulfide with invalid post-relax SG-SG distance: "
                f"{post_distance:.3f} A (required {_DISULFIDE_MIN_DISTANCE:.1f}-"
                f"{_DISULFIDE_MAX_DISTANCE:.1f} A)."
            )

        first_chain = str(item["first_chain_id"]).strip()
        first_resnum = int(item["first_pdb_resnum"])
        first_icode = str(item.get("first_icode", "")).strip()
        first_resname = str(item["first_resname"]).strip()[:3].upper()
        second_chain = str(item["second_chain_id"]).strip()
        second_resnum = int(item["second_pdb_resnum"])
        second_icode = str(item.get("second_icode", "")).strip()
        second_resname = str(item["second_resname"]).strip()[:3].upper()
        if not first_chain or not second_chain or not first_resname or not second_resname:
            raise ValueError(f"Incomplete disulfide PDB endpoint metadata: {item!r}")

        serials_first = _pdb_atom_serials_for_residue_atoms(
            pdb_path,
            chain_id=first_chain,
            residue_number=first_resnum,
            insertion_code=first_icode,
            atom_names=["SG"],
        )
        serials_second = _pdb_atom_serials_for_residue_atoms(
            pdb_path,
            chain_id=second_chain,
            residue_number=second_resnum,
            insertion_code=second_icode,
            atom_names=["SG"],
        )
        serial_first = serials_first.get("SG")
        serial_second = serials_second.get("SG")
        if serial_first is None or serial_second is None:
            raise ValueError(
                "Failed to resolve SG serials for disulfide endpoints "
                f"{first_chain}{first_resnum} and {second_chain}{second_resnum} in {pdb_path}."
            )
        coordinates_first = _pdb_atom_coordinates_for_residue_atoms(
            pdb_path,
            chain_id=first_chain,
            residue_number=first_resnum,
            insertion_code=first_icode,
            atom_names=["SG"],
        ).get("SG")
        coordinates_second = _pdb_atom_coordinates_for_residue_atoms(
            pdb_path,
            chain_id=second_chain,
            residue_number=second_resnum,
            insertion_code=second_icode,
            atom_names=["SG"],
        ).get("SG")
        if coordinates_first is None or coordinates_second is None:
            raise ValueError(
                "Failed to resolve SG coordinates for disulfide endpoints "
                f"{first_chain}{first_resnum} and {second_chain}{second_resnum} in {pdb_path}."
            )
        output_distance = float(np.linalg.norm(coordinates_first - coordinates_second))
        if not np.isfinite(output_distance) or not (
            _DISULFIDE_MIN_DISTANCE <= output_distance <= _DISULFIDE_MAX_DISTANCE
        ):
            raise ValueError(
                "Dumped disulfide PDB has invalid SG-SG geometry: "
                f"{output_distance:.3f} A (required {_DISULFIDE_MIN_DISTANCE:.1f}-"
                f"{_DISULFIDE_MAX_DISTANCE:.1f} A)."
            )
        if abs(output_distance - post_distance) > 0.02:
            raise ValueError(
                "Dumped disulfide PDB geometry disagrees with the relaxed pose: "
                f"pose={post_distance:.3f} A, PDB={output_distance:.3f} A."
            )

        endpoint_pair = tuple(
            sorted(
                (
                    (first_chain, first_resnum, first_icode),
                    (second_chain, second_resnum, second_icode),
                )
            )
        )
        max_ssbond_serial += 1
        formatted_ssbond = (
            f"SSBOND {max_ssbond_serial:>3} {first_resname:>3} {first_chain:1} "
            f"{first_resnum:>4}{first_icode:1}   {second_resname:>3} {second_chain:1} "
            f"{second_resnum:>4}{second_icode:1}                         {output_distance:>5.2f}"
        )
        existing_index = existing_ssbond_indices.get(endpoint_pair)
        if existing_index is None:
            new_lines.append(formatted_ssbond)
            existing_ssbond_indices[endpoint_pair] = -1
            existing_ssbond_serials[endpoint_pair] = max_ssbond_serial
            existing_ssbond_pairs.add(endpoint_pair)
        elif existing_ssbond_serials.get(endpoint_pair) is None:
            lines[existing_index] = formatted_ssbond
            existing_ssbond_serials[endpoint_pair] = max_ssbond_serial
        else:
            max_ssbond_serial -= 1

        serial_pair = tuple(sorted((int(serial_first), int(serial_second))))
        if serial_pair not in existing_conect_pairs:
            new_lines.extend(
                (
                    f"CONECT{int(serial_first):>5}{int(serial_second):>5}",
                    f"CONECT{int(serial_second):>5}{int(serial_first):>5}",
                )
            )
            existing_conect_pairs.add(serial_pair)

    if not new_lines:
        return
    insert_idx = len(lines)
    for index, line in enumerate(lines):
        if line.startswith(("ENDMDL", "END")):
            insert_idx = index
            break
    lines[insert_idx:insert_idx] = new_lines
    with open(pdb_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")


@dataclass(frozen=True)
class InterfaceResidues:
    pdb_ids_csv: str
    aa_counts: dict[str, int]

    @property
    def n_residues(self) -> int:
        return int(sum(self.aa_counts.values()))


def interface_residues_by_distance(
    pose: Any,
    *,
    binder_chain: str = "L",
    target_chains: str,
    atom_distance_cutoff: float = 4.0,
) -> InterfaceResidues:
    pdb_info = pose.pdb_info()
    if pdb_info is None:
        raise ValueError("Pose has no PDBInfo; cannot compute PDB residue IDs.")

    target_chain_ids = set(_parse_chain_ids(target_chains))
    cutoff = float(atom_distance_cutoff)

    target_atom_coords: list[list[float]] = []
    for i in range(1, pose.total_residue() + 1):
        if str(pdb_info.chain(i)) not in target_chain_ids:
            continue
        res = pose.residue(i)
        if not bool(res.is_protein()):
            continue
        for atom_i in range(1, int(res.nheavyatoms()) + 1):
            xyz = res.xyz(atom_i)
            target_atom_coords.append([float(xyz.x), float(xyz.y), float(xyz.z)])

    if not target_atom_coords:
        return InterfaceResidues(pdb_ids_csv="", aa_counts={aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"})

    target_xyz = np.asarray(target_atom_coords, dtype=float)
    target_tree = cKDTree(target_xyz) if cKDTree is not None else None

    interface_resnums: dict[int, str] = {}
    for i in range(1, pose.total_residue() + 1):
        if str(pdb_info.chain(i)) != binder_chain:
            continue
        res = pose.residue(i)
        if not bool(res.is_protein()):
            continue

        binder_atom_coords: list[list[float]] = []
        for atom_i in range(1, int(res.nheavyatoms()) + 1):
            xyz = res.xyz(atom_i)
            binder_atom_coords.append([float(xyz.x), float(xyz.y), float(xyz.z)])
        if not binder_atom_coords:
            continue

        binder_xyz = np.asarray(binder_atom_coords, dtype=float)
        if target_tree is not None:
            close = target_tree.query_ball_point(binder_xyz, r=cutoff)
            is_interface = any(len(nbrs) > 0 for nbrs in close)
        else:
            d2 = ((binder_xyz[:, None, :] - target_xyz[None, :, :]) ** 2).sum(axis=2)
            is_interface = bool((d2 <= cutoff * cutoff).any())

        if is_interface:
            pdb_resnum = int(pdb_info.number(i))
            aa1 = str(res.name1())
            interface_resnums[pdb_resnum] = aa1 if aa1 in "ACDEFGHIKLMNPQRSTVWY" else f"[{res.name3()}]"

    aa_counts = {aa: 0 for aa in "ACDEFGHIKLMNPQRSTVWY"}
    for aa in interface_resnums.values():
        aa_counts[aa] = aa_counts.get(aa, 0) + 1

    pdb_ids_csv = ",".join(f"{binder_chain}{n}" for n in sorted(interface_resnums))
    return InterfaceResidues(pdb_ids_csv=pdb_ids_csv, aa_counts=aa_counts)



def get_scorefxn(scorefxn_name: str):
    """
    Gets the scorefxn with appropriate corrections.
    """
    corrections = {
        'beta_july15': False,
        'beta_nov16': False,
        'gen_potential': False,
        'restore_talaris_behavior': False,
    }
    if 'beta_july15' in scorefxn_name or 'beta_nov15' in scorefxn_name:
        corrections['beta_july15'] = True
    elif 'beta_nov16' in scorefxn_name:
        corrections['beta_nov16'] = True
    elif 'genpot' in scorefxn_name:
        corrections['gen_potential'] = True
        pr.rosetta.basic.options.set_boolean_option('corrections:beta_july15', True)
    elif 'talaris' in scorefxn_name:
        corrections['restore_talaris_behavior'] = True
    for corr, value in corrections.items():
        pr.rosetta.basic.options.set_boolean_option(f'corrections:{corr}', value)
    return pr.create_score_function(scorefxn_name)


def fast_relax_pose2(
    pose: Any,
    *,
    binder_chains: str = "L",
    scorefxn: Literal["ref2015"] | str = "ref2015",
    max_iter: int = 1000,
    subset: Literal["all", "nbrs", "target"] = "nbrs",
    move_bb: bool = True,
    cyclic_first_idx: int | None = None,
    cyclic_last_idx: int | None = None,
) -> None:
    """Advanced FastRelax using TaskFactory and MoveMapFactory to restrict
    packing and minimization to specific subsets (all, nbrs, or target).
    Mutates `pose` in-place.

    When cyclic_first_idx/cyclic_last_idx are provided, applies Rosetta's native
    cutpoint-patch mechanism to enforce correct peptide-bond geometry at the
    head-to-tail junction during relax.
    """
    cyclic = cyclic_first_idx is not None and cyclic_last_idx is not None
    if cyclic:
        add_cyclic_peptide_bond_constraints(
            pose,
            first_idx=int(cyclic_first_idx),
            last_idx=int(cyclic_last_idx),
        )

    fastrelax = FastRelax()

    if subset == 'all':
        fastrelax.constrain_relax_to_start_coords(True)

    sfxn_name = str(scorefxn)
    if cyclic and sfxn_name == "ref2015":
        sfxn_name = "ref2015_cst"
    scorefxn_obj = get_scorefxn(sfxn_name)
    fastrelax.set_scorefxn(scorefxn_obj)
    fastrelax.max_iter(int(max_iter))
    tf = TaskFactory()
    tf.push_back(operation.InitializeFromCommandline())
    tf.push_back(operation.RestrictToRepacking())
    chain_ids = _parse_chain_ids(binder_chains)
    if subset != 'all':
        chain_selectors = [selections.ChainSelector(c) for c in chain_ids]
        if len(chain_selectors) == 1:
            gen_selector = chain_selectors[0]
        elif len(chain_selectors) > 1:
            gen_selector = selections.OrResidueSelector(chain_selectors[0], chain_selectors[1])
            for selector in chain_selectors[2:]:
                gen_selector = selections.OrResidueSelector(gen_selector, selector)
        else:
            gen_selector = selections.TrueResidueSelector()
        nbr_selector = selections.NeighborhoodResidueSelector()
        nbr_selector.set_focus_selector(gen_selector)
        nbr_selector.set_include_focus_in_subset(True)
        if subset == 'nbrs':
            subset_selector = nbr_selector
        elif subset == 'target':
            subset_selector = gen_selector
        else:
            subset_selector = nbr_selector
        prevent_repacking_rlt = operation.PreventRepackingRLT()
        prevent_subset_repacking = operation.OperateOnResidueSubset(
            prevent_repacking_rlt,
            subset_selector,
            flip_subset=True,
        )
        tf.push_back(prevent_subset_repacking)
    else:
        gen_selector = selections.TrueResidueSelector()
        subset_selector = selections.TrueResidueSelector()
    mmf = MoveMapFactory()
    if move_bb:
        mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
    mmf.add_chi_action(move_map_action.mm_enable, subset_selector)
    mm = mmf.create_movemap_from_pose(pose)
    fastrelax.set_movemap(mm)
    fastrelax.set_task_factory(tf)
    fastrelax.apply(pose)
    return None


def add_cyclic_peptide_bond_constraints(
    pose: Any,
    *,
    first_idx: int,
    last_idx: int,
) -> None:
    """Prepare a head-to-tail cyclic junction for FastRelax using Rosetta's native
    cutpoint-patch mechanism (same approach as protocols::relax::cyclize_pose, but
    targeted at an arbitrary binder chain rather than the whole pose).

    Applies CUTPOINT_LOWER to last_idx and CUTPOINT_UPPER to first_idx, which adds
    virtual overlap atoms (OVL1, OVL2 on last; OVU1 on first).  Three HARMONIC(0, 0.1)
    AtomPair constraints between these virtual atoms and their counterparts on the
    opposite residue then enforce correct peptide-bond bond length, bond angles, and
    omega simultaneously — exactly what cyclize_pose does internally.
    """
    add_variant_type_to_pose_residue(pose, CUTPOINT_LOWER, last_idx)
    add_variant_type_to_pose_residue(pose, CUTPOINT_UPPER, first_idx)

    ovl1 = AtomID(pose.residue(last_idx).atom_index("OVL1"), last_idx)
    ovl2 = AtomID(pose.residue(last_idx).atom_index("OVL2"), last_idx)
    ovu1 = AtomID(pose.residue(first_idx).atom_index("OVU1"), first_idx)
    atom_n  = AtomID(pose.residue(first_idx).atom_index("N"),  first_idx)
    atom_ca = AtomID(pose.residue(first_idx).atom_index("CA"), first_idx)
    atom_c  = AtomID(pose.residue(last_idx).atom_index("C"),   last_idx)

    pose.add_constraint(AtomPairConstraint(atom_n,  ovl1, HarmonicFunc(0.0, 0.1)))
    pose.add_constraint(AtomPairConstraint(atom_ca, ovl2, HarmonicFunc(0.0, 0.1)))
    pose.add_constraint(AtomPairConstraint(ovu1, atom_c,  HarmonicFunc(0.0, 0.1)))


def fast_relax_pose(
    pose: Any,
    *,
    scorefxn: Literal["ref2015"] | str = "ref2015",
    max_iter: int = 200,
    move_bb: bool = True,
    move_chi: bool = True,
    allow_jump: bool = False,
    constrain_to_start_coords: bool = True,
    min_type: str = "lbfgs_armijo_nonmonotone",
    cyclic_first_idx: int | None = None,
    cyclic_last_idx: int | None = None,
) -> None:
    """FastRelax a pose in-place.

    When cyclic_first_idx/cyclic_last_idx are provided, applies Rosetta's native
    cutpoint-patch mechanism (CUTPOINT_LOWER/UPPER + OVL/OVU virtual atoms) to
    enforce correct peptide-bond geometry at the head-to-tail junction during relax.
    """
    mm = MoveMap()
    mm.set_bb(bool(move_bb))
    mm.set_chi(bool(move_chi))
    mm.set_jump(bool(allow_jump))

    cyclic = cyclic_first_idx is not None and cyclic_last_idx is not None
    if cyclic:
        add_cyclic_peptide_bond_constraints(
            pose,
            first_idx=int(cyclic_first_idx),
            last_idx=int(cyclic_last_idx),
        )
        sfxn_name = "ref2015_cst" if str(scorefxn) == "ref2015" else str(scorefxn)
    else:
        sfxn_name = str(scorefxn)

    fastrelax = FastRelax()
    scorefxn_obj = pr.create_score_function(sfxn_name)
    fastrelax.set_scorefxn(scorefxn_obj)
    fastrelax.set_movemap(mm)
    fastrelax.max_iter(int(max_iter))
    fastrelax.min_type(str(min_type))
    fastrelax.constrain_relax_to_start_coords(bool(constrain_to_start_coords))
    fastrelax.apply(pose)
    return None


def resolve_relax_settings(
    *,
    relax_mode: Literal["local", "global"] | str = "global",
    relax_subset: Literal["all", "nbrs", "target"] | str | None = None,
    relax_max_iter: int | None = None,
) -> tuple[str, int]:
    mode = str(relax_mode or "global").strip().lower()
    subset = str(relax_subset).strip().lower() if relax_subset is not None else ""

    if subset:
        if subset not in {"all", "nbrs", "target"}:
            raise ValueError(f"Unsupported relax_subset {relax_subset!r}; expected one of all/nbrs/target.")
    elif mode == "global":
        subset = "all"
    else:
        subset = "nbrs"

    if relax_max_iter is None:
        max_iter = 200 if subset == "all" else 1000
    else:
        max_iter = int(relax_max_iter)
    return subset, max_iter


def score_interface(
    pdb_path: str,
    *,
    binder_chain: str = "L",
    binder_is_cyclic: bool = False,
    topology_entries: Any = None,
    target_chains: str | Iterable[str] | None = None,
    scorefxn: Literal["ref2015"] | str = "ref2015",
    atom_distance_cutoff: float = 4.0,
    compute_buns: bool = True,
    buns_probe_radius: float = 1.1,
    buns_burial_cutoff_apo: float = 0.2,
    relax_before_scoring: bool = False,
    relax_mode: Literal["local", "global"] | str = "global",
    relax_subset: Literal["all", "nbrs", "target"] | str | None = None,
    relax_max_iter: int | None = None,
    relax_constrain_to_start_coords: bool = True,
    dump_relaxed_pdb: bool = False,
    relaxed_out_path: str | None = None,
    native_pose: Any = None,
    native_no_pack: bool = False,
) -> dict[str, Any]:

    pose = native_pose if native_pose is not None else pose_from_pdb_no_link(pdb_path)
    cyclization_info: dict[str, Any] | None = None
    native_head_tail = native_pose is not None and any(
        _normalize_topology_type(entry.get("type")) == "head_to_tail"
        for entry in _decode_topology_entries(topology_entries))
    applied_cyclization = native_pose is None and should_cyclize_binder(
        pose,
        binder_chain=binder_chain,
        binder_is_cyclic=binder_is_cyclic,
        topology_entries=topology_entries,
    )
    if applied_cyclization:
        cyclization_info = cyclize_head_tail_binder_pose(pose, binder_chain=binder_chain)

    disulfide_info = [] if native_pose is not None else apply_disulfide_topology_to_pose(
        pose,
        binder_chain=binder_chain,
        topology_entries=topology_entries,
    )
    uses_topology_constraints = cyclization_info is not None or bool(disulfide_info)
    effective_scorefxn = "ref2015_cst" if uses_topology_constraints and str(scorefxn) == "ref2015" else str(scorefxn)
    if native_no_pack:
        from partiflow.postprocess.native import _score_function_for_pose
        scorefxn_obj, _ = _score_function_for_pose(pose)
        if relax_before_scoring:
            raise ValueError("Native nonstandard poses cannot be relaxed or packed by Rosetta")
    else:
        scorefxn_obj = pr.create_score_function(effective_scorefxn)

    if relax_before_scoring:
        subset, max_iter = resolve_relax_settings(
            relax_mode=relax_mode,
            relax_subset=relax_subset,
            relax_max_iter=relax_max_iter,
        )
        fast_relax_pose2(
            pose,
            binder_chains=binder_chain,
            scorefxn=effective_scorefxn,
            max_iter=max_iter,
            subset=subset,
            move_bb=True,
            cyclic_first_idx=cyclization_info["first_pose_idx"] if cyclization_info is not None else None,
            cyclic_last_idx=cyclization_info["last_pose_idx"] if cyclization_info is not None else None,
        )
        if cyclization_info is not None:
            _refresh_head_tail_geometry_after_relax(pose, cyclization_info)
        if disulfide_info:
            disulfide_info = validate_declared_disulfides(pose, disulfide_info)
        if dump_relaxed_pdb:
            if relaxed_out_path is not None:
                out_pdb_path = str(relaxed_out_path)
            elif '_openmm.pdb' in pdb_path:
                out_pdb_path = pdb_path.replace('_openmm.pdb', '_rosetta.pdb')
            else:
                # assume no postfix, just add one
                out_pdb_path = pdb_path.replace('.pdb', '_rosetta.pdb')
            os.makedirs(os.path.dirname(os.path.abspath(out_pdb_path)), exist_ok=True)
            pose.dump_pdb(out_pdb_path)
            clean_pdb(out_pdb_path)
            if disulfide_info:
                append_disulfide_records(out_pdb_path, disulfide_info=disulfide_info)
            if cyclization_info is not None:
                append_head_tail_conect(out_pdb_path, cyclization_info=cyclization_info)
    else:
        subset, max_iter = resolve_relax_settings(
            relax_mode=relax_mode,
            relax_subset=relax_subset,
            relax_max_iter=relax_max_iter,
        )
    if disulfide_info and not relax_before_scoring:
        disulfide_info = validate_declared_disulfides(pose, disulfide_info)
    if cyclization_info is not None and not relax_before_scoring:
        _refresh_head_tail_geometry_after_relax(pose, cyclization_info)

    if target_chains is None:
        target_chain_str = _infer_target_chains_from_pose(pose, binder_chain=binder_chain)
    else:
        target_chain_str = "".join(_parse_chain_ids(target_chains))

    total_score = float(scorefxn_obj(pose))

    # Rosetta interface metrics
    iam = InterfaceAnalyzerMover()
    _set_interface_analyzer_interface(iam, f"{target_chain_str}_{binder_chain}")
    iam.set_scorefunction(scorefxn_obj)
    iam.set_compute_packstat(not native_no_pack)
    iam.set_compute_interface_energy(True)
    iam.set_calc_dSASA(True)
    iam.set_calc_hbond_sasaE(True)
    iam.set_compute_interface_sc(True)
    iam.set_pack_input(False)
    iam.set_pack_separated(not native_no_pack)
    iam.apply(pose)

    interfacescore = iam.get_all_data()
    interface_sc = float(interfacescore.sc_value)
    interface_interface_hbonds = float(interfacescore.interface_hbonds)
    interface_dG = float(iam.get_interface_dG())
    interface_dSASA = float(iam.get_interface_delta_sasa())
    interface_packstat = None if native_no_pack else float(iam.get_interface_packstat())
    interface_dG_SASA_ratio = float(interfacescore.dG_dSASA_ratio) * 100.0

    # Interface residue ids + AA counts (distance-based, robust across Rosetta versions)
    interface = interface_residues_by_distance(
        pose, binder_chain=binder_chain, target_chains=target_chain_str, atom_distance_cutoff=atom_distance_cutoff
    )
    interface_nres = interface.n_residues

    hydrophobic_aa = set("ACFILMPVWY")
    hydrophobic_count = int(sum(interface.aa_counts[aa] for aa in hydrophobic_aa))
    interface_hydrophobicity = (hydrophobic_count / interface_nres) * 100.0 if interface_nres else 0.0

    interface_delta_unsat_hbonds: float | None
    buns_error = None
    if compute_buns:
        try:
            xml = f"""
<ROSETTASCRIPTS>
  <SCOREFXNS>
    <ScoreFunction name="scorefxn" weights="{scorefxn}">
      {'<Reweight scoretype="fa_dun" weight="0"/>' if native_no_pack else ''}
    </ScoreFunction>
  </SCOREFXNS>
  <FILTERS>
    <BuriedUnsatHbonds
      name="buns"
      report_all_heavy_atom_unsats="true"
      scorefxn="scorefxn"
      ignore_surface_res="false"
      use_ddG_style="true"
      dalphaball_sasa="1"
      probe_radius="{buns_probe_radius}"
      burial_cutoff_apo="{buns_burial_cutoff_apo}"
      confidence="0"
    />
  </FILTERS>
  <PROTOCOLS>
    <Add filter="buns"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
""".strip()
            buns = XmlObjects.create_from_string(xml).get_filter("buns")
            interface_delta_unsat_hbonds = float(buns.report_sm(pose))
        except Exception as e:
            buns_error = f"{type(e).__name__}: {e}"
            interface_delta_unsat_hbonds = None
    else:
        interface_delta_unsat_hbonds = None

    if interface_nres:
        interface_hbond_percentage = (interface_interface_hbonds / interface_nres) * 100.0
        interface_bunsch_percentage = (
            (interface_delta_unsat_hbonds / interface_nres) * 100.0 if interface_delta_unsat_hbonds is not None else None
        )
    else:
        interface_hbond_percentage = None
        interface_bunsch_percentage = None

    # Binder energy score (per-chain total energy)
    chain_design = ChainSelector(binder_chain)
    tem = pr.rosetta.core.simple_metrics.metrics.TotalEnergyMetric()
    tem.set_scorefunction(scorefxn_obj)
    tem.set_residue_selector(chain_design)
    binder_score = float(tem.calculate(pose))

    # Binder SASA fraction
    bsasa = pr.rosetta.core.simple_metrics.metrics.SasaMetric()
    bsasa.set_residue_selector(chain_design)
    binder_sasa = float(bsasa.calculate(pose))
    interface_binder_fraction = (interface_dSASA / binder_sasa) * 100.0 if binder_sasa > 0 else 0.0

    # Surface hydrophobicity (binder surface residue composition proxy)
    # When a LINK record merges chains into one conformation chain, split_by_chain()
    # returns fewer entries than unique pdb_info chain letters. Fall back to extracting
    # binder residues directly from the full pose using the residue selector.
    chain_map = {
        pose.pdb_info().chain(pose.conformation().chain_begin(i)): p
        for i, p in zip(range(1, pose.num_chains() + 1), pose.split_by_chain())
    }
    if binder_chain in chain_map:
        binder_pose = chain_map[binder_chain]
    else:
        import pyrosetta.rosetta.core.select as _sel
        binder_subset = ChainSelector(binder_chain).apply(pose)
        binder_indices_all = _sel.get_residues_from_subset(binder_subset)
        binder_pose = pr.Pose()
        pr.rosetta.core.io.pose_from_pose(binder_pose, pose, binder_indices_all)
    pep_internal_sc = float(scorefxn_obj(binder_pose))
    i_sc = float(interface_dG)
    pep_sc = pep_internal_sc + i_sc
    reweighted_sc = total_score + pep_sc + i_sc

    unsupported_backbone = [str(binder_pose.residue_type(i).name())
                            for i in range(1, binder_pose.total_residue()+1)
                            if not binder_pose.residue_type(i).is_alpha_aa()]
    surface_error = ''
    surface_hydrophobicity = None
    if unsupported_backbone:
        surface_error = 'Rosetta LayerSelector requires alpha-amino-acid backbones: ' + ','.join(sorted(set(unsupported_backbone)))
    else:
        layer_sel = pr.rosetta.core.select.residue_selector.LayerSelector()
        layer_sel.set_layers(pick_core=False, pick_boundary=False, pick_surface=True)
        surface_res = layer_sel.apply(binder_pose)

        exp_apol_count = 0
        total_count = 0
        for i in range(1, len(surface_res) + 1):
            if bool(surface_res[i]):
                res = binder_pose.residue(i)
                if bool(res.is_apolar()) or res.name() in {"PHE", "TRP", "TYR"}:
                    exp_apol_count += 1
                total_count += 1
        surface_hydrophobicity = (exp_apol_count / total_count) if total_count else 0.0

    out: dict[str, Any] = {
        "binder_chain": binder_chain,
        "binder_is_cyclic": bool(applied_cyclization or native_head_tail),
        "target_chains": target_chain_str,
        "total_score": round(total_score, 3),
        "I_sc": round(i_sc, 3),
        "pep_internal_sc": round(pep_internal_sc, 3),
        "pep_sc": round(pep_sc, 3),
        "reweighted_sc": round(reweighted_sc, 3),
        "rank_score": round(reweighted_sc, 3),
        "Binder_Energy_Score": round(binder_score, 2),
        "Surface_Hydrophobicity": round(surface_hydrophobicity, 4) if surface_hydrophobicity is not None else None,
        "ShapeComplementarity": round(interface_sc, 3),
        "PackStat": round(interface_packstat, 3) if interface_packstat is not None else None,
        "dG": round(interface_dG, 2),
        "dSASA": round(interface_dSASA, 2),
        "dG/dSASA": round(interface_dG_SASA_ratio, 2),
        "Interface_SASA_%": round(interface_binder_fraction, 2),
        "Interface_Hydrophobicity": None if native_no_pack else round(interface_hydrophobicity, 2),
        "n_InterfaceResidues": int(interface_nres),
        "n_InterfaceHbonds": round(interface_interface_hbonds, 2),
        "InterfaceHbondsPercentage": (round(interface_hbond_percentage, 2) if interface_hbond_percentage is not None else None),
        "n_InterfaceUnsatHbonds": (round(interface_delta_unsat_hbonds, 2) if interface_delta_unsat_hbonds is not None else None),
        "InterfaceUnsatHbondsPercentage": (
            round(interface_bunsch_percentage, 2) if interface_bunsch_percentage is not None else None
        ),
        "InterfaceResidues": interface.pdb_ids_csv,
        "InterfaceAAs": interface.aa_counts,
        "relax_subset": subset,
        "relax_max_iter": int(max_iter),
        "rosetta_scorefxn": effective_scorefxn,
    }
    if buns_error:
        out["unavailable_metric_errors"] = {"interface_unsat_hbonds": buns_error}
    if surface_error:
        out.setdefault("unavailable_metric_errors", {})["surface_hydrophobicity"] = surface_error
    if disulfide_info:
        out["rosetta_disulfide_declared"] = ";".join(str(item["anchors"]) for item in disulfide_info)
        out["rosetta_disulfide_pre_sg_distances"] = ";".join(
            f"{item['anchors']}:{float(item['pre_sg_distance']):.3f}" for item in disulfide_info
        )
        out["rosetta_disulfide_post_sg_distances"] = ";".join(
            f"{item['anchors']}:{float(item['post_sg_distance']):.3f}" for item in disulfide_info
        )
    if cyclization_info is not None:
        out["binder_cyclization"] = "head_to_tail"
        out["binder_head_tail_distance_before"] = round(float(cyclization_info["pre_n_c_distance"]), 3)
        out["binder_head_tail_distance_after"] = round(float(cyclization_info["post_n_c_distance"]), 3)
    else:
        out["binder_cyclization"] = "head_to_tail" if native_head_tail else "linear"
    return out


def fast_relax_pdb(
    pdb_path: str,
    out_path: str,
    *,
    binder_chain: str = "L",
    binder_is_cyclic: bool = False,
    topology_entries: Any = None,
    scorefxn: Literal["ref2015"] | str = "ref2015",
    max_iter: int = 200,
    move_bb: bool = True,
    move_chi: bool = True,
    allow_jump: bool = False,
    constrain_to_start_coords: bool = True,
    min_type: str = "lbfgs_armijo_nonmonotone",
) -> str:

    if os.path.exists(out_path):
        return out_path

    pose = pose_from_pdb_no_link(pdb_path)
    cyclization_info: dict[str, Any] | None = None
    if should_cyclize_binder(
        pose,
        binder_chain=binder_chain,
        binder_is_cyclic=binder_is_cyclic,
        topology_entries=topology_entries,
    ):
        cyclization_info = cyclize_head_tail_binder_pose(pose, binder_chain=binder_chain)

    disulfide_info = apply_disulfide_topology_to_pose(
        pose,
        binder_chain=binder_chain,
        topology_entries=topology_entries,
    )
    effective_scorefxn = "ref2015_cst" if disulfide_info and str(scorefxn) == "ref2015" else str(scorefxn)

    fast_relax_pose(
        pose,
        scorefxn=effective_scorefxn,
        max_iter=max_iter,
        move_bb=move_bb,
        move_chi=move_chi,
        allow_jump=allow_jump,
        constrain_to_start_coords=constrain_to_start_coords,
        min_type=min_type,
        cyclic_first_idx=cyclization_info["first_pose_idx"] if cyclization_info is not None else None,
        cyclic_last_idx=cyclization_info["last_pose_idx"] if cyclization_info is not None else None,
    )
    if cyclization_info is not None:
        _refresh_head_tail_geometry_after_relax(pose, cyclization_info)
    if disulfide_info:
        disulfide_info = validate_declared_disulfides(pose, disulfide_info)

    pose.dump_pdb(out_path)
    clean_pdb(out_path)
    if disulfide_info:
        append_disulfide_records(out_path, disulfide_info=disulfide_info)
    if cyclization_info is not None:
        append_head_tail_conect(out_path, cyclization_info=cyclization_info)
    return out_path

