import itertools


import random


import csv


import json


import ntpath


import os


import stat


import shutil


import uuid


from pathlib import Path


from collections import defaultdict


from typing import Any, Dict, List


from collections.abc import Mapping, Sequence


import numpy as np


import biotite.structure as struc


import biotite.structure.io as strucio


from scipy.spatial.distance import cdist


import lightning as L


import torch


import torch.distributed as dist


from loguru import logger


from omegaconf import OmegaConf


from torch_geometric.transforms import Compose


from partiflow.datasets.transforms import (
    CoordsToNanometers,
    ChainBreakPerResidueTransform,
    CenterStructureTransform,
    MotifMaskTransform,
)


from partiflow.utils.secstruct_utils import (
    BINDER_SECSTRUCT_ALPHA,
    BINDER_SECSTRUCT_BETA,
    BINDER_SECSTRUCT_FEW_SS,
    binder_secstruct_name_from_id,
    classify_secondary_structure_metrics,
    secondary_structure_metrics_from_pdb_string,
)


from partiflow.utils.conditioning_utils import (
    DESIGN_CONDITION_NONE,
    design_condition_mode_to_id,
    design_condition_mode_to_name,
    pair_bin_idx_from_distance_angstrom,
)


from partiflow.utils.nonstandard_residue_registry import (
    UNSUPPORTED_NONSTANDARD_RESIDUES,
    get_nonstandard_residue_spec,
)


from partiflow.utils.pdb_utils import PDB_CHAIN_IDS, create_full_prot, to_pdb


from partiflow.openfold.np.residue_constants import (
    restype_1to3,
    restype_3to1,
    restype_order,
    restypes,
)


INFERENCE_TRANSFORMS = Compose(
    [
        CoordsToNanometers(),
        ChainBreakPerResidueTransform(),
        CenterStructureTransform(),
        MotifMaskTransform(atom_selection_mode="all"),
    ]
)


CYCLIZATION_CA_PAIR_TARGET_ANGSTROM = 3.8


DISULFIDE_CA_PAIR_TARGET_ANGSTROM = 5.5


_PDB_COORD_MIN = -999.999


_PDB_COORD_MAX = 9999.999


SECSTRUCT_ALIASES = {
    "helix": BINDER_SECSTRUCT_ALPHA,
    "alpha": BINDER_SECSTRUCT_ALPHA,
    "strand": BINDER_SECSTRUCT_BETA,
    "sheet": BINDER_SECSTRUCT_BETA,
    "beta": BINDER_SECSTRUCT_BETA,
    "coil": BINDER_SECSTRUCT_FEW_SS,
    "loop": BINDER_SECSTRUCT_FEW_SS,
}


VALID_INFERENCE_SECSTRUCT_IDS = set(SECSTRUCT_ALIASES.values())


SUPPORTED_INFERENCE_SECSTRUCT_NAMES = ("alpha", "beta", "coil")


SUPPORTED_ACTIVE_TASK_TYPES = frozenset(
    {"denovo", "motif", "docking", "inv_folding", "packing"}
)


BINDER_TOPOLOGY_SS_MASK = 3


BINDER_TOPOLOGY_ADJ_MASK = 2


BINDER_TOPOLOGY_SS_BY_NAME = {
    "alpha": 0,
    "helix": 0,
    "h": 0,
    "beta": 1,
    "strand": 1,
    "sheet": 1,
    "e": 1,
    "coil": 2,
    "loop": 2,
    "l": 2,
    "c": 2,
}


BINDER_TOPOLOGY_LIB_CACHE: Dict[str, Any] = {}


BINDER_SCAFFOLD_CATEGORY_ALIASES = {
    "linpep": "linear_peptide",
    "linear": "linear_peptide",
    "linear_peptide": "linear_peptide",
    "peptide": "linear_peptide",
    "cycpep": "head_tail_amide_cyclic_peptide",
    "cyclic": "head_tail_amide_cyclic_peptide",
    "cyclic_peptide": "head_tail_amide_cyclic_peptide",
    "head_tail": "head_tail_amide_cyclic_peptide",
    "head_to_tail": "head_tail_amide_cyclic_peptide",
    "head_tail_amide": "head_tail_amide_cyclic_peptide",
    "head_tail_cyclic": "head_tail_amide_cyclic_peptide",
    "head_tail_cyclic_peptide": "head_tail_amide_cyclic_peptide",
    "head_tail_amide_cyclic": "head_tail_amide_cyclic_peptide",
    "head_tail_amide_cyclic_peptide": "head_tail_amide_cyclic_peptide",
    "linpep_ss": "disulfied_peptide",
    "linear_ss": "disulfied_peptide",
    "linear_disulfide": "disulfied_peptide",
    "linear_disulfide_peptide": "disulfied_peptide",
    "linear_disulfied": "disulfied_peptide",
    "disulfide": "disulfied_peptide",
    "disulfide_peptide": "disulfied_peptide",
    "disulfied": "disulfied_peptide",
    "disulfied_peptide": "disulfied_peptide",
    "ss_peptide": "disulfied_peptide",
    "head_tail_disulfide": "head_tail_disulfied_cyclic_peptide",
    "head_tail_disulfide_cyclic": "head_tail_disulfied_cyclic_peptide",
    "head_tail_disulfide_cyclic_peptide": "head_tail_disulfied_cyclic_peptide",
    "head_tail_disulfied": "head_tail_disulfied_cyclic_peptide",
    "head_tail_disulfied_cyclic": "head_tail_disulfied_cyclic_peptide",
    "head_tail_disulfied_cyclic_peptide": "head_tail_disulfied_cyclic_peptide",
    "cycpep_ss": "head_tail_amide_cyclic_disulfide_peptide",
    "cyclic_ss": "head_tail_amide_cyclic_disulfide_peptide",
    "cyclic_disulfide": "head_tail_amide_cyclic_disulfide_peptide",
    "head_tail_cyclic_disulfide": "head_tail_amide_cyclic_disulfide_peptide",
    "head_tail_cyclic_disulfide_peptide": "head_tail_amide_cyclic_disulfide_peptide",
    "head_tail_amide_cyclic_disulfide": "head_tail_amide_cyclic_disulfide_peptide",
    "head_tail_amide_cyclic_disulfide_peptide": "head_tail_amide_cyclic_disulfide_peptide",
}


BINDER_SCAFFOLD_CYCLIC_CATEGORIES = {
    "head_tail_amide_cyclic_peptide",
    "head_tail_disulfied_cyclic_peptide",
    "head_tail_amide_cyclic_disulfide_peptide",
}


BINDER_SCAFFOLD_AMIDE_CYCLIC_CATEGORIES = {
    "head_tail_amide_cyclic_peptide",
    "head_tail_amide_cyclic_disulfide_peptide",
}


BINDER_SCAFFOLD_ROTATABLE_CATEGORIES = BINDER_SCAFFOLD_AMIDE_CYCLIC_CATEGORIES


BINDER_SCAFFOLD_DISULFIDE_CATEGORIES = {
    "disulfied_peptide",
    "head_tail_disulfied_cyclic_peptide",
    "head_tail_amide_cyclic_disulfide_peptide",
}


def _normalize_scaffold_category(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized or normalized in {"none", "null"}:
        return None
    return BINDER_SCAFFOLD_CATEGORY_ALIASES.get(normalized, normalized)


def _topology_entry_cyclic(entry: Mapping[str, Any]) -> bool | None:
    category = _normalize_scaffold_category(entry.get("category", None))
    if category in BINDER_SCAFFOLD_CYCLIC_CATEGORIES:
        return True
    if category in {"linear_peptide", "disulfied_peptide"}:
        return False
    if "cyclic" in entry and entry["cyclic"] is not None:
        return bool(entry["cyclic"])
    return None


def _topology_entry_uses_amide_cyclization(entry: Mapping[str, Any]) -> bool:
    category = _normalize_scaffold_category(entry.get("category", None))
    if category in BINDER_SCAFFOLD_AMIDE_CYCLIC_CATEGORIES:
        return True
    if category in {
        "head_tail_disulfied_cyclic_peptide",
        "linear_peptide",
        "disulfied_peptide",
    }:
        return False
    cyclization = entry.get("cyclization", None)
    if isinstance(cyclization, Mapping):
        mode = str(cyclization.get("mode", "") or "").strip().lower().replace("-", "_")
        return mode in {
            "head_to_tail",
            "head_tail",
            "amide",
            "amide_head_to_tail",
            "pair",
        }
    return False


def _topology_entry_allows_circular_shift(entry: Mapping[str, Any]) -> bool:
    category = _normalize_scaffold_category(entry.get("category", None))
    if category in BINDER_SCAFFOLD_ROTATABLE_CATEGORIES:
        return True
    if category in {
        "head_tail_disulfied_cyclic_peptide",
        "linear_peptide",
        "disulfied_peptide",
    }:
        return False
    return bool(entry.get("random_circular_shift", False))


def _scaffold_categories_from_cfg(cfg: Mapping[str, Any]) -> set[str] | None:
    raw_categories = cfg.get(
        "category", cfg.get("categories", cfg.get("scaffold_category", None))
    )
    if raw_categories is None:
        return None
    if isinstance(raw_categories, str):
        raw_items = [part.strip() for part in raw_categories.split(",") if part.strip()]
    elif isinstance(raw_categories, Sequence) and not isinstance(
        raw_categories, (bytes, bytearray)
    ):
        raw_items = list(raw_categories)
    else:
        raise ValueError(
            "binder_topology sample category/categories must be a string or list."
        )
    categories = {_normalize_scaffold_category(item) for item in raw_items}
    categories.discard(None)
    return set(categories)


def _normalize_scaffold_preset(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if not normalized or normalized in {"none", "null"}:
        return None
    return normalized


def _scaffold_presets_from_cfg(cfg: Mapping[str, Any]) -> set[str] | None:
    raw_presets = cfg.get(
        "preset",
        cfg.get(
            "presets",
            cfg.get(
                "entry",
                cfg.get(
                    "entries",
                    cfg.get(
                        "name",
                        cfg.get(
                            "names", cfg.get("entry_name", cfg.get("entry_names", None))
                        ),
                    ),
                ),
            ),
        ),
    )
    if raw_presets is None:
        return None
    if isinstance(raw_presets, str):
        raw_items = [part.strip() for part in raw_presets.split(",") if part.strip()]
    elif isinstance(raw_presets, Sequence) and not isinstance(
        raw_presets, (bytes, bytearray)
    ):
        raw_items = list(raw_presets)
    else:
        raise ValueError("binder_topology sample preset/name must be a string or list.")
    presets = {_normalize_scaffold_preset(item) for item in raw_items}
    presets.discard(None)
    return set(presets)


def _scaffold_entry_preset_keys(
    entry: Mapping[str, Any], *, entry_index: int
) -> set[str]:
    keys = {
        _normalize_scaffold_preset(entry.get("name", None)),
        _normalize_scaffold_preset(entry.get("preset", None)),
        _normalize_scaffold_preset(entry.get("topology_preset", None)),
        _normalize_scaffold_preset(entry.get("id", None)),
        _normalize_scaffold_preset(f"entry_{entry_index}"),
    }
    keys.discard(None)
    return set(keys)


def _scaffold_entry_preset_label(entry: Mapping[str, Any], *, entry_index: int) -> str:
    label = entry.get(
        "name",
        entry.get(
            "preset",
            entry.get("topology_preset", entry.get("id", f"entry_{entry_index}")),
        ),
    )
    return str(label)


def _load_topology_tensor(
    path: str | os.PathLike, *, field_name: str, task_name: str
) -> torch.Tensor:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Task '{task_name}': {field_name} file not found: {resolved}"
        )
    tensor = torch.load(resolved, map_location="cpu", weights_only=False)
    if not isinstance(tensor, torch.Tensor):
        raise ValueError(
            f"Task '{task_name}': {field_name} must load to a torch.Tensor, got {type(tensor).__name__}"
        )
    return tensor.long()


def _load_topology_library(
    path: str | os.PathLike, *, task_name: str
) -> list[Dict[str, Any]]:
    resolved = Path(path)
    cache_key = str(resolved.resolve()) if resolved.exists() else str(resolved)
    if cache_key in BINDER_TOPOLOGY_LIB_CACHE:
        return BINDER_TOPOLOGY_LIB_CACHE[cache_key]
    if not resolved.exists():
        raise FileNotFoundError(
            f"Task '{task_name}': binder topology library not found: {resolved}"
        )
    payload = torch.load(resolved, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        raw_entries = payload.get("entries", None)
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raw_entries = None
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"Task '{task_name}': topology library {resolved} must be a list or a mapping with an entries list."
        )
    entries: list[Dict[str, Any]] = []
    for i, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': topology library entry {i} must be a mapping."
            )
        entry = dict(raw_entry)
        if "ss_token" not in entry:
            raise ValueError(
                f"Task '{task_name}': topology library entry {i} is missing ss_token."
            )
        entry["ss_token"] = torch.as_tensor(
            entry["ss_token"], dtype=torch.long
        ).reshape(-1)
        if "block_adj" in entry and entry["block_adj"] is not None:
            entry["block_adj"] = torch.as_tensor(entry["block_adj"], dtype=torch.long)
        elif "adj" in entry and entry["adj"] is not None:
            entry["block_adj"] = torch.as_tensor(entry["adj"], dtype=torch.long)
        else:
            entry["block_adj"] = None
        entry["length"] = int(entry.get("length", entry["ss_token"].numel()))
        if "cyclic" in entry and entry["cyclic"] is not None:
            entry["cyclic"] = bool(entry["cyclic"])
        category = _normalize_scaffold_category(entry.get("category", None))
        if category is None:
            entry_cyclic = entry.get("cyclic", None)
            category = (
                "head_tail_amide_cyclic_peptide"
                if entry_cyclic is True
                else "linear_peptide"
                if entry_cyclic is False
                else None
            )
        entry["category"] = category
        if entry.get("cys_positions", None) is not None:
            entry["cys_positions"] = [int(pos) for pos in entry["cys_positions"]]
        if entry.get("cys_resids", None) is not None:
            entry["cys_resids"] = [int(pos) for pos in entry["cys_resids"]]
        raw_cys_constraints = entry.get("cys_pair_constraints", None)
        if raw_cys_constraints is not None:
            if not isinstance(raw_cys_constraints, Sequence) or isinstance(
                raw_cys_constraints, (str, bytes)
            ):
                raise ValueError(
                    f"Task '{task_name}': topology library entry {i} cys_pair_constraints must be a list."
                )
            normalized_constraints = []
            for j, constraint in enumerate(raw_cys_constraints):
                if not isinstance(constraint, Mapping):
                    raise ValueError(
                        f"Task '{task_name}': topology library entry {i} cys_pair_constraints[{j}] must be a mapping."
                    )
                if "pos_i" not in constraint or "pos_j" not in constraint:
                    raise ValueError(
                        f"Task '{task_name}': topology library entry {i} cys_pair_constraints[{j}] "
                        "must include pos_i and pos_j."
                    )
                pos_i, pos_j = sorted(
                    (int(constraint["pos_i"]), int(constraint["pos_j"]))
                )
                pair_bin_idx = constraint.get(
                    "pair_bin_idx", constraint.get("bin_idx", None)
                )
                if pair_bin_idx is None:
                    ca_distance = constraint.get(
                        "ca_ca_distance_angstrom",
                        constraint.get("target_distance_angstrom", None),
                    )
                    if ca_distance is None:
                        raise ValueError(
                            f"Task '{task_name}': topology library entry {i} cys_pair_constraints[{j}] "
                            "must include pair_bin_idx/bin_idx or ca_ca_distance_angstrom."
                        )
                    pair_bin_idx = pair_bin_idx_from_distance_angstrom(
                        float(ca_distance)
                    )
                normalized = {
                    "pos_i": pos_i,
                    "pos_j": pos_j,
                    "resid_i": int(constraint.get("resid_i", pos_i)),
                    "resid_j": int(constraint.get("resid_j", pos_j)),
                    "pair_bin_idx": int(pair_bin_idx),
                }
                if (
                    "ca_ca_distance_angstrom" in constraint
                    and constraint["ca_ca_distance_angstrom"] is not None
                ):
                    normalized["ca_ca_distance_angstrom"] = float(
                        constraint["ca_ca_distance_angstrom"]
                    )
                if (
                    "sg_sg_distance_angstrom" in constraint
                    and constraint["sg_sg_distance_angstrom"] is not None
                ):
                    normalized["sg_sg_distance_angstrom"] = float(
                        constraint["sg_sg_distance_angstrom"]
                    )
                normalized_constraints.append(normalized)
            entry["cys_pair_constraints"] = normalized_constraints
            if "cys_positions" not in entry or entry["cys_positions"] is None:
                entry["cys_positions"] = sorted(
                    {
                        pos
                        for constraint in normalized_constraints
                        for pos in (constraint["pos_i"], constraint["pos_j"])
                    }
                )
            if "cys_resids" not in entry or entry["cys_resids"] is None:
                entry["cys_resids"] = sorted(
                    {
                        pos
                        for constraint in normalized_constraints
                        for pos in (constraint["resid_i"], constraint["resid_j"])
                    }
                )
        entries.append(entry)
    if not entries:
        raise ValueError(
            f"Task '{task_name}': topology library {resolved} contains no entries."
        )
    BINDER_TOPOLOGY_LIB_CACHE[cache_key] = entries
    return entries


def _task_sampler_implies_cyclic(task: Mapping[str, Any]) -> bool:
    sampler = task.get("sampler", task.get("condition_sampler", None))
    if sampler is not None:
        if isinstance(sampler, str):
            mode = sampler
        elif isinstance(sampler, Mapping):
            mode = sampler.get("mode", sampler.get("type", ""))
        else:
            mode = ""
        if _normalize_sampler_mode_name(mode) == "head_to_tail":
            return True

    topology = task.get("topology", []) or []
    if isinstance(topology, Mapping):
        topology = [topology]
    if isinstance(topology, Sequence) and not isinstance(topology, (str, bytes)):
        for entry in topology:
            if not isinstance(entry, Mapping):
                continue
            entry_type = (
                str(entry.get("type", entry.get("mode", "")) or "").strip().lower()
            )
            if entry_type in {"head_to_tail", "head-tail", "headtail", "cyclic"}:
                return True
    return False


def _resolve_auto_topology_cyclic(
    cfg: Mapping[str, Any],
    task: Mapping[str, Any],
) -> bool | None:
    value = cfg.get("cyclic", cfg.get("binder_is_cyclic", "auto"))
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"auto", ""}:
            return bool(_task_sampler_implies_cyclic(task))
        if text in {"none", "null", "any", "all"}:
            return None
        if text in {
            "true",
            "yes",
            "1",
            "cyclic",
            "head_to_tail",
            "head-tail",
            "headtail",
        }:
            return True
        if text in {"false", "no", "0", "linear"}:
            return False
        raise ValueError(f"Unsupported binder_topology cyclic value {value!r}.")
    return bool(value)


def _rotate_binder_position(pos: int, *, total_len: int, shift: int) -> int:
    return ((int(pos) - 1 + int(shift)) % int(total_len)) + 1


def _rotate_scaffold_constraint(
    constraint: Mapping[str, Any],
    *,
    total_len: int,
    shift: int,
) -> Dict[str, Any]:
    pos_i = _rotate_binder_position(
        int(constraint["pos_i"]), total_len=total_len, shift=shift
    )
    pos_j = _rotate_binder_position(
        int(constraint["pos_j"]), total_len=total_len, shift=shift
    )
    if pos_i > pos_j:
        pos_i, pos_j = pos_j, pos_i
    rotated = dict(constraint)
    rotated["pos_i"] = pos_i
    rotated["pos_j"] = pos_j
    # After a circular shift the old PDB residue ids are no longer meaningful as
    # binder-local identifiers; keep them consistent with the shifted positions.
    rotated["resid_i"] = pos_i
    rotated["resid_j"] = pos_j
    return rotated


def _rotate_sampled_scaffold_entry(
    entry: Mapping[str, Any],
    *,
    total_len: int,
    shift: int,
) -> Dict[str, Any]:
    if int(shift) % int(total_len) == 0:
        return dict(entry)
    rotated = dict(entry)
    rotated["ss_token"] = torch.roll(
        torch.as_tensor(entry["ss_token"], dtype=torch.long), shifts=int(shift), dims=0
    )
    block_adj = entry.get("block_adj", None)
    if block_adj is not None:
        adj_tensor = torch.as_tensor(block_adj, dtype=torch.long)
        rotated["block_adj"] = torch.roll(
            torch.roll(adj_tensor, shifts=int(shift), dims=0), shifts=int(shift), dims=1
        )
    if entry.get("cys_positions", None) is not None:
        rotated_positions = [
            _rotate_binder_position(pos, total_len=total_len, shift=shift)
            for pos in entry["cys_positions"]
        ]
        rotated["cys_positions"] = sorted(set(rotated_positions))
        rotated["cys_resids"] = list(rotated["cys_positions"])
    if entry.get("cys_pair_constraints", None) is not None:
        rotated["cys_pair_constraints"] = [
            _rotate_scaffold_constraint(constraint, total_len=total_len, shift=shift)
            for constraint in entry["cys_pair_constraints"]
        ]
        if "cys_positions" not in rotated or rotated["cys_positions"] is None:
            rotated["cys_positions"] = sorted(
                {
                    pos
                    for constraint in rotated["cys_pair_constraints"]
                    for pos in (int(constraint["pos_i"]), int(constraint["pos_j"]))
                }
            )
            rotated["cys_resids"] = list(rotated["cys_positions"])
    return rotated


def _topology_class_tokens(
    value: Any, *, total_len: int, task_name: str
) -> Dict[str, Any]:
    name = str(value).strip().lower()
    if name in {"", "none", "null"}:
        return {}
    if name not in BINDER_TOPOLOGY_SS_BY_NAME:
        raise ValueError(
            f"Task '{task_name}': unsupported binder_topology class {value!r}. "
            "Supported shorthand classes: alpha, beta, coil."
        )
    ss_token = torch.full(
        (int(total_len),), fill_value=BINDER_TOPOLOGY_SS_BY_NAME[name], dtype=torch.long
    )
    return _normalize_topology_tokens(
        ss_token,
        None,
        total_len=total_len,
        task_name=task_name,
        source=f"ss_class:{name}",
    )


def _normalize_topology_tokens(
    ss_token: torch.Tensor,
    block_adj: torch.Tensor | None,
    *,
    total_len: int,
    task_name: str,
    source: str,
) -> Dict[str, Any]:
    ss_token = ss_token.reshape(-1).long()
    if block_adj is None:
        block_adj = torch.full(
            (int(total_len), int(total_len)),
            fill_value=BINDER_TOPOLOGY_ADJ_MASK,
            dtype=torch.long,
        )
    else:
        block_adj = block_adj.long()
    if int(ss_token.numel()) != int(total_len):
        raise ValueError(
            f"Task '{task_name}': binder topology {source} has ss length {ss_token.numel()}, "
            f"expected binder length {total_len}."
        )
    if block_adj.shape != (total_len, total_len):
        raise ValueError(
            f"Task '{task_name}': binder topology {source} has adj shape {tuple(block_adj.shape)}, "
            f"expected ({total_len}, {total_len})."
        )
    if torch.any((ss_token < 0) | (ss_token > BINDER_TOPOLOGY_SS_MASK)):
        raise ValueError(
            f"Task '{task_name}': binder topology {source} has ss tokens outside [0, 3]."
        )
    if torch.any((block_adj < 0) | (block_adj > BINDER_TOPOLOGY_ADJ_MASK)):
        raise ValueError(
            f"Task '{task_name}': binder topology {source} has adjacency tokens outside [0, 2]."
        )
    return {
        "binder_ss_token": ss_token.clone(),
        "binder_block_adj": block_adj.clone(),
        "binder_topology_source": source,
    }


def _scaffold_ids_from_list(scaffold_list: Any, *, task_name: str) -> List[str]:
    if scaffold_list is None:
        return []
    if isinstance(scaffold_list, (list, tuple)):
        return [str(item).strip() for item in scaffold_list if str(item).strip()]
    list_path = Path(str(scaffold_list))
    if not list_path.exists():
        raise FileNotFoundError(
            f"Task '{task_name}': scaffold_list file not found: {list_path}"
        )
    return [line.strip() for line in list_path.read_text().splitlines() if line.strip()]


def _resolve_scaffold_dir_topology(
    cfg: Mapping[str, Any],
    *,
    total_len: int,
    task_name: str,
    sample_index: int,
) -> Dict[str, Any]:
    scaffold_dir = cfg.get("scaffold_dir", None)
    if scaffold_dir is None:
        raise ValueError(
            f"Task '{task_name}': binder_topology mode=scaffold_dir requires scaffold_dir."
        )
    scaffold_dir = Path(str(scaffold_dir))
    if not scaffold_dir.exists():
        raise FileNotFoundError(
            f"Task '{task_name}': scaffold_dir not found: {scaffold_dir}"
        )

    requested_ids = _scaffold_ids_from_list(
        cfg.get("scaffold_list", None), task_name=task_name
    )
    if requested_ids:
        ss_paths = []
        for item in requested_ids:
            stem = Path(item).name
            if stem.endswith("_ss.pt"):
                stem = stem[:-6]
            ss_paths.append(scaffold_dir / f"{stem}_ss.pt")
    else:
        ss_paths = sorted(scaffold_dir.glob("*_ss.pt"))

    candidates = []
    skipped_lengths = defaultdict(int)
    for ss_path in ss_paths:
        if not ss_path.exists():
            raise FileNotFoundError(
                f"Task '{task_name}': scaffold ss file not found: {ss_path}"
            )
        stem = ss_path.name[:-6]
        adj_path = scaffold_dir / f"{stem}_adj.pt"
        if not adj_path.exists():
            raise FileNotFoundError(
                f"Task '{task_name}': scaffold adj file not found: {adj_path}"
            )
        ss_token = _load_topology_tensor(
            ss_path, field_name="scaffold ss", task_name=task_name
        ).reshape(-1)
        if int(ss_token.numel()) == int(total_len):
            candidates.append((stem, ss_path, adj_path, ss_token))
        else:
            skipped_lengths[int(ss_token.numel())] += 1

    if not candidates:
        observed = ", ".join(
            f"{length}:{count}"
            for length, count in sorted(skipped_lengths.items())[:20]
        )
        raise ValueError(
            f"Task '{task_name}': no scaffold in {scaffold_dir} matches binder length {total_len}. "
            f"Observed scaffold lengths/counts: {observed or 'none'}."
        )

    systematic = bool(cfg.get("systematic", False))
    if systematic:
        stem, ss_path, adj_path, ss_token = candidates[
            int(sample_index) % len(candidates)
        ]
    else:
        stem, ss_path, adj_path, ss_token = random.choice(candidates)
    block_adj = _load_topology_tensor(
        adj_path, field_name="scaffold adj", task_name=task_name
    )
    return _normalize_topology_tokens(
        ss_token,
        block_adj,
        total_len=total_len,
        task_name=task_name,
        source=f"scaffold_dir:{scaffold_dir}/{stem}",
    )


def _scaffold_condition_payload(
    entry: Mapping[str, Any],
    *,
    total_len: int,
    task_name: str,
    source: str,
) -> Dict[str, Any]:
    category = _normalize_scaffold_category(entry.get("category", None))
    pair_constraints = list(entry.get("cys_pair_constraints", []) or [])
    is_disulfide = category in BINDER_SCAFFOLD_DISULFIDE_CATEGORIES or bool(
        pair_constraints
    )
    uses_amide_cyclization = _topology_entry_uses_amide_cyclization(entry)
    known_categories = {
        "linear_peptide",
        *BINDER_SCAFFOLD_CYCLIC_CATEGORIES,
        *BINDER_SCAFFOLD_DISULFIDE_CATEGORIES,
    }
    if (
        category not in known_categories
        and not is_disulfide
        and not uses_amide_cyclization
    ):
        raise ValueError(
            f"Task '{task_name}': sampled binder topology category {category!r} "
            "does not declare classifiable linear, head-to-tail, or disulfide chemistry. "
            "Add an explicit scaffold chemistry adapter before using this category."
        )
    payload: Dict[str, Any] = {
        "source": source,
        "category": category,
        "cys_positions": [],
        "cys_pair_constraints": [],
        "residue_type_constraints": {},
        "optional_ca_pair_constraints": [],
        "cyclization": None,
        "topology_entries": [],
    }

    if is_disulfide:
        if not pair_constraints:
            raise ValueError(
                f"Task '{task_name}': disulfide scaffold {source} must provide cys_pair_constraints."
            )
        aa_c = _normalize_residue_type_constraint_aa("C")
        seen_cys_positions: set[int] = set()
        for idx, constraint in enumerate(pair_constraints):
            pos_i = _parse_binder_position(
                constraint["pos_i"],
                total_len=total_len,
                field_name=f"binder_topology.cys_pair_constraints[{idx}].pos_i",
                task_name=task_name,
            )
            pos_j = _parse_binder_position(
                constraint["pos_j"],
                total_len=total_len,
                field_name=f"binder_topology.cys_pair_constraints[{idx}].pos_j",
                task_name=task_name,
            )
            if pos_i == pos_j:
                raise ValueError(
                    f"Task '{task_name}': disulfide scaffold {source} has identical CYS anchors."
                )
            pos_i, pos_j = sorted((pos_i, pos_j))
            pair_bin_idx = int(
                constraint.get("pair_bin_idx", constraint.get("bin_idx", -1))
            )
            if pair_bin_idx < 0 or pair_bin_idx >= 30:
                raise ValueError(
                    f"Task '{task_name}': disulfide scaffold {source} pair_bin_idx {pair_bin_idx} "
                    "is outside supported range [0, 29]."
                )
            target_distance = float(
                constraint.get(
                    "ca_ca_distance_angstrom",
                    constraint.get(
                        "target_distance_angstrom", DISULFIDE_CA_PAIR_TARGET_ANGSTROM
                    ),
                )
            )
            for pos in (pos_i, pos_j):
                prev = payload["residue_type_constraints"].get(pos, None)
                if prev is not None and int(prev) != aa_c:
                    raise ValueError(
                        f"Task '{task_name}': disulfide scaffold {source} has conflicting residue target at {pos}."
                    )
                payload["residue_type_constraints"][pos] = aa_c
                seen_cys_positions.add(pos)
            normalized_constraint = {
                "pos_i": pos_i,
                "pos_j": pos_j,
                "resid_i": int(constraint.get("resid_i", pos_i)),
                "resid_j": int(constraint.get("resid_j", pos_j)),
                "pair_bin_idx": pair_bin_idx,
                "ca_ca_distance_angstrom": target_distance,
            }
            if (
                "sg_sg_distance_angstrom" in constraint
                and constraint["sg_sg_distance_angstrom"] is not None
            ):
                normalized_constraint["sg_sg_distance_angstrom"] = float(
                    constraint["sg_sg_distance_angstrom"]
                )
            payload["cys_pair_constraints"].append(normalized_constraint)
            payload["optional_ca_pair_constraints"].append(
                {
                    "pos_i": pos_i,
                    "pos_j": pos_j,
                    "bin_idx": pair_bin_idx,
                    "target_distance_angstrom": target_distance,
                }
            )
            payload["topology_entries"].append(
                {"type": "disulfide", "anchors": [pos_i, pos_j]}
            )
        payload["cys_positions"] = sorted(seen_cys_positions)

    if uses_amide_cyclization:
        payload["cyclization"] = {
            "mode": "pair",
            "positions": [1, int(total_len)],
            "shortcut_len": 1,
        }
        payload["topology_entries"].append(
            {"type": "head_to_tail", "anchors": [1, int(total_len)]}
        )

    return payload


def _sampled_topology_candidates(
    task: Dict[str, Any],
    cfg: Mapping[str, Any],
    *,
    total_len: int | set[int] | None,
    task_name: str,
) -> tuple[str, list[tuple[int, Dict[str, Any]]]]:
    """Return every library entry that runtime sampling may select.

    ``total_len`` is one concrete runtime length, a complete preflight set of
    reachable lengths, or ``None`` when no exact set can be known.  Preflight
    never makes a random choice before datamodule or model construction.
    """
    lib_path = cfg.get("lib_path", cfg.get("path", cfg.get("library_path", None)))
    if lib_path is None:
        raise ValueError(
            f"Task '{task_name}': binder_topology mode=sample requires lib_path."
        )
    entries = _load_topology_library(lib_path, task_name=task_name)
    has_explicit_cyclic_filter = "cyclic" in cfg or "binder_is_cyclic" in cfg
    cyclic_filter = (
        _resolve_auto_topology_cyclic(cfg, task) if has_explicit_cyclic_filter else None
    )
    category_filter = _scaffold_categories_from_cfg(cfg)
    preset_filter = _scaffold_presets_from_cfg(cfg)
    min_score = cfg.get("min_score", None)
    if min_score is not None:
        min_score = float(min_score)
    if total_len is None:
        allowed_lengths: set[int] | None = None
    elif isinstance(total_len, set):
        allowed_lengths = {int(length) for length in total_len}
        if not allowed_lengths:
            raise ValueError(
                f"Task '{task_name}': sampled topology length set is empty."
            )
    else:
        allowed_lengths = {int(total_len)}
    candidates = []
    skipped_lengths = defaultdict(int)
    skipped_cyclic = defaultdict(int)
    skipped_categories = defaultdict(int)
    skipped_presets = defaultdict(int)
    skipped_score = 0
    for entry_index, entry in enumerate(entries):
        entry_len = int(entry["length"])
        if allowed_lengths is not None and entry_len not in allowed_lengths:
            skipped_lengths[entry_len] += 1
            continue
        entry_category = _normalize_scaffold_category(entry.get("category", None))
        if category_filter is not None and entry_category not in category_filter:
            skipped_categories[str(entry_category)] += 1
            continue
        entry_cyclic = _topology_entry_cyclic(entry)
        if (
            cyclic_filter is not None
            and entry_cyclic is not None
            and bool(entry_cyclic) != bool(cyclic_filter)
        ):
            skipped_cyclic[bool(entry_cyclic)] += 1
            continue
        if preset_filter is not None and not (
            _scaffold_entry_preset_keys(entry, entry_index=entry_index) & preset_filter
        ):
            skipped_presets[
                _scaffold_entry_preset_label(entry, entry_index=entry_index)
            ] += 1
            continue
        if (
            min_score is not None
            and entry.get("score", None) is not None
            and float(entry["score"]) < min_score
        ):
            skipped_score += 1
            continue
        candidates.append((entry_index, entry))
    if not candidates:
        observed = ", ".join(
            f"{length}:{count}"
            for length, count in sorted(skipped_lengths.items())[:20]
        )
        cyclic_msg = ", ".join(
            f"{key}:{count}" for key, count in sorted(skipped_cyclic.items())
        )
        category_msg = ", ".join(
            f"{key}:{count}" for key, count in sorted(skipped_categories.items())
        )
        preset_msg = ", ".join(
            f"{key}:{count}" for key, count in sorted(skipped_presets.items())
        )
        requested_presets = ",".join(sorted(preset_filter)) if preset_filter else "any"
        length_requirement = (
            ",".join(str(length) for length in sorted(allowed_lengths))
            if allowed_lengths is not None
            else "any"
        )
        message = (
            f"Task '{task_name}': no sampled topology in {lib_path} matches binder length {length_requirement}, "
            f"cyclic={cyclic_filter}, and preset={requested_presets}. Observed lengths/counts: {observed or 'none'}; "
            f"cyclic mismatches: {cyclic_msg or 'none'}; "
            f"category mismatches: {category_msg or 'none'}; "
            f"preset/name mismatches: {preset_msg or 'none'}; score-filtered: {skipped_score}."
        )
        raise ValueError(message)
    return str(lib_path), candidates


def _resolve_sampled_topology(
    task: Dict[str, Any],
    cfg: Mapping[str, Any],
    *,
    total_len: int,
    task_name: str,
    sample_index: int,
) -> Dict[str, Any]:
    lib_path, candidates = _sampled_topology_candidates(
        task,
        cfg,
        total_len=int(total_len),
        task_name=task_name,
    )
    systematic = bool(cfg.get("systematic", False))
    if systematic:
        entry_index, entry = candidates[int(sample_index) % len(candidates)]
    else:
        entry_index, entry = random.choice(candidates)
    circular_shift = 0
    if _topology_entry_allows_circular_shift(entry) and int(total_len) > 1:
        circular_shift = (
            int(sample_index) % int(total_len)
            if systematic
            else random.randrange(int(total_len))
        )
        entry = _rotate_sampled_scaffold_entry(
            entry, total_len=total_len, shift=circular_shift
        )
    source_id = str(entry.get("id", entry.get("name", f"entry_{entry_index}")))
    source = f"sample:{lib_path}:{source_id}"
    if circular_shift:
        source = f"{source}:circular_shift_{circular_shift}"
    resolved = _normalize_topology_tokens(
        entry["ss_token"],
        entry.get("block_adj", None),
        total_len=total_len,
        task_name=task_name,
        source=source,
    )
    scaffold_payload = _scaffold_condition_payload(
        entry,
        total_len=total_len,
        task_name=task_name,
        source=source,
    )
    resolved["_scaffold_condition"] = scaffold_payload
    resolved["binder_topology_category"] = scaffold_payload.get("category")
    resolved["binder_topology_entry_id"] = source_id
    resolved["binder_topology_preset"] = str(entry.get("name", source_id))
    resolved["binder_topology_circular_shift"] = int(circular_shift)
    return resolved


def resolve_binder_topology(
    task: Dict[str, Any],
    *,
    total_len: int,
    task_name: str,
    sample_index: int,
) -> Dict[str, Any]:
    cfg = task.get("binder_topology", None)
    if cfg is None:
        return {}
    if isinstance(cfg, str):
        return _topology_class_tokens(cfg, total_len=total_len, task_name=task_name)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Task '{task_name}': binder_topology must be a mapping.")

    mode = str(cfg.get("mode", "") or "").strip().lower()
    if mode in {"", "null", "none"}:
        return {}
    if mode in BINDER_TOPOLOGY_SS_BY_NAME:
        return _topology_class_tokens(mode, total_len=total_len, task_name=task_name)
    ss_class = cfg.get("ss_class", cfg.get("secondary_class", None))
    if ss_class is not None:
        return _topology_class_tokens(
            ss_class, total_len=total_len, task_name=task_name
        )
    if mode in {"scaffold_pt", "pt", "paths"}:
        ss_path = cfg.get("ss_path", None)
        adj_path = cfg.get("adj_path", None)
        if ss_path is None or adj_path is None:
            raise ValueError(
                f"Task '{task_name}': binder_topology mode=scaffold_pt requires ss_path and adj_path."
            )
        return _normalize_topology_tokens(
            _load_topology_tensor(
                ss_path, field_name="binder_topology.ss_path", task_name=task_name
            ),
            _load_topology_tensor(
                adj_path, field_name="binder_topology.adj_path", task_name=task_name
            ),
            total_len=total_len,
            task_name=task_name,
            source=f"scaffold_pt:{ss_path}",
        )
    if mode in {"ss_token", "ss_tokens", "tokens", "ss"}:
        ss_value = cfg.get("ss_token", cfg.get("ss_tokens", None))
        ss_path = cfg.get("ss_path", None)
        if ss_value is None and ss_path is None:
            raise ValueError(
                f"Task '{task_name}': binder_topology mode=ss_token requires ss_token/ss_tokens or ss_path."
            )
        if ss_path is not None:
            ss_token = _load_topology_tensor(
                ss_path, field_name="binder_topology.ss_path", task_name=task_name
            )
            source = f"ss_token:{ss_path}"
        else:
            ss_token = torch.as_tensor(ss_value, dtype=torch.long)
            source = "ss_token:inline"

        adj_value = cfg.get("block_adj", cfg.get("adj", None))
        adj_path = cfg.get("adj_path", None)
        if adj_path is not None:
            block_adj = _load_topology_tensor(
                adj_path, field_name="binder_topology.adj_path", task_name=task_name
            )
            source = f"{source}+adj:{adj_path}"
        elif adj_value is not None:
            block_adj = torch.as_tensor(adj_value, dtype=torch.long)
            source = f"{source}+adj:inline"
        else:
            block_adj = None
        return _normalize_topology_tokens(
            ss_token,
            block_adj,
            total_len=total_len,
            task_name=task_name,
            source=source,
        )
    if mode in {"sample", "sample_lib", "library", "topology_lib"}:
        return _resolve_sampled_topology(
            task,
            cfg,
            total_len=total_len,
            task_name=task_name,
            sample_index=sample_index,
        )
    if mode == "scaffold_dir":
        return _resolve_scaffold_dir_topology(
            cfg,
            total_len=total_len,
            task_name=task_name,
            sample_index=sample_index,
        )
    raise ValueError(
        f"Task '{task_name}': unsupported binder_topology mode {mode!r}. "
        "Supported modes: scaffold_pt, scaffold_dir, ss_token, sample."
    )


def normalize_omit_aa(value: Any, *, task_name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_items = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw_items = list(value)
    else:
        raise ValueError(
            f"Task '{task_name}': omit_aa must be null, a comma-separated string, "
            "or a list of residue names."
        )

    aa_indices = []
    for raw_item in raw_items:
        token = str(raw_item).strip().upper()
        if not token:
            continue
        if token in restype_order:
            aa_idx = restype_order[token]
        elif token in restype_3to1:
            aa_idx = restype_order[restype_3to1[token]]
        else:
            raise ValueError(
                f"Task '{task_name}': unknown omit_aa residue '{raw_item}'. "
                "Use one-letter or three-letter standard amino acid names."
            )
        if aa_idx not in aa_indices:
            aa_indices.append(aa_idx)
    return aa_indices


def _normalize_binder_secondary_class_value(
    value: Any,
    *,
    task_name: str,
    field_name: str,
) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized not in SECSTRUCT_ALIASES:
            raise ValueError(
                f"Task '{task_name}': unsupported {field_name} '{value}'. "
                f"Supported classes: {', '.join(SUPPORTED_INFERENCE_SECSTRUCT_NAMES)}"
            )
        return int(SECSTRUCT_ALIASES[normalized])
    try:
        class_id = int(value)
    except Exception as exc:
        raise ValueError(
            f"Task '{task_name}': {field_name} must be int/str, got {value!r}"
        ) from exc
    if class_id not in VALID_INFERENCE_SECSTRUCT_IDS:
        raise ValueError(
            f"Task '{task_name}': unsupported {field_name} id {class_id}. "
            f"Supported classes: {', '.join(SUPPORTED_INFERENCE_SECSTRUCT_NAMES)}"
        )
    return class_id


def create_std_ala_residue(chain_id, res_id, ca_coord=None):
    atom_names = ["N", "CA", "C", "O", "CB"]
    elements = ["N", "C", "C", "O", "C"]
    n_atoms = len(atom_names)

    ala = struc.AtomArray(n_atoms)
    ala.chain_id[:] = chain_id
    ala.res_id[:] = res_id
    ala.res_name[:] = "ALA"
    ala.atom_name[:] = atom_names
    ala.element[:] = elements
    ala.coord[:] = 0.0
    if ca_coord is not None:
        ala.coord[1] = ca_coord  # index 1 is CA
    ala.hetero[:] = False

    ala.set_annotation("is_motif", np.array([False] * n_atoms))
    ala.set_annotation("is_hotspot", np.array([False] * n_atoms))
    return ala


def create_proxy_sequence_residue(chain_id: str, res_id: int, aa: str, ca_coord=None):
    aa = str(aa).strip().upper()
    if aa not in restype_1to3:
        raise ValueError(f"Unsupported proxy residue one-letter code: {aa!r}")
    resname = restype_1to3[aa]
    keep_atoms = PROXY_KEEP_ATOMS.get(resname, {"N", "CA", "C", "O", "CB"})
    atom_names = [name for name in ("N", "CA", "C", "O", "CB") if name in keep_atoms]
    elements = [name[0] for name in atom_names]
    n_atoms = len(atom_names)

    residue = struc.AtomArray(n_atoms)
    residue.chain_id[:] = chain_id
    residue.res_id[:] = res_id
    residue.res_name[:] = resname
    residue.atom_name[:] = atom_names
    residue.element[:] = elements
    residue.coord[:] = 0.0
    if ca_coord is not None and "CA" in atom_names:
        residue.coord[atom_names.index("CA")] = ca_coord
    residue.hetero[:] = False
    residue.set_annotation("is_motif", np.array([False] * n_atoms))
    residue.set_annotation("is_hotspot", np.array([False] * n_atoms))
    return residue


PROXY_KEEP_ATOMS = {
    "ALA": {"N", "CA", "C", "O", "CB"},
    "ARG": {"N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"},
    "ASN": {"N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"},
    "ASP": {"N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"},
    "CYS": {"N", "CA", "C", "O", "CB", "SG"},
    "GLN": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"},
    "GLU": {"N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"},
    "GLY": {"N", "CA", "C", "O"},
    "HIS": {"N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
    "ILE": {"N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"},
    "LEU": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"},
    "LYS": {"N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"},
    "MET": {"N", "CA", "C", "O", "CB", "CG", "SD", "CE"},
    "PHE": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "PRO": {"N", "CA", "C", "O", "CB", "CG", "CD"},
    "SER": {"N", "CA", "C", "O", "CB", "OG"},
    "THR": {"N", "CA", "C", "O", "CB", "OG1", "CG2"},
    "TRP": {
        "N",
        "CA",
        "C",
        "O",
        "CB",
        "CG",
        "CD1",
        "CD2",
        "NE1",
        "CE2",
        "CE3",
        "CZ2",
        "CZ3",
        "CH2",
    },
    "TYR": {"N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"},
    "VAL": {"N", "CA", "C", "O", "CB", "CG1", "CG2"},
}


def _proxy_nonstandard_reference_residue(residue: struc.AtomArray) -> struc.AtomArray:
    if len(residue) == 0:
        return residue
    resname = str(residue.res_name[0]).strip().upper()
    spec = get_nonstandard_residue_spec(resname)
    if spec is None and resname != "RTY":
        return residue
    proxy_aa = "A" if resname == "RTY" else spec.proxy_aa
    proxy_resname = restype_1to3.get(proxy_aa, proxy_aa)
    keep_atoms = PROXY_KEEP_ATOMS.get(proxy_resname, {"N", "CA", "C", "O", "CB"})
    atom_mask = np.array(
        [str(atom_name).strip() in keep_atoms for atom_name in residue.atom_name],
        dtype=bool,
    )
    proxied = residue[atom_mask].copy()
    proxied.res_name[:] = proxy_resname
    proxied.hetero[:] = False
    return proxied


_BACKBONE_ATOMS = ("N", "CA", "C", "O")


_BACKBONE_ATOM_SET = set(_BACKBONE_ATOMS)


def nonstandard_positions_from_topology(topology_entries: Any) -> Dict[int, str]:
    """Map 1-indexed binder position -> 3-letter resname for non-standard residues
    that the topology stage will install at known positions.

    Entries covered:
      mk8_staple / nle_staple anchors -> MK8 / NLE
      staple_rty middle anchor (anchors[1]) -> RTY
      residue_template resid -> resname
    """
    if topology_entries is None:
        return {}
    if isinstance(topology_entries, str):
        try:
            topology_entries = json.loads(topology_entries)
        except (json.JSONDecodeError, ValueError):
            return {}
    if not isinstance(topology_entries, (list, tuple)):
        return {}

    out: Dict[int, str] = {}
    for entry in topology_entries:
        if not isinstance(entry, Mapping):
            continue
        topo_type = str(entry.get("type", "")).strip().lower()
        anchors = entry.get("anchors") or []
        if topo_type in {"mk8_staple", "nle_staple"}:
            resname = "MK8" if topo_type == "mk8_staple" else "NLE"
            for a in anchors:
                try:
                    out[int(a)] = resname
                except (TypeError, ValueError):
                    continue
        elif topo_type == "staple_rty":
            if len(anchors) >= 2:
                try:
                    out[int(anchors[1])] = "RTY"
                except (TypeError, ValueError):
                    continue
        elif topo_type == "residue_template":
            resid = entry.get("resid")
            if resid is None and anchors:
                resid = anchors[0]
            resname = str(entry.get("resname", "")).strip().upper()
            if resid is None or not resname:
                continue
            try:
                out[int(resid)] = resname
            except (TypeError, ValueError):
                continue
    return out


def format_sequence_with_nonstandard(sequence: str, positions: Dict[int, str]) -> str:
    """Insert [RESNAME] annotations into a 1-letter sequence at non-standard positions.

    Positions are 1-indexed. The proxy 1-letter at each annotated position is
    replaced with the bracketed 3-letter resname, e.g. ``NAPIV`` + ``{2: "RTY"}``
    -> ``N[RTY]PIV``.
    """
    if not positions:
        return sequence
    parts: List[str] = []
    for i, aa in enumerate(sequence, start=1):
        if i in positions:
            parts.append(f"[{positions[i]}]")
        else:
            parts.append(aa)
    return "".join(parts)


def parse_sequence_with_nonstandard(
    sequence: Any, *, task_name: str
) -> tuple[str, List[Dict[str, Any]]]:
    """Parse sequence strings like ``AC[MK8]D`` into proxy one-letter sequence.

    Bracketed residues use the registered non-standard residue proxy amino acid
    for model conditioning and are returned as residue_template topology entries
    so OpenMM postprocess can reinstall the requested chemistry.
    """
    if sequence is None:
        return "", []
    text = str(sequence).strip()
    if not text:
        raise ValueError(f"Task '{task_name}': sequence must not be empty.")

    proxy_parts: List[str] = []
    templates: List[Dict[str, Any]] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isspace() or ch in {"-", "_"}:
            i += 1
            continue
        if ch == "[":
            end = text.find("]", i + 1)
            if end < 0:
                raise ValueError(
                    f"Task '{task_name}': unterminated bracketed residue in sequence {text!r}."
                )
            resname = text[i + 1 : end].strip().upper()
            if not resname:
                raise ValueError(
                    f"Task '{task_name}': empty bracketed residue in sequence {text!r}."
                )
            if len(resname) == 1 and resname in restype_order:
                proxy_aa = resname
            else:
                spec = get_nonstandard_residue_spec(resname)
                if spec is None:
                    if resname in restype_3to1:
                        proxy_aa = restype_3to1[resname]
                    else:
                        raise ValueError(
                            f"Task '{task_name}': unsupported bracketed residue [{resname}] in sequence. "
                            "Register it in nonstandard_residue_registry.py first."
                        )
                else:
                    if spec.resname in UNSUPPORTED_NONSTANDARD_RESIDUES:
                        raise ValueError(
                            f"Task '{task_name}': non-standard residue '{spec.resname}' in sequence is not "
                            f"supported by the relax pipeline. Unsupported residues: "
                            f"{sorted(UNSUPPORTED_NONSTANDARD_RESIDUES)}."
                        )
                    proxy_aa = spec.proxy_aa
                    templates.append(
                        {
                            "type": "residue_template",
                            "resid": len(proxy_parts) + 1,
                            "resname": spec.resname,
                            "anchors": [len(proxy_parts) + 1],
                        }
                    )
            proxy_parts.append(proxy_aa)
            i = end + 1
            continue
        aa = ch.upper()
        if aa not in restype_order:
            raise ValueError(
                f"Task '{task_name}': unsupported residue '{ch}' in sequence {text!r}. "
                "Use one-letter standard amino acids or bracketed registered 3-letter residues."
            )
        proxy_parts.append(aa)
        i += 1

    if not proxy_parts:
        raise ValueError(f"Task '{task_name}': sequence did not contain any residues.")
    return "".join(proxy_parts), templates


_STAPLE_TARGET_DISTANCE_BY_OFFSET: Dict[int, float] = {
    3: 5.0,
    4: 6.2,
    7: 10.5,
}


def _staple_target_distance_angstrom(pos_i: int, pos_j: int) -> float:
    offset = abs(int(pos_j) - int(pos_i))
    return _STAPLE_TARGET_DISTANCE_BY_OFFSET.get(offset, 6.2)


def _normalize_loaded_structure(pdb_struc: struc.AtomArray) -> struc.AtomArray:
    """Sanitize a freshly loaded AtomArray for inference.

    Two concerns are addressed:

    1. Non-standard residues are often emitted as HETATM, but as long as they
       carry the four backbone atoms (N, CA, C, O) we want to keep them and
       treat them as polymer residues (their sidechain is later proxied to a
       standard amino acid via ``_proxy_nonstandard_reference_residue``).
       Residues missing any of the backbone atoms are dropped (ligands,
       waters, ions, …).

    2. PDB insertion codes collide with the integer-only residue id space the
       rest of the pipeline uses (e.g. graph keys built from
       ``(chain_id, res_id)``). We flatten insertion codes by renumbering each
       chain's surviving residues sequentially starting at 1 in the original
       order, and clear the ``ins_code`` annotation.
    """

    if len(pdb_struc) == 0:
        return pdb_struc

    annotation_categories = pdb_struc.get_annotation_categories()
    has_ins_code = "ins_code" in annotation_categories
    ins_codes = pdb_struc.ins_code if has_ins_code else np.array([""] * len(pdb_struc))

    chain_ids = pdb_struc.chain_id
    res_ids = pdb_struc.res_id
    atom_names = pdb_struc.atom_name

    keep_mask = np.zeros(len(pdb_struc), dtype=bool)
    new_res_ids = np.zeros(len(pdb_struc), dtype=res_ids.dtype)

    # Walk the array in original order and group atoms by
    # (chain_id, res_id, ins_code). Renumber sequentially per chain for the
    # residues that survive the backbone check.
    n_atoms = len(pdb_struc)
    cursor = 0
    per_chain_counter: Dict[str, int] = {}
    while cursor < n_atoms:
        chain = chain_ids[cursor]
        rid = res_ids[cursor]
        icode = ins_codes[cursor]
        end = cursor + 1
        while (
            end < n_atoms
            and chain_ids[end] == chain
            and res_ids[end] == rid
            and ins_codes[end] == icode
        ):
            end += 1

        residue_atom_names = {str(a).strip() for a in atom_names[cursor:end]}
        if all(name in residue_atom_names for name in _BACKBONE_ATOMS):
            chain_key = str(chain)
            next_id = per_chain_counter.get(chain_key, 0) + 1
            per_chain_counter[chain_key] = next_id
            keep_mask[cursor:end] = True
            new_res_ids[cursor:end] = next_id

        cursor = end

    if not keep_mask.any():
        raise ValueError(
            "No residues with full N/CA/C/O backbone survived structure normalization."
        )

    sanitized = pdb_struc[keep_mask].copy()
    sanitized.res_id = new_res_ids[keep_mask]
    # Non-standard but well-formed polymer residues come in as HETATM. After
    # renumbering we treat them as regular polymer residues so downstream
    # parsers (Graphein, PDB writers, etc.) don't drop them.
    sanitized.hetero[:] = False
    if has_ins_code:
        sanitized.ins_code[:] = ""
    return sanitized


def _load_normalized_structure(pdb_path: str) -> struc.AtomArray:
    """Load a PDB and keep only polymer-like residues for inference."""
    return _normalize_loaded_structure(strucio.load_structure(pdb_path))


def _filter_pdb_atom_lines_for_polymer(lines: Sequence[str]) -> List[str]:
    """Keep ATOM records and HETATM residues that have a full backbone.

    Reference receptor preservation works from raw PDB text so that receptor
    sidechains are not regenerated. This helper applies the same input hygiene
    there: waters, ions, salts, and organic ligands are dropped, while
    non-standard amino-acid residues emitted as HETATM are kept when the residue
    contains N/CA/C/O.
    """
    hetatm_groups: Dict[tuple[str, str, str, str], List[str]] = defaultdict(list)
    padded_lines: List[str] = []

    for line in lines:
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        padded = line.rstrip("\n").ljust(80)
        padded_lines.append(padded)
        if padded.startswith("HETATM"):
            key = (padded[21], padded[22:26], padded[26], padded[17:20])
            hetatm_groups[key].append(padded)

    keep_hetatm_keys = set()
    for key, group_lines in hetatm_groups.items():
        atom_names = {line[12:16].strip() for line in group_lines}
        if _BACKBONE_ATOM_SET.issubset(atom_names):
            keep_hetatm_keys.add(key)

    kept: List[str] = []
    for line in padded_lines:
        if line.startswith("ATOM  "):
            kept.append(line)
            continue
        key = (line[21], line[22:26], line[26], line[17:20])
        if key in keep_hetatm_keys:
            kept.append(line)
    return kept


def _surviving_residue_keys_before_normalization(
    pdb_struc: struc.AtomArray,
) -> List[tuple[str, int]]:
    """Return original (chain, res_id) keys for residues kept by normalization."""
    if len(pdb_struc) == 0:
        return []

    annotation_categories = pdb_struc.get_annotation_categories()
    has_ins_code = "ins_code" in annotation_categories
    ins_codes = pdb_struc.ins_code if has_ins_code else np.array([""] * len(pdb_struc))

    chain_ids = pdb_struc.chain_id
    res_ids = pdb_struc.res_id
    atom_names = pdb_struc.atom_name
    keys: List[tuple[str, int]] = []

    n_atoms = len(pdb_struc)
    cursor = 0
    while cursor < n_atoms:
        chain = chain_ids[cursor]
        rid = res_ids[cursor]
        icode = ins_codes[cursor]
        end = cursor + 1
        while (
            end < n_atoms
            and chain_ids[end] == chain
            and res_ids[end] == rid
            and ins_codes[end] == icode
        ):
            end += 1

        residue_atom_names = {str(a).strip() for a in atom_names[cursor:end]}
        if all(name in residue_atom_names for name in _BACKBONE_ATOMS):
            keys.append((str(chain), int(rid)))
        cursor = end

    return keys


def _residue_keys(pdb_struc: struc.AtomArray) -> List[tuple[str, int]]:
    keys: List[tuple[str, int]] = []
    seen = set()
    for chain, res_id in zip(pdb_struc.chain_id.tolist(), pdb_struc.res_id.tolist()):
        key = (str(chain), int(res_id))
        if key in seen:
            continue
        seen.add(key)
        keys.append(key)
    return keys


def _raw_to_normalized_residue_id_map(
    raw_struc: struc.AtomArray,
    normalized_struc: struc.AtomArray,
) -> Dict[tuple[str, int], List[tuple[str, int]]]:
    """Map original PDB residue numbering to normalized residue numbering.

    Explicit task hotspots are user-facing PDB residue ids, while the inference
    structure is internally renumbered to remove gaps/insertion codes. A raw key
    can map to more than one normalized residue when insertion codes were
    flattened and the task did not specify an insertion code.
    """
    raw_keys = _surviving_residue_keys_before_normalization(raw_struc)
    normalized_keys = _residue_keys(normalized_struc)
    if len(raw_keys) != len(normalized_keys):
        raise ValueError(
            "Cannot map raw PDB residue ids to normalized ids: "
            f"{len(raw_keys)} raw residues survived but normalized structure has "
            f"{len(normalized_keys)} residues."
        )

    mapping: Dict[tuple[str, int], List[tuple[str, int]]] = defaultdict(list)
    for raw_key, normalized_key in zip(raw_keys, normalized_keys):
        mapping[raw_key].append(normalized_key)
    return mapping


def _receptor_center_from_reference(
    pdb_struc: struc.AtomArray, pep_chain: str
) -> np.ndarray:
    receptor_ca = pdb_struc[
        (pdb_struc.chain_id != pep_chain) & (pdb_struc.atom_name == "CA")
    ]
    if len(receptor_ca) == 0:
        raise ValueError(
            f"Reference PDB has no receptor CA atoms outside peptide chain '{pep_chain}'."
        )
    return receptor_ca.coord.mean(axis=0)


def _pdb_atom_line_coord(line: str) -> np.ndarray:
    padded = line.rstrip("\n").ljust(80)
    return np.array(
        [
            float(padded[30:38]),
            float(padded[38:46]),
            float(padded[46:54]),
        ],
        dtype=np.float64,
    )


def _pdb_coord_validation_error(
    atom_positions: np.ndarray, atom_mask: np.ndarray
) -> str | None:
    coords = np.asarray(atom_positions)[np.asarray(atom_mask).astype(bool)]
    if coords.size == 0:
        return None
    if not np.isfinite(coords).all():
        return "non-finite coordinates"
    min_coord = float(coords.min())
    max_coord = float(coords.max())
    if min_coord < _PDB_COORD_MIN or max_coord > _PDB_COORD_MAX:
        return (
            f"coordinates outside fixed-width PDB range "
            f"[{_PDB_COORD_MIN:.3f}, {_PDB_COORD_MAX:.3f}] "
            f"(observed min={min_coord:.3f}, max={max_coord:.3f})"
        )
    return None


def _validate_pdb_coord(coord: np.ndarray) -> None:
    coord = np.asarray(coord, dtype=np.float64)
    if not np.isfinite(coord).all():
        raise ValueError(f"non-finite coordinates: {coord.tolist()}")
    min_coord = float(coord.min())
    max_coord = float(coord.max())
    if min_coord < _PDB_COORD_MIN or max_coord > _PDB_COORD_MAX:
        raise ValueError(
            f"coordinates outside fixed-width PDB range "
            f"[{_PDB_COORD_MIN:.3f}, {_PDB_COORD_MAX:.3f}] "
            f"(observed min={min_coord:.3f}, max={max_coord:.3f})"
        )


def _pdb_atom_line_resseq(line: str) -> int:
    return int(line.rstrip("\n").ljust(80)[22:26])


def _rewrite_pdb_atom_line(
    line: str,
    *,
    serial: int,
    resseq: int,
    coord: np.ndarray,
    chain_id: str | None = None,
) -> str:
    padded = line.rstrip("\n").ljust(80)
    chain = padded[21] if chain_id is None else (str(chain_id) + " ")[0]
    return (
        f"{padded[:6]}{serial:5d}{padded[11:21]}{chain:1}{resseq:4d}{padded[26:30]}"
        f"{float(coord[0]):8.3f}{float(coord[1]):8.3f}{float(coord[2]):8.3f}"
        f"{padded[54:80]}"
    ).ljust(80)


def _pdb_ter_line(serial: int, last_atom_line: str) -> str:
    padded = last_atom_line.rstrip("\n").ljust(80)
    resname = padded[17:20]
    chain = padded[21]
    resseq = padded[22:26]
    icode = padded[26]
    return f"TER   {serial:5d}      {resname:>3} {chain:1}{resseq:>4}{icode:1}".ljust(
        80
    )


def _replace_receptor_with_centered_reference(
    pdb_string: str,
    *,
    reference_pdb_path: str | None,
    reference_pep_chain: str,
    reference_receptor_center: np.ndarray | None,
    binder_chain: str = "L",
    generated_peptide_chains: set[str] | None = None,
) -> str:
    if not reference_pdb_path or reference_receptor_center is None:
        return pdb_string
    if not os.path.exists(reference_pdb_path):
        logger.warning(
            f"Reference PDB for receptor preservation does not exist: {reference_pdb_path}"
        )
        return pdb_string

    reference_center = np.asarray(reference_receptor_center, dtype=np.float64).reshape(
        3
    )
    reference_pep_chain = (str(reference_pep_chain or "L") + " ")[0]
    binder_chain = (str(binder_chain or "L") + " ")[0]

    raw_reference_atom_lines = []
    seen_model = False
    with open(reference_pdb_path) as handle:
        for line in handle:
            if line.startswith("MODEL"):
                if seen_model:
                    break
                seen_model = True
                continue
            if seen_model and line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            padded = line.rstrip("\n").ljust(80)
            if padded[21] != reference_pep_chain:
                raw_reference_atom_lines.append(padded)
    reference_atom_lines = _filter_pdb_atom_lines_for_polymer(raw_reference_atom_lines)
    if not reference_atom_lines:
        return pdb_string

    generated_peptide_chains = generated_peptide_chains or {binder_chain}
    generated_peptide_lines = []
    for line in pdb_string.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        padded = line.rstrip("\n").ljust(80)
        if padded[21] in generated_peptide_chains:
            generated_peptide_lines.append(padded)
    if not generated_peptide_lines:
        logger.warning(
            f"Cannot preserve reference receptor for {reference_pdb_path}: "
            f"generated PDB has no binder chain '{binder_chain}'."
        )
        return pdb_string

    chain_offsets: Dict[str, int] = {}
    for line in reference_atom_lines:
        chain = line[21]
        resseq = _pdb_atom_line_resseq(line)
        chain_offsets[chain] = min(chain_offsets.get(chain, resseq), resseq)
    chain_offsets = {
        chain: min_resseq - 1 for chain, min_resseq in chain_offsets.items()
    }

    output_lines = ["MODEL     1".ljust(80)]
    serial = 1
    last_chain: str | None = None
    last_atom_line: str | None = None

    def append_atom_line(
        line: str,
        *,
        coord_shift: np.ndarray | None,
        resseq_offset: int = 0,
        chain_id: str | None = None,
    ) -> None:
        nonlocal serial, last_chain, last_atom_line
        output_chain = line[21] if chain_id is None else (str(chain_id) + " ")[0]
        if last_atom_line is not None and last_chain != output_chain:
            output_lines.append(_pdb_ter_line(serial, last_atom_line))
            serial += 1
        coord = _pdb_atom_line_coord(line)
        if coord_shift is not None:
            coord = coord - coord_shift
        _validate_pdb_coord(coord)
        resseq = _pdb_atom_line_resseq(line) - int(resseq_offset)
        rewritten = _rewrite_pdb_atom_line(
            line, serial=serial, resseq=resseq, coord=coord, chain_id=chain_id
        )
        output_lines.append(rewritten)
        serial += 1
        last_chain = output_chain
        last_atom_line = rewritten

    try:
        for line in reference_atom_lines:
            append_atom_line(
                line,
                coord_shift=reference_center,
                resseq_offset=chain_offsets[line[21]],
            )
        for line in generated_peptide_lines:
            append_atom_line(
                line, coord_shift=None, resseq_offset=0, chain_id=binder_chain
            )
    except ValueError as exc:
        logger.warning(
            f"Cannot preserve reference receptor for {reference_pdb_path}: "
            f"failed to parse or rewrite PDB atom coordinates ({exc})."
        )
        return pdb_string

    if last_atom_line is not None:
        output_lines.append(_pdb_ter_line(serial, last_atom_line))
    output_lines.append("ENDMDL".ljust(80))
    output_lines.append("END".ljust(80))
    return "\n".join(output_lines) + "\n"


def _infer_reference_nonstandard_templates(
    pdb_struc: struc.AtomArray,
    *,
    pep_chain: str,
    task_type: str,
) -> List[Dict[str, Any]]:
    if task_type not in {"docking", "packing"}:
        return []

    pep_residues = pdb_struc[pdb_struc.chain_id == pep_chain]
    pep_res_ids = sorted(set(pep_residues.res_id))
    templates: List[Dict[str, Any]] = []
    for binder_pos, src_rid in enumerate(pep_res_ids, start=1):
        residue = pep_residues[pep_residues.res_id == src_rid]
        if len(residue) == 0:
            continue
        resname = str(residue.res_name[0]).strip().upper()
        spec = get_nonstandard_residue_spec(resname)
        if spec is None:
            continue
        templates.append(
            {"type": "residue_template", "resid": binder_pos, "resname": spec.resname}
        )
    return templates


def parse_binder_secondary_class(task: Dict[str, Any], task_name: str) -> int | None:
    value = task.get("binder_secondary_class", None)
    if value is None:
        return None
    return _normalize_binder_secondary_class_value(
        value,
        task_name=task_name,
        field_name="binder_secondary_class",
    )


def resolve_binder_secondary_classes(
    task: Dict[str, Any],
    *,
    task_name: str,
    n_outputs: int,
) -> List[int | None]:
    if n_outputs < 0:
        raise ValueError(
            f"Task '{task_name}': n_outputs must be >= 0, got {n_outputs}."
        )
    if n_outputs == 0:
        return []
    if task.get("binder_secondary_mix", None) is not None:
        raise ValueError(
            f"Task '{task_name}': binder_secondary_mix is no longer supported. "
            "Use binder_secondary_class or binder_topology."
        )

    single_class = parse_binder_secondary_class(task=task, task_name=task_name)
    return [single_class for _ in range(n_outputs)]


def _standard_residue_index(aa_value: Any) -> int:
    if isinstance(aa_value, int):
        if aa_value < 0 or aa_value >= len(restypes):
            raise ValueError(
                f"Residue type constraint aa index out of range: {aa_value}"
            )
        return aa_value

    aa = str(aa_value).strip().upper()
    if aa in restype_order:
        return int(restype_order[aa])

    if aa in restype_1to3.values():
        one_letter = next((k for k, v in restype_1to3.items() if v == aa), None)
        if one_letter is None:
            raise ValueError(
                f"Unsupported residue type constraint aa value: {aa_value}"
            )
        return int(restype_order[one_letter])

    raise ValueError(
        f"Unsupported residue type constraint aa value: {aa_value}. "
        "Use one-letter code, three-letter code, or residue index."
    )


def _normalize_residue_type_constraint(aa_value: Any) -> tuple[int, str | None]:
    spec = get_nonstandard_residue_spec(aa_value)
    if spec is not None:
        return _standard_residue_index(spec.proxy_aa), spec.resname
    return _standard_residue_index(aa_value), None


def _normalize_residue_type_constraint_aa(aa_value: Any) -> int:
    return _normalize_residue_type_constraint(aa_value)[0]


def _normalize_mode_name(mode_value: Any) -> str:
    if mode_value is None:
        return "none"
    normalized = str(mode_value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"linear", "none", "null", ""}:
        return "none"
    if normalized in {"isopep", "isopeptide", "staple", "stapled"}:
        return "stapled"
    if normalized == "head_tail":
        return "head_to_tail"
    if normalized == "bicyclic":
        return "bicycle"
    if normalized in {
        "residue_template",
        "residue_replacement",
        "linear_residue",
        "nonstandard_residue",
    }:
        return "residue_template"
    if normalized in {"staple_rty", "rty"}:
        return "staple_rty"
    if normalized in {"alkane_staple", "mk8_staple", "staple_mk8", "mk8"}:
        return "mk8_staple"
    if normalized in {"alkene_staple", "nle_staple", "staple_nle", "nle"}:
        return "nle_staple"
    return normalized or "none"


def _normalize_sampler_mode_name(mode_value: Any) -> str:
    if mode_value is None:
        return "none"
    normalized = str(mode_value).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"linear", "none", "null", ""}:
        return "none"
    if normalized == "head_tail":
        return "head_to_tail"
    if normalized in {"isopep", "isopeptide", "staple", "stapled"}:
        return "isopep"
    if normalized == "bicycle":
        return "bicyclic"
    if normalized in {"alkene_staple", "alkene"}:
        return "alkene_staple"
    if normalized in {"alkane_staple", "alkane"}:
        return "alkane_staple"
    return normalized or "none"


def _parse_binder_position(
    value: Any, *, total_len: int, field_name: str, task_name: str
) -> int:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"last", "end"}:
            return int(total_len)
    pos = int(value)
    if pos < 1 or pos > total_len:
        raise ValueError(
            f"Task '{task_name}': {field_name} position {pos} is outside binder length {total_len}."
        )
    return pos


def _constraint_entry_position(entry: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in entry:
            return entry[key]
    raise KeyError(keys[0])


def parse_residue_type_constraints(
    task: Dict[str, Any], total_len: int, task_name: str
) -> Dict[int, int]:
    raw_constraints = task.get("residue_type_constraints", None)
    if raw_constraints is None:
        return {}
    if not isinstance(raw_constraints, Sequence) or isinstance(
        raw_constraints, (str, bytes)
    ):
        raise ValueError(
            f"Task '{task_name}': residue_type_constraints must be a list of {{pos, aa}} entries."
        )

    parsed: Dict[int, int] = {}
    for entry in raw_constraints:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each residue_type_constraints entry must be a mapping, got {entry!r}."
            )
        if "pos" not in entry or "aa" not in entry:
            raise ValueError(
                f"Task '{task_name}': each residue_type_constraints entry must contain both 'pos' and 'aa'."
            )

        pos = int(entry["pos"])
        if pos < 1 or pos > total_len:
            raise ValueError(
                f"Task '{task_name}': residue_type_constraints position {pos} is outside binder length {total_len}."
            )

        aa_idx = _normalize_residue_type_constraint_aa(entry["aa"])
        prev = parsed.get(pos, None)
        if prev is not None and prev != aa_idx:
            raise ValueError(
                f"Task '{task_name}': conflicting residue_type_constraints specified for binder position {pos}."
            )
        parsed[pos] = aa_idx

    return parsed


def parse_nonstandard_residue_type_templates(
    task: Dict[str, Any],
    total_len: int,
    task_name: str,
) -> List[Dict[str, Any]]:
    """Compile legacy NCAA residue constraints into topology templates.

    The legacy parser historically retained only the proxy residue index,
    which made the requested NCAA disappear before postprocess resolution.
    Keep the original validation while returning the chemistry template that
    downstream OpenMM postprocess needs to reinstall.
    """
    raw_constraints = task.get("residue_type_constraints", None)
    if raw_constraints is None:
        return []
    if not isinstance(raw_constraints, Sequence) or isinstance(
        raw_constraints, (str, bytes)
    ):
        raise ValueError(
            f"Task '{task_name}': residue_type_constraints must be a list of {{pos, aa}} entries."
        )

    templates: List[Dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for entry in raw_constraints:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each residue_type_constraints entry must be a mapping, got {entry!r}."
            )
        if "pos" not in entry or "aa" not in entry:
            raise ValueError(
                f"Task '{task_name}': each residue_type_constraints entry must contain both 'pos' and 'aa'."
            )
        pos = int(entry["pos"])
        if pos < 1 or pos > total_len:
            raise ValueError(
                f"Task '{task_name}': residue_type_constraints position {pos} is outside binder length {total_len}."
            )
        spec = get_nonstandard_residue_spec(entry["aa"])
        if spec is None:
            continue
        if spec.resname in UNSUPPORTED_NONSTANDARD_RESIDUES:
            raise ValueError(
                f"Task '{task_name}': non-standard residue '{spec.resname}' at binder position {pos} "
                "is not supported by the relax pipeline (internal-position OpenMM minimize fails). "
                f"Unsupported residues: {sorted(UNSUPPORTED_NONSTANDARD_RESIDUES)}."
            )
        key = (pos, spec.resname)
        if key not in seen:
            templates.append(
                {"type": "residue_template", "resid": pos, "resname": spec.resname}
            )
            seen.add(key)
    return templates


def parse_cyclization(
    task: Dict[str, Any], total_len: int, task_name: str
) -> Dict[str, Any] | None:
    raw_cyclization = task.get("cyclization", None)
    if raw_cyclization is None:
        return None
    if not isinstance(raw_cyclization, Mapping):
        raise ValueError(
            f"Task '{task_name}': cyclization must be a mapping when provided."
        )

    mode = str(raw_cyclization.get("mode", "pair")).strip().lower()
    if mode != "pair":
        raise ValueError(
            f"Task '{task_name}': unsupported cyclization.mode '{mode}'. Only 'pair' is supported."
        )

    positions = raw_cyclization.get("positions", None)
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or len(positions) != 2
    ):
        raise ValueError(
            f"Task '{task_name}': cyclization.positions must be a two-element list of binder positions."
        )
    pos_a = int(positions[0])
    pos_b = int(positions[1])
    if pos_a == pos_b:
        raise ValueError(
            f"Task '{task_name}': cyclization.positions must contain two distinct positions."
        )
    for pos in (pos_a, pos_b):
        if pos < 1 or pos > total_len:
            raise ValueError(
                f"Task '{task_name}': cyclization position {pos} is outside binder length {total_len}."
            )

    shortcut_len = int(raw_cyclization.get("shortcut_len", 1))
    if shortcut_len < 1:
        raise ValueError(f"Task '{task_name}': cyclization.shortcut_len must be >= 1.")

    pos_a, pos_b = sorted((pos_a, pos_b))
    return {
        "mode": mode,
        "positions": [pos_a, pos_b],
        "shortcut_len": shortcut_len,
    }


def parse_optional_ca_pair_constraints(
    task: Dict[str, Any], total_len: int, task_name: str
) -> List[Dict[str, Any]]:
    raw_constraints = task.get("optional_ca_pair_constraints", None)
    if raw_constraints is None:
        return []
    if not isinstance(raw_constraints, Sequence) or isinstance(
        raw_constraints, (str, bytes)
    ):
        raise ValueError(
            f"Task '{task_name}': optional_ca_pair_constraints must be a list of pair constraint entries."
        )

    parsed: List[Dict[str, int]] = []
    seen_pairs = set()
    for entry in raw_constraints:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each optional_ca_pair_constraints entry must be a mapping, got {entry!r}."
            )
        required = {"pos_i", "pos_j", "bin_idx"}
        if not required.issubset(entry):
            raise ValueError(
                f"Task '{task_name}': each optional_ca_pair_constraints entry must contain {sorted(required)}."
            )

        pos_i = int(entry["pos_i"])
        pos_j = int(entry["pos_j"])
        if pos_i == pos_j:
            raise ValueError(
                f"Task '{task_name}': optional_ca_pair_constraints must use two distinct positions."
            )
        for pos in (pos_i, pos_j):
            if pos < 1 or pos > total_len:
                raise ValueError(
                    f"Task '{task_name}': optional_ca_pair_constraints position {pos} is outside binder length {total_len}."
                )

        bin_idx = int(entry["bin_idx"])
        if bin_idx < 0 or bin_idx >= 30:
            raise ValueError(
                f"Task '{task_name}': optional_ca_pair_constraints bin_idx {bin_idx} is outside supported range [0, 29]."
            )

        pos_i, pos_j = sorted((pos_i, pos_j))
        pair_key = (pos_i, pos_j)
        if pair_key in seen_pairs:
            raise ValueError(
                f"Task '{task_name}': duplicate optional_ca_pair_constraints specified for pair {pair_key}."
            )
        seen_pairs.add(pair_key)
        parsed_entry: Dict[str, Any] = {
            "pos_i": pos_i,
            "pos_j": pos_j,
            "bin_idx": bin_idx,
        }
        if (
            "target_distance_angstrom" in entry
            and entry["target_distance_angstrom"] is not None
        ):
            parsed_entry["target_distance_angstrom"] = float(
                entry["target_distance_angstrom"]
            )
        parsed.append(parsed_entry)

    return parsed


def task_has_explicit_constraint_schema(task: Dict[str, Any]) -> bool:
    constraint = task.get("constraint", None)
    if constraint is None:
        return False
    if not isinstance(constraint, Mapping):
        return True
    for key in ("cyclize", "spatial", "restype"):
        value = constraint.get(key, None)
        if value is None:
            continue
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) == 0
        ):
            continue
        return True
    return False


def parse_constraint_schema(
    task: Dict[str, Any],
    *,
    total_len: int,
    task_name: str,
) -> Dict[str, Any]:
    constraint = task.get("constraint", None)
    if constraint is None:
        return {
            "residue_type_constraints": {},
            "nonstandard_residue_templates": [],
            "cyclization": None,
            "optional_ca_pair_constraints": [],
        }
    if not isinstance(constraint, Mapping):
        raise ValueError(
            f"Task '{task_name}': constraint must be a mapping when provided."
        )

    residue_constraints: Dict[int, int] = {}
    nonstandard_templates: List[Dict[str, Any]] = []
    cyclization: Dict[str, Any] | None = None
    pair_constraints: List[Dict[str, int]] = []

    raw_restype = constraint.get("restype", []) or []
    if not isinstance(raw_restype, Sequence) or isinstance(raw_restype, (str, bytes)):
        raise ValueError(
            f"Task '{task_name}': constraint.restype must be a list when provided."
        )
    for entry in raw_restype:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each constraint.restype entry must be a mapping."
            )
        pos_value = _constraint_entry_position(entry, "resid", "pos")
        pos = _parse_binder_position(
            pos_value,
            total_len=total_len,
            field_name="constraint.restype",
            task_name=task_name,
        )
        if "aa" not in entry:
            raise ValueError(
                f"Task '{task_name}': constraint.restype entries must include 'aa'."
            )
        aa_idx, nonstandard_resname = _normalize_residue_type_constraint(entry["aa"])
        if (
            nonstandard_resname is not None
            and nonstandard_resname in UNSUPPORTED_NONSTANDARD_RESIDUES
        ):
            raise ValueError(
                f"Task '{task_name}': non-standard residue '{nonstandard_resname}' at binder position {pos} "
                f"is not supported by the relax pipeline (internal-position OpenMM minimize fails). "
                f"Unsupported residues: {sorted(UNSUPPORTED_NONSTANDARD_RESIDUES)}. "
            )
        prev = residue_constraints.get(pos, None)
        if prev is not None and prev != aa_idx:
            raise ValueError(
                f"Task '{task_name}': conflicting constraint.restype values for binder position {pos}."
            )
        residue_constraints[pos] = aa_idx
        if nonstandard_resname is not None:
            nonstandard_templates.append(
                {
                    "type": "residue_template",
                    "resid": pos,
                    "resname": nonstandard_resname,
                }
            )

    raw_cyclize = constraint.get("cyclize", []) or []
    if not isinstance(raw_cyclize, Sequence) or isinstance(raw_cyclize, (str, bytes)):
        raise ValueError(
            f"Task '{task_name}': constraint.cyclize must be a list when provided."
        )
    if len(raw_cyclize) > 1:
        raise ValueError(
            f"Task '{task_name}': current model only supports one constraint.cyclize entry, got {len(raw_cyclize)}."
        )
    if raw_cyclize:
        entry = raw_cyclize[0]
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each constraint.cyclize entry must be a mapping."
            )
        pos_a = _parse_binder_position(
            _constraint_entry_position(entry, "begin_resid", "begin_pos", "pos_i"),
            total_len=total_len,
            field_name="constraint.cyclize.begin_resid",
            task_name=task_name,
        )
        pos_b = _parse_binder_position(
            _constraint_entry_position(entry, "end_resid", "end_pos", "pos_j"),
            total_len=total_len,
            field_name="constraint.cyclize.end_resid",
            task_name=task_name,
        )
        if pos_a == pos_b:
            raise ValueError(
                f"Task '{task_name}': constraint.cyclize requires two distinct positions."
            )
        shortcut_len = int(entry.get("offset", entry.get("shortcut_len", 1)))
        if shortcut_len < 1:
            raise ValueError(
                f"Task '{task_name}': constraint.cyclize offset must be >= 1."
            )
        cyclization = {
            "mode": "pair",
            "positions": sorted((pos_a, pos_b)),
            "shortcut_len": shortcut_len,
        }

    raw_spatial = constraint.get("spatial", []) or []
    if not isinstance(raw_spatial, Sequence) or isinstance(raw_spatial, (str, bytes)):
        raise ValueError(
            f"Task '{task_name}': constraint.spatial must be a list when provided."
        )
    seen_pairs = set()
    for entry in raw_spatial:
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each constraint.spatial entry must be a mapping."
            )
        pos_i = _parse_binder_position(
            _constraint_entry_position(entry, "begin_resid", "begin_pos", "pos_i"),
            total_len=total_len,
            field_name="constraint.spatial.begin_resid",
            task_name=task_name,
        )
        pos_j = _parse_binder_position(
            _constraint_entry_position(entry, "end_resid", "end_pos", "pos_j"),
            total_len=total_len,
            field_name="constraint.spatial.end_resid",
            task_name=task_name,
        )
        if pos_i == pos_j:
            raise ValueError(
                f"Task '{task_name}': constraint.spatial requires two distinct positions."
            )
        if "distance" not in entry:
            raise ValueError(
                f"Task '{task_name}': constraint.spatial entries must include 'distance'."
            )
        pos_i, pos_j = sorted((pos_i, pos_j))
        pair_key = (pos_i, pos_j)
        if pair_key in seen_pairs:
            raise ValueError(
                f"Task '{task_name}': duplicate constraint.spatial pair {pair_key}."
            )
        seen_pairs.add(pair_key)
        pair_constraints.append(
            {
                "pos_i": pos_i,
                "pos_j": pos_j,
                "bin_idx": pair_bin_idx_from_distance_angstrom(
                    float(entry["distance"])
                ),
            }
        )

    return {
        "residue_type_constraints": residue_constraints,
        "nonstandard_residue_templates": nonstandard_templates,
        "cyclization": cyclization,
        "optional_ca_pair_constraints": pair_constraints,
    }


def parse_topology_schema(
    task: Dict[str, Any],
    *,
    total_len: int,
    task_name: str,
) -> List[Dict[str, Any]]:
    return parse_topology_entries(
        task.get("topology", None),
        total_len=total_len,
        task_name=task_name,
        field_name="topology",
    )


def parse_topology_entries(
    topology: Any,
    *,
    total_len: int,
    task_name: str,
    field_name: str = "topology",
) -> List[Dict[str, Any]]:
    if topology is None:
        return []
    if not isinstance(topology, Sequence) or isinstance(topology, (str, bytes)):
        raise ValueError(
            f"Task '{task_name}': {field_name} must be a list when provided."
        )

    parsed: List[Dict[str, Any]] = []
    for idx, entry in enumerate(topology):
        if not isinstance(entry, Mapping):
            raise ValueError(
                f"Task '{task_name}': each {field_name} entry must be a mapping."
            )
        topo_type = _normalize_mode_name(entry.get("type", None))
        if topo_type not in {
            "head_to_tail",
            "stapled",
            "disulfide",
            "bicycle",
            "staple_rty",
            "mk8_staple",
            "nle_staple",
            "residue_template",
        }:
            raise ValueError(
                f"Task '{task_name}': unsupported {field_name} type '{entry.get('type', None)}' at index {idx}."
            )
        if topo_type == "residue_template":
            resid = _parse_binder_position(
                _constraint_entry_position(entry, "resid", "pos"),
                total_len=total_len,
                field_name=f"{field_name}[{idx}].resid",
                task_name=task_name,
            )
            resname = str(entry.get("resname", entry.get("aa", ""))).strip().upper()
            if not get_nonstandard_residue_spec(resname):
                raise ValueError(
                    f"Task '{task_name}': unsupported {field_name} residue_template resname '{resname}'."
                )
            if resname in UNSUPPORTED_NONSTANDARD_RESIDUES:
                raise ValueError(
                    f"Task '{task_name}': {field_name} residue_template resname '{resname}' is registered but is not "
                    f"supported by the relax pipeline (internal-position OpenMM minimize fails). "
                    f"Unsupported residues: {sorted(UNSUPPORTED_NONSTANDARD_RESIDUES)}."
                )
            parsed.append(
                {
                    "type": "residue_template",
                    "resid": resid,
                    "resname": resname,
                    "anchors": [resid],
                }
            )
            continue
        if topo_type in {"bicycle", "staple_rty"}:
            anchors = entry.get("anchors", None)
            if (
                not isinstance(anchors, Sequence)
                or isinstance(anchors, (str, bytes))
                or len(anchors) != 3
            ):
                raise ValueError(
                    f"Task '{task_name}': {topo_type} {field_name} requires anchors=[i,j,k]."
                )
            parsed.append(
                {
                    "type": topo_type,
                    "anchors": [
                        _parse_binder_position(
                            v,
                            total_len=total_len,
                            field_name=f"{field_name}[{idx}].anchors",
                            task_name=task_name,
                        )
                        for v in anchors
                    ],
                }
            )
            continue
        if topo_type in {"mk8_staple", "nle_staple"}:
            anchors = entry.get("anchors", None)
            if anchors is not None:
                if (
                    not isinstance(anchors, Sequence)
                    or isinstance(anchors, (str, bytes))
                    or len(anchors) != 2
                ):
                    raise ValueError(
                        f"Task '{task_name}': {topo_type} {field_name} requires anchors=[i,j]."
                    )
                pos_a, pos_b = [
                    _parse_binder_position(
                        v,
                        total_len=total_len,
                        field_name=f"{field_name}[{idx}].anchors",
                        task_name=task_name,
                    )
                    for v in anchors
                ]
            else:
                pos_a = _parse_binder_position(
                    _constraint_entry_position(
                        entry, "begin_resid", "begin_pos", "pos_i"
                    ),
                    total_len=total_len,
                    field_name=f"{field_name}[{idx}].begin_resid",
                    task_name=task_name,
                )
                pos_b = _parse_binder_position(
                    _constraint_entry_position(entry, "end_resid", "end_pos", "pos_j"),
                    total_len=total_len,
                    field_name=f"{field_name}[{idx}].end_resid",
                    task_name=task_name,
                )
            if pos_a == pos_b:
                raise ValueError(
                    f"Task '{task_name}': {field_name} entry {idx} requires two distinct positions."
                )
            parsed.append({"type": topo_type, "anchors": sorted((pos_a, pos_b))})
            continue
        anchors = entry.get("anchors", None)
        if anchors is not None:
            if (
                not isinstance(anchors, Sequence)
                or isinstance(anchors, (str, bytes))
                or len(anchors) != 2
            ):
                raise ValueError(
                    f"Task '{task_name}': {topo_type} {field_name} requires anchors=[i,j]."
                )
            pos_a, pos_b = [
                _parse_binder_position(
                    v,
                    total_len=total_len,
                    field_name=f"{field_name}[{idx}].anchors",
                    task_name=task_name,
                )
                for v in anchors
            ]
        else:
            pos_a = _parse_binder_position(
                _constraint_entry_position(entry, "begin_resid", "begin_pos", "pos_i"),
                total_len=total_len,
                field_name=f"{field_name}[{idx}].begin_resid",
                task_name=task_name,
            )
            pos_b = _parse_binder_position(
                _constraint_entry_position(entry, "end_resid", "end_pos", "pos_j"),
                total_len=total_len,
                field_name=f"{field_name}[{idx}].end_resid",
                task_name=task_name,
            )
        if pos_a == pos_b:
            raise ValueError(
                f"Task '{task_name}': {field_name} entry {idx} requires two distinct positions."
            )
        # For stapled (isopep) anchors carry semantic order: pos_i must be K,
        # pos_j must be D or E. Sorting would erase that distinction, breaking
        # any force-fix that needs to know which side is which.
        anchor_list = (
            [pos_a, pos_b] if topo_type == "stapled" else sorted((pos_a, pos_b))
        )
        parsed.append({"type": topo_type, "anchors": anchor_list})
    return parsed


def _merge_nonstandard_residue_templates(
    topology_entries: Sequence[Mapping[str, Any]],
    residue_templates: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    merged = [dict(entry) for entry in topology_entries]
    existing: set[tuple[int, str]] = set()
    active_anchor_residues: dict[int, str] = {}
    active_residue_names = {
        "mk8_staple": "MK8",
        "nle_staple": "NLE",
    }
    for entry in merged:
        topo_type = _normalize_mode_name(entry.get("type", None))
        anchors = [int(v) for v in entry.get("anchors", []) if int(v) > 0]
        active_resname = active_residue_names.get(topo_type)
        if active_resname is not None:
            for anchor in anchors:
                active_anchor_residues[anchor] = active_resname
        if topo_type == "residue_template":
            resid = entry.get("resid", anchors[0] if anchors else None)
            resname = entry.get("resname", None)
            if resid is not None and resname is not None:
                existing.add((int(resid), str(resname).strip().upper()))
    for template in residue_templates:
        resid = int(template["resid"])
        resname = str(template["resname"]).strip().upper()
        active_resname = active_anchor_residues.get(resid)
        if active_resname is not None:
            if resname != active_resname:
                raise ValueError(
                    f"Non-standard residue constraint at binder position {resid} requests {resname}, "
                    f"but active topology requires {active_resname}."
                )
            continue
        key = (resid, resname)
        if key in existing:
            continue
        merged.append(
            {
                "type": "residue_template",
                "resid": resid,
                "resname": resname,
                "anchors": [resid],
            }
        )
        existing.add(key)
    return merged


def _strict_boolish(value: Any, *, field_name: str) -> bool:
    """Parse config booleans without treating the string ``"false"`` as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean (true/false), got {value!r}.")


def parse_postprocess_schema(task: Dict[str, Any]) -> Dict[str, Any]:
    raw_postprocess = task.get("postprocess", None)
    if raw_postprocess is None:
        return {
            "enabled": False,
            "enabled_explicit": False,
            "method": "auto",
            "relax_linear": False,
            "explicit": False,
        }
    if isinstance(raw_postprocess, bool):
        return {
            "enabled": bool(raw_postprocess),
            "enabled_explicit": True,
            "method": "auto",
            "relax_linear": False,
            "explicit": True,
        }
    if not isinstance(raw_postprocess, Mapping):
        raise ValueError(
            "Task postprocess must be either bool or mapping when provided."
        )
    enabled_explicit = "enabled" in raw_postprocess
    # A mapping carrying a chemistry method is a request to run it unless the
    # caller explicitly opts out with enabled=false.  This keeps
    # ``{method: auto}`` distinct from an explicit disabled setting.
    enabled = _strict_boolish(
        raw_postprocess.get("enabled", True),
        field_name="postprocess.enabled",
    )
    method = str(raw_postprocess.get("method", "auto")).strip().lower() or "auto"
    method_aliases = {
        "auto": "auto",
        "openmm": "openmm",
        "rosetta": "rosetta_topology",
        "pyrosetta": "rosetta_topology",
        "pyrosetta_disulfide": "rosetta_topology",
        "rosetta_disulfide": "rosetta_topology",
        "rosetta_topology": "rosetta_topology",
    }
    method = method_aliases.get(method, method)
    if method not in {"auto", "openmm", "rosetta_topology"}:
        raise ValueError(
            f"Task postprocess.method supports 'auto', 'openmm', or 'rosetta_topology', got '{method}'."
        )
    return {
        "enabled": enabled,
        "enabled_explicit": enabled_explicit,
        "method": method,
        "relax_linear": _strict_boolish(
            raw_postprocess.get("relax_linear", False),
            field_name="postprocess.relax_linear",
        ),
        "explicit": True,
    }


_ROSETTA_POSTPROCESS_TOPOLOGY_TYPES = frozenset({"head_to_tail", "disulfide"})


_OPENMM_POSTPROCESS_TOPOLOGY_TYPES = frozenset(
    {"residue_template", "stapled", "bicycle", "staple_rty", "mk8_staple", "nle_staple"}
)


def _active_postprocess_topology_entries(
    topology_entries: Sequence[Mapping[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """Return topology entries that require chemistry restoration/relaxation."""
    active: List[Dict[str, Any]] = []
    for entry in topology_entries or []:
        if not isinstance(entry, Mapping):
            continue
        if _normalize_mode_name(entry.get("type", None)) == "none":
            continue
        active.append(dict(entry))
    return active


def resolve_postprocess_contract(
    task: Dict[str, Any],
    topology_entries: Sequence[Mapping[str, Any]] | None,
) -> Dict[str, Any]:
    """Resolve task-local chemistry postprocess policy before inference starts.

    Standard linear samples are skipped by default.  Head-to-tail and
    disulfide-only chemistry is Rosetta-relaxable; any non-standard residue or
    topology that Rosetta cannot represent is OpenMM-owned.  Explicit task
    settings are validated here so an incompatible method or an explicit
    ``enabled=false`` cannot silently bypass mandatory chemistry restoration.
    """
    cfg = parse_postprocess_schema(task)
    active_entries = _active_postprocess_topology_entries(topology_entries)
    topology_types = {
        _normalize_mode_name(entry.get("type", None)) for entry in active_entries
    }
    unknown_types = topology_types - (
        _ROSETTA_POSTPROCESS_TOPOLOGY_TYPES | _OPENMM_POSTPROCESS_TOPOLOGY_TYPES
    )
    if unknown_types:
        raise ValueError(
            "Task postprocess cannot classify topology types "
            f"{sorted(unknown_types)}; add an explicit chemistry policy first."
        )

    has_mandatory_chemistry = bool(active_entries)
    if has_mandatory_chemistry:
        required_method = (
            "rosetta_topology"
            if topology_types <= _ROSETTA_POSTPROCESS_TOPOLOGY_TYPES
            else "openmm"
        )
    else:
        # A canonical linear sample remains a no-op by default.  An explicit
        # enabled=true request, however, must resolve to an executable backend
        # instead of being swallowed by the default-skip branch.
        required_method = (
            "rosetta_topology"
            if cfg["relax_linear"]
            else ("openmm" if cfg["explicit"] and cfg["enabled"] else "auto")
        )

    if cfg["explicit"]:
        if (
            has_mandatory_chemistry
            and cfg.get("enabled_explicit", False)
            and not cfg["enabled"]
        ):
            raise ValueError(
                "Task postprocess.enabled=false is invalid for mandatory special "
                f"chemistry ({sorted(topology_types)}); enable postprocess so it can be restored."
            )
        requested_method = str(cfg["method"])
        if (
            requested_method != "auto"
            and has_mandatory_chemistry
            and requested_method != required_method
        ):
            raise ValueError(
                f"Task postprocess.method='{requested_method}' is incompatible with "
                f"topology {sorted(topology_types)}; use '{required_method}' or 'auto'."
            )
        resolved_method = (
            required_method if requested_method == "auto" else requested_method
        )
        # ``auto`` on an ordinary linear sample remains a no-op unless the
        # task explicitly asks for linear relaxation.  A non-auto method is
        # an explicit request and is therefore allowed to force that work.
        resolved_enabled = bool(cfg["enabled"]) and (
            has_mandatory_chemistry
            or bool(cfg.get("relax_linear", False))
            or requested_method != "auto"
            or (cfg["explicit"] and bool(cfg.get("enabled", False)))
        )
    else:
        resolved_method = required_method
        resolved_enabled = bool(has_mandatory_chemistry or cfg["relax_linear"])

    return {
        **cfg,
        "resolved_enabled": resolved_enabled,
        "resolved_method": resolved_method,
        "mandatory_chemistry": has_mandatory_chemistry,
        "topology_types": sorted(topology_types),
    }


def validate_task_postprocess_config(
    task: Dict[str, Any],
    task_name: str,
    *,
    inspect_reference_chemistry: bool = False,
) -> Dict[str, Any]:
    """Validate task-local postprocess semantics before model construction.

    ``build_atom_array`` later compiles sampler/constraint state into concrete
    topology entries.  Configuration validation must nevertheless reject an
    explicit opt-out or incompatible backend before a checkpoint is loaded.
    This helper collects chemistry that is statically visible in the task and
    delegates the actual policy decision to the same resolver used by runtime
    sample construction, keeping the two paths on one contract.  Direct
    inference may additionally request a lightweight reference-PDB inspection
    so an explicit opt-out cannot survive until dataloader iteration.
    """
    topology_entries: list[Dict[str, Any]] = []

    # Infer a static binder length for legacy position validation.  Sequence
    # length is authoritative when present; otherwise the largest explicit
    # position is the strongest bound available before a reference PDB is
    # loaded.
    static_total_len = 0
    static_exact_total_len: int | None = None
    sequence_proxy = None
    sequence = task.get("sequence", None)
    if sequence is not None:
        sequence_proxy, sequence_templates = parse_sequence_with_nonstandard(
            sequence,
            task_name=task_name,
        )
        static_total_len = len(sequence_proxy)
        static_exact_total_len = int(static_total_len)
        topology_entries.extend(sequence_templates)

    def _position_values(value: Any) -> list[int]:
        def _coerce_position(item: Any) -> int | None:
            if isinstance(item, str) and item.strip().lower() in {"last", "end"}:
                # Legacy cyclization schemas use ``last``/``end`` for the
                # terminal binder residue.  Resolve it during static
                # validation whenever a length is already known (the caller
                # keeps a conservative lower bound of one otherwise).
                return int(static_total_len) if int(static_total_len) > 0 else None
            try:
                return int(item)
            except (TypeError, ValueError):
                return None

        if isinstance(value, Mapping):
            values: list[int] = []
            for key in (
                "pos",
                "resid",
                "begin_resid",
                "begin_pos",
                "pos_i",
                "end_resid",
                "end_pos",
                "pos_j",
                "positions",
            ):
                if key not in value:
                    continue
                item = value[key]
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    for nested in item:
                        position = _coerce_position(nested)
                        if position is not None:
                            values.append(position)
                else:
                    position = _coerce_position(item)
                    if position is not None:
                        values.append(position)
            anchors = value.get("anchors", None)
            if isinstance(anchors, Sequence) and not isinstance(anchors, (str, bytes)):
                for item in anchors:
                    position = _coerce_position(item)
                    if position is not None:
                        values.append(position)
            known_keys = {
                "pos",
                "resid",
                "begin_resid",
                "begin_pos",
                "pos_i",
                "end_resid",
                "end_pos",
                "pos_j",
                "positions",
                "anchors",
            }
            for key, item in value.items():
                if key in known_keys or not isinstance(item, (Mapping, Sequence)):
                    continue
                if isinstance(item, (str, bytes)):
                    continue
                values.extend(_position_values(item))
            return values
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            values: list[int] = []
            for item in value:
                values.extend(_position_values(item))
            return values
        return []

    for key in ("nres", "length", "peptide_length", "binder_length"):
        try:
            candidate_len = int(task.get(key, 0) or 0)
            static_total_len = max(static_total_len, candidate_len)
            if static_exact_total_len is None and candidate_len > 0:
                static_exact_total_len = candidate_len
        except (TypeError, ValueError):
            pass
    static_total_len = max(
        static_total_len,
        *_position_values(task.get("topology", None)),
        *_position_values(task.get("constraint", None)),
        *_position_values(task.get("cyclization", None)),
        *_position_values(task.get("residue_type_constraints", None)),
        1,
    )

    def _is_static_head_tail_pair(value: Any) -> bool:
        positions = _position_values(value)
        if len(positions) < 2:
            return False
        return sorted(positions[:2]) == [1, int(static_total_len)]

    raw_topology = task.get("topology", None)
    if raw_topology is not None:
        if not isinstance(raw_topology, Sequence) or isinstance(
            raw_topology, (str, bytes)
        ):
            # ``validate_task`` already reports the schema error with the
            # task name; let it remain the source of that diagnostic.
            topology_entries = []
        else:
            topology_entries.extend(
                dict(entry) for entry in raw_topology if isinstance(entry, Mapping)
            )

    # Explicit restype constraints can request a registered non-standard
    # residue without putting it in ``topology``.
    constraint = task.get("constraint", None)
    if isinstance(constraint, Mapping):
        raw_restype = constraint.get("restype", None) or []
        if isinstance(raw_restype, Sequence) and not isinstance(
            raw_restype, (str, bytes)
        ):
            for entry in raw_restype:
                if not isinstance(entry, Mapping) or "aa" not in entry:
                    continue
                spec = get_nonstandard_residue_spec(entry["aa"])
                if spec is not None:
                    if spec.resname in UNSUPPORTED_NONSTANDARD_RESIDUES:
                        raise ValueError(
                            f"Task '{task_name}': non-standard residue '{spec.resname}' "
                            "is not supported by the relax pipeline. "
                            f"Unsupported residues: {sorted(UNSUPPORTED_NONSTANDARD_RESIDUES)}."
                        )
                    topology_entries.append(
                        {
                            "type": "residue_template",
                            "resid": entry.get("resid", entry.get("pos", 1)),
                            "resname": spec.resname,
                        }
                    )
        raw_cyclize = constraint.get("cyclize", None) or []
        if (
            isinstance(raw_cyclize, Sequence)
            and not isinstance(raw_cyclize, (str, bytes))
            and raw_cyclize
            and _is_static_head_tail_pair(raw_cyclize[0])
        ):
            topology_entries.append({"type": "head_to_tail", "anchors": []})

    if _is_static_head_tail_pair(task.get("cyclization", None)):
        topology_entries.append({"type": "head_to_tail", "anchors": []})

    # Preserve non-standard requests from the legacy list schema instead of
    # allowing proxy amino-acid indices to erase their chemistry.
    topology_entries.extend(
        parse_nonstandard_residue_type_templates(
            task,
            static_total_len,
            task_name,
        )
    )

    # Sampler modes are resolved stochastically later, but their chemistry
    # class is already explicit enough to enforce the mandatory policy now.
    sampler = task.get("sampler", None)
    if sampler is None:
        sampler = task.get("condition_sampler", None)
    if task_has_explicit_constraint_schema(task):
        # Runtime gives an explicit constraint schema precedence over either
        # sampler alias.  Preflight must not invent chemistry that the sample
        # builder will deliberately ignore.
        sampler = None
    sampler_mode = None
    if isinstance(sampler, str):
        sampler_mode = sampler
    elif isinstance(sampler, Mapping):
        sampler_mode = sampler.get("mode", sampler.get("type", None))
    normalized_sampler_mode = str(sampler_mode or "").strip().lower().replace("-", "_")
    sampler_topology = {
        "head_to_tail": "head_to_tail",
        "disulfide": "disulfide",
        "isopep": "stapled",
        "stapled": "stapled",
        "alkane_staple": "mk8_staple",
        "alkene_staple": "nle_staple",
        "bicycle": "bicycle",
        "bicyclic": "bicycle",
        "staple_rty": "staple_rty",
    }.get(normalized_sampler_mode)
    if sampler_topology is not None:
        topology_entries.append({"type": sampler_topology, "anchors": []})

    # ``build_atom_array()`` lets a non-null motif contig determine each output
    # length.  Enumerate the complete deterministic length set so preflight
    # neither misses reachable chemistry nor rejects an unreachable library
    # length.  A reference contig is exact when reference inspection is enabled.
    sampled_topology_total_len: int | set[int] | None = static_exact_total_len
    motif_cfg = task.get("motif_cfg", None)
    if isinstance(motif_cfg, Mapping) and motif_cfg.get("contigs", None) is not None:
        sampled_topology_total_len = _possible_motif_total_lengths(
            task,
            task_name=task_name,
            inspect_reference=inspect_reference_chemistry,
        )

    binder_topology = task.get("binder_topology", None)
    if isinstance(binder_topology, Mapping):
        cys_pairs = binder_topology.get("cys_pair_constraints", None)
        if (
            isinstance(cys_pairs, Sequence)
            and not isinstance(cys_pairs, (str, bytes))
            and cys_pairs
        ):
            topology_entries.append({"type": "disulfide", "anchors": []})
        if (
            binder_topology.get("cyclization", None)
            or binder_topology.get("cyclic", False) is True
        ):
            topology_entries.append({"type": "head_to_tail", "anchors": []})
        binder_topology_mode = (
            str(binder_topology.get("mode", "") or "").strip().lower()
        )
        if binder_topology_mode in {"sample", "sample_lib", "library", "topology_lib"}:
            lib_path, sampled_candidates = _sampled_topology_candidates(
                task,
                binder_topology,
                total_len=sampled_topology_total_len,
                task_name=task_name,
            )
            # A task-level postprocess policy must be valid for every entry
            # that sampling can choose.  Build the deterministic chemistry
            # union with the same adapter used by runtime; never pick a random
            # library entry during configuration validation.
            for entry_index, entry in sampled_candidates:
                source_id = str(
                    entry.get("id", entry.get("name", f"entry_{entry_index}"))
                )
                preflight_source = f"preflight:{lib_path}:{source_id}"
                _normalize_topology_tokens(
                    entry["ss_token"],
                    entry.get("block_adj", None),
                    total_len=int(entry["length"]),
                    task_name=task_name,
                    source=preflight_source,
                )
                scaffold_payload = _scaffold_condition_payload(
                    entry,
                    total_len=int(entry["length"]),
                    task_name=task_name,
                    source=preflight_source,
                )
                topology_entries.extend(
                    scaffold_payload.get("topology_entries", []) or []
                )

    resolved = resolve_postprocess_contract(task, topology_entries)
    if not inspect_reference_chemistry or task.get("postprocess", None) is None:
        return resolved

    task_type = str(task.get("task_type", "")).strip().lower()
    sequence_based_docking = task_type == "docking" and bool(sequence_proxy)
    if task_type not in {"docking", "packing"} or sequence_based_docking:
        return resolved

    # Reference-derived NCAA identities are not present in the YAML topology,
    # but they are concrete chemistry and therefore must participate in the
    # same configuration-phase mandatory-postprocess decision.  This is only a
    # PDB parse: it deliberately does not construct a datamodule or load a
    # checkpoint.  Sequence-based docking ignores reference peptide chemistry,
    # matching ``build_atom_array`` exactly.
    reference_pdb_path = os.path.abspath(str(task.get("pdb_path", "")))
    reference_templates = _infer_reference_nonstandard_templates(
        _load_normalized_structure(reference_pdb_path),
        pep_chain=str(task.get("pep_chain") or "L"),
        task_type=task_type,
    )
    if not reference_templates:
        return resolved
    return resolve_postprocess_contract(
        task,
        [*topology_entries, *reference_templates],
    )


def topology_entries_to_condition_metadata(
    *,
    topology_entries: Sequence[Mapping[str, Any]],
    residue_constraints: Dict[int, int],
    pair_constraints: List[Dict[str, int]],
) -> Dict[str, Any]:
    if not topology_entries:
        return _build_condition_metadata(
            mode_name=None,
            anchor_positions=[],
            pair_entries=pair_constraints,
            residue_constraints=residue_constraints,
        )
    active_entries = [
        dict(entry)
        for entry in topology_entries
        if _normalize_mode_name(entry.get("type", None)) != "residue_template"
    ]
    if not active_entries:
        return _build_condition_metadata(
            mode_name=None,
            anchor_positions=[],
            pair_entries=pair_constraints,
            residue_constraints=residue_constraints,
        )
    topo = active_entries[0]
    mode_name = _normalize_mode_name(topo.get("type", None))
    anchors = topo.get("anchors", [])
    pair_entries = []
    anchor_set = set(int(v) for v in anchors)
    for entry in pair_constraints:
        pair_set = {int(entry["pos_i"]), int(entry["pos_j"])}
        if pair_set.issubset(anchor_set):
            pair_entries.append(entry)
    return _build_condition_metadata(
        mode_name=mode_name,
        anchor_positions=anchors,
        pair_entries=pair_entries or pair_constraints,
        residue_constraints=residue_constraints,
    )


def _merge_residue_type_constraints(
    base_constraints: Dict[int, int],
    extra_constraints: Dict[int, int],
    *,
    task_name: str,
) -> Dict[int, int]:
    merged = dict(base_constraints)
    for pos, aa_idx in extra_constraints.items():
        prev = merged.get(int(pos), None)
        if prev is not None and int(prev) != int(aa_idx):
            raise ValueError(
                f"Task '{task_name}': conflicting residue_type_constraints for binder position {int(pos)} "
                f"({int(prev)} vs {int(aa_idx)})."
            )
        merged[int(pos)] = int(aa_idx)
    return merged


def _merge_optional_pair_constraints(
    base_constraints: List[Dict[str, Any]],
    extra_constraints: List[Dict[str, Any]],
    *,
    task_name: str,
) -> List[Dict[str, Any]]:
    merged: Dict[tuple[int, int], Dict[str, Any]] = {}
    for entry in list(base_constraints) + list(extra_constraints):
        pos_i, pos_j = sorted((int(entry["pos_i"]), int(entry["pos_j"])))
        bin_idx = int(entry["bin_idx"])
        key = (pos_i, pos_j)
        prev = merged.get(key, None)
        if prev is not None and int(prev["bin_idx"]) != bin_idx:
            raise ValueError(
                f"Task '{task_name}': conflicting optional_ca_pair_constraints for pair {key} "
                f"({prev['bin_idx']} vs {bin_idx})."
            )
        merged_entry: Dict[str, Any] = {
            "pos_i": pos_i,
            "pos_j": pos_j,
            "bin_idx": bin_idx,
        }
        if (
            "target_distance_angstrom" in entry
            and entry["target_distance_angstrom"] is not None
        ):
            merged_entry["target_distance_angstrom"] = float(
                entry["target_distance_angstrom"]
            )
        elif prev is not None and "target_distance_angstrom" in prev:
            merged_entry["target_distance_angstrom"] = float(
                prev["target_distance_angstrom"]
            )
        merged[key] = merged_entry
    return [merged[key] for key in sorted(merged)]


def _merge_cyclization(
    base_cyclization: Dict[str, Any] | None,
    extra_cyclization: Dict[str, Any] | None,
    *,
    task_name: str,
    source_name: str,
) -> Dict[str, Any] | None:
    if extra_cyclization is None:
        return base_cyclization
    normalized_extra = {
        "mode": str(extra_cyclization.get("mode", "pair")),
        "positions": sorted(int(pos) for pos in extra_cyclization.get("positions", [])),
        "shortcut_len": int(extra_cyclization.get("shortcut_len", 1)),
    }
    if base_cyclization is not None:
        normalized_base = {
            "mode": str(base_cyclization.get("mode", "pair")),
            "positions": sorted(
                int(pos) for pos in base_cyclization.get("positions", [])
            ),
            "shortcut_len": int(base_cyclization.get("shortcut_len", 1)),
        }
        if normalized_base != normalized_extra:
            raise ValueError(
                f"Task '{task_name}': {source_name} cyclization conflicts with explicit cyclization "
                f"({normalized_base} vs {normalized_extra})."
            )
        return dict(base_cyclization)
    return normalized_extra


def _topology_entry_signature(entry: Mapping[str, Any]) -> tuple[Any, ...]:
    topo_type = _normalize_mode_name(entry.get("type", None))
    if topo_type == "residue_template":
        resid = entry.get("resid", None)
        anchors = entry.get("anchors", [])
        if resid is None and anchors:
            resid = anchors[0]
        return (topo_type, int(resid), str(entry.get("resname", "")).strip().upper())
    return (topo_type, tuple(int(v) for v in entry.get("anchors", [])))


def _merge_topology_entries(
    base_entries: Sequence[Mapping[str, Any]],
    extra_entries: Sequence[Mapping[str, Any]],
    *,
    task_name: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    merged = [dict(entry) for entry in (base_entries or [])]
    signatures = {_topology_entry_signature(entry) for entry in merged}
    single_instance_types = {"head_to_tail", "disulfide"}
    active_singletons: Dict[str, tuple[Any, ...]] = {}
    for entry in merged:
        topo_type = _normalize_mode_name(entry.get("type", None))
        if topo_type in single_instance_types:
            active_singletons[topo_type] = _topology_entry_signature(entry)

    for entry in extra_entries or []:
        normalized_entry = dict(entry)
        topo_type = _normalize_mode_name(normalized_entry.get("type", None))
        normalized_entry["type"] = topo_type
        signature = _topology_entry_signature(normalized_entry)
        if signature in signatures:
            continue
        prev_signature = active_singletons.get(topo_type, None)
        if prev_signature is not None and prev_signature != signature:
            raise ValueError(
                f"Task '{task_name}': {source_name} topology entry {normalized_entry} conflicts with "
                f"existing {topo_type} topology {prev_signature}."
            )
        merged.append(normalized_entry)
        signatures.add(signature)
        if topo_type in single_instance_types:
            active_singletons[topo_type] = signature
    return merged


def _cyclization_ca_pair_constraint(
    cyclization: Dict[str, Any] | None,
) -> List[Dict[str, int]]:
    if cyclization is None:
        return []
    positions = cyclization.get("positions", None)
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or len(positions) != 2
    ):
        return []
    pos_i, pos_j = sorted(int(pos) for pos in positions)
    if pos_i == pos_j:
        return []
    return [
        {
            "pos_i": pos_i,
            "pos_j": pos_j,
            "bin_idx": pair_bin_idx_from_distance_angstrom(
                CYCLIZATION_CA_PAIR_TARGET_ANGSTROM
            ),
            "target_distance_angstrom": CYCLIZATION_CA_PAIR_TARGET_ANGSTROM,
        }
    ]


def _head_tail_cyclization_from_topology(
    *,
    total_len: int,
    topology_entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any] | None:
    for entry in topology_entries or []:
        if not isinstance(entry, Mapping):
            continue
        topo_type = _normalize_mode_name(entry.get("type", None))
        if topo_type != "head_to_tail":
            continue
        anchors = entry.get("anchors") or []
        if len(anchors) != 2:
            continue
        pos_i, pos_j = sorted(int(pos) for pos in anchors)
        if pos_i != 1 or pos_j != int(total_len):
            continue
        return {
            "mode": "pair",
            "positions": [1, int(total_len)],
            "shortcut_len": 1,
        }
    return None


def _skip_existing_pair_constraints(
    base_constraints: Sequence[Mapping[str, Any]],
    extra_constraints: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    existing_pairs = {
        tuple(sorted((int(entry["pos_i"]), int(entry["pos_j"]))))
        for entry in base_constraints or []
    }
    out = []
    for entry in extra_constraints or []:
        key = tuple(sorted((int(entry["pos_i"]), int(entry["pos_j"]))))
        if key in existing_pairs:
            continue
        out.append(dict(entry))
    return out


def _build_condition_metadata(
    *,
    mode_name: str | None,
    anchor_positions: Sequence[int] | None,
    pair_entries: Sequence[Mapping[str, Any]] | None,
    residue_constraints: Dict[int, int],
) -> Dict[str, Any]:
    anchor_positions = [int(pos) for pos in (anchor_positions or [])]
    pair_entries = list(pair_entries or [])

    anchor_positions_padded = [-1, -1, -1]
    residue_targets = [-1, -1, -1]
    residue_target_mask = [False, False, False]
    for idx, pos in enumerate(anchor_positions[:3]):
        anchor_positions_padded[idx] = int(pos)
        if int(pos) in residue_constraints:
            residue_targets[idx] = int(residue_constraints[int(pos)])
            residue_target_mask[idx] = True

    pair_positions = [[-1, -1], [-1, -1], [-1, -1]]
    pair_bin_idx = [0, 0, 0]
    pair_target_angstrom = [0.0, 0.0, 0.0]
    pair_valid_mask = [False, False, False]
    for idx, entry in enumerate(pair_entries[:3]):
        pair_positions[idx] = [int(entry["pos_i"]), int(entry["pos_j"])]
        pair_bin_idx[idx] = int(entry["bin_idx"])
        pair_target_angstrom[idx] = float(
            entry.get("target_distance_angstrom", 0.0) or 0.0
        )
        pair_valid_mask[idx] = True

    return {
        "condition_mode_id": int(design_condition_mode_to_id(mode_name)),
        "condition_mode_name": design_condition_mode_to_name(mode_name),
        "condition_anchor_positions": anchor_positions_padded,
        "condition_pair_positions": pair_positions,
        "condition_pair_bin_idx": pair_bin_idx,
        "condition_pair_target_angstrom": pair_target_angstrom,
        "condition_pair_valid_mask": pair_valid_mask,
        "condition_residue_targets": residue_targets,
        "condition_residue_target_mask": residue_target_mask,
    }


def _sample_condition_sampler(
    *,
    task: Dict[str, Any],
    total_len: int,
    task_name: str,
    forbidden_positions: set[int] | None = None,
    rng: random.Random | None = None,
) -> Dict[str, Any]:
    rng = rng or random
    sampler = task.get("sampler", None)
    if sampler is None:
        sampler = task.get("condition_sampler", None)
    if task_has_explicit_constraint_schema(task) and sampler is not None:
        logger.warning(
            "Task '{}' provides both sampler/condition_sampler and explicit constraint; ignoring sampler and respecting constraint.",
            task_name,
        )
        sampler = None
    if sampler is None:
        return _build_condition_metadata(
            mode_name=None,
            anchor_positions=[],
            pair_entries=[],
            residue_constraints={},
        ) | {
            "cyclization": None,
            "residue_type_constraints": {},
            "optional_ca_pair_constraints": [],
            "topology": [],
        }
    if isinstance(sampler, str):
        sampler = {"mode": sampler}
    if not isinstance(sampler, Mapping):
        raise ValueError(
            f"Task '{task_name}': sampler/condition_sampler must be a mapping or string when provided."
        )
    extra_sampler_keys = set(sampler.keys()) - {
        "mode",
        "type",
        "resname",
        "residue",
        "aa",
        "pair_policy",
        "placement",
        "anchors",
        "positions",
        "min_seq_sep",
        "min_sep",
        "max_seq_sep",
        "max_sep",
        "termini_probability",
        "terminal_probability",
        "target_distance_angstrom",
        "ca_pair_target_angstrom",
        "ca_pair_target_distance",
        "pair_target_angstrom",
    }
    if extra_sampler_keys:
        raise ValueError(
            f"Task '{task_name}': sampler/condition_sampler got unsupported keys. "
            f"Unexpected keys: {sorted(extra_sampler_keys)}"
        )

    forbidden_positions = set(int(pos) for pos in (forbidden_positions or set()))
    mode_name = _normalize_sampler_mode_name(
        sampler.get("mode", sampler.get("type", ""))
    )
    residue_type_constraints: Dict[int, int] = {}
    pair_entries: List[Dict[str, Any]] = []
    anchor_positions: List[int] = []
    cyclization: Dict[str, Any] | None = None
    topology_entries: List[Dict[str, Any]] = []

    def _sampler_int(*keys: str, default: int | None = None) -> int | None:
        for key in keys:
            if key in sampler and sampler[key] is not None:
                return int(sampler[key])
        return default

    def _sampler_float(*keys: str, default: float) -> float:
        for key in keys:
            if key in sampler and sampler[key] is not None:
                return float(sampler[key])
        return default

    def _explicit_anchor_pair() -> tuple[int, int] | None:
        anchors = sampler.get("anchors", sampler.get("positions", None))
        if anchors is None:
            return None
        if (
            not isinstance(anchors, Sequence)
            or isinstance(anchors, (str, bytes))
            or len(anchors) != 2
        ):
            raise ValueError(
                f"Task '{task_name}': disulfide sampler anchors/positions must be [i,j]."
            )
        pos_i = _parse_binder_position(
            anchors[0],
            total_len=total_len,
            field_name="sampler.anchors",
            task_name=task_name,
        )
        pos_j = _parse_binder_position(
            anchors[1],
            total_len=total_len,
            field_name="sampler.anchors",
            task_name=task_name,
        )
        if pos_i == pos_j:
            raise ValueError(
                f"Task '{task_name}': disulfide sampler anchors must be distinct."
            )
        pos_i, pos_j = sorted((pos_i, pos_j))
        if pos_i in forbidden_positions or pos_j in forbidden_positions:
            raise ValueError(
                f"Task '{task_name}': disulfide sampler anchors overlap forbidden motif positions."
            )
        return pos_i, pos_j

    def _valid_linear_pairs(
        offsets: Sequence[int] | None = None,
        *,
        min_sep: int | None = None,
        max_sep: int | None = None,
        exclude_termini: bool = False,
    ) -> List[tuple[int, int]]:
        candidates: List[tuple[int, int]] = []
        if offsets is None:
            min_sep = int(1 if min_sep is None else min_sep)
            max_sep = int(total_len - 1 if max_sep is None else max_sep)
            offsets = range(min_sep, max_sep + 1)
        for offset in offsets:
            offset = int(offset)
            max_start = total_len - offset
            for pos_i in range(1, max_start + 1):
                pos_j = pos_i + offset
                if pos_i < 1 or pos_j > total_len:
                    continue
                if exclude_termini and (pos_i == 1 or pos_j == total_len):
                    continue
                if pos_i in forbidden_positions or pos_j in forbidden_positions:
                    continue
                candidates.append((pos_i, pos_j))
        return candidates

    if mode_name == "none":
        metadata = _build_condition_metadata(
            mode_name=None,
            anchor_positions=[],
            pair_entries=[],
            residue_constraints={},
        )
        return metadata | {
            "cyclization": None,
            "residue_type_constraints": {},
            "optional_ca_pair_constraints": [],
            "topology": [],
        }
    if mode_name == "isopep":
        candidates = _valid_linear_pairs([3, 4, 7])
        if not candidates:
            raise ValueError(
                f"Task '{task_name}': no valid isopep condition positions for length {total_len}."
            )
        pos_i, pos_j = rng.choice(candidates)
        aa_i = _normalize_residue_type_constraint_aa("K")
        aa_j = _normalize_residue_type_constraint_aa(rng.choice(["D", "E"]))
        target_distance = _staple_target_distance_angstrom(pos_i, pos_j)
        residue_type_constraints = {pos_i: aa_i, pos_j: aa_j}
        pair_entries = [
            {
                "pos_i": pos_i,
                "pos_j": pos_j,
                "bin_idx": pair_bin_idx_from_distance_angstrom(target_distance),
                "target_distance_angstrom": target_distance,
            }
        ]
        anchor_positions = [pos_i, pos_j]
        topology_entries = [{"type": "stapled", "anchors": [pos_i, pos_j]}]
        metadata_mode_name = "stapled"
    elif mode_name in {"alkene_staple", "alkane_staple"}:
        candidates = _valid_linear_pairs([3, 4])
        if not candidates:
            raise ValueError(
                f"Task '{task_name}': no valid {mode_name} condition positions for length {total_len}."
            )
        pos_i, pos_j = rng.choice(candidates)
        requested_resname = sampler.get(
            "resname", sampler.get("residue", sampler.get("aa", None))
        )
        if requested_resname is not None:
            resname = str(requested_resname).strip().upper()
            if resname not in {"MK8", "NLE"}:
                raise ValueError(
                    f"Task '{task_name}': {mode_name} sampler resname must be MK8 or NLE, got {requested_resname!r}."
                )
        else:
            resname = rng.choice(["MK8", "NLE"])
        topology_type = "mk8_staple" if resname == "MK8" else "nle_staple"
        aa_idx = _normalize_residue_type_constraint_aa(resname)
        target_distance = _staple_target_distance_angstrom(pos_i, pos_j)
        residue_type_constraints = {pos_i: aa_idx, pos_j: aa_idx}
        pair_entries = [
            {
                "pos_i": pos_i,
                "pos_j": pos_j,
                "bin_idx": pair_bin_idx_from_distance_angstrom(target_distance),
                "target_distance_angstrom": target_distance,
            }
        ]
        anchor_positions = [pos_i, pos_j]
        topology_entries = [{"type": topology_type, "anchors": [pos_i, pos_j]}]
        metadata_mode_name = "stapled"
    elif mode_name == "head_to_tail":
        pos_i, pos_j = 1, total_len
        if pos_i in forbidden_positions or pos_j in forbidden_positions:
            raise ValueError(
                f"Task '{task_name}': head-to-tail condition overlaps forbidden motif positions."
            )
        anchor_positions = [pos_i, pos_j]
        cyclization = {
            "mode": "pair",
            "positions": [pos_i, pos_j],
            "shortcut_len": 1,
        }
        topology_entries = [{"type": "head_to_tail", "anchors": [pos_i, pos_j]}]
    elif mode_name == "disulfide":
        explicit_pair = _explicit_anchor_pair()
        if explicit_pair is not None:
            pos_i, pos_j = explicit_pair
        else:
            pair_policy = (
                str(sampler.get("pair_policy", sampler.get("placement", "boltzgen")))
                .strip()
                .lower()
            )
            pair_policy = pair_policy.replace("-", "_")
            deterministic_pair: tuple[int, int] | None = None
            if pair_policy in {
                "boltzgen",
                "fixed_internal",
                "second_penultimate",
                "second_to_penultimate",
            }:
                if total_len < 4:
                    raise ValueError(
                        f"Task '{task_name}': BoltzGen-style disulfide anchors [2, L-1] "
                        f"require binder length >= 4, got {total_len}."
                    )
                fixed_pair = (2, total_len - 1)
                if (
                    fixed_pair[0] in forbidden_positions
                    or fixed_pair[1] in forbidden_positions
                ):
                    raise ValueError(
                        f"Task '{task_name}': BoltzGen-style disulfide anchors {fixed_pair} "
                        "overlap forbidden motif positions."
                    )
                deterministic_pair = fixed_pair
                candidates = []
            elif pair_policy in {
                "terminal",
                "termini",
                "n_c",
                "n_to_c",
                "head_tail",
                "head_to_tail",
            }:
                terminal_pair = (1, total_len)
                if (
                    terminal_pair[0] in forbidden_positions
                    or terminal_pair[1] in forbidden_positions
                ):
                    raise ValueError(
                        f"Task '{task_name}': terminal disulfide overlaps forbidden motif positions."
                    )
                candidates = [terminal_pair]
            else:
                min_sep = _sampler_int("min_seq_sep", "min_sep", default=3)
                max_sep = _sampler_int("max_seq_sep", "max_sep", default=total_len - 1)
                if min_sep is None or min_sep < 1:
                    raise ValueError(
                        f"Task '{task_name}': disulfide min_seq_sep must be >= 1."
                    )
                if max_sep is None:
                    max_sep = total_len - 1
                max_sep = min(int(max_sep), total_len - 1)
                if min_sep > max_sep:
                    raise ValueError(
                        f"Task '{task_name}': disulfide min_seq_sep ({min_sep}) "
                        f"cannot exceed max_seq_sep ({max_sep})."
                    )
                candidates = _valid_linear_pairs(min_sep=min_sep, max_sep=max_sep)
                if pair_policy in {"internal", "random_internal"}:
                    candidates = [pair for pair in candidates if pair != (1, total_len)]
                elif pair_policy in {"random", "uniform", "mixed", "auto"}:
                    pass
                else:
                    raise ValueError(
                        f"Task '{task_name}': unsupported disulfide pair_policy '{pair_policy}'. "
                        "Use boltzgen, mixed, random, internal, termini, or explicit anchors."
                    )
            if deterministic_pair is not None:
                pos_i, pos_j = deterministic_pair
            else:
                if not candidates:
                    raise ValueError(
                        f"Task '{task_name}': no valid disulfide condition positions for length {total_len}."
                    )
                terminal_pair = (1, total_len)
                terminal_probability = _sampler_float(
                    "termini_probability",
                    "terminal_probability",
                    default=0.6,
                )
                if (
                    pair_policy in {"mixed", "auto"}
                    and terminal_pair in candidates
                    and rng.random() < max(0.0, min(1.0, terminal_probability))
                ):
                    pos_i, pos_j = terminal_pair
                else:
                    pos_i, pos_j = rng.choice(candidates)
        aa_c = _normalize_residue_type_constraint_aa("C")
        target_distance = _sampler_float(
            "ca_pair_target_angstrom",
            "ca_pair_target_distance",
            "pair_target_angstrom",
            "target_distance_angstrom",
            default=DISULFIDE_CA_PAIR_TARGET_ANGSTROM,
        )
        residue_type_constraints = {pos_i: aa_c, pos_j: aa_c}
        pair_entries = [
            {
                "pos_i": pos_i,
                "pos_j": pos_j,
                "bin_idx": pair_bin_idx_from_distance_angstrom(target_distance),
                "target_distance_angstrom": target_distance,
            }
        ]
        anchor_positions = [pos_i, pos_j]
        topology_entries = [{"type": "disulfide", "anchors": [pos_i, pos_j]}]
    elif mode_name == "bicyclic":
        offset_a, offset_b = 6, 12
        candidates: List[tuple[int, int, int]] = []
        for pos_i in range(1, total_len + 1):
            pos_j = pos_i + offset_a
            pos_k = pos_i + offset_b
            if pos_k > total_len:
                continue
            if any(pos in forbidden_positions for pos in (pos_i, pos_j, pos_k)):
                continue
            candidates.append((pos_i, pos_j, pos_k))
        if not candidates:
            raise ValueError(
                f"Task '{task_name}': no valid bicycle condition positions for length {total_len}."
            )
        pos_i, pos_j, pos_k = rng.choice(candidates)
        aa_c = _normalize_residue_type_constraint_aa("C")
        target_distance = 8.0
        residue_type_constraints = {pos_i: aa_c, pos_j: aa_c, pos_k: aa_c}
        pair_bin = pair_bin_idx_from_distance_angstrom(target_distance)
        pair_entries = [
            {
                "pos_i": pos_i,
                "pos_j": pos_j,
                "bin_idx": pair_bin,
                "target_distance_angstrom": target_distance,
            },
            {
                "pos_i": pos_i,
                "pos_j": pos_k,
                "bin_idx": pair_bin,
                "target_distance_angstrom": target_distance,
            },
            {
                "pos_i": pos_j,
                "pos_j": pos_k,
                "bin_idx": pair_bin,
                "target_distance_angstrom": target_distance,
            },
        ]
        anchor_positions = [pos_i, pos_j, pos_k]
        topology_entries = [{"type": "bicycle", "anchors": [pos_i, pos_j, pos_k]}]
        metadata_mode_name = "bicycle"
    else:
        raise ValueError(
            f"Task '{task_name}': unsupported sampler/condition_sampler mode '{mode_name}'. "
            "Supported modes: linear, head_to_tail, disulfide, isopep, bicyclic, "
            "alkene-staple, alkane-staple."
        )

    metadata_mode_name = locals().get("metadata_mode_name", mode_name)
    metadata = _build_condition_metadata(
        mode_name=metadata_mode_name,
        anchor_positions=anchor_positions,
        pair_entries=pair_entries,
        residue_constraints=residue_type_constraints,
    )
    return metadata | {
        "cyclization": cyclization,
        "residue_type_constraints": residue_type_constraints,
        "optional_ca_pair_constraints": [
            {
                "pos_i": int(entry["pos_i"]),
                "pos_j": int(entry["pos_j"]),
                "bin_idx": int(entry["bin_idx"]),
                **(
                    {
                        "target_distance_angstrom": float(
                            entry["target_distance_angstrom"]
                        )
                    }
                    if "target_distance_angstrom" in entry
                    else {}
                ),
            }
            for entry in pair_entries
        ],
        "topology": topology_entries,
    }


def _is_head_tail_cyclic_sample(
    *,
    total_len: int,
    cyclization: Dict[str, Any] | None,
    topology_entries: Sequence[Mapping[str, Any]] | None,
) -> bool:
    for entry in topology_entries or []:
        topo_type = _normalize_mode_name(entry.get("type", None))
        anchors = [int(v) for v in entry.get("anchors", [])]
        if (
            topo_type == "head_to_tail"
            and len(anchors) == 2
            and sorted(anchors) == [1, int(total_len)]
        ):
            return True
    if cyclization is None:
        return False
    positions = cyclization.get("positions", None)
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or len(positions) != 2
    ):
        return False
    pos_a, pos_b = sorted(int(v) for v in positions)
    shortcut_len = int(cyclization.get("shortcut_len", 1))
    return pos_a == 1 and pos_b == int(total_len) and shortcut_len == 1


def _uses_head_tail_offset_cyclization(
    *,
    total_len: int,
    cyclization: Dict[str, Any] | None,
) -> bool:
    if cyclization is None:
        return False
    positions = cyclization.get("positions", None)
    if (
        not isinstance(positions, Sequence)
        or isinstance(positions, (str, bytes))
        or len(positions) != 2
    ):
        return False
    pos_a, pos_b = sorted(int(v) for v in positions)
    shortcut_len = int(cyclization.get("shortcut_len", 1))
    return pos_a == 1 and pos_b == int(total_len) and shortcut_len == 1


def parse_nlens_cfg(task):
    lens_cfg = task["lens_cfg"]
    n_samples = task["sample_num"]
    if lens_cfg["nres"]:
        nres = [[int(n) for _ in range(n_samples)] for n in lens_cfg["nres"]]
    elif lens_cfg["min_len"]:
        nres = np.arange(
            lens_cfg["min_len"], lens_cfg["max_len"] + 1, lens_cfg["step_len"]
        ).tolist()
        nres = [[n for _ in range(n_samples)] for n in nres]
    else:
        nres = None
    if nres is not None:
        nres = list(itertools.chain.from_iterable(nres))
    return nres  # a list indicates peptide length to generate for denovo design task


def generate_combinations(min_cost, max_cost, ranges):
    result = []
    ranges = [[x] if isinstance(x, int) else range(x[0], x[1] + 1) for x in ranges]
    for combination in itertools.product(*ranges):
        total_cost = sum(combination)
        if min_cost <= total_cost <= max_cost:
            padded_combination = list(combination) + [0] * (
                len(ranges) - len(combination)
            )
            result.append(padded_combination)
    return result


_MOTIF_CONTIG_ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"


_REFERENCE_MOTIF_CONTIG = "reference"


def _motif_cfg_layout(motif_cfg: Mapping[str, Any]):
    """Parse one explicit motif contig into the same complete runtime choices."""
    contig = str(motif_cfg["contigs"])
    min_length = int(motif_cfg["min_len"])
    max_length = int(motif_cfg["max_len"])
    components = contig.split("/")
    ranges = []
    motif_length = 0
    orig_motif_indices = []
    for part in components:
        if not part:
            raise ValueError("Motif contigs must not contain empty components.")
        if part[0] in _MOTIF_CONTIG_ALPHABET:
            if "-" in part:
                start, end = map(int, part[1:].split("-"))
            else:
                start = end = int(part[1:])
            length = end - start + 1
            motif_length += length
            for res_id in range(start, end + 1):
                orig_motif_indices.append((part[0], res_id))
        elif "-" in part:
            bounds = part.split("-")
            assert int(bounds[0]) <= int(bounds[-1])
            ranges.append((int(bounds[0]), int(bounds[-1])))
        else:
            ranges.append(int(part))
    combinations = generate_combinations(
        min_length - motif_length,
        max_length - motif_length,
        ranges,
    )
    if not combinations:
        raise ValueError(
            "No Motif combinations to sample from please update the max and min lengths"
        )
    return components, motif_length, orig_motif_indices, combinations


def _possible_motif_total_lengths(
    task: Mapping[str, Any],
    *,
    task_name: str,
    inspect_reference: bool,
) -> set[int] | None:
    """Return every total length reachable by runtime motif construction."""
    motif_cfg = task.get("motif_cfg", None)
    if not isinstance(motif_cfg, Mapping) or motif_cfg.get("contigs", None) is None:
        return None
    if motif_cfg["contigs"] == _REFERENCE_MOTIF_CONTIG:
        if not inspect_reference:
            return None
        reference = _load_normalized_structure(
            os.path.abspath(str(task.get("pdb_path", "")))
        )
        pep_chain = str(task.get("pep_chain") or "L")
        pep_ca = reference[
            (reference.atom_name == "CA") & (reference.chain_id == pep_chain)
        ]
        if len(pep_ca) == 0:
            raise ValueError(
                f"Task '{task_name}': reference motif requires peptide CA atoms for chain '{pep_chain}'."
            )
        return {int(len(pep_ca))}
    _, motif_length, _, combinations = _motif_cfg_layout(motif_cfg)
    return {int(motif_length + sum(combination)) for combination in combinations}


def parse_motif_cfg(task):
    motif_cfg = task["motif_cfg"]
    nsamples = task["sample_num"]
    components, _, orig_motif_indices, combinations = _motif_cfg_layout(motif_cfg)

    overall_lengths = []
    motif_indices = []
    out_strs = []

    combos = random.choices(combinations, k=nsamples)
    for combo in combos:
        combo_idx = 0
        current_position = 1  # Start positions at 1 for 1-based indexing
        motif_index = []
        output_string = ""
        for part in components:
            if part[0] in _MOTIF_CONTIG_ALPHABET:
                # Motif part
                if "-" in part:
                    start, end = map(int, part[1:].split("-"))
                else:
                    start = end = int(part[1:])
                length = end - start + 1
                motif_index.extend(range(current_position, current_position + length))
                new_part = "L" + str(current_position)
                if length > 1:
                    new_part += "-" + str(current_position + length - 1)
                output_string += new_part + "/"
            else:
                # Scaffold part
                length = int(combo[combo_idx])
                combo_idx += 1
                output_string += str(length) + "/"
            current_position += (
                length  # Update the current position after processing each part
            )
        overall_lengths.append(
            current_position - 1
        )  # current_position is 1 past the last residue
        motif_indices.append(motif_index)
        out_strs.append(output_string[:-1])
    return (overall_lengths, motif_indices, out_strs, orig_motif_indices)


def sample_motif_sequence_mask_reference(
    n_res: int,
    motif_cfg: Dict[str, Any],
    rng: random.Random | None = None,
) -> List[bool]:
    rng = rng or random
    motif_min_pct_res = float(motif_cfg.get("motif_min_pct_res", 0.05))
    motif_max_pct_res = float(motif_cfg.get("motif_max_pct_res", 0.5))
    motif_min_n_seg = int(motif_cfg.get("motif_min_n_seg", 1))
    motif_max_n_seg = int(motif_cfg.get("motif_max_n_seg", 4))

    motif_n_res = int(
        rng.random() * (n_res * motif_max_pct_res - n_res * motif_min_pct_res)
        + n_res * motif_min_pct_res
    )
    motif_n_res = min(motif_n_res, n_res)

    high = min(motif_n_res, motif_max_n_seg)
    if high < motif_min_n_seg:
        motif_n_seg = 1
    else:
        motif_n_seg = int(rng.random() * (high - motif_min_n_seg + 1) + motif_min_n_seg)
    motif_n_seg = max(1, min(motif_n_seg, motif_n_res))

    if motif_n_seg > 1:
        cuts = sorted(rng.sample(range(1, motif_n_res), motif_n_seg - 1))
        indices = [0, *cuts, motif_n_res]
    else:
        indices = [0, motif_n_res]

    segment_lengths = [
        int(indices[i + 1] - indices[i]) for i in range(len(indices) - 1)
    ]
    total_gap_res = max(0, n_res - motif_n_res)
    gap_lengths = [0] * (motif_n_seg + 1)
    for _ in range(total_gap_res):
        gap_lengths[rng.randrange(len(gap_lengths))] += 1

    mask: List[bool] = []
    for seg_idx, seg_len in enumerate(segment_lengths):
        mask.extend([False] * gap_lengths[seg_idx])
        mask.extend([True] * seg_len)
    mask.extend([False] * gap_lengths[-1])

    if len(mask) != n_res:
        raise ValueError(
            f"Reference motif mask length mismatch: expected {n_res}, got {len(mask)}."
        )
    return mask


def mask_to_reference_contig(
    chain_id: str, res_ids: List[int], motif_mask: List[bool]
) -> str:
    parts: List[str] = []
    n = len(res_ids)
    i = 0
    while i < n:
        if not motif_mask[i]:
            j = i
            while j < n and not motif_mask[j]:
                j += 1
            parts.append(str(j - i))
            i = j
            continue

        start = i
        end = i
        while (
            end + 1 < n and motif_mask[end + 1] and res_ids[end + 1] == res_ids[end] + 1
        ):
            end += 1

        r0 = int(res_ids[start])
        r1 = int(res_ids[end])
        if r0 == r1:
            parts.append(f"{chain_id}{r0}")
        else:
            parts.append(f"{chain_id}{r0}-{r1}")
        i = end + 1

    return "/".join(parts) if parts else str(n)


def parse_reference_motif_cfg(
    task: Dict[str, Any],
    pdb_struc: struc.AtomArray,
    pep_chain: str,
):
    motif_cfg = task["motif_cfg"]
    nsamples = int(task["sample_num"])

    ca_mask = pdb_struc.atom_name == "CA"
    pep_mask = pdb_struc.chain_id == pep_chain
    pep_ca = pdb_struc[ca_mask & pep_mask]
    pep_res_ids = pep_ca.res_id.tolist()
    if len(pep_res_ids) == 0:
        raise ValueError(
            f"Reference motif requires peptide CA atoms for chain '{pep_chain}'."
        )

    overall_lengths = [len(pep_res_ids)] * nsamples
    motif_indices_list: List[List[int]] = []
    out_strs: List[str] = []
    orig_motif_indices_list: List[List[tuple]] = []

    for _ in range(nsamples):
        motif_mask = sample_motif_sequence_mask_reference(
            len(pep_res_ids),
            motif_cfg,
        )
        target_motif_idxs = [i + 1 for i, m in enumerate(motif_mask) if m]
        orig_motif_indices = [
            (pep_chain, int(pep_res_ids[i])) for i, m in enumerate(motif_mask) if m
        ]
        out_str = mask_to_reference_contig(pep_chain, pep_res_ids, motif_mask)

        motif_indices_list.append(target_motif_idxs)
        out_strs.append(out_str)
        orig_motif_indices_list.append(orig_motif_indices)

    return overall_lengths, motif_indices_list, out_strs, orig_motif_indices_list


def generate_hotspot_indices(contig: str):
    ALPHABET = "ABCDEFGHJKLMNOPQRSTUVWXYZ"
    components = contig.split("/")
    expanded_indices = []
    for part in components:
        assert part[0] in ALPHABET, (
            f"Hotspot contig part must start with a chain identifier from {ALPHABET}, got {part}"
        )
        if "-" in part:
            start, end = map(int, part[1:].split("-"))
        else:
            start = end = int(part[1:])
        for res_id in range(start, end + 1):
            expanded_indices.append((part[0], res_id))
    return expanded_indices


def annotate_hotspots(task):
    raw_pdb_struc = strucio.load_structure(task["pdb_path"])
    pdb_struc = _normalize_loaded_structure(raw_pdb_struc)
    raw_to_norm_residue_ids = _raw_to_normalized_residue_id_map(
        raw_pdb_struc, pdb_struc
    )
    raw_res_id_by_norm_key = {}
    for (raw_chain, raw_res_id), norm_keys in raw_to_norm_residue_ids.items():
        for norm_chain, norm_res_id in norm_keys:
            raw_res_id_by_norm_key[(str(norm_chain), int(norm_res_id))] = int(
                raw_res_id
            )
    pdb_struc.set_annotation(
        "raw_res_id",
        np.array(
            [
                raw_res_id_by_norm_key.get((str(chain), int(res_id)), int(res_id))
                for chain, res_id in zip(
                    pdb_struc.chain_id.tolist(), pdb_struc.res_id.tolist()
                )
            ],
            dtype=np.int32,
        ),
    )
    is_complex = task["is_complex"]
    pep_chain = task.get("pep_chain") or "L"
    receptor_ca_coords = pdb_struc[
        (pdb_struc.chain_id != pep_chain) & (pdb_struc.atom_name == "CA")
    ].coord
    centered_coord = pdb_struc.coord - receptor_ca_coords.mean(axis=0, keepdims=True)
    pdb_struc.coord = centered_coord
    if task["hotspots"] == "reference":
        assert is_complex, (
            "Reference hotspots can only be annotated automatically for complex structures."
        )
        cutoff = 8.0
        ca_mask = pdb_struc.atom_name == "CA"
        pep_mask = pdb_struc.chain_id == pep_chain
        rec_mask = ~pep_mask
        rec_ca_indices = np.where(ca_mask & rec_mask)[0]
        pep_ca_indices = np.where(ca_mask & pep_mask)[0]
        if len(rec_ca_indices) == 0 or len(pep_ca_indices) == 0:
            raise ValueError(
                f"Cannot annotate reference hotspots: missing CA atoms for receptor or peptide chain '{pep_chain}'. "
                f"Found receptor CA={len(rec_ca_indices)}, peptide CA={len(pep_ca_indices)}."
            )
        rec_ca_coords = pdb_struc.coord[rec_ca_indices]
        pep_ca_coords = pdb_struc.coord[pep_ca_indices]
        dists = cdist(rec_ca_coords, pep_ca_coords)
        is_hotspot_local = (dists < cutoff).any(axis=1)
        hotspot_indices = rec_ca_indices[is_hotspot_local]
        hotspot_res_ids = pdb_struc.res_id[hotspot_indices]
        hotspot_chain_ids = pdb_struc.chain_id[hotspot_indices]
        is_hotspot = np.zeros(len(pdb_struc), dtype=bool)
        hotspot_set = set(zip(hotspot_chain_ids, hotspot_res_ids))
        for chain, res in hotspot_set:
            match_mask = (pdb_struc.chain_id == chain) & (pdb_struc.res_id == res)
            is_hotspot[match_mask] = True
        pdb_struc.set_annotation("is_hotspot", is_hotspot)
    elif task["hotspots"] is not None:
        # Explicit hotspots are specified in the input PDB's residue numbering.
        # Map them onto the normalized residue ids used downstream.
        hotspot_indices = generate_hotspot_indices(task["hotspots"])
        is_hotspot = np.zeros(len(pdb_struc), dtype=bool)
        unresolved = []
        for chain, res_id in hotspot_indices:
            normalized_keys = raw_to_norm_residue_ids.get((str(chain), int(res_id)), [])
            if not normalized_keys:
                unresolved.append((str(chain), int(res_id)))
                continue
            for norm_chain, norm_res_id in normalized_keys:
                match_mask = (pdb_struc.chain_id == norm_chain) & (
                    pdb_struc.res_id == norm_res_id
                )
                is_hotspot[match_mask] = True
        if unresolved:
            logger.warning(
                f"Explicit hotspot residues not found after structure normalization: {unresolved}. "
                "Check chain ids, residue ids, and whether the residues have complete N/CA/C/O backbone atoms."
            )
        pdb_struc.set_annotation("is_hotspot", is_hotspot)
    else:
        pdb_struc.set_annotation("is_hotspot", np.array([False] * len(pdb_struc)))
    return pdb_struc


def _output_residue_pdb_idx_by_key(
    rec_struc: struc.AtomArray,
    full_pep: struc.AtomArray,
) -> Dict[tuple[str, int], int]:
    """Map normalized graph residue keys to PDB residue ids used only for output."""
    mapping: Dict[tuple[str, int], int] = {}

    rec_ca = rec_struc[rec_struc.atom_name == "CA"]
    if len(rec_ca) > 0:
        if "raw_res_id" in rec_ca.get_annotation_categories():
            rec_output_ids = rec_ca.get_annotation("raw_res_id").tolist()
        else:
            rec_output_ids = rec_ca.res_id.tolist()
        for chain, res_id, output_res_id in zip(
            rec_ca.chain_id.tolist(),
            rec_ca.res_id.tolist(),
            rec_output_ids,
        ):
            mapping[(str(chain), int(res_id))] = int(output_res_id)

    pep_ca = full_pep[full_pep.atom_name == "CA"]
    for chain, res_id in zip(pep_ca.chain_id.tolist(), pep_ca.res_id.tolist()):
        mapping[(str(chain), int(res_id))] = int(res_id)

    return mapping


def validate_task(task, task_name):
    task_type = str(task.get("task_type", "")).strip().lower()
    if task_type not in SUPPORTED_ACTIVE_TASK_TYPES:
        raise ValueError(
            f"Task '{task_name}': unsupported task_type '{task.get('task_type', None)}'. "
            f"Supported active task types: {sorted(SUPPORTED_ACTIVE_TASK_TYPES)}."
        )
    is_complex = task["is_complex"]
    has_sequence = task.get("sequence", None) is not None
    hotspots = task.get("hotspots", None)
    motif_cfg = task.get("motif_cfg", {})
    constraint = task.get("constraint", None)
    sampler = task.get("sampler", None)
    topology = task.get("topology", None)
    postprocess = task.get("postprocess", None)
    residue_type_constraints = task.get("residue_type_constraints", None)
    cyclization = task.get("cyclization", None)
    optional_ca_pair_constraints = task.get("optional_ca_pair_constraints", None)
    condition_sampler = task.get("condition_sampler", None)
    sampler_allowed_keys = {
        "mode",
        "type",
        "resname",
        "residue",
        "aa",
        "pair_policy",
        "placement",
        "anchors",
        "positions",
        "min_seq_sep",
        "min_sep",
        "max_seq_sep",
        "max_sep",
        "termini_probability",
        "terminal_probability",
        "target_distance_angstrom",
        "ca_pair_target_angstrom",
        "ca_pair_target_distance",
        "pair_target_angstrom",
    }

    if isinstance(motif_cfg, Mapping):
        motif_contigs = motif_cfg.get("contigs", None)
        if (
            isinstance(motif_contigs, str)
            and motif_contigs.strip().lower() == _REFERENCE_MOTIF_CONTIG
            and motif_contigs != _REFERENCE_MOTIF_CONTIG
        ):
            raise ValueError(
                f"Task '{task_name}': motif_cfg.contigs reference sentinel must be exactly "
                f"'{_REFERENCE_MOTIF_CONTIG}'."
            )

    if "ref_center" in task:
        raise ValueError(
            f"Task '{task_name}': `ref_center` has been removed from the task schema. "
            "Use receptor-centered defaults; do not specify ref_center manually."
        )
    if "cyclic" in task:
        raise ValueError(
            f"Task '{task_name}': `cyclic` has been removed from the task schema. "
            "Use `sampler: head_to_tail` or explicit `constraint.cyclize + topology` instead."
        )

    if constraint is not None and not isinstance(constraint, Mapping):
        raise ValueError(
            f"Task '{task_name}': constraint must be a mapping when provided."
        )
    if sampler is not None and not isinstance(sampler, (str, Mapping)):
        raise ValueError(
            f"Task '{task_name}': sampler must be a string or mapping when provided."
        )
    if isinstance(sampler, Mapping):
        extra_sampler_keys = set(sampler.keys()) - sampler_allowed_keys
        if extra_sampler_keys:
            raise ValueError(
                f"Task '{task_name}': sampler has unsupported keys. "
                f"Unexpected keys: {sorted(extra_sampler_keys)}"
            )
    if topology is not None and (
        not isinstance(topology, Sequence) or isinstance(topology, (str, bytes))
    ):
        raise ValueError(f"Task '{task_name}': topology must be a list when provided.")
    if postprocess is not None and not isinstance(postprocess, (bool, Mapping)):
        raise ValueError(
            f"Task '{task_name}': postprocess must be bool or mapping when provided."
        )

    if residue_type_constraints is not None and (
        not isinstance(residue_type_constraints, Sequence)
        or isinstance(residue_type_constraints, (str, bytes))
    ):
        raise ValueError(
            f"Task '{task_name}': residue_type_constraints must be a list when provided."
        )
    if cyclization is not None and not isinstance(cyclization, Mapping):
        raise ValueError(
            f"Task '{task_name}': cyclization must be a mapping when provided."
        )
    if optional_ca_pair_constraints is not None and (
        not isinstance(optional_ca_pair_constraints, Sequence)
        or isinstance(optional_ca_pair_constraints, (str, bytes))
    ):
        raise ValueError(
            f"Task '{task_name}': optional_ca_pair_constraints must be a list when provided."
        )
    if condition_sampler is not None and not isinstance(
        condition_sampler, (str, Mapping)
    ):
        raise ValueError(
            f"Task '{task_name}': condition_sampler must be a string or mapping when provided."
        )
    if isinstance(condition_sampler, Mapping):
        extra_sampler_keys = set(condition_sampler.keys()) - sampler_allowed_keys
        if extra_sampler_keys:
            raise ValueError(
                f"Task '{task_name}': condition_sampler has unsupported keys. "
                f"Unexpected keys: {sorted(extra_sampler_keys)}"
            )
    if sampler is not None and condition_sampler is not None:
        raise ValueError(
            f"Task '{task_name}': use either sampler or condition_sampler, not both."
        )

    if task_type == "motif":
        if not is_complex:
            raise ValueError(
                f"Task '{task_name}': is_complex must be True for motif tasks."
            )

    if task_type == "inv_folding":
        if not is_complex:
            raise ValueError(
                f"Task '{task_name}': is_complex must be True for inv_folding tasks."
            )
        if hotspots is None:
            logger.info(
                f"Task '{task_name}': hotspots is set to None, but hotspots can be specified for inv_folding tasks."
            )

    if task_type == "docking":
        if not is_complex and not has_sequence:
            raise ValueError(
                f"Task '{task_name}': is_complex must be True for docking tasks unless sequence is provided."
            )
        if (
            has_sequence
            and isinstance(motif_cfg, Mapping)
            and motif_cfg.get("contigs", None) is not None
        ):
            raise ValueError(
                f"Task '{task_name}': sequence-based docking cannot be combined with "
                "motif_cfg.contigs. Use sequence docking without motif contigs, or "
                "reference/contig docking without sequence."
            )
        if has_sequence and hotspots == "reference":
            raise ValueError(
                f"Task '{task_name}': hotspots='reference' requires a reference peptide chain and "
                "is not supported for sequence-based docking tasks."
            )
        if motif_cfg and len(motif_cfg) > 0:
            logger.info(
                f"Task '{task_name}': motif_cfg is not null for docking tasks, u must know what you are doing!"
            )

    if hotspots == "reference" and not is_complex:
        raise ValueError(
            f"Task '{task_name}': is_complex must be True when hotspots is 'reference'."
        )


def build_atom_array(task, task_name):
    validate_task(task, task_name)
    ret_list = []
    task_type = str(task["task_type"]).strip().lower()
    is_complex = task["is_complex"]
    pep_chain = task.get("pep_chain") or "L"
    sequence_proxy, sequence_nonstandard_templates = parse_sequence_with_nonstandard(
        task.get("sequence", None),
        task_name=task_name,
    )
    sequence_based_docking = task_type == "docking" and bool(sequence_proxy)
    reference_pdb_path = os.path.abspath(str(task["pdb_path"]))
    reference_receptor_center = _receptor_center_from_reference(
        _load_normalized_structure(reference_pdb_path),
        pep_chain,
    )
    if task_type == "inv_folding":
        task_type_short = "if"
    elif task_type == "docking":
        task_type_short = "dock"
    else:
        task_type_short = task_type
    pdb_struc = annotate_hotspots(task)
    rec_struc = pdb_struc[pdb_struc.chain_id != pep_chain]
    rec_struc.set_annotation(
        "is_motif", np.array([True] * len(rec_struc))
    )  # receptor must be motif
    ref_center = None
    inferred_nonstandard_templates = _infer_reference_nonstandard_templates(
        pdb_struc,
        pep_chain=pep_chain,
        task_type=task_type,
    )
    if sequence_based_docking:
        inferred_nonstandard_templates = []
    _pep_ca_coords = {}
    if task_type == "inv_folding":
        _all_pep_ca = pdb_struc[
            (pdb_struc.chain_id == pep_chain) & (pdb_struc.atom_name == "CA")
        ]
        _pep_ca_coords = {j + 1: _all_pep_ca.coord[j] for j in range(len(_all_pep_ca))}
    elif task_type in ("docking", "packing"):
        _pep_ca_coords = {}

    if task_type in ("motif", "inv_folding", "docking", "packing"):
        if not (is_complex or sequence_based_docking):
            raise ValueError(
                f"Task '{task_name}': is_complex must be True unless sequence-based docking is used."
            )
        motif_cfg = task.get("motif_cfg") or {}
        contigs = motif_cfg.get("contigs") if motif_cfg else None

        if not motif_cfg or contigs is None:
            pep_ca = pdb_struc[
                (pdb_struc.chain_id == pep_chain) & (pdb_struc.atom_name == "CA")
            ]
            nsamples = int(task["sample_num"])
            if sequence_based_docking:
                overall_lengths = [len(sequence_proxy)] * nsamples
            else:
                if len(pep_ca) == 0:
                    raise ValueError(
                        f"Task '{task_name}': docking task has no peptide CA atoms in chain '{pep_chain}'. "
                        "Provide sequence for sequence-based docking or a complex PDB with pep_chain."
                    )
                overall_lengths = [len(pep_ca)] * nsamples
            motif_indices_list = [[] for _ in range(nsamples)]
            out_strs = [None] * nsamples
            orig_motif_indices_list = [[] for _ in range(nsamples)]
        elif contigs == _REFERENCE_MOTIF_CONTIG:
            overall_lengths, motif_indices_list, out_strs, orig_motif_indices_list = (
                parse_reference_motif_cfg(
                    task=task, pdb_struc=pdb_struc, pep_chain=pep_chain
                )
            )
        else:
            overall_lengths, motif_indices_list, out_strs, orig_motif_indices = (
                parse_motif_cfg(task)
            )
            orig_motif_indices_list = [
                orig_motif_indices for _ in range(len(overall_lengths))
            ]
    else:
        overall_lengths = parse_nlens_cfg(task)
        if overall_lengths is None:
            raise ValueError(
                "Invalid lens_cfg for non-motif task: expected either 'nres' or "
                "'min_len'/'max_len'/'step_len' to be set."
            )
        motif_indices_list = [[] for _ in range(len(overall_lengths))]
        out_strs = [None] * len(overall_lengths)
        orig_motif_indices_list = [[] for _ in range(len(overall_lengths))]
    binder_secondary_classes = resolve_binder_secondary_classes(
        task=task,
        task_name=task_name,
        n_outputs=len(overall_lengths),
    )
    for i in range(len(overall_lengths)):
        key_name = f"{task_type_short}_{task_name}_sample_{i}"
        binder_secondary_class = binder_secondary_classes[i]
        out_str = out_strs[i] if out_strs[i] is not None else None
        total_len = overall_lengths[i]
        binder_topology = resolve_binder_topology(
            task=task,
            total_len=total_len,
            task_name=task_name,
            sample_index=i,
        )
        scaffold_condition = dict(binder_topology.pop("_scaffold_condition", {}) or {})
        target_motif_idxs = motif_indices_list[i]
        orig_motif_indices = orig_motif_indices_list[i]
        explicit_constraint_schema = task_has_explicit_constraint_schema(task)
        nonstandard_residue_templates: List[Dict[str, Any]] = []
        legacy_nonstandard_residue_templates: List[Dict[str, Any]] = []
        if explicit_constraint_schema:
            compiled_constraints = parse_constraint_schema(
                task=task,
                total_len=total_len,
                task_name=task_name,
            )
            residue_type_constraints = compiled_constraints["residue_type_constraints"]
            nonstandard_residue_templates = list(
                compiled_constraints.get("nonstandard_residue_templates", [])
            )
            cyclization = compiled_constraints["cyclization"]
            optional_ca_pair_constraints = compiled_constraints[
                "optional_ca_pair_constraints"
            ]
        else:
            residue_type_constraints = parse_residue_type_constraints(
                task=task,
                total_len=total_len,
                task_name=task_name,
            )
            legacy_nonstandard_residue_templates = (
                parse_nonstandard_residue_type_templates(
                    task=task,
                    total_len=total_len,
                    task_name=task_name,
                )
            )
            cyclization = parse_cyclization(
                task=task,
                total_len=total_len,
                task_name=task_name,
            )
            optional_ca_pair_constraints = parse_optional_ca_pair_constraints(
                task=task,
                total_len=total_len,
                task_name=task_name,
            )
        topology_entries = parse_topology_schema(
            task=task,
            total_len=total_len,
            task_name=task_name,
        )
        if cyclization is not None:
            positions = cyclization.get("positions", None)
            if (
                isinstance(positions, Sequence)
                and not isinstance(positions, (str, bytes))
                and len(positions) == 2
                and sorted(int(pos) for pos in positions) == [1, int(total_len)]
            ):
                topology_entries = _merge_topology_entries(
                    topology_entries,
                    [{"type": "head_to_tail", "anchors": [1, int(total_len)]}],
                    task_name=task_name,
                    source_name="cyclization",
                )
        topology_entries = _merge_nonstandard_residue_templates(
            topology_entries,
            list(inferred_nonstandard_templates)
            + list(sequence_nonstandard_templates)
            + list(nonstandard_residue_templates)
            + list(legacy_nonstandard_residue_templates),
        )
        residue_type_constraints = _merge_residue_type_constraints(
            residue_type_constraints,
            scaffold_condition.get("residue_type_constraints", {}) or {},
            task_name=task_name,
        )
        cyclization = _merge_cyclization(
            cyclization,
            scaffold_condition.get("cyclization", None),
            task_name=task_name,
            source_name="scaffold",
        )
        optional_ca_pair_constraints = _merge_optional_pair_constraints(
            optional_ca_pair_constraints,
            scaffold_condition.get("optional_ca_pair_constraints", []) or [],
            task_name=task_name,
        )
        topology_entries = _merge_topology_entries(
            topology_entries,
            scaffold_condition.get("topology_entries", []) or [],
            task_name=task_name,
            source_name="scaffold",
        )
        sampled_condition = _sample_condition_sampler(
            task=task,
            total_len=total_len,
            task_name=task_name,
            forbidden_positions=set(target_motif_idxs),
        )
        omit_aa_indices = normalize_omit_aa(
            task.get("omit_aa", None), task_name=task_name
        )
        sampled_cyclization = sampled_condition.get("cyclization", None)
        sampled_topology_entries = list(sampled_condition.get("topology", []))
        residue_type_constraints = _merge_residue_type_constraints(
            residue_type_constraints,
            sampled_condition["residue_type_constraints"],
            task_name=task_name,
        )
        if sampled_cyclization is not None:
            cyclization = _merge_cyclization(
                cyclization,
                sampled_cyclization,
                task_name=task_name,
                source_name="sampler",
            )
        optional_ca_pair_constraints = _merge_optional_pair_constraints(
            optional_ca_pair_constraints,
            sampled_condition["optional_ca_pair_constraints"],
            task_name=task_name,
        )
        topology_entries = _merge_topology_entries(
            topology_entries,
            sampled_topology_entries,
            task_name=task_name,
            source_name="sampler",
        )
        topology_cyclization = _head_tail_cyclization_from_topology(
            total_len=total_len,
            topology_entries=topology_entries,
        )
        cyclization = _merge_cyclization(
            cyclization,
            topology_cyclization,
            task_name=task_name,
            source_name="topology",
        )
        optional_ca_pair_constraints = _merge_optional_pair_constraints(
            optional_ca_pair_constraints,
            _skip_existing_pair_constraints(
                optional_ca_pair_constraints,
                _cyclization_ca_pair_constraint(cyclization),
            ),
            task_name=task_name,
        )
        condition_metadata = topology_entries_to_condition_metadata(
            topology_entries=topology_entries,
            residue_constraints=residue_type_constraints,
            pair_constraints=optional_ca_pair_constraints,
        )
        inferred_cyclic = _is_head_tail_cyclic_sample(
            total_len=total_len,
            cyclization=cyclization,
            topology_entries=topology_entries,
        )
        uses_head_tail_offset_cyclization = _uses_head_tail_offset_cyclization(
            total_len=total_len,
            cyclization=cyclization,
        )
        postprocess_cfg = resolve_postprocess_contract(task, topology_entries)
        resolved_postprocess_enabled = bool(postprocess_cfg["resolved_enabled"])
        if task_type == "motif":
            overlap = sorted(set(target_motif_idxs) & set(residue_type_constraints))
            if overlap:
                logger.info(
                    "Task '{}': residue_type_constraints overlap motif-fixed binder positions {}; "
                    "keeping both so topology anchors remain explicit.",
                    task_name,
                    overlap,
                )
        motif_map = {}
        if len(target_motif_idxs) > 0:
            if len(target_motif_idxs) != len(orig_motif_indices):
                raise ValueError(
                    f"Sample {i}: Target motif length {len(target_motif_idxs)} != Source motif length {len(orig_motif_indices)}"
                )
            for t_idx, src_info in zip(target_motif_idxs, orig_motif_indices):
                motif_map[t_idx] = src_info
        residues_buffer = []
        for res_id in range(1, total_len + 1):
            if res_id in motif_map:
                src_chain, src_rid = motif_map[res_id]
                mask = (pdb_struc.chain_id == src_chain) & (pdb_struc.res_id == src_rid)
                motif_res = pdb_struc[mask].copy()
                if len(motif_res) == 0:
                    raise ValueError(
                        f"Motif residue {src_chain}:{src_rid} not found in PDB!"
                    )
                motif_res.chain_id[:] = "L"
                motif_res.res_id[:] = res_id
                motif_res.set_annotation("is_motif", np.array([True] * len(motif_res)))
                residues_buffer.append(motif_res)
            else:
                if sequence_based_docking:
                    aa = sequence_proxy[res_id - 1]
                    seq_res = create_proxy_sequence_residue(
                        "L",
                        res_id,
                        aa,
                        ca_coord=_pep_ca_coords.get(res_id),
                    )
                    residues_buffer.append(seq_res)
                elif task_type in ("docking", "packing"):
                    pep_residues = pdb_struc[(pdb_struc.chain_id == pep_chain)]
                    pep_res_ids = sorted(set(pep_residues.res_id))
                    if res_id - 1 < len(pep_res_ids):
                        src_rid = pep_res_ids[res_id - 1]
                        mask = (pdb_struc.chain_id == pep_chain) & (
                            pdb_struc.res_id == src_rid
                        )
                        real_res = pdb_struc[mask].copy()
                        real_res = _proxy_nonstandard_reference_residue(real_res)
                        real_res.chain_id[:] = "L"
                        real_res.res_id[:] = res_id
                        real_res.set_annotation(
                            "is_motif", np.array([False] * len(real_res))
                        )
                        residues_buffer.append(real_res)
                    else:
                        ala_res = create_std_ala_residue(
                            "L", res_id, ca_coord=_pep_ca_coords.get(res_id)
                        )
                        residues_buffer.append(ala_res)
                else:
                    ala_res = create_std_ala_residue(
                        "L", res_id, ca_coord=_pep_ca_coords.get(res_id)
                    )
                    residues_buffer.append(ala_res)
        full_pep = residues_buffer[0]
        for res in residues_buffer[1:]:
            full_pep += res
        complex_struc = rec_struc + full_pep
        ret_list.append(
            {
                "struc_input": complex_struc,
                "output_residue_pdb_idx_by_key": _output_residue_pdb_idx_by_key(
                    rec_struc, full_pep
                ),
                "contig": out_str,
                "task_name": task_name,
                "sample_fn": key_name,
                "ref_center": ref_center,
                "reference_pdb_path": reference_pdb_path,
                "reference_pep_chain": pep_chain,
                "reference_receptor_center": reference_receptor_center,
                "task_type": task_type,
                "pep_chain": pep_chain,
                "binder_chain": "L",
                "cyclic": bool(inferred_cyclic),
                "binder_secondary_class": binder_secondary_class,
                **binder_topology,
                "residue_type_constraints": residue_type_constraints,
                "cyclization": cyclization,
                "optional_ca_pair_constraints": optional_ca_pair_constraints,
                "condition_mode_id": int(condition_metadata["condition_mode_id"]),
                "condition_mode_name": condition_metadata["condition_mode_name"],
                "condition_anchor_positions": condition_metadata[
                    "condition_anchor_positions"
                ],
                "condition_pair_positions": condition_metadata[
                    "condition_pair_positions"
                ],
                "condition_pair_bin_idx": condition_metadata["condition_pair_bin_idx"],
                "condition_pair_target_angstrom": condition_metadata[
                    "condition_pair_target_angstrom"
                ],
                "condition_pair_valid_mask": condition_metadata[
                    "condition_pair_valid_mask"
                ],
                "condition_residue_targets": condition_metadata[
                    "condition_residue_targets"
                ],
                "condition_residue_target_mask": condition_metadata[
                    "condition_residue_target_mask"
                ],
                "topology_entries": topology_entries,
                "postprocess_enabled": bool(resolved_postprocess_enabled),
                "postprocess_method": str(postprocess_cfg["method"]),
                "postprocess_resolved_method": str(postprocess_cfg["resolved_method"]),
                "postprocess_relax_linear": bool(
                    postprocess_cfg.get("relax_linear", False)
                ),
                "omit_aa_indices": omit_aa_indices,
            }
        )
    return ret_list
