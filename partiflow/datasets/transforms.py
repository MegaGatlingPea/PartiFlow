import json
import random
import itertools
from typing import List, Literal, Tuple
import numpy as np
import torch
import einops
import biotite.structure as struc
from loguru import logger
from torch_geometric import transforms as T
from torch_geometric.data import Data

from partiflow.openfold.np import residue_constants
from partiflow.utils.coors_utils import ang_to_nm, nm_to_ang, sample_uniform_rotation
from partiflow.utils.conditioning_utils import (
    CHEMICAL_BOND_DISULFIDE,
    CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    CHEMICAL_BOND_ISOPEPTIDE,
    CHEMICAL_BOND_OTHER_COVALENT,
    chemical_bond_type_to_id,
    pair_bin_idx_from_distance_angstrom,
)
from partiflow.utils.pdb_utils import create_full_prot, to_pdb
from partiflow.utils.secstruct_utils import (
    atom_array_from_pdb_string,
    binder_secstruct_fold_code_from_id,
    classify_secondary_structure_metrics,
    secondary_structure_metrics_from_pdb_string,
)


def mean_w_mask(a, mask, keepdim=True):
    """
    Computes the mean of point cloud a accounting for the mask.

    Args:
        a: Input point cloud of shape [*, n, d]
        mask: Input mask of shape [*, n] of boolean values
        keepdim: whether to keep the dimension across which we're computing the mean
            like normal pytorch mean

    Returns:
        Masked mean of a across dimension -2 (or n)
    """
    mask = mask[..., None]  # [*, n, 1]
    num_elements = torch.sum(mask, dim=-2, keepdim=True)  # [*, 1, 1]
    num_elements = torch.where(
        num_elements == 0, torch.tensor(1.0), num_elements
    )  # [*, 1, 1]
    a_masked = torch.masked_fill(a, ~mask, 0.0)  # [*, n, d]
    mean = torch.sum(a_masked, dim=-2, keepdim=True) / num_elements  # [*, 1, d]
    mean = torch.masked_fill(mean, num_elements == 0, 0.0)  # [*, 1, d]
    if not keepdim:
        mean = einops.rearrange(mean, "... () d -> ... d")
    return mean


# if we add more transforms, consider calling this in the first place
class CopyCoordinatesTransform(T.BaseTransform):
    """Copies coords to coords_unmodified. Useful if other transforms like noising or rotations/translations are applied later on."""

    def __call__(self, graph: Data) -> Data:
        graph.coords_unmodified = graph.coords.clone()
        return graph


# gets coords in nanometers
class CoordsToNanometers(T.BaseTransform):
    """Gets cordinates in nanometers."""

    def __call__(self, graph: Data) -> Data:
        graph.coords_nm = ang_to_nm(graph.coords)
        return graph


class AutoDisulfideAnnotationTransform(T.BaseTransform):
    """Detect peptide CYS SG-SG bonds and expose them as forced disulfide conditions."""

    def __init__(
        self,
        min_sg_sg_distance_angstrom: float = 1.5,
        max_sg_sg_distance_angstrom: float = 2.3,
        min_sequence_separation: int = 1,
        scope: Literal["peptide", "all"] = "peptide",
        skip_if_topology_exists: bool = True,
    ):
        self.min_sg_sg_distance_angstrom = float(min_sg_sg_distance_angstrom)
        self.max_sg_sg_distance_angstrom = float(max_sg_sg_distance_angstrom)
        self.min_sequence_separation = int(min_sequence_separation)
        self.scope = str(scope)
        self.skip_if_topology_exists = bool(skip_if_topology_exists)
        if self.scope not in {"peptide", "all"}:
            raise ValueError(f"scope must be 'peptide' or 'all', got {scope!r}.")
        if self.min_sg_sg_distance_angstrom < 0.0:
            raise ValueError("min_sg_sg_distance_angstrom must be non-negative.")
        if self.max_sg_sg_distance_angstrom < self.min_sg_sg_distance_angstrom:
            raise ValueError("max_sg_sg_distance_angstrom must be >= min_sg_sg_distance_angstrom.")

    @staticmethod
    def _decode_topology_entries(raw_entries) -> list[dict]:
        if raw_entries is None:
            return []
        if isinstance(raw_entries, str):
            try:
                entries = json.loads(raw_entries)
            except (json.JSONDecodeError, ValueError):
                return []
        elif isinstance(raw_entries, list):
            entries = raw_entries
        else:
            return []
        if not isinstance(entries, list):
            return []
        return [entry for entry in entries if isinstance(entry, dict)]

    @staticmethod
    def _has_valid_forced_topology(graph: Data) -> bool:
        forced = getattr(graph, "forced_topology_anchor_resids", None)
        if forced is None:
            return False
        forced = torch.as_tensor(forced)
        return forced.numel() > 0 and bool((forced.long() >= 0).any().item())

    def _has_existing_topology(self, graph: Data) -> bool:
        entries = self._decode_topology_entries(getattr(graph, "topology_entries", None))
        return len(entries) > 0 or self._has_valid_forced_topology(graph)

    def _scope_mask(self, graph: Data, device: torch.device) -> torch.Tensor:
        num_res = int(graph.residue_type.shape[0])
        if self.scope == "all" or not hasattr(graph, "is_peptide"):
            return torch.ones(num_res, dtype=torch.bool, device=device)
        return graph.is_peptide.to(device=device).long() == 1

    @staticmethod
    def _set_empty_auto_fields(graph: Data, device: torch.device | None = None) -> None:
        if device is None:
            coords = getattr(graph, "coords", None)
            device = coords.device if isinstance(coords, torch.Tensor) else torch.device("cpu")
        num_res = int(graph.residue_type.shape[0]) if hasattr(graph, "residue_type") else 0
        graph.auto_disulfide_pair_count = torch.zeros((1,), dtype=torch.long, device=device)
        graph.auto_disulfide_anchor_resids = torch.empty((0, 2), dtype=torch.long, device=device)
        graph.auto_disulfide_sg_sg_distances_angstrom = torch.empty((0,), dtype=torch.float32, device=device)
        graph.auto_disulfide_ca_ca_distances_angstrom = torch.empty((0,), dtype=torch.float32, device=device)
        if not hasattr(graph, "topology_entries"):
            graph.topology_entries = "[]"
        if not hasattr(graph, "forced_topology_anchor_resids"):
            graph.forced_topology_anchor_resids = torch.empty((0, 2), dtype=torch.long, device=device)
        if not hasattr(graph, "forced_optional_ca_pair_indices"):
            graph.forced_optional_ca_pair_indices = torch.empty((0, 2), dtype=torch.long, device=device)
        if not hasattr(graph, "forced_optional_ca_pair_bin_idx"):
            graph.forced_optional_ca_pair_bin_idx = torch.empty((0,), dtype=torch.long, device=device)
        if not hasattr(graph, "forced_optional_ca_pair_valid_mask"):
            graph.forced_optional_ca_pair_valid_mask = torch.empty((0,), dtype=torch.bool, device=device)
        if num_res > 0 and not hasattr(graph, "forced_residue_type_condition"):
            graph.forced_residue_type_condition = torch.full((num_res,), fill_value=-1, dtype=torch.long, device=device)
        if num_res > 0 and not hasattr(graph, "forced_residue_type_condition_mask"):
            graph.forced_residue_type_condition_mask = torch.zeros(num_res, dtype=torch.bool, device=device)
        if num_res > 0 and not hasattr(graph, "chemical_bond_type"):
            graph.chemical_bond_type = torch.zeros((num_res, num_res), dtype=torch.long, device=device)

    @staticmethod
    def _resid_is_unique_in_scope(
        residue_pdb_idx: torch.Tensor,
        scope_mask: torch.Tensor,
        resid: int,
    ) -> bool:
        matches = (residue_pdb_idx.long() == int(resid)) & scope_mask
        return int(matches.sum().item()) == 1

    def __call__(self, graph: Data) -> Data:
        self._set_empty_auto_fields(graph)
        if self.skip_if_topology_exists and self._has_existing_topology(graph):
            return graph
        required = ("coords", "coord_mask", "residue_type", "residue_pdb_idx")
        if any(not hasattr(graph, name) for name in required):
            return graph

        coords = graph.coords
        device = coords.device
        self._set_empty_auto_fields(graph, device=device)
        coord_mask = graph.coord_mask.to(device=device).bool()
        residue_type = graph.residue_type.to(device=device).long()
        residue_pdb_idx = graph.residue_pdb_idx.to(device=device).long()
        num_res = int(residue_type.shape[0])
        ca_idx = residue_constants.atom_order["CA"]
        sg_idx = residue_constants.atom_order["SG"]
        cys_type = residue_constants.restype_order["C"]

        scope_mask = self._scope_mask(graph, device=device)
        has_atoms = coord_mask[:, ca_idx] & coord_mask[:, sg_idx]
        cys_mask = scope_mask & has_atoms & (residue_type == int(cys_type))
        cys_indices = torch.nonzero(cys_mask, as_tuple=True)[0].tolist()
        if len(cys_indices) < 2:
            return graph

        candidates = []
        for idx_i, idx_j in itertools.combinations(cys_indices, 2):
            if abs(idx_i - idx_j) < self.min_sequence_separation:
                continue
            resid_i = int(residue_pdb_idx[idx_i].item())
            resid_j = int(residue_pdb_idx[idx_j].item())
            if not self._resid_is_unique_in_scope(residue_pdb_idx, scope_mask, resid_i):
                continue
            if not self._resid_is_unique_in_scope(residue_pdb_idx, scope_mask, resid_j):
                continue
            sg_dist = float(torch.norm(coords[idx_i, sg_idx] - coords[idx_j, sg_idx], dim=-1).item())
            if not (
                self.min_sg_sg_distance_angstrom
                <= sg_dist
                <= self.max_sg_sg_distance_angstrom
            ):
                continue
            ca_dist = float(torch.norm(coords[idx_i, ca_idx] - coords[idx_j, ca_idx], dim=-1).item())
            candidates.append((sg_dist, ca_dist, idx_i, idx_j, resid_i, resid_j))
        if len(candidates) == 0:
            return graph

        candidates.sort(key=lambda item: item[0])
        selected = []
        used_residue_indices: set[int] = set()
        for candidate in candidates:
            _, _, idx_i, idx_j, _, _ = candidate
            if idx_i in used_residue_indices or idx_j in used_residue_indices:
                continue
            selected.append(candidate)
            used_residue_indices.update((idx_i, idx_j))
        if len(selected) == 0:
            return graph

        pair_indices = []
        anchor_resids = []
        pair_bins = []
        sg_distances = []
        ca_distances = []
        topology_entries = self._decode_topology_entries(getattr(graph, "topology_entries", None))
        for sg_dist, ca_dist, idx_i, idx_j, resid_i, resid_j in selected:
            pair_indices.append([idx_i, idx_j])
            anchor_resids.append([resid_i, resid_j])
            pair_bins.append(pair_bin_idx_from_distance_angstrom(ca_dist))
            sg_distances.append(sg_dist)
            ca_distances.append(ca_dist)
            topology_entries.append({"type": "disulfide", "anchors": [resid_i, resid_j]})

        graph.topology_entries = json.dumps(topology_entries)
        graph.forced_topology_anchor_resids = torch.tensor(anchor_resids, dtype=torch.long, device=device)
        graph.forced_optional_ca_pair_indices = torch.tensor(pair_indices, dtype=torch.long, device=device)
        graph.forced_optional_ca_pair_bin_idx = torch.tensor(pair_bins, dtype=torch.long, device=device)
        graph.forced_optional_ca_pair_valid_mask = torch.ones(len(pair_indices), dtype=torch.bool, device=device)

        graph.forced_residue_type_condition = torch.full((num_res,), fill_value=-1, dtype=torch.long, device=device)
        graph.forced_residue_type_condition_mask = torch.zeros(num_res, dtype=torch.bool, device=device)
        for idx_i, idx_j in pair_indices:
            graph.forced_residue_type_condition[[idx_i, idx_j]] = residue_type[[idx_i, idx_j]]
            graph.forced_residue_type_condition_mask[[idx_i, idx_j]] = True

        graph.chemical_bond_type = torch.zeros((num_res, num_res), dtype=torch.long, device=device)
        for idx_i, idx_j in pair_indices:
            graph.chemical_bond_type[int(idx_i), int(idx_j)] = CHEMICAL_BOND_DISULFIDE
            graph.chemical_bond_type[int(idx_j), int(idx_i)] = CHEMICAL_BOND_DISULFIDE

        graph.auto_disulfide_pair_count = torch.tensor([len(pair_indices)], dtype=torch.long, device=device)
        graph.auto_disulfide_anchor_resids = graph.forced_topology_anchor_resids.clone()
        graph.auto_disulfide_sg_sg_distances_angstrom = torch.tensor(sg_distances, dtype=torch.float32, device=device)
        graph.auto_disulfide_ca_ca_distances_angstrom = torch.tensor(ca_distances, dtype=torch.float32, device=device)
        return graph


# assert a chain break if either the distance between CA atoms is too large or if there is a change in is_peptide flag
class ChainBreakPerResidueTransform(T.BaseTransform):
    """Creates a binary mask indicating whether residue has chain break or not."""

    def __init__(
        self,
        chain_break_cutoff: float = 4.0,
    ):
        self.chain_break_cutoff = chain_break_cutoff

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        distance_chain_breaks = torch.cat([ca_dists > self.chain_break_cutoff, torch.tensor([False], dtype=torch.bool, device=ca_dists.device)])
        receptor_peptide_break = torch.cat([torch.abs(torch.diff(graph.is_peptide)), torch.tensor([False], dtype=torch.bool, device=graph.is_peptide.device)])
        chain_breaks_per_residue = distance_chain_breaks | receptor_peptide_break
        graph.chain_breaks_per_residue = chain_breaks_per_residue.bool()
        return graph
    

class CenterStructureTransform(T.BaseTransform):
    """Centers the structure on receptor CA coordinates with optional global translation."""

    def __init__(
        self,
        random_translation: bool = False,
        translation_strategy: Literal["normal", "uniform"] = "normal",
        translation_std_nm: float = 0.1,
        translation_max_nm: float | None = None,
    ):
        self.random_translation = bool(random_translation)
        self.translation_strategy = str(translation_strategy)
        self.translation_std_nm = float(translation_std_nm)
        self.translation_max_nm = (
            None if translation_max_nm is None else float(translation_max_nm)
        )
        if self.translation_strategy not in {"normal", "uniform"}:
            raise ValueError(
                f"translation_strategy must be 'normal' or 'uniform', got {translation_strategy!r}."
            )

    def __call__(self, graph: Data) -> Data:
        ca_coords = graph.coords_nm[:, 1, :]  # [n, 3]
        is_peptide = graph.is_peptide.bool()  # [n,]
        mask = ~is_peptide
        com = mean_w_mask(ca_coords, mask, keepdim=True)  # [1, 3]
        graph.coords_nm = graph.coords_nm - com[None, ...]  # [n, 37, 3] - [1, 3]
        if self.random_translation:
            if self.translation_strategy == "normal":
                translation = torch.randn(
                    3, dtype=graph.coords_nm.dtype, device=graph.coords_nm.device
                ) * self.translation_std_nm
            else:
                max_nm = self.translation_max_nm
                if max_nm is None:
                    max_nm = self.translation_std_nm
                translation = torch.empty(
                    3, dtype=graph.coords_nm.dtype, device=graph.coords_nm.device
                ).uniform_(-max_nm, max_nm)
            graph.coords_nm = graph.coords_nm + translation[None, None, :]
            graph.center_random_translation_nm = translation
        graph.coords = nm_to_ang(graph.coords_nm)
        return graph


# apply global rotation to the entire structure
class GlobalRotationTransform(T.BaseTransform):
    """Modifies the global rotation of the atom37 representation randomly.

    Should be used as the first transform in the pipeline that modifies coordinates in order to keep
    e.g. frame construction or other things consistent down the pipeline."""

    def __init__(self, rotation_strategy: Literal["uniform"] = "uniform"):
        self.rotation_strategy = rotation_strategy

    def __call__(self, graph: Data) -> Data:
        if self.rotation_strategy == "uniform":
            rot = sample_uniform_rotation(
                dtype=graph.coords_nm.dtype, device=graph.coords_nm.device
            )
        else:
            raise ValueError(
                f"Rotation strategy {self.rotation_strategy} not supported"
            )
        graph.coords_nm = torch.matmul(
            graph.coords_nm, rot
        )  # [n, 37, 3] * [3, 3] -> [n, 37, 3]
        graph.coords = nm_to_ang(graph.coords_nm)
        return graph


# should only add noise to peptide atoms. maybe we dont want this for now while testing on small datasets.
class StructureNoiseTransform(T.BaseTransform):
    """Adds noise to the coordinates of the peptide part of a complex structure.

    Sets the following attributes on the protein data object:
        coords_uncorrupted (torch.Tensor): The original coordinates of the protein.
        noise (torch.Tensor): The noise added to the coordinates.
        coords (torch.Tensor): The original coordinates with added noise.

    Args:
        corruption_rate (float): Magnitude of corruption to apply to the coordinates.
        corruption_strategy (str): Noise strategy to use for corruption. Must be
            either "uniform" or "gaussian".
        gaussian_mean (float, optional): Mean of the Gaussian distribution.
            Defaults to 0.0.
        gaussian_std (float, optional): Standard deviation of the Gaussian
            distribution. Defaults to 1.0.
        uniform_min (float, optional): Minimum value of the uniform distribution.
            Defaults to -1.0.
        uniform_max (float, optional): Maximum value of the uniform distribution.
            Defaults to 1.0.

    Raises:
        ValueError: If the corruption strategy is not supported.
    """

    def __init__(
        self,
        corruption_strategy: Literal["uniform", "gaussian"] = "gaussian",
        gaussian_mean: float = 0.0,
        gaussian_std: float = 0.1,
        uniform_min: float = -0.15,
        uniform_max: float = 0.15,
    ):
        self.corruption_strategy = corruption_strategy
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        self.uniform_min = uniform_min
        self.uniform_max = uniform_max

    def __call__(self, graph: Data) -> Data:
        """Adds noise to the coordinates of a protein structure.

        Args:
            graph (torch_geometric.data.Data): Protein data object.

        Returns:
            torch_geometric.data.Data: Protein data object with corrupted coordinates.
        """

        if self.corruption_strategy == "uniform":
            noise = torch.empty_like(graph.coords_nm).uniform_(
                self.uniform_min, self.uniform_max
            )
        elif self.corruption_strategy == "gaussian":
            noise = torch.normal(
                mean=self.gaussian_mean,
                std=self.gaussian_std,
                size=graph.coords_nm.size(),
            )
        else:
            raise ValueError(
                f"Corruption strategy '{self.corruption_strategy}' not supported."
            )
        
        graph.noise = noise.to(graph.coords_nm.dtype).to(graph.coords_nm.device)
        mask = (graph.coord_mask == 0) | (graph.is_peptide == 0).unsqueeze(-1)
        graph.noise[mask] = 0
        
        # we dont add noise to coords, as we need to calculate loss in angstroms
        graph.coords_nm = graph.coords_nm + graph.noise

        return graph

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(corruption_strategy={self.corruption_strategy}, "
            f"gaussian_mean={self.gaussian_mean}, "
            f"gaussian_std={self.gaussian_std}, uniform_min={self.uniform_min}, "
            f"uniform_max={self.uniform_max})"
        )
    

# this need to be further modified, as out complex is given as a single graph with is_peptide flag
class ReceptorHotspotTransform(T.BaseTransform):
    """
    Computes receptor hotspots in-place on a single complex graph using the `is_peptide`
    flag to distinguish receptor vs peptide residues. With probability `drop_prob`
    all hotspots are set to zero to avoid overleaking ground-truth contacts.
    """

    def __init__(
        self,
        drop_prob: float = 0.5,
        cutoff_nm: float = 0.8,
        keep_n_range: Tuple[int, int] | None = (4, 8),
    ):
        self.drop_prob = float(drop_prob)
        self.cutoff_nm = cutoff_nm
        self.keep_n_range = keep_n_range

    def __call__(self, graph: Data) -> Data:
        if graph is None:
            return graph

        num_res = graph.coords.shape[0]
        device = graph.coords.device
        hotspot_mask = torch.zeros(num_res, dtype=torch.bool, device=device)

        drop_prob = self.drop_prob
        if drop_prob > 1.0:
            logger.warning(
                f"ReceptorHotspotTransform.drop_prob should be in [0, 1], got {drop_prob}. "
                "Clamping to 1.0."
            )
            drop_prob = 1.0
        if drop_prob < 0.0:
            logger.warning(
                f"ReceptorHotspotTransform.drop_prob should be in [0, 1], got {drop_prob}. "
                "Clamping to 0.0."
            )
            drop_prob = 0.0

        if torch.rand(1).item() < drop_prob:
            graph.hotspot = hotspot_mask.float().unsqueeze(-1)
            return graph

        is_peptide = graph.is_peptide.to(device=device).long()
        receptor_mask = is_peptide == 0
        peptide_mask = is_peptide == 1

        rec_ca_coords_nm = graph.coords_nm[:, 1, :][receptor_mask]
        pep_ca_coords_nm = graph.coords_nm[:, 1, :][peptide_mask]

        dists = torch.cdist(rec_ca_coords_nm, pep_ca_coords_nm)
        rec_to_pep_min_dist = dists.min(dim=1).values
        rec_hotspots = rec_to_pep_min_dist < self.cutoff_nm

        if self.keep_n_range is not None:
            low, high = self.keep_n_range
            if low <= 0 or high <= 0 or low > high:
                raise ValueError(
                    f"keep_n_range must be a positive (low, high) tuple with low<=high, got {self.keep_n_range}"
                )
            true_indices = torch.where(rec_hotspots)[0]
            num_true = len(true_indices)
            if num_true > 0:
                n_to_keep = torch.randint(low, high + 1, (1,), device=device).item()
                n_to_keep = min(n_to_keep, num_true)
                if num_true > n_to_keep:
                    shuffled = torch.randperm(num_true, device=device)
                    drop_idx = true_indices[shuffled[n_to_keep:]]
                    rec_hotspots[drop_idx] = False

        hotspot_mask[receptor_mask] = rec_hotspots
        graph.hotspot = hotspot_mask.float().unsqueeze(-1)
        return graph


class BinderSecondaryAnnotationTransform(T.BaseTransform):
    """
    Annotates the cropped binder with a coarse secondary-structure class.

    Class ids:
      0 -> null
      1 -> alpha
      2 -> beta
      3 -> alpha_beta
      4 -> few_ss

    The label is computed from the cropped binder only, then randomly dropped to
    the null class with probability ``null_prob`` so the model also learns to
    operate without the condition.
    """

    NULL_CLASS = 0
    ALPHA_CLASS = 1
    BETA_CLASS = 2
    ALPHA_BETA_CLASS = 3
    FEW_SS_CLASS = 4

    def __init__(
        self,
        alpha_threshold: float = 0.45,
        alpha_max_sheet_fraction: float = 0.15,
        beta_threshold: float = 0.30,
        beta_max_helix_fraction: float = 0.20,
        alpha_beta_min_helix_fraction: float = 0.20,
        alpha_beta_min_sheet_fraction: float = 0.15,
        alpha_beta_min_structured_fraction: float = 0.55,
        few_ss_max_structured_fraction: float = 0.45,
        few_ss_min_coil_fraction: float = 0.50,
        min_longest_helix_run: int = 4,
        min_longest_sheet_run: int = 3,
        null_prob: float = 1.0 / 3.0,
        fold_mask_detail_prob: float = 0.0,
        fold_mask_family_prob: float = 0.0,
        fold_mask_struct_prob: float = 0.0,
        use_unmodified_coords: bool = True,
    ):
        self.alpha_threshold = float(alpha_threshold)
        self.alpha_max_sheet_fraction = float(alpha_max_sheet_fraction)
        self.beta_threshold = float(beta_threshold)
        self.beta_max_helix_fraction = float(beta_max_helix_fraction)
        self.alpha_beta_min_helix_fraction = float(alpha_beta_min_helix_fraction)
        self.alpha_beta_min_sheet_fraction = float(alpha_beta_min_sheet_fraction)
        self.alpha_beta_min_structured_fraction = float(alpha_beta_min_structured_fraction)
        self.few_ss_max_structured_fraction = float(few_ss_max_structured_fraction)
        self.few_ss_min_coil_fraction = float(few_ss_min_coil_fraction)
        self.min_longest_helix_run = int(min_longest_helix_run)
        self.min_longest_sheet_run = int(min_longest_sheet_run)
        self.null_prob = float(null_prob)
        self.fold_mask_detail_prob = float(fold_mask_detail_prob)
        self.fold_mask_family_prob = float(fold_mask_family_prob)
        self.fold_mask_struct_prob = float(fold_mask_struct_prob)
        self.use_unmodified_coords = bool(use_unmodified_coords)

    def _default_metrics(self) -> dict:
        return {
            "helix_fraction": 0.0,
            "sheet_fraction": 0.0,
            "coil_fraction": 0.0,
            "longest_helix_run": 0.0,
            "longest_sheet_run": 0.0,
            "num_residues": 0.0,
        }

    def _binder_pdb_string(self, graph: Data) -> str | None:
        binder_mask = (graph.is_peptide == 1)
        if int(binder_mask.sum().item()) == 0:
            return None

        coords = (
            graph.coords_unmodified
            if self.use_unmodified_coords and hasattr(graph, "coords_unmodified")
            else graph.coords
        )
        atom37 = coords[binder_mask].detach().cpu().numpy()
        atom37_mask = graph.coord_mask[binder_mask].detach().cpu().numpy()
        aatype = graph.residue_type[binder_mask].detach().cpu().numpy()
        chain_index = np.zeros(atom37.shape[0], dtype=np.int32)

        prot = create_full_prot(
            atom37=atom37,
            atom37_mask=atom37_mask,
            chain_index=chain_index,
            aatype=aatype,
        )
        return to_pdb(prot, model=1, add_end=True)

    def _classify(self, metrics: dict) -> tuple[int, bool]:
        class_id, _, threshold_hit = classify_secondary_structure_metrics(
            metrics=metrics,
            alpha_threshold=self.alpha_threshold,
            alpha_max_sheet_fraction=self.alpha_max_sheet_fraction,
            beta_threshold=self.beta_threshold,
            beta_max_helix_fraction=self.beta_max_helix_fraction,
            alpha_beta_min_helix_fraction=self.alpha_beta_min_helix_fraction,
            alpha_beta_min_sheet_fraction=self.alpha_beta_min_sheet_fraction,
            alpha_beta_min_structured_fraction=self.alpha_beta_min_structured_fraction,
            few_ss_max_structured_fraction=self.few_ss_max_structured_fraction,
            few_ss_min_coil_fraction=self.few_ss_min_coil_fraction,
            min_longest_helix_run=self.min_longest_helix_run,
            min_longest_sheet_run=self.min_longest_sheet_run,
        )
        return class_id, threshold_hit

    def _mask_fold_code(self, fold_code: tuple[int, int, int]) -> tuple[int, int, int]:
        struct_level, family_level, detail_level = fold_code
        if random.random() < self.fold_mask_detail_prob:
            detail_level = self.NULL_CLASS
            if random.random() < self.fold_mask_family_prob:
                family_level = self.NULL_CLASS
                if random.random() < self.fold_mask_struct_prob:
                    struct_level = self.NULL_CLASS
        return struct_level, family_level, detail_level

    def __call__(self, graph: Data) -> Data:
        metrics = self._default_metrics()
        class_id = self.NULL_CLASS
        raw_class_id = self.NULL_CLASS
        threshold_hit = False

        try:
            pdb_string = self._binder_pdb_string(graph)
            if pdb_string is not None:
                full_metrics = secondary_structure_metrics_from_pdb_string(
                    pdb_string=pdb_string,
                    peptide_chain="A",
                )
                metrics = {
                    "helix_fraction": float(full_metrics.get("helix_fraction", 0.0)),
                    "sheet_fraction": float(full_metrics.get("sheet_fraction", 0.0)),
                    "coil_fraction": float(full_metrics.get("coil_fraction", 0.0)),
                    "longest_helix_run": float(full_metrics.get("longest_helix_run", 0.0)),
                    "longest_sheet_run": float(full_metrics.get("longest_sheet_run", 0.0)),
                    "num_residues": float(full_metrics.get("num_residues", 0.0)),
                }
                raw_class_id, threshold_hit = self._classify(metrics)
                class_id = raw_class_id
        except Exception as exc:
            logger.warning(
                f"BinderSecondaryAnnotationTransform failed on graph {getattr(graph, 'id', '<unknown>')}: {exc}. "
                "Falling back to null secondary-structure label."
            )

        if random.random() < self.null_prob:
            class_id = self.NULL_CLASS

        fold_code = self._mask_fold_code(binder_secstruct_fold_code_from_id(class_id))
        raw_fold_code = binder_secstruct_fold_code_from_id(raw_class_id)

        graph.binder_secondary_class = torch.tensor([class_id], dtype=torch.long)
        graph.binder_secondary_class_raw = torch.tensor([raw_class_id], dtype=torch.long)
        graph.binder_secondary_fold_code = torch.tensor([fold_code], dtype=torch.long)
        graph.binder_secondary_fold_code_raw = torch.tensor([raw_fold_code], dtype=torch.long)
        graph.binder_helix_fraction = torch.tensor([metrics["helix_fraction"]], dtype=torch.float32)
        graph.binder_sheet_fraction = torch.tensor([metrics["sheet_fraction"]], dtype=torch.float32)
        graph.binder_coil_fraction = torch.tensor([metrics["coil_fraction"]], dtype=torch.float32)
        graph.binder_structured_fraction = torch.tensor(
            [metrics["helix_fraction"] + metrics["sheet_fraction"]], dtype=torch.float32
        )
        graph.binder_longest_helix_run = torch.tensor([metrics["longest_helix_run"]], dtype=torch.float32)
        graph.binder_longest_sheet_run = torch.tensor([metrics["longest_sheet_run"]], dtype=torch.float32)
        graph.binder_num_residues = torch.tensor([metrics["num_residues"]], dtype=torch.float32)
        graph.binder_secondary_threshold_hit = torch.tensor([threshold_hit], dtype=torch.bool)
        return graph


class BinderSecondaryStructureAdjacencyTransform(T.BaseTransform):
    """
    Adds RFdiffusion-style binder topology conditioning.

    Receptor residues/pairs are left as mask tokens. This matches RFdiffusion's
    PPI scaffold-guided default where binder ss/adj are provided, target ss/adj
    are mask unless explicit target_ss/target_adj inputs are supplied, and
    binder-target pair features remain mask.

    Secondary-structure token ids follow RFdiffusion:
      0 -> helix
      1 -> strand
      2 -> loop
      3 -> mask/unknown

    Block adjacency ids follow RFdiffusion:
      0 -> not adjacent / unconditioned none
      1 -> adjacent secondary-structure blocks
      2 -> mask/unknown
    """

    HELIX = 0
    STRAND = 1
    LOOP = 2
    MASK = 3

    ADJ_NONE = 0
    ADJ_CONTACT = 1
    ADJ_MASK = 2

    def __init__(
        self,
        adjacency_cutoff_angstrom: float = 6.0,
        include_loops_in_adjacency: bool = False,
        unconditioned_prob: float = 0.5,
        ss_only_prob: float = 0.4,
        full_condition_prob: float = 0.1,
        ss_mask: int = 0,
        use_unmodified_coords: bool = True,
    ):
        self.adjacency_cutoff_angstrom = float(adjacency_cutoff_angstrom)
        self.include_loops_in_adjacency = bool(include_loops_in_adjacency)
        probs = {
            "unconditioned_prob": float(unconditioned_prob),
            "ss_only_prob": float(ss_only_prob),
            "full_condition_prob": float(full_condition_prob),
        }
        for name, value in probs.items():
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}.")
        total_prob = sum(probs.values())
        if total_prob <= 0.0:
            raise ValueError("At least one binder topology conditioning probability must be positive.")
        self.unconditioned_prob = probs["unconditioned_prob"] / total_prob
        self.ss_only_prob = probs["ss_only_prob"] / total_prob
        self.full_condition_prob = probs["full_condition_prob"] / total_prob
        self.ss_mask = int(ss_mask)
        self.use_unmodified_coords = bool(use_unmodified_coords)

    @staticmethod
    def _generate_cb(n: torch.Tensor, ca: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        b = ca - n
        c_vec = c - ca
        a = torch.cross(b, c_vec, dim=-1)
        return -0.57910144 * a + 0.5689693 * b - 0.5441217 * c_vec + ca

    @staticmethod
    def _segments(ss: torch.Tensor, residue_indices: torch.Tensor | None = None) -> list[tuple[int, torch.Tensor]]:
        if ss.numel() == 0:
            return []
        if residue_indices is None:
            residue_indices = torch.arange(ss.numel(), device=ss.device)
        segments: list[tuple[int, torch.Tensor]] = []
        begin = 0
        for i in range(1, int(ss.numel())):
            if int(ss[i].item()) != int(ss[i - 1].item()):
                segments.append((int(ss[i - 1].item()), residue_indices[begin:i]))
                begin = i
        segments.append((int(ss[-1].item()), residue_indices[begin:int(ss.numel())]))
        return segments

    def _ss_from_pdb(self, graph: Data, residue_mask: torch.Tensor) -> torch.Tensor:
        coords = (
            graph.coords_unmodified
            if self.use_unmodified_coords and hasattr(graph, "coords_unmodified")
            else graph.coords
        )
        atom37 = coords[residue_mask].detach().cpu().numpy()
        atom37_mask = graph.coord_mask[residue_mask].detach().cpu().numpy()
        aatype = graph.residue_type[residue_mask].detach().cpu().numpy()
        chain_index = np.zeros(atom37.shape[0], dtype=np.int32)
        prot = create_full_prot(
            atom37=atom37,
            atom37_mask=atom37_mask,
            chain_index=chain_index,
            aatype=aatype,
        )
        pdb_string = to_pdb(prot, model=1, add_end=True)
        atom_array = atom_array_from_pdb_string(pdb_string)
        chain = atom_array[atom_array.chain_id == "A"]
        if chain.array_length() == 0:
            raise ValueError("empty topology atom array")
        sse = struc.annotate_sse(chain)
        sse_arr = np.asarray(sse)
        ss = torch.full((len(sse_arr),), self.MASK, dtype=torch.long, device=graph.coords.device)
        ss[torch.from_numpy(sse_arr == "a").to(device=graph.coords.device)] = self.HELIX
        ss[torch.from_numpy(sse_arr == "b").to(device=graph.coords.device)] = self.STRAND
        ss[torch.from_numpy(sse_arr == "c").to(device=graph.coords.device)] = self.LOOP
        return ss

    def _block_adjacency(
        self,
        ss: torch.Tensor,
        coords: torch.Tensor,
        coord_mask: torch.Tensor,
        segment_masks: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        n = coords[:, residue_constants.atom_order["N"], :]
        ca = coords[:, residue_constants.atom_order["CA"], :]
        c = coords[:, residue_constants.atom_order["C"], :]
        backbone_mask = (
            coord_mask[:, residue_constants.atom_order["N"]].bool()
            & coord_mask[:, residue_constants.atom_order["CA"]].bool()
            & coord_mask[:, residue_constants.atom_order["C"]].bool()
        )

        cb = self._generate_cb(n, ca, c)
        dists = torch.cdist(cb, cb)
        dists[~backbone_mask[:, None] | ~backbone_mask[None, :]] = float("inf")
        dists.fill_diagonal_(float("inf"))

        adj = torch.zeros((ss.numel(), ss.numel()), dtype=torch.long, device=coords.device)
        if segment_masks is None:
            segments = self._segments(ss)
        else:
            segments = []
            for segment_mask in segment_masks:
                segment_indices = torch.where(segment_mask.bool())[0]
                segments.extend(self._segments(ss[segment_indices], residue_indices=segment_indices))

        for i, (ss_i, idx_i) in enumerate(segments):
            if ss_i == self.LOOP and not self.include_loops_in_adjacency:
                continue
            if ss_i == self.MASK:
                continue
            for ss_j, idx_j in segments[i + 1:]:
                if idx_i.numel() == 0 or idx_j.numel() == 0:
                    continue
                if ss_j == self.LOOP and not self.include_loops_in_adjacency:
                    continue
                if ss_j == self.MASK:
                    continue
                if torch.any(dists[idx_i[:, None], idx_j[None, :]] < self.adjacency_cutoff_angstrom):
                    adj[idx_i[:, None], idx_j[None, :]] = self.ADJ_CONTACT
                    adj[idx_j[:, None], idx_i[None, :]] = self.ADJ_CONTACT
        return adj

    def _mask_ss_adj(self, ss: torch.Tensor, adj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        condition_mask = ss != self.LOOP
        if self.ss_mask > 0 and condition_mask.numel() > 0:
            original_mask = condition_mask.clone()
            for i in range(1, self.ss_mask + 1):
                condition_mask[i:] &= original_mask[:-i]
                condition_mask[:-i] &= original_mask[i:]

        ss = ss.clone()
        adj = adj.clone()
        ss[~condition_mask] = self.MASK
        adj[~condition_mask, :] = self.ADJ_MASK
        adj[:, ~condition_mask] = self.ADJ_MASK

        mode_sample = random.random()
        if mode_sample < self.unconditioned_prob:
            ss[:] = self.MASK
            adj[:] = self.ADJ_MASK
        elif mode_sample < self.unconditioned_prob + self.ss_only_prob:
            adj[:] = self.ADJ_MASK
        return ss, adj

    def __call__(self, graph: Data) -> Data:
        num_res = int(graph.coords.shape[0])
        ss_token = torch.full((num_res,), self.MASK, dtype=torch.long, device=graph.coords.device)
        block_adj = torch.full((num_res, num_res), self.ADJ_MASK, dtype=torch.long, device=graph.coords.device)

        binder_mask = (graph.is_peptide.to(device=graph.coords.device) == 1)
        if int(binder_mask.sum().item()) == 0:
            graph.binder_ss_token = ss_token
            graph.binder_block_adj = block_adj
            return graph

        coords = (
            graph.coords_unmodified
            if self.use_unmodified_coords and hasattr(graph, "coords_unmodified")
            else graph.coords
        )

        try:
            binder_ss = self._ss_from_pdb(graph, binder_mask)
            binder_indices = torch.where(binder_mask)[0]
            if binder_ss.numel() != binder_indices.numel():
                raise ValueError(f"SS length {binder_ss.numel()} != binder length {binder_indices.numel()}")

            binder_adj = self._block_adjacency(
                ss=binder_ss,
                coords=coords[binder_mask],
                coord_mask=graph.coord_mask[binder_mask],
            )
            binder_ss, binder_adj = self._mask_ss_adj(binder_ss, binder_adj)
            ss_token[binder_indices] = binder_ss
            block_adj[binder_indices[:, None], binder_indices[None, :]] = binder_adj
        except Exception as exc:
            logger.warning(
                f"BinderSecondaryStructureAdjacencyTransform failed on graph "
                f"{getattr(graph, 'id', '<unknown>')}: {exc}. Falling back to masked topology labels."
            )

        graph.binder_ss_token = ss_token
        graph.binder_block_adj = block_adj
        return graph

# ConditionMaskTransform
# given a graph with peptide and receptor parts, create two masks:
# 1. ca_cond_mask: used for fixing ca (add no noise to ca mode) when training, used for inv_folding mode
# 2. latent_cond_mask: normal mode, we want out motif to be fixed in both ca and latent space 
# motif mode: return the same ca_cond_mask and latent_cond_mask, both with only condition part being true
# inv folding mode: return all true ca_cond_mask and latent_cond_mask with only condition part (receptor + condition motif) being true, which allow redesign option just like ProteinMPNN
# docking mode: maybe that is not that easy, further logic need tobe in training_step hook, for here we should just return ca_cond_mask and latent_cond_mask as is_receptor_mask

class ConditionMaskTransform(T.BaseTransform):
    """
    Creates a motif mask for a protein structure with multiple strategies.

    Args:
        training_mode (Literal["motif", "inv_folding", "docking"], optional): Training mode to determine conditioning strategy. 
            Defaults to "motif".
        motif_prob (float, optional): Probability of creating a motif. Defaults to 1.0.
        motif_min_pct_res (float, optional): Minimum percentage of residues in motif. Defaults to 0.05.
        motif_max_pct_res (float, optional): Maximum percentage of residues in motif. Defaults to 0.5.
        motif_min_n_seg (int, optional): Minimum number of segments in motif. Defaults to 1.
        motif_max_n_seg (int, optional): Maximum number of segments in motif. Defaults to 4.

    Returns:
        The input graph with the following attributes added:
            - ca_cond_mask (torch.BoolTensor): Mask for CA conditioning.
            - latent_cond_mask (torch.BoolTensor): Mask for latent conditioning.
    """

    def __init__(
        self,
        training_mode: Literal[
            "motif", "inv_folding", "docking"
        ] = "motif",
        motif_prob: float = 1.0,
        motif_min_pct_res: float = 0.05,
        motif_max_pct_res: float = 0.5,
        motif_min_n_seg: int = 1,
        motif_max_n_seg: int = 4,
    ):
        self.training_mode = training_mode
        self.motif_prob = motif_prob
        self.motif_min_pct_res = motif_min_pct_res
        self.motif_max_pct_res = motif_max_pct_res
        self.motif_min_n_seg = motif_min_n_seg
        self.motif_max_n_seg = motif_max_n_seg


    def __call__(self, graph: Data) -> Data:
        if random.random() > self.motif_prob:
            condition_mask = graph.is_peptide == 0  # receptor only
            if self.training_mode == "motif":
                graph.ca_cond_mask = condition_mask.bool()
                graph.latent_cond_mask = condition_mask.bool()
            elif self.training_mode == "inv_folding":
                graph.ca_cond_mask = torch.ones_like(condition_mask).bool()
                graph.latent_cond_mask = condition_mask.bool()
            elif self.training_mode == "docking":
                graph.ca_cond_mask = condition_mask.bool()
                graph.latent_cond_mask = condition_mask.bool()
            else:
                raise ValueError(f"Unknown training mode: {self.training_mode}")
            return graph

        num_res = sum(graph.is_peptide)
        motif_n_res = int(
            random.random()
            * (num_res * self.motif_max_pct_res - num_res * self.motif_min_pct_res)
            + num_res * self.motif_min_pct_res
        )
        motif_n_res = min(motif_n_res, num_res)
        motif_n_seg = int(
            random.random()
            * (min(motif_n_res, self.motif_max_n_seg) - self.motif_min_n_seg + 1)
            + self.motif_min_n_seg
        )
        indices = (
            np.sort(random.sample(range(1, motif_n_res), motif_n_seg - 1))
            if motif_n_seg > 1
            else []
        )
        indices = (
            np.concatenate([[0], indices, [motif_n_res]]).astype(int)
            if motif_n_seg > 1
            else np.array([0, motif_n_res])
        )
        segments = []
        for i in range(len(indices) - 1):
            start, end = indices[i], indices[i + 1]
            segment_length = end - start
            segments.append("".join(["1"] * segment_length))
        segments.extend(["0"] * (num_res - motif_n_res))
        random.shuffle(segments)
        motif_sequence_mask = torch.tensor(
            [int(elt) for elt in "".join(segments)]
        ).bool()
        
        condition_mask = torch.ones(size=(graph.coords_nm.shape[0],))
        condition_mask[graph.is_peptide == 1] = motif_sequence_mask.float()

        if self.training_mode == "motif":
            graph.ca_cond_mask = condition_mask.bool()
            graph.latent_cond_mask = condition_mask.bool()
        elif self.training_mode == "inv_folding":
            graph.ca_cond_mask = torch.ones_like(condition_mask).bool()
            graph.latent_cond_mask = condition_mask.bool()
        elif self.training_mode == "docking":
            receptor_mask = graph.is_peptide == 0
            graph.ca_cond_mask = receptor_mask.bool()
            graph.latent_cond_mask = receptor_mask.bool()
        else:
            raise ValueError(f"Unknown training mode: {self.training_mode}")

        return graph


class OptionalConditionTransform(T.BaseTransform):
    """
    Sample optional conditioning protocols on top of the existing context task.

    This transform is intended to run after ``ConditionMaskTransform`` so it can
    preferentially place sparse conditions on designable binder residues.

    Condition type ids:
      0 -> none / unapplied
      1 -> generic sparse joint
      2 -> generic sparse independent none
      3 -> generic sparse independent residue only
      4 -> generic sparse independent pair only
      5 -> generic sparse independent residue + pair
      6 -> legacy inv_folding
      7 -> legacy folding
    """

    TYPE_NONE = 0
    TYPE_GENERIC_JOINT = 1
    TYPE_GENERIC_INDEP_NONE = 2
    TYPE_GENERIC_INDEP_RES = 3
    TYPE_GENERIC_INDEP_PAIR = 4
    TYPE_GENERIC_INDEP_BOTH = 5
    TYPE_LEGACY_INV = 6
    TYPE_LEGACY_FOLD = 7

    def __init__(
        self,
        apply_prob: float = 0.30,
        generic_sparse_prob: float = 0.80,
        legacy_inv_folding_prob: float = 0.10,
        legacy_folding_prob: float = 0.10,
        joint_prob: float = 0.50,
        independent_prob: float = 0.50,
        joint_min_anchor_k: int = 2,
        joint_max_anchor_k: int = 4,
        indep_residue_prob: float = 0.50,
        indep_pair_prob: float = 0.50,
        residue_min_k: int = 1,
        residue_max_k: int = 4,
        pair_min_k: int = 0,
        pair_max_k: int = 3,
        min_seq_sep: int = 3,
        pair_nbins: int = 30,
        pair_min_dist_nm: float = 0.1,
        pair_max_dist_nm: float = 3.0,
        bin_reweight: bool = True,
        prefer_designable_region: bool = True,
        non_head_tail_forced_condition_drop_prob: float = 0.0,
    ):
        self.apply_prob = float(apply_prob)
        self.generic_sparse_prob = float(generic_sparse_prob)
        self.legacy_inv_folding_prob = float(legacy_inv_folding_prob)
        self.legacy_folding_prob = float(legacy_folding_prob)
        self.joint_prob = float(joint_prob)
        self.independent_prob = float(independent_prob)
        self.joint_min_anchor_k = int(joint_min_anchor_k)
        self.joint_max_anchor_k = int(joint_max_anchor_k)
        self.indep_residue_prob = float(indep_residue_prob)
        self.indep_pair_prob = float(indep_pair_prob)
        self.residue_min_k = int(residue_min_k)
        self.residue_max_k = int(residue_max_k)
        self.pair_min_k = int(pair_min_k)
        self.pair_max_k = int(pair_max_k)
        self.min_seq_sep = int(min_seq_sep)
        self.pair_nbins = int(pair_nbins)
        self.pair_min_dist_nm = float(pair_min_dist_nm)
        self.pair_max_dist_nm = float(pair_max_dist_nm)
        self.bin_reweight = bool(bin_reweight)
        self.prefer_designable_region = bool(prefer_designable_region)
        self.non_head_tail_forced_condition_drop_prob = float(non_head_tail_forced_condition_drop_prob)
        self._bin_limits_cache = None

        if self.joint_min_anchor_k < 2:
            raise ValueError("joint_min_anchor_k must be >= 2")
        if self.joint_max_anchor_k < self.joint_min_anchor_k:
            raise ValueError("joint_max_anchor_k must be >= joint_min_anchor_k")
        if self.residue_min_k < 1:
            raise ValueError("residue_min_k must be >= 1")
        if self.residue_max_k < self.residue_min_k:
            raise ValueError("residue_max_k must be >= residue_min_k")
        if self.pair_max_k < self.pair_min_k:
            raise ValueError("pair_max_k must be >= pair_min_k")
        if self.pair_nbins < 2:
            raise ValueError("pair_nbins must be >= 2")
        if not 0.0 <= self.non_head_tail_forced_condition_drop_prob <= 1.0:
            raise ValueError("non_head_tail_forced_condition_drop_prob must be in [0, 1]")

    def _bin_limits(self, device: torch.device) -> torch.Tensor:
        if self._bin_limits_cache is None or self._bin_limits_cache.device != device:
            self._bin_limits_cache = torch.linspace(
                self.pair_min_dist_nm,
                self.pair_max_dist_nm,
                self.pair_nbins - 1,
                device=device,
            )
        return self._bin_limits_cache

    def _reset_optional_fields(self, graph: Data) -> None:
        num_res = int(graph.residue_type.shape[0])
        self._ensure_topology_defaults(graph)
        graph.has_cyclization_pair = torch.zeros((1,), dtype=torch.bool)
        graph.cyclization_pair_indices = torch.full((1, 2), fill_value=-1, dtype=torch.long)
        graph.cyclization_shortcut_len = torch.ones((1,), dtype=torch.long)
        graph.use_ca_coors_nm_feature = torch.tensor([False], dtype=torch.bool)
        graph.use_residue_type_feature = torch.tensor([False], dtype=torch.bool)
        graph.residue_type_condition = torch.full(
            (num_res,), fill_value=-1, dtype=torch.long
        )
        graph.residue_type_condition_mask = torch.zeros(num_res, dtype=torch.bool)
        graph.optional_ca_pair_indices = torch.full((1, 2), fill_value=-1, dtype=torch.long)
        graph.optional_ca_pair_bin_idx = torch.zeros((1,), dtype=torch.long)
        graph.optional_ca_pair_valid_mask = torch.zeros((1,), dtype=torch.bool)
        graph.chemical_bond_type = torch.zeros((num_res, num_res), dtype=torch.long)
        graph.optional_condition_type = torch.tensor([self.TYPE_NONE], dtype=torch.long)

    @staticmethod
    def _ensure_topology_defaults(graph: Data) -> None:
        if not hasattr(graph, "cyclic"):
            graph.cyclic = torch.tensor([False], dtype=torch.bool)
        else:
            graph.cyclic = graph.cyclic.bool().reshape(-1)[:1]

        if not hasattr(graph, "has_cyclization_pair"):
            graph.has_cyclization_pair = torch.zeros((1,), dtype=torch.bool)
        else:
            graph.has_cyclization_pair = graph.has_cyclization_pair.bool().reshape(-1)[:1]

        if not hasattr(graph, "cyclization_pair_indices"):
            graph.cyclization_pair_indices = torch.full((1, 2), fill_value=-1, dtype=torch.long)
        else:
            graph.cyclization_pair_indices = graph.cyclization_pair_indices.long().reshape(-1, 2)[:1]

        if not hasattr(graph, "cyclization_shortcut_len"):
            graph.cyclization_shortcut_len = torch.ones((1,), dtype=torch.long)
        else:
            graph.cyclization_shortcut_len = graph.cyclization_shortcut_len.long().reshape(-1)[:1]

    def _peptide_ca_mask(self, graph: Data) -> torch.Tensor:
        has_ca = graph.coord_mask[:, residue_constants.atom_types.index("CA")].bool()
        return (graph.is_peptide == 1) & has_ca

    def _designable_binder_mask(self, graph: Data, min_required: int) -> torch.Tensor:
        binder_mask = self._peptide_ca_mask(graph)
        if not self.prefer_designable_region or not hasattr(graph, "latent_cond_mask"):
            return binder_mask

        latent_cond_mask = graph.latent_cond_mask.bool()
        designable_mask = binder_mask & (~latent_cond_mask)
        if int(designable_mask.sum().item()) >= min_required:
            return designable_mask
        return binder_mask

    @staticmethod
    def _sample_condition_mode(prob_triplet: List[float]) -> int:
        total = sum(max(v, 0.0) for v in prob_triplet)
        if total <= 0:
            return 0
        r = random.random() * total
        acc = 0.0
        for idx, prob in enumerate(prob_triplet):
            acc += max(prob, 0.0)
            if r <= acc:
                return idx
        return len(prob_triplet) - 1

    @staticmethod
    def _pairwise_seq_sep_ok(selected: List[int], min_seq_sep: int) -> bool:
        if min_seq_sep <= 0:
            return True
        for i, j in itertools.combinations(selected, 2):
            if abs(i - j) < min_seq_sep:
                return False
        return True

    def _sample_residue_subset(
        self,
        eligible_indices: List[int],
        min_k: int,
        max_k: int,
        min_seq_sep: int,
    ) -> List[int]:
        if len(eligible_indices) < min_k:
            return []
        max_k = min(max_k, len(eligible_indices))
        if max_k < min_k:
            return []

        target_sizes = list(range(min_k, max_k + 1))
        random.shuffle(target_sizes)
        for target_k in target_sizes:
            for _ in range(64):
                candidate = sorted(random.sample(eligible_indices, target_k))
                if self._pairwise_seq_sep_ok(candidate, min_seq_sep=min_seq_sep):
                    return candidate
        return []

    def _encode_pair_bins(self, graph: Data, pairs: List[Tuple[int, int]]) -> List[int]:
        if len(pairs) == 0:
            return []
        ca_index = residue_constants.atom_types.index("CA")
        ca_coords = graph.coords_nm[:, ca_index, :]
        device = ca_coords.device
        bin_limits = self._bin_limits(device=device)
        dists = []
        for idx_i, idx_j in pairs:
            dist_nm = torch.norm(ca_coords[idx_i] - ca_coords[idx_j], dim=-1)
            dists.append(dist_nm)
        dists_tensor = torch.stack(dists, dim=0)
        bin_idx = torch.bucketize(dists_tensor, bin_limits)
        return [int(v.item()) for v in bin_idx]

    def _set_sparse_pair_fields(
        self,
        graph: Data,
        pair_indices: List[Tuple[int, int]],
        pair_bin_idx: List[int],
    ) -> None:
        if len(pair_indices) == 0:
            graph.optional_ca_pair_indices = torch.full((1, 2), fill_value=-1, dtype=torch.long)
            graph.optional_ca_pair_bin_idx = torch.zeros((1,), dtype=torch.long)
            graph.optional_ca_pair_valid_mask = torch.zeros((1,), dtype=torch.bool)
            return
        graph.optional_ca_pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        graph.optional_ca_pair_bin_idx = torch.tensor(pair_bin_idx, dtype=torch.long)
        graph.optional_ca_pair_valid_mask = torch.ones(len(pair_indices), dtype=torch.bool)

    def _refresh_forced_topology_indices(self, graph: Data) -> None:
        forced_resids = getattr(graph, "forced_topology_anchor_resids", None)
        if forced_resids is None or not hasattr(graph, "residue_pdb_idx"):
            return
        forced_resids = forced_resids.long().reshape(-1, 2)
        if forced_resids.numel() == 0 or (forced_resids < 0).all():
            return
        pair_indices = []
        pair_valid = []
        peptide_anchor_mask = graph.is_peptide.long() == 1 if hasattr(graph, "is_peptide") else torch.ones_like(graph.residue_pdb_idx, dtype=torch.bool)
        for resid_i, resid_j in forced_resids.tolist():
            matches_i = torch.nonzero(
                (graph.residue_pdb_idx.long() == int(resid_i)) & peptide_anchor_mask,
                as_tuple=True,
            )[0]
            matches_j = torch.nonzero(
                (graph.residue_pdb_idx.long() == int(resid_j)) & peptide_anchor_mask,
                as_tuple=True,
            )[0]
            if matches_i.numel() == 1 and matches_j.numel() == 1:
                pair_indices.append([int(matches_i[0]), int(matches_j[0])])
                pair_valid.append(True)
            else:
                pair_indices.append([-1, -1])
                pair_valid.append(False)
        graph.forced_optional_ca_pair_indices = torch.tensor(pair_indices, dtype=torch.long)
        graph.forced_optional_ca_pair_valid_mask = torch.tensor(pair_valid, dtype=torch.bool)
        if hasattr(graph, "forced_optional_ca_pair_bin_idx"):
            graph.forced_optional_ca_pair_bin_idx = graph.forced_optional_ca_pair_bin_idx.long().reshape(-1)[: len(pair_indices)]
        else:
            graph.forced_optional_ca_pair_bin_idx = torch.zeros((len(pair_indices),), dtype=torch.long)
        bond_type_id = self._forced_chemical_bond_type_id(graph)
        if bond_type_id > 0:
            if not hasattr(graph, "chemical_bond_type"):
                num_res = int(graph.residue_type.shape[0])
                graph.chemical_bond_type = torch.zeros((num_res, num_res), dtype=torch.long)
            for (idx_i, idx_j), is_valid in zip(pair_indices, pair_valid):
                if is_valid:
                    graph.chemical_bond_type[int(idx_i), int(idx_j)] = int(bond_type_id)
                    graph.chemical_bond_type[int(idx_j), int(idx_i)] = int(bond_type_id)
        if bond_type_id == CHEMICAL_BOND_HEAD_TAIL_AMIDE and hasattr(graph, "cyclization_pair_indices"):
            graph.cyclization_pair_indices = graph.forced_optional_ca_pair_indices[:1].clone()
        if bond_type_id == CHEMICAL_BOND_HEAD_TAIL_AMIDE and hasattr(graph, "has_cyclization_pair"):
            graph.has_cyclization_pair = graph.forced_optional_ca_pair_valid_mask[:1].clone()
        if hasattr(graph, "forced_residue_type_condition"):
            num_res = int(graph.residue_type.shape[0])
            graph.forced_residue_type_condition = torch.full((num_res,), fill_value=-1, dtype=torch.long)
            graph.forced_residue_type_condition_mask = torch.zeros(num_res, dtype=torch.bool)
            for (idx_i, idx_j), is_valid in zip(pair_indices, pair_valid):
                if is_valid:
                    graph.forced_residue_type_condition[[idx_i, idx_j]] = graph.residue_type[[idx_i, idx_j]].long()
                    graph.forced_residue_type_condition_mask[[idx_i, idx_j]] = True

    @staticmethod
    def _string_metadata(value) -> str:
        if value is None:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="ignore").lower()
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return ""
            return str(value.reshape(-1)[0].item()).lower()
        return str(value).lower()

    def _forced_chemical_bond_type_id(self, graph: Data) -> int:
        cyclic_type = self._string_metadata(getattr(graph, "cpcore_cyclic_type", None))
        normalized_cyclic_type = cyclic_type.replace("-", "").replace("_", "").replace(" ", "")
        if normalized_cyclic_type == "headtail":
            return CHEMICAL_BOND_HEAD_TAIL_AMIDE
        if normalized_cyclic_type in {"disulfide", "disulfied"}:
            return CHEMICAL_BOND_DISULFIDE
        if normalized_cyclic_type in {"isopeptide", "isopep"}:
            return CHEMICAL_BOND_ISOPEPTIDE

        raw_entries = getattr(graph, "topology_entries", None)
        if isinstance(raw_entries, str):
            try:
                entries = json.loads(raw_entries)
            except (json.JSONDecodeError, ValueError):
                entries = []
        else:
            entries = []
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                topo_type = str(entry.get("type", "")).strip().lower()
                bond_id = chemical_bond_type_to_id(topo_type)
                if bond_id > 0:
                    return bond_id

        topology_entries = self._string_metadata(raw_entries)
        if "head_tail" in topology_entries or "head-to-tail" in topology_entries or "headtail" in topology_entries:
            return CHEMICAL_BOND_HEAD_TAIL_AMIDE
        if "disulfide" in topology_entries or "disulfied" in topology_entries:
            return CHEMICAL_BOND_DISULFIDE
        if "isopeptide" in topology_entries or "isopep" in topology_entries or "stapled" in topology_entries:
            return CHEMICAL_BOND_ISOPEPTIDE
        if any(token in topology_entries for token in ("mk8", "nle", "staple_rty", "bicycle")):
            return CHEMICAL_BOND_OTHER_COVALENT
        return 0

    def _is_head_tail_forced_topology(self, graph: Data) -> bool:
        cyclic_type = self._string_metadata(getattr(graph, "cpcore_cyclic_type", None))
        if cyclic_type.replace("-", "").replace("_", "") == "headtail":
            return True

        topology_entries = self._string_metadata(getattr(graph, "topology_entries", None))
        return "head_tail" in topology_entries or "head-to-tail" in topology_entries or "headtail" in topology_entries

    def _should_drop_forced_condition(self, graph: Data) -> bool:
        if self.non_head_tail_forced_condition_drop_prob <= 0.0:
            return False
        has_forced_condition = (
            hasattr(graph, "forced_residue_type_condition_mask")
            or hasattr(graph, "forced_optional_ca_pair_valid_mask")
            or hasattr(graph, "forced_topology_anchor_resids")
        )
        if not has_forced_condition or self._is_head_tail_forced_topology(graph):
            return False
        return random.random() < self.non_head_tail_forced_condition_drop_prob

    @staticmethod
    def _clear_active_forced_topology_condition(graph: Data) -> None:
        graph.has_cyclization_pair = torch.zeros((1,), dtype=torch.bool)
        graph.cyclization_pair_indices = torch.full((1, 2), fill_value=-1, dtype=torch.long)
        num_res = int(graph.residue_type.shape[0])
        graph.chemical_bond_type = torch.zeros((num_res, num_res), dtype=torch.long)

    def _merge_forced_condition_fields(self, graph: Data) -> Data:
        if self._should_drop_forced_condition(graph):
            self._clear_active_forced_topology_condition(graph)
            return graph

        self._refresh_forced_topology_indices(graph)
        forced_res_mask = getattr(graph, "forced_residue_type_condition_mask", None)
        forced_res = getattr(graph, "forced_residue_type_condition", None)
        if forced_res_mask is not None and forced_res is not None:
            forced_res_mask = forced_res_mask.bool()
            graph.residue_type_condition = torch.where(
                forced_res_mask,
                forced_res.long(),
                graph.residue_type_condition.long(),
            )
            graph.residue_type_condition_mask = graph.residue_type_condition_mask.bool() | forced_res_mask

        forced_pair_valid = getattr(graph, "forced_optional_ca_pair_valid_mask", None)
        forced_pair_indices = getattr(graph, "forced_optional_ca_pair_indices", None)
        forced_pair_bin_idx = getattr(graph, "forced_optional_ca_pair_bin_idx", None)
        if (
            forced_pair_valid is not None
            and forced_pair_indices is not None
            and forced_pair_bin_idx is not None
            and bool(forced_pair_valid.bool().any().item())
        ):
            if bool(graph.optional_ca_pair_valid_mask.bool().any().item()):
                graph.optional_ca_pair_indices = torch.cat(
                    [graph.optional_ca_pair_indices.long(), forced_pair_indices.long()],
                    dim=0,
                )
                graph.optional_ca_pair_bin_idx = torch.cat(
                    [graph.optional_ca_pair_bin_idx.long(), forced_pair_bin_idx.long()],
                    dim=0,
                )
                graph.optional_ca_pair_valid_mask = torch.cat(
                    [graph.optional_ca_pair_valid_mask.bool(), forced_pair_valid.bool()],
                    dim=0,
                )
            else:
                graph.optional_ca_pair_indices = forced_pair_indices.long()
                graph.optional_ca_pair_bin_idx = forced_pair_bin_idx.long()
                graph.optional_ca_pair_valid_mask = forced_pair_valid.bool()

        has_forced_condition = (
            hasattr(graph, "forced_residue_type_condition_mask")
            or hasattr(graph, "forced_optional_ca_pair_valid_mask")
        )
        if has_forced_condition and int(graph.optional_condition_type.reshape(-1)[0].item()) == self.TYPE_NONE:
            graph.optional_condition_type = torch.tensor([self.TYPE_GENERIC_JOINT], dtype=torch.long)
        return graph

    def _sample_independent_pairs(self, graph: Data, eligible_indices: List[int]) -> Tuple[List[Tuple[int, int]], List[int]]:
        if len(eligible_indices) < 2 or self.pair_max_k == 0:
            return [], []

        candidate_pairs = []
        for idx_i, idx_j in itertools.combinations(sorted(eligible_indices), 2):
            if abs(idx_i - idx_j) < self.min_seq_sep:
                continue
            candidate_pairs.append((idx_i, idx_j))
        if len(candidate_pairs) == 0:
            return [], []

        pair_bin_idx = self._encode_pair_bins(graph, candidate_pairs)
        max_k = min(self.pair_max_k, len(candidate_pairs))
        if max_k < self.pair_min_k:
            return [], []
        target_k = random.randint(self.pair_min_k, max_k)
        if target_k == 0:
            return [], []

        if self.bin_reweight:
            counts = {}
            for b in pair_bin_idx:
                counts[b] = counts.get(b, 0) + 1
            weights = torch.tensor(
                [1.0 / counts[b] for b in pair_bin_idx],
                dtype=torch.float32,
            )
            sampled = torch.multinomial(weights, num_samples=target_k, replacement=False).tolist()
        else:
            sampled = random.sample(range(len(candidate_pairs)), target_k)
        sampled_pairs = [candidate_pairs[i] for i in sampled]
        sampled_bins = [pair_bin_idx[i] for i in sampled]
        return sampled_pairs, sampled_bins

    def _apply_joint_sparse_condition(self, graph: Data) -> int:
        eligible_mask = self._designable_binder_mask(graph, min_required=self.joint_min_anchor_k)
        eligible_indices = torch.nonzero(eligible_mask, as_tuple=True)[0].tolist()
        anchor_indices = self._sample_residue_subset(
            eligible_indices=eligible_indices,
            min_k=self.joint_min_anchor_k,
            max_k=self.joint_max_anchor_k,
            min_seq_sep=self.min_seq_sep,
        )
        if len(anchor_indices) < 2:
            return self.TYPE_NONE

        graph.residue_type_condition[anchor_indices] = graph.residue_type[anchor_indices].long()
        graph.residue_type_condition_mask[anchor_indices] = True

        pair_indices = list(itertools.combinations(anchor_indices, 2))
        pair_bin_idx = self._encode_pair_bins(graph, pair_indices)
        self._set_sparse_pair_fields(graph, pair_indices, pair_bin_idx)
        return self.TYPE_GENERIC_JOINT

    def _apply_independent_sparse_condition(self, graph: Data) -> int:
        apply_residue = random.random() < self.indep_residue_prob
        apply_pair = random.random() < self.indep_pair_prob

        if apply_residue:
            residue_mask = self._designable_binder_mask(graph, min_required=self.residue_min_k)
            residue_indices = torch.nonzero(residue_mask, as_tuple=True)[0].tolist()
            picked_residue_indices = self._sample_residue_subset(
                eligible_indices=residue_indices,
                min_k=self.residue_min_k,
                max_k=self.residue_max_k,
                min_seq_sep=self.min_seq_sep,
            )
            if len(picked_residue_indices) > 0:
                graph.residue_type_condition[picked_residue_indices] = graph.residue_type[picked_residue_indices].long()
                graph.residue_type_condition_mask[picked_residue_indices] = True
            else:
                apply_residue = False

        if apply_pair:
            pair_mask = self._designable_binder_mask(graph, min_required=2)
            pair_indices = torch.nonzero(pair_mask, as_tuple=True)[0].tolist()
            sampled_pairs, sampled_bins = self._sample_independent_pairs(
                graph=graph,
                eligible_indices=pair_indices,
            )
            self._set_sparse_pair_fields(graph, sampled_pairs, sampled_bins)
            if len(sampled_pairs) == 0:
                apply_pair = False

        if not apply_pair:
            self._set_sparse_pair_fields(graph, [], [])

        if apply_residue and apply_pair:
            return self.TYPE_GENERIC_INDEP_BOTH
        if apply_residue:
            return self.TYPE_GENERIC_INDEP_RES
        if apply_pair:
            return self.TYPE_GENERIC_INDEP_PAIR
        return self.TYPE_GENERIC_INDEP_NONE

    def __call__(self, graph: Data) -> Data:
        self._reset_optional_fields(graph)

        if random.random() >= self.apply_prob:
            return self._merge_forced_condition_fields(graph)

        condition_mode = self._sample_condition_mode(
            [
                self.generic_sparse_prob,
                self.legacy_inv_folding_prob,
                self.legacy_folding_prob,
            ]
        )
        if condition_mode == 1:
            graph.use_ca_coors_nm_feature = torch.tensor([True], dtype=torch.bool)
            graph.optional_condition_type = torch.tensor([self.TYPE_LEGACY_INV], dtype=torch.long)
            return self._merge_forced_condition_fields(graph)
        if condition_mode == 2:
            graph.use_residue_type_feature = torch.tensor([True], dtype=torch.bool)
            graph.optional_condition_type = torch.tensor([self.TYPE_LEGACY_FOLD], dtype=torch.long)
            return self._merge_forced_condition_fields(graph)

        total_sparse_prob = max(self.joint_prob, 0.0) + max(self.independent_prob, 0.0)
        use_joint = total_sparse_prob <= 0 or (random.random() * total_sparse_prob < max(self.joint_prob, 0.0))
        condition_type = (
            self._apply_joint_sparse_condition(graph)
            if use_joint
            else self._apply_independent_sparse_condition(graph)
        )
        graph.optional_condition_type = torch.tensor([condition_type], dtype=torch.long)
        return self._merge_forced_condition_fields(graph)


class MotifMaskTransform(T.BaseTransform):
    """
    Creates a motif mask for a protein structure, supporting multiple residue and atom selection strategies.

    Args:
        atom_selection_mode (str): How to select atoms within each motif residue. Options:
            - "all": Select all available atoms.
            - "ca_only": Select only CA atom if available.

    Returns:
        The input graph with the following attributes added:
            - motif_mask: Binary tensor of shape (num_res, 37) indicating which atoms are in the motif.
            - x_motif: Tensor containing fixed coordinates for the motif.
            - seq_motif_mask: Binary tensor of shape (num_res) indicating which residues are in the motif.
            - seq_motif: Tensor containing the motif sequence.
    """

    def __init__(
        self,
        atom_selection_mode: Literal[
            "ca_only",
            "all",
        ] = "all",
    ):
        self.atom_selection_mode = atom_selection_mode
        self.ca_index = residue_constants.atom_types.index("CA")

    def _select_atoms(
        self, available_atoms: torch.Tensor, residue_idx: int = None, graph: Data = None
    ) -> List[int]:
        """Select atoms for a residue based on the specified mode.

        Args:
            available_atoms (torch.Tensor): Tensor of available atom indices.
            residue_idx (int, optional): Residue index in the graph (needed for bond_graph mode).
            graph (Data, optional): The full graph (needed for bond_graph mode).

        Returns:
            List[int]: List of selected atom indices.
        """
        if self.atom_selection_mode == "all":
            # Select all available atoms
            return available_atoms.tolist()

        elif self.atom_selection_mode == "ca_only":
            # Select only CA atom if available
            return [self.ca_index] if self.ca_index in available_atoms else []

        else:
            raise ValueError(f"Unknown atom selection mode: {self.atom_selection_mode}")

    def __call__(self, graph: Data) -> Data:
        assert graph.latent_cond_mask is not None, "latent_cond_mask must be set by ConditionMaskTransform before MotifMaskTransform"
        motif_sequence_mask = graph.latent_cond_mask # only select residues marked in latent_cond_mask, including receptor.
        motif_mask = torch.zeros_like(graph.coord_mask)
        for res_idx in torch.where(motif_sequence_mask)[0]:
            available_atoms = torch.where(graph.coord_mask[res_idx])[0]
            if len(available_atoms) == 0:
                continue
            else:
                selected_atoms = self._select_atoms(available_atoms)
            motif_mask[res_idx, selected_atoms] = True
        motif_mask = motif_mask.bool()
        graph.motif_mask = motif_mask.bool()  # [n, 37]
        graph.x_motif = graph.coords_nm * graph.motif_mask[..., None]  # [n, 37, 3]
        graph.seq_motif_mask = motif_mask.sum(dim=-1).bool()  # [n]
        graph.seq_motif = graph.residue_type * graph.seq_motif_mask  # [n]
        return graph
    

class CroppingTransform(T.BaseTransform):
    """
    Contiguous cropping on the binder chain, and both contiguous and spatial cropping on the target chains.

    1. Based on the interface threshold, select interface residues.
    2. Randomly select one residue from the interface residues and use its corresponding chain as the binder chain, and all the other chains as the target chains.
    3. Do contiguous cropping on the binder chain, with its length being randomly sampled from Uniform(binder_min_length, binder_max_length).
        Minimal number of residues (=binder_padding_length) on the both sides of the binder seed residue on the binder chain will be included.
    4. For the target chains, will do both contiguous cropping and spatial cropping.
        a): First do spatial cropping on the target chains, select target residues with distance to the whole binder chain less than target_spatial_crop_threshold.
            If there are more than (crop_size - binder_length) residues selected, take the top (crop_size - binder_length) residues.
        b): Then do contiguous cropping on the target chains for each segments of the target chains
    """

    def __init__(
        self,
        crop_size: int = 384,
        interface_threshold: float = 8.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        data_mode: Literal["bb_ca", "all-atom"] = "all-atom",
        binder_min_length: int = 12,
        binder_max_length: int = 192,
        binder_padding_length: int = 0,
        target_spatial_crop_threshold: float = 15.0,
        target_min_length: int = 50,
        enforce_target_min_length: bool = False,
        max_num_target_chains: int = -1,  # -1 for no limit
        shuffle_dimer_chains: bool = True,
        max_residues: int | None = None,
        short_binder_ratio_threshold: float = 2.5,
        short_binder_min_length_cutoff: int = 100,
        auto_assign_short_peptide_chain: bool = True,
        short_peptide_max_length: int | None = 30,
        short_peptide_receptor_min_length: int = 30,
        no_crop_if_all_chains_below_length: int | None = 30,
        preserve_binder_below_length: int | None = 30,
    ):
        """
        Args:
            crop_size: the total number of residues to keep after cropping. Default is 384.
            interface_threshold: the threshold (in Angstrom) for interface residues. Default is 5.0.
            dist_mode: the distance mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom for computing the distance between residues. Default is "bb_ca".
            data_mode: the data mode to use, "bb_ca" for backbone-CA, "all-atom" for all-atom. Default is "all-atom".
            binder_min_length: the minimum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 50.
            binder_max_length: the maximum length of the binder chain, binder length will be randomly sampled from Uniform(binder_min_length, binder_max_length). Default is 200.
            binder_padding_length: the minimal number of residues to include on each side of the binder seed residue when cropping the binder chain. Default is 10.
            target_spatial_crop_threshold: the threshold (in Angstrom) for target spatial cropping, which is the cutoff distanceof target residues to the whole binder chain. Default is 15.0.
            enforce_target_min_length: whether to enforce the target chain to have at least target_min_length residues after cropping.
                If True, target might have more chains than max_num_target_chains. Default is True.
            max_num_target_chains: the maximum number of target chains to include in the cropping, -1 for no limit. Default is -1.
            max_residues: optional hard cap for output residues, usually synced from datamodule.pad_max_size.
            short_binder_ratio_threshold: for imbalanced dimers, force the shorter chain to be binder when max/min exceeds this value.
            short_binder_min_length_cutoff: only force the shorter chain as binder when the shorter chain has fewer residues than this cutoff.
            auto_assign_short_peptide_chain: for two-chain complexes, mark the unique short chain as binder/peptide.
            short_peptide_max_length: maximum length for the automatically assigned peptide chain.
            short_peptide_receptor_min_length: minimum length for the partner chain when auto-assigning a peptide.
            no_crop_if_all_chains_below_length: for two-chain complexes where both chains are shorter than this, skip cropping.
            preserve_binder_below_length: keep the full binder chain when it is shorter than this length.
        """
        self.crop_size = crop_size
        self.max_residues = max_residues
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.data_mode = data_mode
        self.binder_min_length = binder_min_length
        self.binder_max_length = binder_max_length
        self.binder_padding_length = binder_padding_length
        self.target_spatial_crop_threshold = target_spatial_crop_threshold
        self.target_min_length = target_min_length
        self.enforce_target_min_length = enforce_target_min_length
        self.max_num_target_chains = max_num_target_chains
        self.shuffle_dimer_chains = shuffle_dimer_chains
        self.short_binder_ratio_threshold = short_binder_ratio_threshold
        self.short_binder_min_length_cutoff = short_binder_min_length_cutoff
        self.auto_assign_short_peptide_chain = bool(auto_assign_short_peptide_chain)
        self.short_peptide_max_length = (
            None if short_peptide_max_length is None else int(short_peptide_max_length)
        )
        self.short_peptide_receptor_min_length = int(short_peptide_receptor_min_length)
        self.no_crop_if_all_chains_below_length = (
            None if no_crop_if_all_chains_below_length is None else int(no_crop_if_all_chains_below_length)
        )
        self.preserve_binder_below_length = (
            None if preserve_binder_below_length is None else int(preserve_binder_below_length)
        )
        assert self.binder_min_length >= 2 * self.binder_padding_length + 1, (
            "binder_min_length must be >= 2 * binder_padding_length + 1"
        )

    def set_pad_max_size(self, pad_max_size: int | None) -> None:
        self.max_residues = pad_max_size

    def _effective_crop_size(self) -> int:
        if self.max_residues is None:
            return self.crop_size
        return min(self.crop_size, int(self.max_residues))

    def _swap_peptide_target(self, graph: Data) -> None:
        chain_ids = [str(c) for c in graph.orig_chain_id]
        is_peptide = graph.is_peptide.detach().cpu().tolist()
        peptide_chain_ids = sorted({c for c, flag in zip(chain_ids, is_peptide) if flag == 1})
        target_chain_ids = sorted({c for c, flag in zip(chain_ids, is_peptide) if flag == 0})
        peptide_chain_id = peptide_chain_ids[0] if len(peptide_chain_ids) == 1 else "L"
        target_chain_id = target_chain_ids[0] if len(target_chain_ids) == 1 else "A"
        graph.is_peptide = 1 - graph.is_peptide
        graph.orig_chain_id = [
            target_chain_id if c == peptide_chain_id else peptide_chain_id if c == target_chain_id else c
            for c in chain_ids
        ]

    def _num_orig_chains(self, graph: Data) -> int:
        orig_chain_id = getattr(graph, "orig_chain_id", None)
        if orig_chain_id is not None:
            return len(set(map(str, orig_chain_id)))
        return int(torch.unique(graph.chains).numel())

    @staticmethod
    def _chain_lengths(graph: Data) -> dict[int, int]:
        asym_id = graph.chains
        return {
            int(chain_id): int((asym_id == chain_id).sum().item())
            for chain_id in torch.unique(asym_id).sort().values.tolist()
        }

    def _should_skip_short_dimer_crop(self, graph: Data) -> bool:
        if self.no_crop_if_all_chains_below_length is None:
            return False
        chain_lengths = self._chain_lengths(graph)
        if len(chain_lengths) != 2:
            return False
        return all(
            length < self.no_crop_if_all_chains_below_length
            for length in chain_lengths.values()
        )

    def _mark_chain_as_peptide(self, graph: Data, peptide_chain_id: int) -> None:
        asym_id = graph.chains
        peptide_mask = asym_id == int(peptide_chain_id)
        graph.is_peptide = peptide_mask.to(dtype=torch.long)

        orig_chain_id = getattr(graph, "orig_chain_id", None)
        if isinstance(orig_chain_id, list) and len(orig_chain_id) == len(asym_id):
            graph.orig_chain_id = [
                "L" if bool(is_peptide) else ("A" if str(chain_id) == "L" else str(chain_id))
                for chain_id, is_peptide in zip(orig_chain_id, peptide_mask.tolist())
            ]

    def _auto_assign_short_peptide_chain(self, graph: Data) -> None:
        if not self.auto_assign_short_peptide_chain or self.short_peptide_max_length is None:
            return
        chain_lengths = self._chain_lengths(graph)
        if len(chain_lengths) != 2:
            return
        sorted_chains = sorted(chain_lengths.items(), key=lambda item: item[1])
        (short_chain_id, short_len), (_, long_len) = sorted_chains
        if short_len >= self.short_peptide_max_length:
            return
        if long_len < self.short_peptide_receptor_min_length:
            return
        if short_len == long_len:
            return
        self._mark_chain_as_peptide(graph, peptide_chain_id=short_chain_id)

    def _orient_dimer_binder(self, graph: Data) -> None:
        num_chains = self._num_orig_chains(graph)
        if num_chains != 2:
            if self.shuffle_dimer_chains:
                raise ValueError(
                    f"shuffle_dimer_chains=True requires exactly two chains, "
                    f"but got {num_chains} chains for {getattr(graph, 'id', '<unknown>')}."
                )
            return

        peptide_len = int((graph.is_peptide == 1).sum().item())
        target_len = int((graph.is_peptide == 0).sum().item())
        if peptide_len > 0 and target_len > 0:
            min_len = min(peptide_len, target_len)
            max_len = max(peptide_len, target_len)
            ratio = max_len / min_len
            force_short_binder = (
                ratio > self.short_binder_ratio_threshold
                and min_len < self.short_binder_min_length_cutoff
            )
            if force_short_binder:
                if peptide_len > target_len:
                    self._swap_peptide_target(graph)
                return

        if self.shuffle_dimer_chains and random.random() < 0.5:
            self._swap_peptide_target(graph)

    def __call__(self, graph: Data) -> Data:
        crop_size = self._effective_crop_size()
        if crop_size < self.binder_min_length:
            raise ValueError(
                f"CroppingTransform effective crop_size={crop_size} is smaller than "
                f"binder_min_length={self.binder_min_length}. Increase pad_max_size/crop_size "
                "or lower binder_min_length."
            )
        if self._should_skip_short_dimer_crop(graph):
            return graph
        self._auto_assign_short_peptide_chain(graph)
        self._orient_dimer_binder(graph)
        asym_id = graph.chains
        n_res = len(graph.chains)
        residues_idxs = torch.arange(n_res)
        chain_lens = torch.bincount(asym_id)
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        remaining_idxs = torch.ones(n_res, dtype=torch.bool)
        chain_residue_ranges = {}  # {chain_id: [start_idx, end_idx]}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [
                int(chain_indices.min()),
                int(chain_indices.max()),
            ]

        # get interface residues, and residue-wise distances: (n_res, n_res)
        interface_residues, min_dist_per_res = self._get_interface_residues(graph)
        # filter out interface residues that are from too short chains
        chain_of_interface_residues = asym_id[interface_residues]
        binder_candidate_seed_residues = interface_residues[
            chain_lens[chain_of_interface_residues] >= self.binder_min_length
        ]

        if len(binder_candidate_seed_residues) == 0:
            # Select the top k residues in min_dist_per_res and generate the chain_of_interface_residues
            k = min(10, min_dist_per_res.shape[0])  # or another value for k as appropriate
            # Get the indices of the k smallest min_dist_per_res values (i.e., closest interface residues)
            topk_vals, topk_indices = torch.topk(min_dist_per_res, k, largest=False)
            interface_residues = topk_indices
            chain_of_interface_residues = asym_id[interface_residues]
            binder_candidate_seed_residues = interface_residues[
                chain_lens[chain_of_interface_residues] >= self.binder_min_length
            ]
            # raise ValueError(f"No binder candidate seed residues found for pdb: {graph.id}")
            logger.warning(
                f"No binder candidate seed residues found for pdb: {graph.id}, using top {k} interface residues as binder candidate seed residues, with min interface distance: {topk_vals.min().item():.2f}A"
            )

        peptide_mask = graph.is_peptide == 1
        target_mask = graph.is_peptide == 0
        if peptide_mask.any():
            peptide_residues = torch.nonzero(peptide_mask, as_tuple=True)[0]
            peptide_interface_residues = interface_residues[peptide_mask[interface_residues]]
            if len(peptide_interface_residues) > 0:
                binder_seed_residue = random.choice(peptide_interface_residues.tolist())
            else:
                binder_seed_residue = random.choice(peptide_residues.tolist())
            binder_chain_id = int(asym_id[peptide_mask][0])
        else:
            raise ValueError(
                "No peptide residues found in the PDB structure. "
                "Please check the PDB structure and try again."
            )

        if (n_res - int(chain_lens[binder_chain_id])) < self.target_min_length and self.enforce_target_min_length:
            raise ValueError(f"Target chain too short for pdb: {graph.id}")

        if len(graph.chains) < crop_size:
            return graph

        # Contiguous cropping on the binder chain
        binder_chain_start = chain_residue_ranges[binder_chain_id][0]
        binder_chain_end = chain_residue_ranges[binder_chain_id][1]
        binder_chain_length = binder_chain_end - binder_chain_start + 1
        if (
            self.preserve_binder_below_length is not None
            and binder_chain_length < self.preserve_binder_below_length
        ):
            binder_chain_crop_idxs = torch.arange(binder_chain_start, binder_chain_end + 1)
        else:
            binder_max_length = min(self.binder_max_length, crop_size)
            binder_length = int(torch.randint(self.binder_min_length, binder_max_length + 1, (1,))[0])
            binder_left_length = int(
                torch.randint(
                    self.binder_padding_length,
                    binder_length - self.binder_padding_length - 1 + 1,
                    (1,),
                )[0]
            )
            binder_chain_crop_start = max(binder_chain_start, binder_seed_residue - binder_left_length)
            binder_chain_crop_end = min(binder_chain_end + 1, binder_chain_crop_start + binder_length)
            if binder_chain_crop_end == binder_chain_end + 1:
                binder_chain_crop_start = max(binder_chain_start, binder_chain_crop_end - binder_length)
            binder_chain_crop_idxs = torch.arange(binder_chain_crop_start, binder_chain_crop_end)
        selected_idxs[binder_chain_crop_idxs] = True

        # Spatial cropping on the target chains first to get the target seed residues
        to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
        target_seed_residues = residues_idxs[target_mask][to_binder_dist < self.target_spatial_crop_threshold]
        if len(target_seed_residues) == 0 and to_binder_dist.numel() > 0:
            target_seed_residues = residues_idxs[target_mask][torch.argmin(to_binder_dist).reshape(1)]
        target_seed_residue_chains = asym_id[target_seed_residues]
        target_chains = target_seed_residue_chains.unique()
        target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        if self.max_num_target_chains > 0 and len(target_chains) > self.max_num_target_chains:
            # can't use random.shuffle to shuffle torch tensors.
            target_chains = (
                target_chains[torch.randperm(len(target_chains))][: self.max_num_target_chains].sort().values
            )
            target_seed_residues = target_seed_residues[
                (target_seed_residue_chains[:, None] == target_chains[None, :]).any(dim=-1)
            ]
            target_seed_residue_chains = asym_id[target_seed_residues]
            target_mask = (asym_id[:, None] == target_chains[None, :]).any(dim=-1)

        # put chains not included in the target_chains to be False in remaining_idxs
        remaining_idxs[(asym_id[:, None] != target_chains[None, :]).all(dim=-1)] = False
        num_budget = crop_size - int(selected_idxs.sum())
        num_remaining = int(remaining_idxs.sum())

        if num_remaining <= num_budget:
            selected_idxs[residues_idxs[target_mask]] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        if len(target_seed_residues) >= num_budget:
            to_binder_dist = (min_dist_per_res[target_mask][:, binder_chain_crop_idxs]).min(dim=-1)[0]
            target_seed_residues = residues_idxs[target_mask][torch.argsort(to_binder_dist)[:num_budget]]
            selected_idxs[target_seed_residues] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

        # Get the segments of the target chains
        segments = []
        for chain_id in target_chains:
            for segment in self._convert_to_segments(
                target_seed_residues[target_seed_residue_chains == chain_id].tolist()
            ):
                segments.append(segment)

        # Contiguous cropping on the target chains from the target seed residues/segments
        remaining_idxs[target_seed_residues] = False
        selected_idxs[target_seed_residues] = True
        segment_idxs = list(range(len(segments)))
        segment_used_mask = torch.zeros((len(segments), 2), dtype=torch.bool)  # [left_crop, right_crop]
        random.shuffle(segment_idxs)

        for segment_idx in segment_idxs:
            segment_start, segment_end = segments[segment_idx]
            chain_id = int(asym_id[segment_start])
            chain_start = chain_residue_ranges[chain_id][0]
            chain_end = chain_residue_ranges[chain_id][1]

            orders = [0, 1]
            random.shuffle(orders)
            for order in orders:
                if order == 0:
                    # left crop
                    segment_used_mask[segment_idx, 0] = True
                    if segment_idx == 0:
                        remaining_idxs[chain_start:segment_start] = False
                    else:
                        if segment_used_mask[segment_idx - 1, 1]:  # and segment_used_mask[segment_idx, 0]
                            remaining_idxs[segments[segment_idx - 1][1] : segment_start] = False
                        elif segments[segment_idx - 1][1] < chain_start:
                            remaining_idxs[chain_start:segment_start] = False
                    num_budget = crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx >= 1:
                        segment_crop_size_max = min(
                            num_budget,
                            segment_start - chain_start,
                            int((~selected_idxs[(segments[segment_idx - 1][1] + 1) : segment_start]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, segment_start - chain_start)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_start - segment_crop_size) : (segment_start)] = True
                    remaining_idxs[(segment_start - segment_crop_size) : (segment_start)] = False

                else:
                    # right crop
                    segment_used_mask[segment_idx, 1] = True
                    if segment_idx == len(segment_idxs) - 1:
                        remaining_idxs[segment_end : chain_end + 1] = False
                    else:
                        if segment_used_mask[segment_idx + 1, 0]:  # and segment_used_mask[segment_idx, 1]
                            remaining_idxs[segment_end : segments[segment_idx + 1][0]] = False
                        elif segments[segment_idx + 1][0] > chain_end:
                            remaining_idxs[segment_end : chain_end + 1] = False
                    num_budget = crop_size - int(selected_idxs.sum())
                    num_remaining = int(remaining_idxs.sum())
                    if segment_idx < len(segment_idxs) - 1:
                        segment_crop_size_max = min(
                            num_budget,
                            chain_end - segment_end,
                            int((~selected_idxs[(segment_end + 1) : segments[segment_idx + 1][0]]).sum()),
                        )
                    else:
                        segment_crop_size_max = min(num_budget, chain_end - segment_end)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                    selected_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = True
                    remaining_idxs[(segment_end + 1) : (segment_end + 1 + segment_crop_size)] = False

        graph = self._crop_graph(graph, residues_idxs[selected_idxs])

        return graph

    @staticmethod
    def _convert_to_segments(sorted_integers):
        """
        Convert a list of sorted integers to a combination of ranges.

        Args:
            sorted_integers: List of sorted integers

        Returns:
            List of tuples, each tuple is a range of integers
        """
        segments = []
        start = sorted_integers[0]
        end = sorted_integers[0]

        for i in range(1, len(sorted_integers)):
            if sorted_integers[i] == end + 1:
                end = sorted_integers[i]
            else:
                segments.append((start, end))
                start = sorted_integers[i]
                end = sorted_integers[i]

        # Add the last range
        segments.append((start, end))

        return segments

    def _get_interface_residues(self, graph: Data) -> torch.Tensor:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = residue_constants.atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)  # .min(dim=-1)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(
                    positions.view(n_res * n_atom, -1),
                    positions.view(n_res * n_atom, -1),
                )
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")

        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]

        return graph


# I am not so sure if I should use that
class InterfaceCroppingTransform(T.BaseTransform):
    """
    Interface-aware cropping that guarantees all interface residues are preserved.

    1. Identifies all interface residues (< interface_threshold Angstrom)
    2. For binder: finds minimal contiguous interval covering all interface residues, then expands
    3. For target: uses interface residues as seeds, then does spatial + contiguous expansion
    """

    def __init__(
        self,
        crop_size: int = 384,
        interface_threshold: float = 8.0,
        dist_mode: Literal["bb_ca", "all-atom"] = "bb_ca",
        binder_min_length: int = 10,
        binder_max_length: int = 80,
        binder_min_interface_ratio: float = 0.5,
        binder_enforce_mode: Literal["ratio", "length"] = "length",
        target_spatial_crop_threshold: float = 15.0,
        target_min_length: int = 50,
        enforce_target_min_length: bool = False,
        max_num_target_chains: int = -1,
        force_complete_binder: bool = False,
    ):
        self.crop_size = crop_size
        self.interface_threshold = interface_threshold
        self.dist_mode = dist_mode
        self.binder_min_length = binder_min_length
        self.binder_max_length = binder_max_length
        self.binder_min_interface_ratio = binder_min_interface_ratio
        self.binder_enforce_mode = binder_enforce_mode
        self.target_spatial_crop_threshold = target_spatial_crop_threshold
        self.target_min_length = target_min_length
        self.enforce_target_min_length = enforce_target_min_length
        self.max_num_target_chains = max_num_target_chains
        self.force_complete_binder = force_complete_binder

    def __call__(self, graph: Data) -> Data:
        if graph.id.startswith("pdb") or len(graph.id) == 6:
            return self._crop_peptide_data(graph)
        else:
            asym_id = graph.chains
            n_res = len(graph.chains)
            residues_idxs = torch.arange(n_res)
            peptide_mask = graph.is_peptide == 1
            target_mask = graph.is_peptide == 0
            if not peptide_mask.any():
                raise ValueError("No peptide residues found")
            interface_residues, min_dist_per_res = self._get_interface_residues(graph)
            if len(interface_residues) == 0:
                raise ValueError(f"No interface residues found for {graph.id}")
            binder_interface = interface_residues[peptide_mask[interface_residues]]
            target_interface = interface_residues[target_mask[interface_residues]]
            if len(binder_interface) == 0:
                raise ValueError(f"No binder interface residues for {graph.id}")
            if len(target_interface) == 0:
                raise ValueError(f"No target interface residues for {graph.id}")
            if self.force_complete_binder:
                binder_crop_idxs = self._get_complete_binder_crop_idxs(
                    graph=graph,
                    peptide_mask=peptide_mask,
                )
            else:
                binder_crop_idxs = self._crop_binder(
                    graph, binder_interface, peptide_mask, asym_id, min_dist_per_res
                )
            target_crop_idxs = self._crop_target(
                graph, target_interface, binder_crop_idxs, target_mask, asym_id, residues_idxs, min_dist_per_res
            )
            selected_idxs = torch.zeros(n_res, dtype=torch.bool)
            selected_idxs[binder_crop_idxs] = True
            selected_idxs[target_crop_idxs] = True
            graph = self._crop_graph(graph, residues_idxs[selected_idxs])
            return graph

    def _crop_peptide_data(self, graph: Data) -> Data:
        asym_id = graph.chains
        n_res = len(graph.chains)
        if n_res <= self.crop_size:
            return graph
        residues_idxs = torch.arange(n_res)
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        peptide_mask = graph.is_peptide == 1
        target_mask = graph.is_peptide == 0
        if not peptide_mask.any():
            raise ValueError("No peptide residues found")
        if self.force_complete_binder:
            binder_idxs = self._get_complete_binder_crop_idxs(
                graph=graph,
                peptide_mask=peptide_mask,
            )
        else:
            binder_idxs = torch.nonzero(peptide_mask, as_tuple=True)[0]
            if len(binder_idxs) > self.crop_size:
                binder_idxs = binder_idxs[:self.crop_size]
        selected_idxs[binder_idxs] = True
        num_budget = self.crop_size - int(selected_idxs.sum())
        if num_budget <= 0:
            return self._crop_graph(graph, residues_idxs[selected_idxs])
        _, min_dist_per_res = self._get_interface_residues(graph)
        to_binder_dist = min_dist_per_res[target_mask][:, binder_idxs].min(dim=-1)[0]
        target_residues = residues_idxs[target_mask]
        core_interface_mask = to_binder_dist < self.interface_threshold
        core_interface_residues = target_residues[core_interface_mask]
        core_dists = to_binder_dist[core_interface_mask]
        sorted_core_idxs = torch.argsort(core_dists)
        core_interface_sorted = core_interface_residues[sorted_core_idxs]
        actual_core_to_add = core_interface_sorted[:num_budget]
        selected_idxs[actual_core_to_add] = True
        num_budget -= len(actual_core_to_add)
        if num_budget <= 0:
            return self._crop_graph(graph, residues_idxs[selected_idxs])
        chain_residue_ranges = {}
        for chain_id in torch.unique(asym_id).sort().values.tolist():
            chain_indices = residues_idxs[asym_id == chain_id]
            if len(chain_indices) > 0:
                chain_residue_ranges[chain_id] = [int(chain_indices.min()), int(chain_indices.max())]
        target_chains = asym_id[actual_core_to_add].unique()
        for chain_id in target_chains.tolist():
            if num_budget <= 0: break
            chain_start, chain_end = chain_residue_ranges[chain_id]
            chain_seed_residues = actual_core_to_add[asym_id[actual_core_to_add] == chain_id]
            segments = self._convert_to_segments(sorted(chain_seed_residues.tolist()))
            for segment_start, segment_end in segments:
                if num_budget <= 0: break
                expand_left = min(num_budget, segment_start - chain_start)
                if expand_left > 0:
                    selected_idxs[segment_start - expand_left : segment_start] = True
                    num_budget -= expand_left
                if num_budget <= 0: break
                expand_right = min(num_budget, chain_end - segment_end)
                if expand_right > 0:
                    selected_idxs[segment_end + 1 : segment_end + 1 + expand_right] = True
                    num_budget -= expand_right
        if num_budget > 0:
            sorted_dist_all_idxs = torch.argsort(to_binder_dist)
            sorted_all_targets = target_residues[sorted_dist_all_idxs]
            unselected_mask = ~selected_idxs[sorted_all_targets]
            unselected_targets = sorted_all_targets[unselected_mask]
            selected_idxs[unselected_targets[:num_budget]] = True
        return self._crop_graph(graph, residues_idxs[selected_idxs])

    def _get_complete_binder_crop_idxs(
        self,
        graph: Data,
        peptide_mask: torch.Tensor,
    ) -> torch.Tensor:
        binder_idxs = torch.nonzero(peptide_mask, as_tuple=True)[0]
        if len(binder_idxs) == 0:
            raise ValueError("No peptide residues found")
        if len(binder_idxs) > self.crop_size:
            raise ValueError(
                f"Cannot keep full binder for {graph.id}: binder length {len(binder_idxs)} "
                f"exceeds crop_size {self.crop_size}"
            )
        return binder_idxs
    
    def _crop_binder(self, graph: Data, binder_interface: torch.Tensor, peptide_mask: torch.Tensor,
                     asym_id: torch.Tensor, min_dist_per_res: torch.Tensor) -> torch.Tensor:
        binder_chain_id = int(asym_id[peptide_mask][0])
        binder_chain_mask = asym_id == binder_chain_id
        binder_residues = torch.nonzero(binder_chain_mask, as_tuple=True)[0]
        chain_start = int(binder_residues.min())
        chain_end = int(binder_residues.max())
        interface_segments = self._convert_to_segments(sorted(binder_interface.tolist()))
        interface_set = set(binder_interface.tolist())
        long_segments = [[s[0], s[1]] for s in interface_segments if (s[1] - s[0] + 1) > 5]
        if not long_segments:
            longest_seg = max(interface_segments, key=lambda s: s[1] - s[0] + 1)
            long_segments = [[longest_seg[0], longest_seg[1]]]
        if len(long_segments) >= 2 and random.random() < 0.5:
            for i in range(len(long_segments) - 1):
                seg1_start, seg1_end = long_segments[i][0], long_segments[i][1]
                seg2_start, seg2_end = long_segments[i + 1][0], long_segments[i + 1][1]
                merged_start, merged_end = seg1_start, seg2_end
                merged_length = merged_end - merged_start + 1
                merged_n_interface = sum(1 for j in range(merged_start, merged_end + 1) if j in interface_set)
                merged_ratio = merged_n_interface / merged_length
                if merged_ratio >= self.binder_min_interface_ratio:
                    long_segments = [long_segments[i][:]]
                    long_segments[0][1] = seg2_end
                    break
        selected_seg = random.choice(long_segments)
        crop_start, crop_end = selected_seg[0], selected_seg[1] + 1
        n_interface = sum(1 for i in range(crop_start, crop_end) if i in interface_set)
        target_length = int(torch.randint(self.binder_min_length, self.binder_max_length + 1, (1,))[0])
        if self.binder_enforce_mode == "length":
            while (crop_end - crop_start) < self.binder_min_length:
                can_expand_left = crop_start > chain_start
                can_expand_right = crop_end <= chain_end
                if not can_expand_left and not can_expand_right:
                    break
                expand_left = can_expand_left and (not can_expand_right or random.random() < 0.5)
                if expand_left:
                    crop_start -= 1
                    if crop_start in interface_set:
                        n_interface += 1
                else:
                    if crop_end in interface_set:
                        n_interface += 1
                    crop_end += 1
            while (crop_end - crop_start) < target_length:
                can_expand_left = crop_start > chain_start
                can_expand_right = crop_end <= chain_end
                if not can_expand_left and not can_expand_right:
                    break
                expand_left = can_expand_left and (not can_expand_right or random.random() < 0.5)
                if expand_left:
                    new_interface = n_interface + (1 if crop_start - 1 in interface_set else 0)
                    new_ratio = new_interface / (crop_end - crop_start + 1)
                    if new_ratio < self.binder_min_interface_ratio:
                        break
                    crop_start -= 1
                    n_interface = new_interface
                else:
                    new_interface = n_interface + (1 if crop_end in interface_set else 0)
                    new_ratio = new_interface / (crop_end - crop_start + 1)
                    if new_ratio < self.binder_min_interface_ratio:
                        break
                    n_interface = new_interface
                    crop_end += 1

        else:  # enforce_mode == "ratio"
            while (crop_end - crop_start) < target_length:
                can_expand_left = crop_start > chain_start
                can_expand_right = crop_end <= chain_end
                if not can_expand_left and not can_expand_right:
                    break
                expand_left = can_expand_left and (not can_expand_right or random.random() < 0.5)
                if expand_left:
                    new_interface = n_interface + (1 if crop_start - 1 in interface_set else 0)
                    new_ratio = new_interface / (crop_end - crop_start + 1)
                    if new_ratio < self.binder_min_interface_ratio:
                        break
                    crop_start -= 1
                    n_interface = new_interface
                else:
                    new_interface = n_interface + (1 if crop_end in interface_set else 0)
                    new_ratio = new_interface / (crop_end - crop_start + 1)
                    if new_ratio < self.binder_min_interface_ratio:
                        break
                    n_interface = new_interface
                    crop_end += 1

        return torch.arange(crop_start, crop_end)

    def _crop_target(self, graph: Data, target_interface: torch.Tensor, binder_crop_idxs: torch.Tensor,
                     target_mask: torch.Tensor, asym_id: torch.Tensor, residues_idxs: torch.Tensor,
                     min_dist_per_res: torch.Tensor) -> torch.Tensor:
        n_res = len(asym_id)
        selected_idxs = torch.zeros(n_res, dtype=torch.bool)
        remaining_idxs = torch.ones(n_res, dtype=torch.bool)
        remaining_idxs[~target_mask] = False
        selected_idxs[target_interface] = True
        to_binder_dist = (min_dist_per_res[target_mask][:, binder_crop_idxs]).min(dim=-1)[0]
        spatial_residues = residues_idxs[target_mask][to_binder_dist < self.target_spatial_crop_threshold]
        target_chains = asym_id[spatial_residues].unique()
        if self.max_num_target_chains > 0 and len(target_chains) > self.max_num_target_chains:
            target_chains = target_chains[torch.randperm(len(target_chains))][:self.max_num_target_chains].sort().values
        chain_residue_ranges = {}
        for chain_id in target_chains.tolist():
            chain_mask = asym_id == chain_id
            chain_indices = residues_idxs[chain_mask]
            chain_residue_ranges[chain_id] = [int(chain_indices.min()), int(chain_indices.max())]
        num_budget = self.crop_size - len(binder_crop_idxs)
        num_remaining = int(remaining_idxs.sum())
        if num_remaining <= num_budget:
            selected_idxs[residues_idxs[target_mask]] = True
            return torch.nonzero(selected_idxs, as_tuple=True)[0]
        if len(spatial_residues) >= num_budget:
            spatial_dists = to_binder_dist[to_binder_dist < self.target_spatial_crop_threshold]
            spatial_residues = spatial_residues[torch.argsort(spatial_dists)[:num_budget]]
            selected_idxs[spatial_residues] = True
            return torch.nonzero(selected_idxs, as_tuple=True)[0]
        segments = []
        for chain_id in target_chains.tolist():
            chain_spatial = spatial_residues[asym_id[spatial_residues] == chain_id]
            if len(chain_spatial) > 0:
                for seg in self._convert_to_segments(chain_spatial.tolist()):
                    segments.append(seg)
        remaining_idxs[spatial_residues] = False
        selected_idxs[spatial_residues] = True
        segment_idxs = list(range(len(segments)))
        random.shuffle(segment_idxs)
        for segment_idx in segment_idxs:
            segment_start, segment_end = segments[segment_idx]
            chain_id = int(asym_id[segment_start])
            chain_start, chain_end = chain_residue_ranges[chain_id]
            for order in [0, 1] if random.random() < 0.5 else [1, 0]:
                num_budget = self.crop_size - int(selected_idxs.sum())
                num_remaining = int(remaining_idxs.sum())
                if num_budget <= 0:
                    break
                if order == 0:  # Left
                    segment_crop_size_max = min(num_budget, segment_start - chain_start)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    if segment_crop_size_max > 0:
                        segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                        selected_idxs[(segment_start - segment_crop_size):segment_start] = True
                        remaining_idxs[(segment_start - segment_crop_size):segment_start] = False
                else:  # Right
                    segment_crop_size_max = min(num_budget, chain_end - segment_end)
                    segment_crop_size_min = min(segment_crop_size_max, max(0, num_budget - num_remaining))
                    if segment_crop_size_max > 0:
                        segment_crop_size = int(torch.randint(segment_crop_size_min, segment_crop_size_max + 1, (1,))[0])
                        selected_idxs[(segment_end + 1):(segment_end + 1 + segment_crop_size)] = True
                        remaining_idxs[(segment_end + 1):(segment_end + 1 + segment_crop_size)] = False
        return torch.nonzero(selected_idxs, as_tuple=True)[0]

    @staticmethod
    def _convert_to_segments(sorted_integers):
        if len(sorted_integers) == 0:
            return []
        segments = []
        start = sorted_integers[0]
        end = sorted_integers[0]
        for i in range(1, len(sorted_integers)):
            if sorted_integers[i] == end + 1:
                end = sorted_integers[i]
            else:
                segments.append((start, end))
                start = sorted_integers[i]
                end = sorted_integers[i]
        segments.append((start, end))
        return segments

    def _get_interface_residues(self, graph: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        asym_id = graph.chains
        diff_chain_mask = asym_id[..., None, :] != asym_id[..., :, None]
        if self.dist_mode == "bb_ca":
            ca_idx = residue_constants.atom_order["CA"]
            ca_positions = graph.coords[..., ca_idx, :]
            ca_pairwise_dists = torch.cdist(ca_positions, ca_positions)
            min_dist_per_res = torch.where(diff_chain_mask, ca_pairwise_dists, torch.inf)
        elif self.dist_mode == "all-atom":
            positions = graph.coords
            n_res, n_atom, _ = positions.shape
            atom_mask = graph.coord_mask
            pairwise_dists = (
                torch.cdist(positions.view(n_res * n_atom, -1), positions.view(n_res * n_atom, -1))
                .view(n_res, n_atom, n_res, n_atom)
                .permute(0, 2, 1, 3)
            )
            pair_mask = atom_mask[None, :, None, :] * atom_mask[:, None, :, None]
            mask = diff_chain_mask[:, :, None, None] * pair_mask
            min_dist_per_res = torch.where(mask, pairwise_dists, torch.inf).min(dim=-1).values.min(dim=-1).values
        else:
            raise ValueError(f"Invalid dist mode: {self.dist_mode}")
        valid_interfaces = torch.sum((min_dist_per_res < self.interface_threshold).float(), dim=-1)
        interface_residues_idxs = torch.nonzero(valid_interfaces, as_tuple=True)[0]
        return interface_residues_idxs, min_dist_per_res

    def _crop_graph(self, graph: Data, crop_idxs: torch.Tensor) -> Data:
        num_residues = graph.coords.size(0)
        for key, value in graph:
            if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_residues:
                graph[key] = value[crop_idxs]
            elif isinstance(value, list) and len(value) == num_residues:
                graph[key] = [value[i] for i in crop_idxs]
        return graph
