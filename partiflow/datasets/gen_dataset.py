import json
import tempfile
import random
from typing import Callable, Dict, List, Optional
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from tqdm import tqdm
from loguru import logger

import biotite.structure.io as strucio
try:
    from graphein.protein.tensor.io import protein_to_pyg
except ImportError as e:
    protein_to_pyg = None
    logger.warning(
        "Graphein is not available; inference dataset generation from raw PDB will fail. "
        "Install graphein if you need to process raw PDB structures."
    )

from partiflow.datasets.base_data import BaseLightningDataModule
from partiflow.openfold.np.residue_constants import resname_to_idx, restypes
from partiflow.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from partiflow.utils.conditioning_utils import chemical_bond_type_to_id
from partiflow.design.tasks import build_atom_array, INFERENCE_TRANSFORMS
from partiflow.utils.secstruct_utils import binder_secstruct_fold_code_from_id


BINDER_SS_TOKEN_MASK = 3
BINDER_BLOCK_ADJ_MASK = 2


def _attach_binder_topology(
    graph: Data,
    *,
    binder_ss_token,
    binder_block_adj=None,
    sample_fn: str,
    task_name: str,
) -> None:
    binder_indices = torch.where(graph.is_peptide == 1)[0]
    binder_len = int(binder_indices.numel())
    n_res = int(graph.residue_type.shape[0])

    ss_token = torch.as_tensor(binder_ss_token, dtype=torch.long).reshape(-1)
    if int(ss_token.numel()) != binder_len:
        raise ValueError(
            f"binder_ss_token length {ss_token.numel()} does not match binder length {binder_len} "
            f"for sample_fn={sample_fn} task_name={task_name}"
        )

    if binder_block_adj is None:
        block_adj = torch.full((binder_len, binder_len), fill_value=BINDER_BLOCK_ADJ_MASK, dtype=torch.long)
    else:
        block_adj = torch.as_tensor(binder_block_adj, dtype=torch.long)
    if tuple(block_adj.shape) != (binder_len, binder_len):
        raise ValueError(
            f"binder_block_adj shape {tuple(block_adj.shape)} does not match "
            f"({binder_len}, {binder_len}) for sample_fn={sample_fn} task_name={task_name}"
        )

    full_ss_token = torch.full((n_res,), fill_value=BINDER_SS_TOKEN_MASK, dtype=torch.long)
    full_ss_token[binder_indices] = ss_token
    full_block_adj = torch.full((n_res, n_res), fill_value=BINDER_BLOCK_ADJ_MASK, dtype=torch.long)
    full_block_adj[binder_indices[:, None], binder_indices[None, :]] = block_adj

    graph.binder_ss_token = full_ss_token
    graph.binder_block_adj = full_block_adj


def _attach_chemical_bond_type(
    graph: Data,
    *,
    topology_entries,
    binder_pos_to_graph_idx: Dict[int, int],
    sample_fn: str,
    task_name: str,
) -> None:
    n_res = int(graph.residue_type.shape[0])
    chemical_bond_type = torch.zeros((n_res, n_res), dtype=torch.long)
    for entry in topology_entries or []:
        if not isinstance(entry, dict):
            continue
        topo_type = str(entry.get("type", "")).strip().lower().replace("-", "_").replace(" ", "_")
        if topo_type == "residue_template":
            continue
        bond_id = chemical_bond_type_to_id(topo_type)
        if bond_id <= 0:
            continue
        anchors = entry.get("anchors") or []
        if len(anchors) < 2:
            continue
        anchor_indices = []
        unresolved = []
        for pos in anchors:
            try:
                pos_i = int(pos)
            except (TypeError, ValueError):
                unresolved.append(pos)
                continue
            idx = binder_pos_to_graph_idx.get(pos_i, None)
            if idx is None:
                unresolved.append(pos_i)
            else:
                anchor_indices.append(idx)
        if unresolved:
            raise ValueError(
                f"Failed to align topology entry {entry} onto binder residues "
                f"for sample_fn={sample_fn} task_name={task_name}; unresolved anchors={unresolved}"
            )
        for i, idx_i in enumerate(anchor_indices):
            for idx_j in anchor_indices[i + 1 :]:
                chemical_bond_type[idx_i, idx_j] = int(bond_id)
                chemical_bond_type[idx_j, idx_i] = int(bond_id)
    graph.chemical_bond_type = chemical_bond_type


def _attach_residue_type_condition(
    graph: Data,
    *,
    task_type: str,
    residue_type_constraints: Dict[int, int],
    sample_fn: str,
    task_name: str,
) -> None:
    """Attach the authoritative peptide residue-identity condition.

    Known-sequence docking conditions the complete binder sequence. Sparse
    ``constraint.restype`` entries remain the condition source for other task
    types and serve as consistency checks when they overlap a docking sequence.
    """
    residue_type_condition = torch.full_like(graph.residue_type, fill_value=-1)
    residue_type_condition_mask = torch.zeros_like(graph.residue_type, dtype=torch.bool)

    peptide_mask = graph.is_peptide.bool()
    if task_type == "docking":
        invalid_target_mask = peptide_mask & (
            (graph.residue_type < 0) | (graph.residue_type >= len(restypes))
        )
        if invalid_target_mask.any():
            invalid_positions = graph.residue_pdb_idx[invalid_target_mask].tolist()
            raise ValueError(
                f"Task '{task_name}': docking peptide residue identities at positions "
                f"{invalid_positions} cannot be represented by the decoder for "
                f"sample_fn={sample_fn}. Register a supported non-standard residue proxy."
            )
        residue_type_condition[peptide_mask] = graph.residue_type[peptide_mask].long()
        residue_type_condition_mask[peptide_mask] = True

    unresolved_constraint_positions = set(int(pos) for pos in residue_type_constraints)
    for idx_res, (chain_id, res_id) in enumerate(
        zip(graph.orig_chain_id, graph.residue_pdb_idx.tolist())
    ):
        if chain_id != "L":
            continue
        aa_idx = residue_type_constraints.get(int(res_id), None)
        if aa_idx is None:
            continue
        if (
            residue_type_condition_mask[idx_res]
            and int(residue_type_condition[idx_res]) != int(aa_idx)
        ):
            raise ValueError(
                f"Task '{task_name}': explicit residue_type_constraint at binder position "
                f"{int(res_id)} conflicts with the docking sequence for sample_fn={sample_fn} "
                f"({int(aa_idx)} vs {int(residue_type_condition[idx_res])})."
            )
        residue_type_condition[idx_res] = int(aa_idx)
        residue_type_condition_mask[idx_res] = True
        unresolved_constraint_positions.discard(int(res_id))

    if unresolved_constraint_positions:
        raise ValueError(
            f"Failed to align residue_type_constraints {sorted(unresolved_constraint_positions)} "
            f"onto binder residues for sample_fn={sample_fn} task_name={task_name}"
        )

    graph.residue_type_condition = residue_type_condition
    graph.residue_type_condition_mask = residue_type_condition_mask


class InferenceDataset(Dataset):
    """Lightweight dataset that reads processed protein graphs from disk."""

    def __init__(
        self,
        processed_tasks: List[Dict],
        transform: Optional[Callable] = None,
        preload: bool = False,
    ) -> None:
        
        self.processed_tasks = processed_tasks
        self.transform = transform
        self.preload = bool(preload)
        self.memory: Optional[List[Data]] = None
        if self.preload:
            self.memory = [
                self._apply_transform(self._process_single_task(i))
                for i in tqdm(range(len(self.processed_tasks)))
            ]

    def __len__(self) -> int:
        return len(self.processed_tasks)

    def _apply_transform(self, sample: Dict[str, Data] | Data):
        if self.transform is None:
            return sample
        return self.transform(sample)
    
    def _process_single_task(self, idx: int) -> Data:
        if protein_to_pyg is None:
            raise ImportError(
                "Missing dependency: graphein. Install graphein to parse raw PDB into PyG graphs."
            )
        input_data = self.processed_tasks[idx]
        atom_array = input_data['struc_input']
        contig = input_data['contig'] if input_data['contig'] is not None else None
        task_name = input_data['task_name']
        sample_fn = input_data['sample_fn']
        ref_center = input_data['ref_center']
        task_type = input_data['task_type']
        pep_chain = input_data.get('pep_chain', 'L')
        binder_chain = str(input_data.get('binder_chain', 'L') or 'L')
        reference_pdb_path = input_data.get('reference_pdb_path', None)
        reference_pep_chain = input_data.get('reference_pep_chain', pep_chain)
        reference_receptor_center = input_data.get('reference_receptor_center', None)
        cyclic = bool(input_data.get('cyclic', False))
        binder_secondary_class = input_data.get('binder_secondary_class', None)
        binder_ss_token = input_data.get('binder_ss_token', None)
        binder_block_adj = input_data.get('binder_block_adj', None)
        binder_topology_source = input_data.get('binder_topology_source', None)
        binder_topology_entry_id = input_data.get('binder_topology_entry_id', None)
        binder_topology_preset = input_data.get('binder_topology_preset', None)
        residue_type_constraints = input_data.get('residue_type_constraints', {}) or {}
        cyclization = input_data.get('cyclization', None)
        optional_ca_pair_constraints = input_data.get('optional_ca_pair_constraints', []) or []
        condition_mode_id = int(input_data.get('condition_mode_id', 0) or 0)
        condition_mode_name = str(input_data.get('condition_mode_name', 'none') or 'none')
        condition_anchor_positions = input_data.get('condition_anchor_positions', [-1, -1, -1])
        condition_pair_positions = input_data.get('condition_pair_positions', [[-1, -1], [-1, -1], [-1, -1]])
        condition_pair_bin_idx = input_data.get('condition_pair_bin_idx', [0, 0, 0])
        condition_pair_target_angstrom = input_data.get('condition_pair_target_angstrom', [0.0, 0.0, 0.0])
        condition_pair_valid_mask = input_data.get('condition_pair_valid_mask', [False, False, False])
        condition_residue_targets = input_data.get('condition_residue_targets', [-1, -1, -1])
        condition_residue_target_mask = input_data.get('condition_residue_target_mask', [False, False, False])
        topology_entries = input_data.get('topology_entries', []) or []
        postprocess_enabled = bool(input_data.get('postprocess_enabled', False))
        postprocess_method = str(input_data.get('postprocess_method', 'openmm') or 'openmm')
        omit_aa_indices = input_data.get('omit_aa_indices', []) or []
        output_residue_pdb_idx_by_key = input_data.get('output_residue_pdb_idx_by_key', {}) or {}
        
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=True) as tmp_file:
            strucio.save_structure(tmp_file.name, atom_array)
            try:
                graph = protein_to_pyg(path=tmp_file.name)
            except Exception as e:
                logger.error(f"Graphein parsing failed for {sample_fn}: {e}")
                raise e
        
        # basic information
        coord_mask = graph.coords != 1e-5
        graph.coord_mask = coord_mask[..., 0]
        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        residue_indices = []
        fallback_idx = resname_to_idx.get("UNK", resname_to_idx.get("ALA", 0))
        for resname in graph.residues:
            residue_indices.append(resname_to_idx.get(resname, fallback_idx))
        graph.residue_type = torch.tensor(residue_indices, dtype=torch.long)
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
        )
        graph.orig_chain_id = [
            s.split(":")[0] for s in graph.residue_id
        ]
        graph.output_residue_pdb_idx = torch.tensor(
            [
                int(output_residue_pdb_idx_by_key.get((str(chain_id), int(res_id)), int(res_id)))
                for chain_id, res_id in zip(graph.orig_chain_id, graph.residue_pdb_idx.tolist())
            ],
            dtype=torch.long,
        )
        graph.is_peptide = torch.tensor(
            [1 if s.split(":")[0] == 'L' else 0 for s in graph.residue_id],
            dtype=torch.long,
        )
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
        graph.id = sample_fn
        graph.task_name = task_name
        graph.contig = contig if contig is not None else "null"
        graph.binder_chain = binder_chain
        graph.reference_pdb_path = str(reference_pdb_path) if reference_pdb_path is not None else ""
        graph.reference_pep_chain = str(reference_pep_chain or pep_chain)
        if reference_receptor_center is not None:
            graph.reference_receptor_center = torch.as_tensor(
                reference_receptor_center,
                dtype=torch.float32,
            ).reshape(1, 3)
        else:
            graph.reference_receptor_center = torch.full((1, 3), float("nan"), dtype=torch.float32)
        graph.cyclic = torch.tensor([cyclic], dtype=torch.bool)
        if binder_ss_token is not None or binder_block_adj is not None:
            if binder_ss_token is None:
                raise ValueError(
                    f"sample_fn={sample_fn} task_name={task_name} must provide binder_ss_token "
                    "when binder topology conditioning is requested."
                )
            _attach_binder_topology(
                graph,
                binder_ss_token=binder_ss_token,
                binder_block_adj=binder_block_adj,
                sample_fn=sample_fn,
                task_name=task_name,
            )
            graph.binder_topology_source = str(binder_topology_source or "")
            graph.binder_topology_entry_id = str(binder_topology_entry_id or "")
            graph.binder_topology_preset = str(binder_topology_preset or "")
        if binder_secondary_class is not None:
            binder_secondary_class = int(binder_secondary_class)
            graph.binder_secondary_class = torch.tensor([binder_secondary_class], dtype=torch.long)
            graph.binder_secondary_fold_code = torch.tensor(
                [binder_secstruct_fold_code_from_id(binder_secondary_class)],
                dtype=torch.long,
            )

        _attach_residue_type_condition(
            graph,
            task_type=task_type,
            residue_type_constraints=residue_type_constraints,
            sample_fn=sample_fn,
            task_name=task_name,
        )

        cyclization_pair_indices = torch.full((1, 2), fill_value=-1, dtype=torch.long)
        cyclization_shortcut_len = torch.ones((1,), dtype=torch.long)
        has_cyclization_pair = torch.zeros((1,), dtype=torch.bool)
        if cyclization is not None:
            positions = cyclization.get("positions", None)
            if positions is None:
                raise ValueError(f"Missing cyclization.positions for sample_fn={sample_fn} task_name={task_name}")
            idxs = []
            for pos in positions:
                matches = [
                    idx_res
                    for idx_res, (chain_id, res_id) in enumerate(zip(graph.orig_chain_id, graph.residue_pdb_idx.tolist()))
                    if chain_id == 'L' and int(res_id) == int(pos)
                ]
                if len(matches) != 1:
                    raise ValueError(
                        f"Failed to align cyclization position {pos} onto binder residues "
                        f"for sample_fn={sample_fn} task_name={task_name}"
                    )
                idxs.append(matches[0])
            cyclization_pair_indices[0] = torch.tensor(idxs, dtype=torch.long)
            cyclization_shortcut_len[0] = int(cyclization.get("shortcut_len", 1))
            has_cyclization_pair[0] = True
        graph.cyclization_pair_indices = cyclization_pair_indices
        graph.cyclization_shortcut_len = cyclization_shortcut_len
        graph.has_cyclization_pair = has_cyclization_pair

        optional_ca_pair_dist_condition = torch.zeros(
            (graph.residue_type.shape[0], graph.residue_type.shape[0], 30),
            dtype=torch.float32,
        )
        optional_ca_pair_dist_mask = torch.zeros(
            (graph.residue_type.shape[0], graph.residue_type.shape[0]),
            dtype=torch.bool,
        )
        unresolved_optional_pairs = {(int(entry["pos_i"]), int(entry["pos_j"])): int(entry["bin_idx"]) for entry in optional_ca_pair_constraints}
        binder_pos_to_graph_idx = {}
        for idx_res, (chain_id, res_id) in enumerate(zip(graph.orig_chain_id, graph.residue_pdb_idx.tolist())):
            if chain_id == 'L':
                binder_pos_to_graph_idx[int(res_id)] = idx_res
        for (pos_i, pos_j), bin_idx in unresolved_optional_pairs.items():
            if pos_i not in binder_pos_to_graph_idx or pos_j not in binder_pos_to_graph_idx:
                raise ValueError(
                    f"Failed to align optional_ca_pair_constraints pair ({pos_i}, {pos_j}) "
                    f"onto binder residues for sample_fn={sample_fn} task_name={task_name}"
                )
            idx_i = binder_pos_to_graph_idx[pos_i]
            idx_j = binder_pos_to_graph_idx[pos_j]
            optional_ca_pair_dist_condition[idx_i, idx_j, bin_idx] = 1.0
            optional_ca_pair_dist_condition[idx_j, idx_i, bin_idx] = 1.0
            optional_ca_pair_dist_mask[idx_i, idx_j] = True
            optional_ca_pair_dist_mask[idx_j, idx_i] = True
        graph.optional_ca_pair_dist_condition = optional_ca_pair_dist_condition
        graph.optional_ca_pair_dist_mask = optional_ca_pair_dist_mask
        _attach_chemical_bond_type(
            graph,
            topology_entries=topology_entries,
            binder_pos_to_graph_idx=binder_pos_to_graph_idx,
            sample_fn=sample_fn,
            task_name=task_name,
        )
        graph.condition_mode_id = torch.tensor([condition_mode_id], dtype=torch.long)
        graph.condition_mode_name = condition_mode_name
        graph.condition_anchor_positions = torch.tensor(condition_anchor_positions, dtype=torch.long)
        graph.condition_pair_positions = torch.tensor(condition_pair_positions, dtype=torch.long)
        graph.condition_pair_bin_idx = torch.tensor(condition_pair_bin_idx, dtype=torch.long)
        graph.condition_pair_target_angstrom = torch.tensor(condition_pair_target_angstrom, dtype=torch.float32)
        graph.condition_pair_valid_mask = torch.tensor(condition_pair_valid_mask, dtype=torch.bool)
        graph.condition_residue_targets = torch.tensor(condition_residue_targets, dtype=torch.long)
        graph.condition_residue_target_mask = torch.tensor(condition_residue_target_mask, dtype=torch.bool)
        graph.topology_entries = json.dumps(topology_entries, ensure_ascii=False)
        graph.postprocess_enabled = torch.tensor([postprocess_enabled], dtype=torch.bool)
        graph.postprocess_method = postprocess_method
        graph.omit_aa_indices = torch.tensor([int(i) for i in omit_aa_indices], dtype=torch.long)
        
        # handle hotspot and motif annotations
        ca_atoms = atom_array[atom_array.atom_name == "CA"]
        aa_keys = list(zip(ca_atoms.chain_id.tolist(), ca_atoms.res_id.tolist()))
        g_keys  = list(zip(graph.orig_chain_id, graph.residue_pdb_idx.cpu().tolist()))
        is_hotspot_arr = ca_atoms.get_annotation("is_hotspot")
        is_motif_arr = ca_atoms.get_annotation("is_motif")
        if aa_keys != g_keys:
            aa_index = {}
            dup = None
            for i, k in enumerate(aa_keys):
                if k in aa_index:
                    dup = k
                    break
                aa_index[k] = i
            if dup is not None:
                raise ValueError(f"Duplicate (chain,res_id) keys in AtomArray CA list: {dup}; sample_fn={sample_fn} task_name={task_name}")
            try:
                idxs = [aa_index[k] for k in g_keys]
            except KeyError as e:
                raise AssertionError(
                    f"Cannot align residues: Graphein key {e.args[0]} missing from AtomArray; sample_fn={sample_fn} task_name={task_name}"
                ) from None
            is_hotspot_arr = is_hotspot_arr[idxs]
            is_motif_arr = is_motif_arr[idxs]
        is_motif_tensor = torch.from_numpy(is_motif_arr).bool()
        is_receptor = (graph.is_peptide == 0).bool()
        # Keep the peptide-only motif annotation available to the decoder.  The
        # legacy ``latent_cond_mask`` also includes the receptor, so it cannot
        # distinguish sequence-fixed peptide motif positions from receptor
        # positions when decoding a docking sample.
        graph.peptide_motif_mask = is_motif_tensor & graph.is_peptide.bool()
        cond_mask = is_receptor | is_motif_tensor
        graph.hotspot = torch.from_numpy(is_hotspot_arr).float().unsqueeze(-1)

        graph.is_inv_folding_task = (task_type == 'inv_folding')
        graph.is_docking_task = (task_type == 'docking')
        # Legacy model consumers used this feature flag for the known-sequence
        # residue-type conditioning path.  Keep it as a docking compatibility
        # alias while rejecting external task_type='folding' in validate_task.
        graph.is_folding_task = graph.is_docking_task
        graph.is_packing_task = (task_type == 'packing')
        graph.use_ca_coors_nm_feature = torch.tensor(
            [task_type in ('inv_folding', 'packing')],
            dtype=torch.bool,
        )
        graph.use_residue_type_feature = torch.tensor(
            [task_type in ('docking', 'packing')],
            dtype=torch.bool,
        )

        if task_type == 'inv_folding':
            graph.ca_cond_mask = torch.ones_like(cond_mask)
            graph.latent_cond_mask = cond_mask
        elif task_type == 'docking':
            graph.ca_cond_mask = is_receptor
            graph.latent_cond_mask = is_receptor
        elif task_type == 'packing':
            graph.ca_cond_mask = torch.ones_like(cond_mask)
            graph.latent_cond_mask = is_receptor
        else:
            graph.ca_cond_mask = cond_mask
            graph.latent_cond_mask = cond_mask
        if ref_center is not None:
            graph.ref_center = 0.1 * torch.from_numpy(ref_center).float().unsqueeze(0)  # [1, 3] nanometers
            graph.has_ref_center = torch.ones(1, dtype=torch.bool)
        else:
            graph.ref_center = torch.zeros(1, 3, dtype=torch.float32)  # [1, 3] sentinel
            graph.has_ref_center = torch.zeros(1, dtype=torch.bool)

        return graph

    def __getitem__(self, idx: int):
        if self.memory is not None:
            return self.memory[idx]
        return self._apply_transform(self._process_single_task(idx))


class InferenceLightningDataModule(BaseLightningDataModule):
    """Minimal LightningDataModule that materialises processed graphs from a structure folder."""

    def __init__(
        self,
        raw_tasks: Dict,
        batch_padding: bool = True,
        transforms: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        interleave: bool = False,
        multiprocessing_context: Optional[str] = None,
        pad_max_size: Optional[int] = None,
        truncate_to_pad_max_size: bool = False,
        pad_partial_batch: bool = False,
    ) -> None:

        super().__init__(
            batch_padding=batch_padding,
            transforms=transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            multiprocessing_context=multiprocessing_context,
            pad_max_size=pad_max_size,
            truncate_to_pad_max_size=truncate_to_pad_max_size,
        )

        self.pad_partial_batch = pad_partial_batch
        self.pep_chain = "L"
        self.raw_tasks = raw_tasks
        self.processed_tasks = []
        self.interleave = interleave
        self.transform = (
            INFERENCE_TRANSFORMS if transforms is None else self._compose_transforms(transforms)
        )

    def setup(self, stage = 'predict') -> None:
        self._process_tasks()
        super().setup(stage)

    def _process_tasks(self) -> List[Dict]:
        if len(self.processed_tasks) == 0:
            groups = defaultdict(list)
            for task_name, task in self.raw_tasks.items():
                task_input_list = build_atom_array(task, task_name)
                groups[task_name].extend(task_input_list)
            rng = random.Random(2026)
            for group in groups.values():
                rng.shuffle(group)

            if self.interleave:
                group_lists = list(groups.values())
                rng.shuffle(group_lists)
                result = []
                while group_lists:
                    next_round = []
                    for group in group_lists:
                        result.append(group.pop(0))
                        if group:
                            next_round.append(group)
                    group_lists = next_round
                logger.info(f"Interleaved {len(groups)} task groups to maximize per-batch task diversity.")
            else:
                result = []
                for group in groups.values():
                    result.extend(group)
                logger.info(f"Concatenated {len(groups)} task groups without interleaving.")

            self.processed_tasks = result
        return self.processed_tasks

    def _get_dataset(self, split: str = "predict") -> InferenceDataset:
        assert split == "predict", "InferenceLightningDataModule only supports 'predict' split"
        processed_tasks_list = self._process_tasks()
        return InferenceDataset(
            processed_tasks=processed_tasks_list,
            transform=self.transform,
            preload=False,
        )

    
