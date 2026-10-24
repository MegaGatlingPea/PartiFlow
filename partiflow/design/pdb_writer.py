from __future__ import annotations
from typing import Any
import numpy as np
import torch
from loguru import logger
from partiflow.utils.pdb_utils import PDB_CHAIN_IDS, create_full_prot, to_pdb
from .tasks import (
    _pdb_coord_validation_error,
    _replace_receptor_with_centered_reference,
)


class PDBWriter:
    def __init__(self, receptor_source="decoded"):
        self.receptor_source = self._normalize_receptor_source(receptor_source)

    @staticmethod
    def _normalize_receptor_source(value: Any) -> str:
        source = str(value or "decoded").strip().lower()
        aliases = {
            "ae": "decoded",
            "autoencoder": "decoded",
            "autoencoder_decoded": "decoded",
            "generated": "decoded",
            "model": "decoded",
            "model_decoded": "decoded",
            "reference_pdb": "reference",
            "original": "reference",
            "preserve_reference": "reference",
        }
        source = aliases.get(source, source)
        if source not in {"decoded", "reference"}:
            raise ValueError(
                f"Unsupported output.receptor_source={value!r}; expected 'decoded' or 'reference'."
            )
        return source

    def coors37_to_pdb_string(
        self,
        coors37: torch.Tensor,
        residue_type: torch.Tensor,
        is_peptide: torch.Tensor,
        batch: Any,
        sample_idx: int,
        chain_index_override: torch.Tensor | None = None,
        residue_index_override: torch.Tensor | np.ndarray | None = None,
        reference_pdb_path: str | None = None,
        reference_pep_chain: str = "L",
        reference_receptor_center: Any = None,
        binder_chain: str = "L",
    ) -> str:
        atom37 = coors37.detach().float().cpu().numpy()  # [n, 37, 3]
        aatype = residue_type.detach().long().cpu().numpy()  # [n]

        atom37_mask = (np.abs(atom37) > 1e-7).any(axis=-1)  # [n, 37]
        residue_mask = atom37_mask.any(axis=-1)  # [n]
        if residue_mask.sum() == 0:
            return ""

        chain_index = None
        if chain_index_override is not None:
            chain_index = chain_index_override.detach().long().cpu().numpy()
        else:
            orig_chain_ids = getattr(batch, "orig_chain_id", None)
            if orig_chain_ids is not None:
                try:
                    chain_ids_i = orig_chain_ids[sample_idx]
                    if isinstance(chain_ids_i, (list, tuple)) and len(chain_ids_i) > 0:
                        n = atom37.shape[0]
                        chain_ids_i = list(chain_ids_i)
                        if len(chain_ids_i) < n:
                            chain_ids_i = chain_ids_i + [chain_ids_i[-1]] * (
                                n - len(chain_ids_i)
                            )
                        if len(chain_ids_i) > n:
                            chain_ids_i = chain_ids_i[:n]
                        chain_index = np.array(
                            [
                                PDB_CHAIN_IDS.index(cid) if cid in PDB_CHAIN_IDS else 0
                                for cid in chain_ids_i
                            ],
                            dtype=np.int64,
                        )
                except Exception:
                    chain_index = None

        if chain_index is None:
            chain_index = is_peptide.detach().long().cpu().numpy().astype(np.int64)

        atom37 = atom37[residue_mask]
        atom37_mask = atom37_mask[residue_mask]
        aatype = aatype[residue_mask]
        chain_index = chain_index[residue_mask]
        peptide_mask = is_peptide.detach().bool().cpu().numpy()[residue_mask]
        generated_peptide_chains = {
            PDB_CHAIN_IDS[int(idx)]
            for idx in np.unique(chain_index[peptide_mask])
            if int(idx) < len(PDB_CHAIN_IDS)
        }
        residue_index = residue_index_override
        if residue_index is not None:
            if isinstance(residue_index, torch.Tensor):
                residue_index = residue_index.detach().long().cpu().numpy()
            else:
                residue_index = np.asarray(residue_index)
            residue_index = residue_index[residue_mask]

        prot = create_full_prot(
            atom37=atom37,
            atom37_mask=atom37_mask.astype(np.float32),
            chain_index=chain_index,
            aatype=aatype,
            residue_index=residue_index,
        )
        preserve_reference = (
            self.receptor_source == "reference"
            and bool(reference_pdb_path)
            and reference_receptor_center is not None
        )
        coord_error = _pdb_coord_validation_error(
            prot.atom_positions[peptide_mask]
            if preserve_reference
            else prot.atom_positions,
            prot.atom_mask[peptide_mask] if preserve_reference else prot.atom_mask,
        )
        if coord_error is not None:
            logger.warning(
                f"Cannot write PDB for sample_idx={sample_idx}: {coord_error}."
            )
            return ""
        pdb_string = to_pdb(prot=prot)
        output_pdb_string = pdb_string
        if preserve_reference:
            output_pdb_string = _replace_receptor_with_centered_reference(
                pdb_string,
                reference_pdb_path=reference_pdb_path,
                reference_pep_chain=reference_pep_chain,
                reference_receptor_center=reference_receptor_center,
                binder_chain=binder_chain,
                generated_peptide_chains=generated_peptide_chains,
            )
        if preserve_reference and output_pdb_string == pdb_string:
            coord_error = _pdb_coord_validation_error(
                prot.atom_positions, prot.atom_mask
            )
            if coord_error is not None:
                logger.warning(
                    f"Cannot write fallback generated-complex PDB for sample_idx={sample_idx}: "
                    f"{coord_error}."
                )
                return ""
        return output_pdb_string

    @staticmethod
    def _select_output_residue_index(record: Dict[str, Any]) -> Any:
        output_residue_index = record.get("output_residue_pdb_idx", None)
        if output_residue_index is not None:
            return output_residue_index
        return record.get("residue_pdb_idx", None)
