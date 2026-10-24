import json
import os
import random
import copy
import re
from functools import partial
from typing import Any, Dict, List, Literal, Mapping, Tuple, Union
import ray

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from omegaconf import OmegaConf

from partiflow.flow.product_space_flow_matcher import ProductSpaceFlowMatcher
from partiflow.flow.local_latents_tf import LocalLatentsTransformer
from partiflow.vae.autoencoder import AutoEncoder
from partiflow.utils.coors_utils import nm_to_ang, trans_nm_to_atom37
from partiflow.utils.align_utils import kabsch_align
from partiflow.utils.pdb_utils import (
    create_full_prot,
    to_pdb,
    PDB_CHAIN_IDS
)
from partiflow.utils.openfold_utils import optimize_sidechains
from partiflow.utils.secstruct_utils import (
    binder_secstruct_name_from_id,
    classify_secondary_structure_metrics,
    secondary_structure_metrics_from_pdb_string,
)
from partiflow.utils.conditioning_utils import (
    CHEMICAL_BOND_DISULFIDE,
    CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    CHEMICAL_BOND_ISOPEPTIDE,
    CHEMICAL_BOND_OTHER_COVALENT,
    design_condition_mode_to_name,
)
from partiflow.openfold.np.residue_constants import RESTYPE_ATOM37_MASK, atom_order, restypes
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def _plain_batch_mapping(batch):
    """Convert PyG Data-like batches to plain mappings before torch.compile tracing."""
    if isinstance(batch, Mapping):
        return {
            key: _plain_batch_mapping(value) if isinstance(value, Mapping) else value
            for key, value in batch.items()
        }
    if hasattr(batch, "_store") and hasattr(batch._store, "_mapping"):
        return {
            key: _plain_batch_mapping(value) if isinstance(value, Mapping) else value
            for key, value in batch._store._mapping.items()
        }
    return batch


@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


class PartiFlow(L.LightningModule):
    def __init__(self, cfg_exp, store_dir=None, autoencoder_ckpt_path=None, create_val_tmp=True):
        super().__init__()
        self.save_hyperparameters()
        self.cfg_exp = cfg_exp
        self.inf_cfg = None  # Only for inference runs
        self.output_receptor_source = "decoded"
        self.validation_output_lens = {}
        self.validation_output_data = []
        self.validation_inference_tasks = []
        self.store_dir = store_dir if store_dir is not None else "val_samples"
        self.val_path_tmp = os.path.join("./tmp", self.store_dir)
        if create_val_tmp:
            create_dir(self.val_path_tmp)
        self.predict_ret_mode = "coors37_n_aatype"

        if autoencoder_ckpt_path is not None:
            logger.info(f"Manually setting autoencoder_ckpt_path to {autoencoder_ckpt_path}")
            OmegaConf.set_struct(cfg_exp, False)
            cfg_exp.autoencoder_ckpt_path = autoencoder_ckpt_path
            OmegaConf.set_struct(cfg_exp, True)

        self.autoencoder, latent_dim = self.load_autoencoder(cfg_exp, freeze_params=True)
        
        # Add right latent dimensionality in the config file, needed to instantiate the flow matcher below
        if latent_dim is not None:
            self.latent_dim = latent_dim
        else:
            self.latent_dim = cfg_exp.product_flowmatcher.local_latents.get("dim", 8)
            
        if self.autoencoder is not None:
            try:
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
            except:
                OmegaConf.set_struct(cfg_exp, False)
                # Update the configuration with the new key-value pair
                cfg_exp.product_flowmatcher.local_latents.dim = self.latent_dim
                # Re-enable struct mode if needed
                OmegaConf.set_struct(cfg_exp, True)

        self.fm = ProductSpaceFlowMatcher(cfg_exp)
        self.motif_constraint_mode = cfg_exp.get("motif_constraint_mode", "rigid")
        logger.info(f"Motif constraint mode: {self.motif_constraint_mode}")
        logger.info(f"cfg_exp.network: {cfg_exp.network}")

        # Neural network
        if cfg_exp.network.name == "transformer_fm_large":
            self.nn = LocalLatentsTransformer(**cfg_exp.network, latent_dim=self.latent_dim)
        elif cfg_exp.network.name == "transformer_fm":
            self.nn = LocalLatentsTransformer(**cfg_exp.network, latent_dim=self.latent_dim)
        elif cfg_exp.network.name == "transformer_fm_small":
            self.nn = LocalLatentsTransformer(**cfg_exp.network, latent_dim=self.latent_dim)
        else:
            raise IOError(f"Wrong nn selected for CAFlow {cfg_exp.network.name}")

        # Scaling laws stuff
        self.nflops = 0
        self.nsamples_processed = 0
        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        self.nn_ag = None
        self.search_cfg = None
        self.reward_model = None
        self._search_instance = None

    def load_autoencoder(self, cfg_exp, freeze_params=True):
        """Loads autoencoder, if required."""
        if ("autoencoder_ckpt_path" in cfg_exp):
            # for new runs trained with refactored codebase
            ae_ckp_path = cfg_exp.autoencoder_ckpt_path
        elif ("autoencoder_ckpt_path" in cfg_exp.product_flowmatcher.local_latents):
            # for old runs trained with old codebase
            ae_ckp_path = cfg_exp.product_flowmatcher.local_latents.autoencoder_ckpt_path
        else:
            raise ValueError("No autoencoder checkpoint path provided")

        if ae_ckp_path is None:
            return None, None
        
        logger.info(f"Loading autoencoder from {ae_ckp_path}")
        autoencoder = AutoEncoder.load_from_checkpoint(ae_ckp_path, strict=False, create_val_tmp=False)
        if freeze_params:
            for param in autoencoder.parameters():
                param.requires_grad = False
        return autoencoder, autoencoder.latent_dim

    def configure_optimizers(self):
        optimizer_type = self.cfg_exp.opt.get("optimizer", "adam").lower()
        weight_decay = self.cfg_exp.opt.get("weight_decay", 0.0)

        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.cfg_exp.opt.lr,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                [p for p in self.parameters() if p.requires_grad],
                lr=self.cfg_exp.opt.lr
            )

        if self.cfg_exp.opt.get("use_scheduler", False):

            warmup_steps = self.cfg_exp.opt.get("warmup_steps", 1000)
            total_steps = self.cfg_exp.opt.get("total_steps", 100000)

            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            main_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=self.cfg_exp.opt.get("min_lr", 1e-6)
            )

            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_steps]
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            }

        return optimizer

    def on_save_checkpoint(self, checkpoint):
        """Adds additional variables to checkpoint."""
        checkpoint["nflops"] = self.nflops
        checkpoint["nsamples_processed"] = self.nsamples_processed

    def on_load_checkpoint(self, checkpoint):
        """Loads additional variables from checkpoint."""
        try:
            self.nflops = checkpoint["nflops"]
            self.nsamples_processed = checkpoint["nsamples_processed"]
        except:
            logger.info("Failed to load nflops and nsamples_processed from checkpoint")
            self.nflops = 0
            self.nsamples_processed = 0

    def call_nn(
        self,
        batch: Dict[str, torch.Tensor],
        n_recycle: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Calls NN with recycling. Should this be here or in the NN? Possibly better here,
        in case we want to recycle using decoder for some approach, etc, and this is akin
        to self conditioning, also here.
        Also, if we want to recycle clean sample predictions... Then we'd need this here,
        as the nn does not know about relations between v, x1, ...
        """
        return self._call_nn_module(self.nn, batch, n_recycle=n_recycle)

    def _call_nn_module(
        self,
        nn_module: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        n_recycle: int,
    ) -> Dict[str, torch.Tensor]:
        """Run a specific denoiser with the same recycling logic as ``call_nn``."""
        batch = _plain_batch_mapping(batch)

        # First call
        nn_out = nn_module(batch)

        # Recycle n_recycle times detaching gradients and updating input
        # Note that recycling is supported by the codebase, but the models provided 
        # with the La-Proteina paper do not use it, nor were trained with it.
        for _ in range(n_recycle):
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch[f"x_recycle"] = {dm: x_1_pred[dm].detach() for dm in x_1_pred}
            nn_out = nn_module(batch)

        # Final prediction
        return nn_out

    @staticmethod
    def _maybe_remove_batch_key(batch_like: Dict, key: str, removed_keys: List[str]) -> None:
        if key in batch_like:
            batch_like.pop(key)
            removed_keys.append(key)

    @staticmethod
    def _cfg_bool_arg(args: Dict, key: str, default: bool = False) -> bool:
        value = args.get(key, default)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "on"}
        return bool(value)

    def build_unconditional_batch(self, batch: Dict) -> Tuple[Dict, List[str]]:
        """
        Build the unconditional branch used by classifier-free guidance.

        In the current inference setup we use CFG to weaken coarse binder-side
        semantic conditioning while preserving hard structural context. We keep
        motif / receptor structural conditioning intact, but strip the coarse
        binder secondary-structure labels and receptor hotspot annotations.
        """
        uncond_batch = copy.copy(batch)
        removed_keys: List[str] = []

        removable_condition_keys = [
            "binder_secondary_class",
            "binder_secondary_class_raw",
            "binder_secondary_fold_code",
            "binder_secondary_fold_code_raw",
            "hotspot",
        ]
        for key in removable_condition_keys:
            self._maybe_remove_batch_key(uncond_batch, key, removed_keys)

        sample_args = getattr(self.inf_cfg, "args", {}) or {}
        if self._cfg_bool_arg(sample_args, "cfg_drop_pair_condition", False):
            for key in (
                "optional_ca_pair_indices",
                "optional_ca_pair_bin_idx",
                "optional_ca_pair_valid_mask",
                "optional_ca_pair_dist_condition",
                "optional_ca_pair_dist_mask",
            ):
                self._maybe_remove_batch_key(uncond_batch, key, removed_keys)

        if self._cfg_bool_arg(sample_args, "cfg_drop_chemical_bond", False):
            self._maybe_remove_batch_key(uncond_batch, "chemical_bond_type", removed_keys)

        return uncond_batch, removed_keys

    # TODO: need to be modified, no cath code is needed anymore
    def predict_for_sampling(
        self,
        batch: Dict,
        mode: Literal["full", "ucond", "ag"],
        n_recycle: int,
    ) -> Tuple[Union[Dict[str, torch.Tensor], float, None]]:
        """
        This function predicts clean samples for multiple models:
        x_pred, the 'original' model, if mode == full
        x_pred_ucond, the unconditional model, , if mode == ucond

        TODO: Need to update to include autoguidance again

        These predictions will later be used to sample with guidance and autoguidance.

        Args:
            batch: Dict
            mode: str

        Returns:
            x_pred (tensor) for the requested mode
        """
        if mode == "full":
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
        elif mode == "ag":
            if self.nn_ag is None:
                raise ValueError("Requested autoguidance branch but `nn_ag` is None.")
            nn_module = self.nn_ag.nn if hasattr(self.nn_ag, "nn") else self.nn_ag
            nn_out = self._call_nn_module(nn_module, copy.copy(batch), n_recycle=n_recycle)
        elif mode == "ucond":
            uncond_batch, removed_keys = self.build_unconditional_batch(batch)
            if len(removed_keys) == 0:
                logger.warning(
                    "CFG requested but no explicit conditioning keys were found in batch; "
                    "unconditional branch will match the conditional branch."
                )
            nn_out = self.call_nn(uncond_batch, n_recycle=n_recycle)
        else:
            raise IOError(f"Wrong {mode} passed to `predict_for_sampling`")

        return nn_out

    def training_step(self, batch: Dict, batch_idx: int):
        """
        Computes training loss for batch of samples.

        Args:
            batch: Data batch.

        Returns:
            Training loss averaged over batch dimension.
        """
        val_step = batch_idx == -1  # validation step is indicated with batch_idx -1
        log_prefix = "validation_loss" if val_step else "train"

        # Add clean samples for all data modes / spaces we are working on
        batch = self.add_clean_samples(batch)

        # Corrupt the batch
        # TODO: product_space_flowmatcher.process_batch should add `is_peptide` mask indicate which part needs to be denoised
        # TODO: apply mask indicates places both exist and need to be denoised while sample noise
        # TODO: which mask should we deliver when interpolating between x0 and x1?
        batch = self.fm.corrupt_batch(batch)  # adds x_1, t, x_0, x_t, mask
        bs, n = batch["mask"].shape 
        
        # we need to preprocess the batch x_1, t, x_0, x_t, mask before handle self_cond.
        # ... code need to be implemented here
        if self.motif_constraint_mode in ["rigid", "rigid_receptor"]:
            for mode in batch["x_0"]:
                diff_x0 = (batch["x_0"][mode] - batch["x_1"][mode]) * batch["cond_mask"][mode][..., None]
                diff_xt = (batch["x_t"][mode] - batch["x_1"][mode]) * batch["cond_mask"][mode][..., None]
                assert torch.all(diff_x0.abs() < 1e-5), f"x_0 mismatch for condition/receptor residues in mode {mode}"
                assert torch.all(diff_xt.abs() < 1e-5), f"x_t mismatch for condition/receptor residues in mode {mode}"

        # Handle conditioning variables
        # TODO: we should ensure x_sc contains ground truth for those parts we do not want to denoise
        # maybe we need to design the feature we gonna use in feature_factory
        batch = self.handle_self_cond(
            batch
        )  # self conditioning, adds ["x_sc"] to batch prob 0.5, already perform call_nn inside 

        if "x_sc" in batch and self.motif_constraint_mode in ["rigid", "rigid_receptor"]:
            for mode in batch["x_sc"]:
                diff_xsc = (batch["x_sc"][mode] - batch["x_1"][mode]) * batch["cond_mask"][mode][..., None]
                assert torch.all(diff_xsc.abs() < 1e-5), f"x_sc mismatch for condition/receptor residues in mode {mode}"

        batch = self.handle_folding_n_inverse_folding(
            batch
        )  # folding and inverse folding iterations

        # Number of recycling steps
        n_recycle = self.handle_recycling()

        nn_out = self.call_nn(batch, n_recycle=n_recycle) # it should satisfy the condition/receptor part unchanged while using v param
        # assert ((batch["x_0"]["bb_ca"] + nn_out["bb_ca"]["v"] - batch["x_1"]["bb_ca"]) * is_receptor_mask).any() is False
        
        # we need to add mask to get logic here right
        losses = self.fm.compute_loss(
            batch=batch,
            nn_out=nn_out,
        )  # Dict[str, Tensor w.batch shape [*]]

        self.log_losses(bs=bs, losses=losses, log_prefix=log_prefix, batch=batch)
        train_loss = sum([torch.mean(losses[k]) for k in losses if "_justlog" not in k])

        self.log(
            f"{log_prefix}/loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=bs,
            sync_dist=True,
            add_dataloader_idx=False,
        )

        if not val_step:  # Don't log these for val step
            self.log_train_loss_n_prog_bar(bs, train_loss)
            self.update_n_log_flops(bs, n)
            self.update_n_log_nsamples_processed(bs)
            self.log_nparams()

        return train_loss

    def add_clean_samples(self, batch: Dict) -> Dict:
        """
        Adds clean sample for all data modes / spaces we are working on. For instance, if we have two
        data modes, bb_ca and local_latents, it adds the clean data to the batch
        x_1 = {
            "bb_ca": Corresponding tensor with clean bb_ca coordinates, shape [b, n, 3]
            "local_latents": Corresponding tensor with clean local_latents, shape [b, n, d]
        }

        For local_latents, x_1 is constructed as:
        - rigid mode: fix positions use gt_cond_latents (masked encode), denoise positions use full_latents
        - flex mode: all positions use full_latents (no conditioning)

        Args:
            batch: Batch to add clean samples to.

        Returns:
            Batch with clean sample added.
        """
        # Compute is_fix for _get_cond_latents
        mask = batch["mask_dict"]["coords"][..., 0, 0]
        is_peptide = (batch["is_peptide"] == 1) & mask
        is_receptor = (batch["is_peptide"] == 0) & mask

        if self.motif_constraint_mode == "flex":
            batch["is_fix"] = torch.zeros_like(mask, dtype=torch.bool)
        elif self.motif_constraint_mode == "rigid":
            latent_cond_mask = batch.get("latent_cond_mask", is_receptor)
            batch["is_fix"] = latent_cond_mask & mask
        elif self.motif_constraint_mode == "rigid_receptor":
            batch["is_fix"] = is_receptor
        else:
            raise ValueError(f"Unknown motif_constraint_mode: {self.motif_constraint_mode}")

        batch["x_1"] = {}

        for dm in self.cfg_exp.product_flowmatcher:
            if dm == "bb_ca":
                batch["x_1"][dm] = batch["coords_nm"][:, :, 1, :]  # [b, n, 3]
            elif dm == "local_latents":
                full_encoded = self.autoencoder.encode(batch)
                full_latents = full_encoded["z_latent"]  # [b, n, d]

                if self.motif_constraint_mode == "flex":
                    # Flex mode: use full_latents for all positions (no conditioning)
                    batch["x_1"][dm] = full_latents
                    batch["gt_cond_latents"] = torch.zeros_like(full_latents)  # Placeholder, not used
                else:
                    # Rigid mode: use gt_cond_latents for fix positions, full_latents for denoise positions
                    is_fix = batch["is_fix"]
                    gt_cond_latents = self._get_cond_latents(batch)  # [b, n, d]
                    batch["gt_cond_latents"] = gt_cond_latents
                    cond_mask_expanded = is_fix[..., None]  # [b, n, 1]
                    combined_latents = torch.where(cond_mask_expanded, gt_cond_latents, full_latents)
                    batch["x_1"][dm] = combined_latents
            else:
                raise ValueError(f"Loading clean samples from data mode {dm} not supported.")

        return batch

    def _get_cond_latents(self, batch: Dict) -> torch.Tensor:
        """
        Encodes a masked version of the structure where only fixed positions have valid info.
        This simulates the inference scenario where we don't have ground truth for positions to generate.

        Args:
            batch: Batch containing structure and is_fix mask.

        Returns:
            Latent vectors encoded from masked structure, shape [b, n, d].
        """
        if self.motif_constraint_mode == "flex":
            b, n = batch["coords_nm"].shape[:2]
            return torch.zeros(b, n, self.latent_dim, device=batch["coords_nm"].device)

        is_fix = batch.get("is_fix")

        # Create masked batch: zero out structure info at denoise positions
        masked_batch = {}
        masked_batch.update(batch)

        # Mask coordinates: set denoise positions to zero
        cond_mask_expanded = is_fix[..., None, None]  # [b, n, 1, 1]
        masked_coords_nm = batch["coords_nm"] * cond_mask_expanded  # [b, n, 37, 3]
        masked_batch["coords_nm"] = masked_coords_nm

        # Mask residue types: set to zero at denoise positions
        masked_residue_type = torch.where(
            is_fix,
            batch["residue_type"],
            torch.zeros_like(batch["residue_type"])
        )
        masked_batch["residue_type"] = masked_residue_type

        # Encode the masked structure
        with torch.no_grad():
            encoded_masked = self.autoencoder.encode(masked_batch)

        return encoded_masked["z_latent"]  # [b, n, d]

    @staticmethod
    def _aligned_region_rmsd_ang(
        pred_coords_nm: torch.Tensor,
        ref_coords_nm: torch.Tensor,
        align_mask: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = int(pred_coords_nm.shape[0])
        pred_flat = pred_coords_nm.reshape(b, -1, 3)
        ref_flat = ref_coords_nm.reshape(b, -1, 3)
        align_flat = align_mask.reshape(b, -1).bool()
        pred_aligned = kabsch_align(pred_flat, ref_flat, mask=align_flat).reshape_as(pred_coords_nm)

        region_mask = region_mask.bool()
        atom_mask = align_mask & region_mask[:, :, None]
        sq_diff = ((pred_aligned - ref_coords_nm) ** 2).sum(dim=-1)
        denom = atom_mask.sum(dim=(1, 2)).clamp(min=1)
        rmsd_nm = torch.sqrt((sq_diff * atom_mask).sum(dim=(1, 2)) / denom)
        return rmsd_nm * 10.0, pred_aligned

    @staticmethod
    def _aligned_ca_rmsd_ang(
        pred_coords_nm: torch.Tensor,
        ref_coords_nm: torch.Tensor,
        atom_mask: torch.Tensor,
        align_residue_mask: torch.Tensor,
        region_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ca_idx = atom_order["CA"]
        pred_ca = pred_coords_nm[:, :, ca_idx, :]
        ref_ca = ref_coords_nm[:, :, ca_idx, :]
        ca_mask = atom_mask[:, :, ca_idx].bool()
        align_mask = align_residue_mask.bool() & ca_mask
        pred_aligned_ca = kabsch_align(pred_ca, ref_ca, mask=align_mask)

        region_ca_mask = region_mask.bool() & ca_mask
        sq_diff = ((pred_aligned_ca - ref_ca) ** 2).sum(dim=-1)
        denom = region_ca_mask.sum(dim=1).clamp(min=1)
        rmsd_nm = torch.sqrt((sq_diff * region_ca_mask).sum(dim=1) / denom)
        return rmsd_nm * 10.0, pred_aligned_ca


    def handle_self_cond(self, batch: Dict) -> Dict:
        n_recycle = self.cfg_exp.training.get(
            "n_recycle", 0
        )
        if random.random() > 0.5 and self.cfg_exp.training.self_cond:
            nn_out = self.call_nn(batch, n_recycle=n_recycle)
            x_1_pred = self.fm.nn_out_to_clean_sample_prediction(
                batch=batch, nn_out=nn_out
            )
            batch["x_sc"] = {k: x_1_pred[k].detach() for k in x_1_pred}

        return batch

    def handle_recycling(self):
        n_recycle = self.cfg_exp.training.get("n_recycle", 0)
        if n_recycle == 0:
            return 0
        return random.randint(0, n_recycle)  # 0 and n_recycle included

    @staticmethod
    def _coerce_optional_feature_flag(flag_value, default: bool = False) -> bool:
        if flag_value is None:
            return default
        if isinstance(flag_value, torch.Tensor):
            if flag_value.numel() == 0:
                return default
            return bool(flag_value.bool().all().item())
        return bool(flag_value)

    @staticmethod
    def _normalize_optional_feature_flag(
        flag_value,
        batch_size: int,
        device: torch.device,
        default: bool = False,
    ):
        if flag_value is None:
            return torch.full((batch_size,), default, dtype=torch.bool, device=device)
        if isinstance(flag_value, torch.Tensor):
            if flag_value.numel() == 0:
                return torch.full((batch_size,), default, dtype=torch.bool, device=device)
            return flag_value.to(device=device).bool().reshape(batch_size, -1).any(dim=-1)
        return torch.full((batch_size,), bool(flag_value), dtype=torch.bool, device=device)

    def _infer_task_mode_from_batch(self, batch: Dict) -> str:
        has_explicit_optional_features = (
            "use_ca_coors_nm_feature" in batch
            or "use_residue_type_feature" in batch
        )
        if has_explicit_optional_features:
            use_ca_feature = self._coerce_optional_feature_flag(
                batch.get("use_ca_coors_nm_feature", False)
            )
            use_residue_type_feature = self._coerce_optional_feature_flag(
                batch.get("use_residue_type_feature", False)
            )
            is_docking_tensor = batch.get("is_docking_task", torch.tensor([False]))
            is_docking = bool(is_docking_tensor.all()) if isinstance(is_docking_tensor, torch.Tensor) else bool(is_docking_tensor)
            if use_ca_feature and use_residue_type_feature:
                return "packing"
            if use_ca_feature:
                return "invfold"
            if use_residue_type_feature:
                return "docking" if is_docking else "folding"
            return "generation"

        is_inv_folding_tensor = batch.get("is_inv_folding_task", torch.tensor([False]))
        is_docking_tensor = batch.get("is_docking_task", torch.tensor([False]))
        is_folding_tensor = batch.get("is_folding_task", torch.tensor([False]))
        is_packing_tensor = batch.get("is_packing_task", torch.tensor([False]))
        is_inv_folding = bool(is_inv_folding_tensor.all()) if isinstance(is_inv_folding_tensor, torch.Tensor) else bool(is_inv_folding_tensor)
        is_docking = bool(is_docking_tensor.all()) if isinstance(is_docking_tensor, torch.Tensor) else bool(is_docking_tensor)
        is_folding = bool(is_folding_tensor.all()) if isinstance(is_folding_tensor, torch.Tensor) else bool(is_folding_tensor)
        is_packing = bool(is_packing_tensor.all()) if isinstance(is_packing_tensor, torch.Tensor) else bool(is_packing_tensor)

        if is_packing:
            return "packing"
        if is_inv_folding:
            return "invfold"
        if is_docking:
            return "docking"
        # Older serialized batches may carry only the legacy alias.  New
        # inference data always sets is_docking_task, and validate_task rejects
        # external task_type='folding'.
        if is_folding:
            return "folding"
        return "generation"

    @staticmethod
    def _classify_validation_category(task_mode: str, has_hotspot: bool, has_motif: bool) -> str:
        if has_motif:
            condition_suffix = "motif"
        elif has_hotspot:
            condition_suffix = "hotspot"
        else:
            condition_suffix = "freegen"

        if task_mode == "generation":
            return condition_suffix
        return f"{task_mode}-{condition_suffix}"

    @staticmethod
    def _extract_optional_class_label(batch: Dict, key: str, src_idx: int):
        if key not in batch:
            return None
        value = batch[key]
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return int(value[src_idx].reshape(-1)[0].detach().cpu().item())
        if isinstance(value, (list, tuple)):
            item = value[src_idx]
            if isinstance(item, torch.Tensor):
                if item.numel() == 0:
                    return None
                return int(item.reshape(-1)[0].detach().cpu().item())
            return int(item)
        return int(value)

    @staticmethod
    def _get_batch_metadata_list(batch: Dict, key: str, expected_len: int, default_value):
        if hasattr(batch, key):
            value = getattr(batch, key)
        elif key in batch:
            value = batch[key]
        else:
            return [default_value for _ in range(expected_len)]

        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().tolist()
        elif isinstance(value, tuple):
            value = list(value)
        elif not isinstance(value, list):
            value = [value for _ in range(expected_len)]

        if len(value) < expected_len:
            value = list(value) + [default_value for _ in range(expected_len - len(value))]
        return value[:expected_len]

    def _optional_pair_bin_limits(self, device: torch.device) -> torch.Tensor:
        return torch.linspace(0.1, 3.0, 29, device=device)

    @staticmethod
    def _optional_condition_type_name(type_id: Any) -> str:
        mapping = {
            0: "none",
            1: "generic_joint",
            2: "generic_indep_none",
            3: "generic_indep_residue_only",
            4: "generic_indep_pair_only",
            5: "generic_indep_both",
            6: "legacy_inv_folding",
            7: "legacy_folding",
        }
        try:
            return mapping.get(int(type_id), f"unknown_{type_id}")
        except Exception:
            return "unknown"

    def _compute_optional_condition_metrics_for_sample(
        self,
        pred_residue_type: torch.Tensor,
        pred_coors_nm: torch.Tensor,
        pred_residue_mask: torch.Tensor,
        extra_features: Dict,
        sample_idx: int,
    ) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "conditioning_optional_type": self._extract_optional_class_label(
                extra_features, "optional_condition_type", sample_idx
            ),
            "conditioning_residue_count": 0,
            "conditioning_pair_count": 0,
            "conditioning_residue_hit_rate": None,
            "conditioning_pair_bin_hit_rate": None,
            "conditioning_pair_bin_hit_rate_pm1": None,
            "conditioning_pair_bin_mae": None,
            "conditioning_joint_all_hit": None,
            "binder_receptor_ca_center_distance_angstrom": None,
            "binder_ca_center_norm_angstrom": None,
            "receptor_ca_center_norm_angstrom": None,
            "chemical_bond_expected_count": 0,
            "chemical_bond_recovered_count": 0,
            "chemical_bond_recovery_rate": None,
            "chemical_bond_disulfide_expected_count": 0,
            "chemical_bond_disulfide_recovered_count": 0,
            "chemical_bond_disulfide_sg_sg_distances_angstrom": None,
            "chemical_bond_head_tail_expected_count": 0,
            "chemical_bond_head_tail_recovered_count": 0,
            "chemical_bond_head_tail_c_n_distances_angstrom": None,
            "chemical_bond_isopeptide_expected_count": 0,
            "chemical_bond_isopeptide_recovered_count": 0,
            "chemical_bond_isopeptide_distances_angstrom": None,
            "chemical_bond_other_expected_count": 0,
        }

        residue_condition_mask = getattr(extra_features, "residue_type_condition_mask", None)
        residue_condition = getattr(extra_features, "residue_type_condition", None)
        residue_hit_rate = None
        residue_all_hit = None
        if residue_condition_mask is not None and residue_condition is not None:
            cond_mask = residue_condition_mask[sample_idx].bool() & pred_residue_mask.bool()
            cond_count = int(cond_mask.sum().item())
            metrics["conditioning_residue_count"] = cond_count
            if cond_count > 0:
                cond_target = residue_condition[sample_idx][cond_mask].long()
                cond_pred = pred_residue_type[cond_mask].long()
                hits = (cond_target == cond_pred).float()
                residue_hit_rate = float(hits.mean().item())
                residue_all_hit = bool(hits.all().item())
                metrics["conditioning_residue_hit_rate"] = residue_hit_rate

        pair_valid_mask = getattr(extra_features, "optional_ca_pair_valid_mask", None)
        pair_indices = getattr(extra_features, "optional_ca_pair_indices", None)
        pair_bin_idx = getattr(extra_features, "optional_ca_pair_bin_idx", None)
        pair_condition = getattr(extra_features, "optional_ca_pair_dist_condition", None)
        pair_condition_mask = getattr(extra_features, "optional_ca_pair_dist_mask", None)
        pair_hit_rate = None
        pair_hit_rate_pm1 = None
        pair_mae = None
        pair_all_hit = None
        idx_pairs = None
        target_bins = None
        if pair_valid_mask is not None and pair_indices is not None and pair_bin_idx is not None:
            valid_mask = pair_valid_mask[sample_idx].bool()
            if valid_mask.any():
                idx_pairs = pair_indices[sample_idx][valid_mask].long()
                target_bins = pair_bin_idx[sample_idx][valid_mask].long()
        elif pair_condition is not None and pair_condition_mask is not None:
            condition_i = pair_condition[sample_idx].float()
            condition_mask_i = pair_condition_mask[sample_idx].bool()
            if condition_mask_i.ndim == 2:
                condition_mask_i = torch.triu(condition_mask_i, diagonal=1)
            if condition_mask_i.any():
                idx_pairs = torch.nonzero(condition_mask_i, as_tuple=False).long()
                target_bins = condition_i[idx_pairs[:, 0], idx_pairs[:, 1]].argmax(dim=-1).long()

        if idx_pairs is not None and target_bins is not None:
            pair_count = int(idx_pairs.shape[0])
            metrics["conditioning_pair_count"] = pair_count
            if pair_count > 0:
                ca_pred = pred_coors_nm[:, 1, :]
                idx_i = idx_pairs[:, 0].clamp(min=0, max=max(ca_pred.shape[0] - 1, 0))
                idx_j = idx_pairs[:, 1].clamp(min=0, max=max(ca_pred.shape[0] - 1, 0))
                pred_dist_nm = torch.norm(ca_pred[idx_i] - ca_pred[idx_j], dim=-1)
                pred_bins = torch.bucketize(pred_dist_nm, self._optional_pair_bin_limits(device=pred_dist_nm.device))
                bin_error = (pred_bins - target_bins).abs()
                pair_hit_rate = float((bin_error == 0).float().mean().item())
                pair_hit_rate_pm1 = float((bin_error <= 1).float().mean().item())
                pair_mae = float(bin_error.float().mean().item())
                pair_all_hit = bool((bin_error == 0).all().item())
                metrics["conditioning_pair_bin_hit_rate"] = pair_hit_rate
                metrics["conditioning_pair_bin_hit_rate_pm1"] = pair_hit_rate_pm1
                metrics["conditioning_pair_bin_mae"] = pair_mae

        if metrics["conditioning_residue_count"] > 0 or metrics["conditioning_pair_count"] > 0:
            residue_ok = True if metrics["conditioning_residue_count"] == 0 else bool(residue_all_hit)
            pair_ok = True if metrics["conditioning_pair_count"] == 0 else bool(pair_all_hit)
            metrics["conditioning_joint_all_hit"] = bool(residue_ok and pair_ok)

        is_peptide = getattr(extra_features, "is_peptide", None)
        if is_peptide is not None:
            is_peptide_i = is_peptide[sample_idx].to(device=pred_coors_nm.device).bool()
            valid_residue = pred_residue_mask.to(device=pred_coors_nm.device).bool()
            binder_mask = is_peptide_i & valid_residue
            receptor_mask = (~is_peptide_i) & valid_residue
            if binder_mask.any() and receptor_mask.any():
                ca_pred = pred_coors_nm[:, atom_order["CA"], :]
                binder_center = ca_pred[binder_mask].mean(dim=0)
                receptor_center = ca_pred[receptor_mask].mean(dim=0)
                metrics["binder_receptor_ca_center_distance_angstrom"] = float(
                    torch.norm(binder_center - receptor_center).detach().cpu().item() * 10.0
                )
                metrics["binder_ca_center_norm_angstrom"] = float(
                    torch.norm(binder_center).detach().cpu().item() * 10.0
                )
                metrics["receptor_ca_center_norm_angstrom"] = float(
                    torch.norm(receptor_center).detach().cpu().item() * 10.0
                )

        chemical_bond_type = getattr(extra_features, "chemical_bond_type", None)
        if chemical_bond_type is not None:
            bond_matrix = chemical_bond_type[sample_idx].long()
            valid_pair_mask = pred_residue_mask.bool()[:, None] & pred_residue_mask.bool()[None, :]
            upper_bond_mask = torch.triu((bond_matrix > 0) & valid_pair_mask, diagonal=1)
            bond_pairs = torch.nonzero(upper_bond_mask, as_tuple=False).long()
            if bond_pairs.numel() > 0:
                cys_idx = restypes.index("C")
                lys_idx = restypes.index("K")
                asp_idx = restypes.index("D")
                glu_idx = restypes.index("E")
                sg_idx = atom_order["SG"]
                n_idx = atom_order["N"]
                c_idx = atom_order["C"]
                nz_idx = atom_order["NZ"]
                cg_idx = atom_order["CG"]
                cd_idx = atom_order["CD"]
                recovered_count = 0
                disulfide_sg_dists = []
                head_tail_cn_dists = []
                isopeptide_dists = []
                for idx_i_t, idx_j_t in bond_pairs.tolist():
                    idx_i = int(idx_i_t)
                    idx_j = int(idx_j_t)
                    bond_id = int(bond_matrix[idx_i, idx_j].item())
                    if bond_id == CHEMICAL_BOND_DISULFIDE:
                        metrics["chemical_bond_disulfide_expected_count"] += 1
                        is_cys_pair = (
                            int(pred_residue_type[idx_i].item()) == cys_idx
                            and int(pred_residue_type[idx_j].item()) == cys_idx
                        )
                        if is_cys_pair:
                            dist_ang = float(
                                torch.norm(
                                    pred_coors_nm[idx_i, sg_idx] - pred_coors_nm[idx_j, sg_idx],
                                    dim=-1,
                                ).detach().cpu().item()
                                * 10.0
                            )
                            disulfide_sg_dists.append(dist_ang)
                            if dist_ang <= 2.3:
                                recovered_count += 1
                                metrics["chemical_bond_disulfide_recovered_count"] += 1
                    elif bond_id == CHEMICAL_BOND_HEAD_TAIL_AMIDE:
                        metrics["chemical_bond_head_tail_expected_count"] += 1
                        n_term_idx, c_term_idx = sorted((idx_i, idx_j))
                        dist_ang = float(
                            torch.norm(
                                pred_coors_nm[c_term_idx, c_idx] - pred_coors_nm[n_term_idx, n_idx],
                                dim=-1,
                            ).detach().cpu().item()
                            * 10.0
                        )
                        head_tail_cn_dists.append(dist_ang)
                        if dist_ang <= 2.0:
                            recovered_count += 1
                            metrics["chemical_bond_head_tail_recovered_count"] += 1
                    elif bond_id == CHEMICAL_BOND_ISOPEPTIDE:
                        metrics["chemical_bond_isopeptide_expected_count"] += 1
                        aa_i = int(pred_residue_type[idx_i].item())
                        aa_j = int(pred_residue_type[idx_j].item())
                        lys_pos = acid_pos = acid_atom_idx = None
                        if aa_i == lys_idx and aa_j in {asp_idx, glu_idx}:
                            lys_pos = idx_i
                            acid_pos = idx_j
                            acid_atom_idx = cg_idx if aa_j == asp_idx else cd_idx
                        elif aa_j == lys_idx and aa_i in {asp_idx, glu_idx}:
                            lys_pos = idx_j
                            acid_pos = idx_i
                            acid_atom_idx = cg_idx if aa_i == asp_idx else cd_idx
                        if lys_pos is not None and acid_pos is not None and acid_atom_idx is not None:
                            dist_ang = float(
                                torch.norm(
                                    pred_coors_nm[lys_pos, nz_idx] - pred_coors_nm[acid_pos, acid_atom_idx],
                                    dim=-1,
                                ).detach().cpu().item()
                                * 10.0
                            )
                            isopeptide_dists.append(dist_ang)
                            if abs(dist_ang - 1.335) <= 0.35:
                                recovered_count += 1
                                metrics["chemical_bond_isopeptide_recovered_count"] += 1
                    elif bond_id == CHEMICAL_BOND_OTHER_COVALENT:
                        metrics["chemical_bond_other_expected_count"] += 1

                expected_count = int(bond_pairs.shape[0])
                metrics["chemical_bond_expected_count"] = expected_count
                metrics["chemical_bond_recovered_count"] = int(recovered_count)
                metrics["chemical_bond_recovery_rate"] = float(recovered_count) / float(expected_count)
                if disulfide_sg_dists:
                    metrics["chemical_bond_disulfide_sg_sg_distances_angstrom"] = ";".join(
                        f"{dist:.3f}" for dist in disulfide_sg_dists
                    )
                if head_tail_cn_dists:
                    metrics["chemical_bond_head_tail_c_n_distances_angstrom"] = ";".join(
                        f"{dist:.3f}" for dist in head_tail_cn_dists
                    )
                if isopeptide_dists:
                    metrics["chemical_bond_isopeptide_distances_angstrom"] = ";".join(
                        f"{dist:.3f}" for dist in isopeptide_dists
                    )

        return metrics

    def handle_folding_n_inverse_folding(self, batch: Dict) -> Dict:
        """
        With 15% probability either a folding or inverse folding iteration.
        If one such iteration (ie 15% of the times), with 50% probability set
        set folding_mode to true, otherwise set inverse_folding_mode to true.

        For inverse folding, we just provide CA.

        Applies to the whole batch.

        With 85% probability sets both to false.

        Adds entries 'folding_mode' and 'inverse_folding_ca_mode' to batch, with
        values being boolean variables (True or False).
        """
        if (
            "use_ca_coors_nm_feature" in batch
            or "use_residue_type_feature" in batch
        ):
            batch_size = int(batch["coords"].shape[0])
            device = batch["coords"].device
            batch["use_ca_coors_nm_feature"] = self._normalize_optional_feature_flag(
                batch.get("use_ca_coors_nm_feature", False),
                batch_size=batch_size,
                device=device,
            )
            batch["use_residue_type_feature"] = self._normalize_optional_feature_flag(
                batch.get("use_residue_type_feature", False),
                batch_size=batch_size,
                device=device,
            )
            return batch

        batch["use_ca_coors_nm_feature"] = False
        batch["use_residue_type_feature"] = False
        is_inv_folding_training = self.cfg_exp.training.get("inv_folding_only", False)
        if not is_inv_folding_training:
            prob = self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0)
            r1 = random.random()  # float
            if r1 < prob:  # with p=prob
                r2 = random.random()
                if r2 < 0.5:  # with p=0.5
                    batch["use_ca_coors_nm_feature"] = True
                else:
                    batch["use_residue_type_feature"] = True
        else:
            batch["use_ca_coors_nm_feature"] = True
        return batch

    # I need a handle condition func here? or maybe just in transform before training step?
    # I think the latter is better

    def log_losses(
        self,
        bs: int,
        losses: Dict[str, Float[torch.Tensor, "b"]],
        log_prefix: str,
        batch: Dict,
    ):
        for k in losses:
            log_name = k[: -len("_justlog")] if k.endswith("_justlog") else k

            self.log(
                f"{log_prefix}/loss_{log_name}",
                torch.mean(losses[k]),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=bs,
                sync_dist=True,
                add_dataloader_idx=False,
            )

            if self.cfg_exp.training.get("p_folding_n_inv_folding_iters", 0.0) > 0.0:
                # Log also for folding and inverse folding iters
                # divides by p_aux to account for the fact that for most steps loss will be just zero
                p_aux = self.cfg_exp.training["p_folding_n_inv_folding_iters"] / 2
                f_inv_fold = batch["use_ca_coors_nm_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_invfold_ca_iter/loss_{log_name}",
                    torch.mean(losses[k] * f_inv_fold),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

                f_fold = batch["use_residue_type_feature"] * 1.0 / p_aux
                self.log(
                    f"{log_prefix}_fold_iter/loss_{log_name}",
                    torch.mean(losses[k] * f_fold),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    logger=True,
                    batch_size=bs,
                    sync_dist=True,
                    add_dataloader_idx=False,
                )

    def log_train_loss_n_prog_bar(self, b: int, train_loss: torch.Tensor):
        self.log(
            f"train_loss",
            train_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=b,
            sync_dist=True,
            add_dataloader_idx=False,
        )

    def log_nparams(self):
        self.log(
            "scaling/nparams",
            self.nparams * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )  # constant line

    def update_n_log_nsamples_processed(self, b: int):
        self.nsamples_processed = self.nsamples_processed + b * self.trainer.world_size
        self.log(
            "scaling/nsamples_processed",
            self.nsamples_processed * 1.0,
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            logger=True,
            batch_size=1,
            sync_dist=True,
        )

    def update_n_log_flops(self, b: int, n: int):
        """
        Updates and logs flops, if available
        """
        try:
            nflops_step = self.nn.nflops_computer(
                b, n
            )  # nn should implement this function if we want to see nflops
        except:
            nflops_step = None

        if nflops_step is not None:
            self.nflops = (
                self.nflops + nflops_step * self.trainer.world_size
            )  # Times number of processes so it logs sum across devices
            self.log(
                "scaling/nflops",
                self.nflops * 1.0,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                logger=True,
                batch_size=1,
                sync_dist=True,
            )

    def validation_step(self, batch: Dict, batch_idx: int):
        """
        Validation step.

        Args:
            batch: batch from dataset (see last argument)
            batch_idx: batch index (unused)
        """
        self.validation_step_data(batch, batch_idx)

    def validation_step_data(self, batch: Dict, batch_idx: int):
        """Evaluates the training loss on validation data and collects batches for inference."""
        batch = self.handle_folding_n_inverse_folding(batch)
        run_generation = bool(self.cfg_exp.get("val_run_generation", False))
        batch_for_inference = copy.deepcopy(batch) if run_generation else None
        if (
            batch_for_inference is not None
            and bool(self.cfg_exp.get("val_generation_disable_chemical_bond", False))
            and "chemical_bond_type" in batch_for_inference
        ):
            batch_for_inference["chemical_bond_type"] = torch.zeros_like(
                batch_for_inference["chemical_bond_type"]
            )

        with torch.no_grad():
            loss = self.training_step(batch, batch_idx=-1)
            self.validation_output_data.append(loss.item())
            if run_generation:
                self.validation_inference_tasks.append(batch_for_inference)

    def on_validation_epoch_end(self):
        """
        Takes the samples produced in the validation step, stores them as pdb files, and computes validation metrics.
        It also cleans results.
        """
        self.on_validation_epoch_end_data()

    def on_validation_epoch_end_data(self):
        """
        Generate samples and store as PDB strings with metadata.
        """
        if not bool(self.cfg_exp.get("val_run_generation", False)):
            self.validation_output_data = []
            self.validation_inference_tasks = []
            return

        val_gen_cfg = self.cfg_exp.get("generation", None)
        if val_gen_cfg is None:
            logger.warning("No generation config found, skipping sample generation")
            self.validation_output_data = []
            self.validation_inference_tasks = []
            return

        self.configure_inference(val_gen_cfg, nn_ag=None)
        val_root_path = os.path.join(self.val_path_tmp, f"epoch_{self.current_epoch}")
        os.makedirs(val_root_path, exist_ok=True)

        metrics_records = []

        for batch in tqdm(self.validation_inference_tasks, desc="Generating validation samples"):
            task_mode = self._infer_task_mode_from_batch(batch)
            pdb_list = self.predict_step(batch, batch_idx=0, ret_mode='pdb_string')

            for pdb_entry in pdb_list:
                task_id = pdb_entry["task_id"]
                entry_name = f"{task_id}"
                pdb_filename = f"{entry_name}.pdb"
                pdb_path_complex = os.path.join(val_root_path, pdb_filename)
                with open(pdb_path_complex, "w") as f:
                    f.write(pdb_entry["pdb_string"])

                has_hotspot = pdb_entry.get("has_hotspot", False)
                has_motif = pdb_entry.get("has_motif", False)
                category = self._classify_validation_category(
                    task_mode=task_mode,
                    has_hotspot=has_hotspot,
                    has_motif=has_motif,
                )
                
                metrics_records.append({
                    "task_id": task_id,
                    "task_mode": task_mode,
                    "category": category,
                    "nres": pdb_entry.get("nres", 0),
                    "peptide_seq_recovery": pdb_entry.get("peptide_seq_recovery", 0.0),
                    "peptide_rmsd": pdb_entry.get("peptide_rmsd", 0.0),
                    "receptor_rmsd": pdb_entry.get("receptor_rmsd", 0.0),
                    "complex_ca_rmsd": pdb_entry.get("complex_ca_rmsd", 0.0),
                    "has_hotspot": has_hotspot,
                    "has_motif": has_motif,
                    "conditioning_optional_type": pdb_entry.get("conditioning_optional_type", None),
                    "conditioning_residue_count": pdb_entry.get("conditioning_residue_count", 0),
                    "conditioning_pair_count": pdb_entry.get("conditioning_pair_count", 0),
                    "conditioning_residue_hit_rate": pdb_entry.get("conditioning_residue_hit_rate", None),
                    "conditioning_pair_bin_hit_rate": pdb_entry.get("conditioning_pair_bin_hit_rate", None),
                    "conditioning_pair_bin_hit_rate_pm1": pdb_entry.get("conditioning_pair_bin_hit_rate_pm1", None),
                    "conditioning_pair_bin_mae": pdb_entry.get("conditioning_pair_bin_mae", None),
                    "conditioning_joint_all_hit": pdb_entry.get("conditioning_joint_all_hit", None),
                    "binder_receptor_ca_center_distance_angstrom": pdb_entry.get("binder_receptor_ca_center_distance_angstrom", None),
                    "binder_ca_center_norm_angstrom": pdb_entry.get("binder_ca_center_norm_angstrom", None),
                    "receptor_ca_center_norm_angstrom": pdb_entry.get("receptor_ca_center_norm_angstrom", None),
                    "chemical_bond_expected_count": pdb_entry.get("chemical_bond_expected_count", 0),
                    "chemical_bond_recovered_count": pdb_entry.get("chemical_bond_recovered_count", 0),
                    "chemical_bond_recovery_rate": pdb_entry.get("chemical_bond_recovery_rate", None),
                    "chemical_bond_disulfide_expected_count": pdb_entry.get("chemical_bond_disulfide_expected_count", 0),
                    "chemical_bond_disulfide_recovered_count": pdb_entry.get("chemical_bond_disulfide_recovered_count", 0),
                    "chemical_bond_disulfide_sg_sg_distances_angstrom": pdb_entry.get("chemical_bond_disulfide_sg_sg_distances_angstrom", None),
                    "chemical_bond_head_tail_expected_count": pdb_entry.get("chemical_bond_head_tail_expected_count", 0),
                    "chemical_bond_head_tail_recovered_count": pdb_entry.get("chemical_bond_head_tail_recovered_count", 0),
                    "chemical_bond_head_tail_c_n_distances_angstrom": pdb_entry.get("chemical_bond_head_tail_c_n_distances_angstrom", None),
                    "chemical_bond_isopeptide_expected_count": pdb_entry.get("chemical_bond_isopeptide_expected_count", 0),
                    "chemical_bond_isopeptide_recovered_count": pdb_entry.get("chemical_bond_isopeptide_recovered_count", 0),
                    "chemical_bond_isopeptide_distances_angstrom": pdb_entry.get("chemical_bond_isopeptide_distances_angstrom", None),
                    "chemical_bond_other_expected_count": pdb_entry.get("chemical_bond_other_expected_count", 0),
                })

                requested_secstruct_class = pdb_entry.get("conditioning_binder_secondary_class", None)
                requested_secstruct_class_raw = pdb_entry.get("conditioning_binder_secondary_class_raw", None)
                if requested_secstruct_class is not None:
                    secstruct_update = {
                        "conditioning_binder_secondary_class": requested_secstruct_class,
                        "conditioning_binder_secondary_name": binder_secstruct_name_from_id(requested_secstruct_class),
                        "conditioning_binder_secondary_class_raw": requested_secstruct_class_raw,
                        "conditioning_binder_secondary_name_raw": (
                            binder_secstruct_name_from_id(requested_secstruct_class_raw)
                            if requested_secstruct_class_raw is not None
                            else None
                        ),
                    }
                    try:
                        gen_ss_metrics = secondary_structure_metrics_from_pdb_string(
                            pdb_string=pdb_entry["pdb_string"],
                            peptide_chain="L",
                        )
                        gen_ss_class_id, gen_ss_class_name, gen_ss_threshold_hit = classify_secondary_structure_metrics(
                            gen_ss_metrics
                        )
                        secstruct_update.update(
                            {
                                "generated_binder_secondary_class": gen_ss_class_id,
                                "generated_binder_secondary_name": gen_ss_class_name,
                                "generated_binder_secondary_threshold_hit": bool(gen_ss_threshold_hit),
                                "generated_binder_helix_fraction": float(gen_ss_metrics["helix_fraction"]),
                                "generated_binder_sheet_fraction": float(gen_ss_metrics["sheet_fraction"]),
                                "generated_binder_coil_fraction": float(gen_ss_metrics["coil_fraction"]),
                                "generated_binder_structured_fraction": float(
                                    gen_ss_metrics["helix_fraction"] + gen_ss_metrics["sheet_fraction"]
                                ),
                                "generated_binder_longest_helix_run_count": float(
                                    gen_ss_metrics["longest_helix_run"] * gen_ss_metrics["num_residues"]
                                ),
                                "generated_binder_longest_sheet_run_count": float(
                                    gen_ss_metrics["longest_sheet_run"] * gen_ss_metrics["num_residues"]
                                ),
                                "generated_binder_secstruct_match": bool(
                                    int(requested_secstruct_class) == int(gen_ss_class_id)
                                ),
                                "generated_binder_secstruct_match_raw": bool(
                                    requested_secstruct_class_raw is not None
                                    and int(requested_secstruct_class_raw) == int(gen_ss_class_id)
                                ),
                            }
                        )
                    except Exception as exc:
                        logger.warning(
                            f"Failed to compute generated binder secondary-structure metrics "
                            f"for validation sample {task_id}: {exc}"
                        )
                        secstruct_update.update(
                            {
                                "generated_binder_secondary_class": None,
                                "generated_binder_secondary_name": None,
                                "generated_binder_secondary_threshold_hit": None,
                                "generated_binder_helix_fraction": None,
                                "generated_binder_sheet_fraction": None,
                                "generated_binder_coil_fraction": None,
                                "generated_binder_structured_fraction": None,
                                "generated_binder_longest_helix_run_count": None,
                                "generated_binder_longest_sheet_run_count": None,
                                "generated_binder_secstruct_match": None,
                                "generated_binder_secstruct_match_raw": None,
                            }
                        )
                    metrics_records[-1].update(secstruct_update)
        
        if dist.is_available() and dist.is_initialized():
            gathered_metrics = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_metrics, metrics_records)
            all_metrics_records = [item for sublist in gathered_metrics for item in sublist]
        else:
            all_metrics_records = metrics_records
        
        if self.trainer.is_global_zero:
            if all_metrics_records:
                df_metrics = pd.DataFrame(all_metrics_records)
                df_metrics = df_metrics.drop_duplicates(subset=["task_id"])
                csv_path = os.path.join(val_root_path, "validation_metrics.csv")
                df_metrics.to_csv(csv_path, index=False)
                logger.info(f"Saved per-sample metrics to {csv_path} (Total samples: {len(df_metrics)})")
                avg_rmsd = df_metrics["peptide_rmsd"].mean()
                avg_recovery = df_metrics["peptide_seq_recovery"].mean()
                logger.info(f"Epoch {self.current_epoch} Summary | Avg RMSD: {avg_rmsd:.4f} Å | Avg Recovery: {avg_recovery:.4f}")

                df_res_cond = df_metrics[df_metrics["conditioning_residue_count"] > 0]
                if not df_res_cond.empty:
                    logger.info(
                        f"Epoch {self.current_epoch} Condition | "
                        f"ResidueHit: {df_res_cond['conditioning_residue_hit_rate'].mean():.4f} "
                        f"(n={len(df_res_cond)}, avg_count={df_res_cond['conditioning_residue_count'].mean():.2f})"
                    )

                df_pair_cond = df_metrics[df_metrics["conditioning_pair_count"] > 0]
                if not df_pair_cond.empty:
                    logger.info(
                        f"Epoch {self.current_epoch} Condition | "
                        f"PairBinHit: {df_pair_cond['conditioning_pair_bin_hit_rate'].mean():.4f} | "
                        f"PairBinHit±1: {df_pair_cond['conditioning_pair_bin_hit_rate_pm1'].mean():.4f} | "
                        f"PairBinMAE: {df_pair_cond['conditioning_pair_bin_mae'].mean():.4f} "
                        f"(n={len(df_pair_cond)}, avg_count={df_pair_cond['conditioning_pair_count'].mean():.2f})"
                    )

                df_joint_cond = df_metrics[
                    (df_metrics["conditioning_residue_count"] > 0)
                    & (df_metrics["conditioning_pair_count"] > 0)
                    & df_metrics["conditioning_joint_all_hit"].notna()
                ]
                if not df_joint_cond.empty:
                    logger.info(
                        f"Epoch {self.current_epoch} Condition | "
                        f"JointAllHit: {df_joint_cond['conditioning_joint_all_hit'].mean():.4f} "
                        f"(n={len(df_joint_cond)})"
                    )

                if "conditioning_optional_type" in df_metrics.columns:
                    for cond_type in sorted(df_metrics["conditioning_optional_type"].dropna().unique().tolist()):
                        df_type = df_metrics[df_metrics["conditioning_optional_type"] == cond_type]
                        if df_type.empty:
                            continue
                        msg = (
                            f"Epoch {self.current_epoch} Condition "
                            f"[{self._optional_condition_type_name(cond_type)}] | "
                            f"Count: {len(df_type)}"
                        )
                        df_type_res = df_type[df_type["conditioning_residue_count"] > 0]
                        if not df_type_res.empty:
                            msg += f" | ResidueHit: {df_type_res['conditioning_residue_hit_rate'].mean():.4f}"
                        df_type_pair = df_type[df_type["conditioning_pair_count"] > 0]
                        if not df_type_pair.empty:
                            msg += (
                                f" | PairHit: {df_type_pair['conditioning_pair_bin_hit_rate'].mean():.4f}"
                                f" | PairHit±1: {df_type_pair['conditioning_pair_bin_hit_rate_pm1'].mean():.4f}"
                            )
                        logger.info(msg)

                if "generated_binder_secstruct_match" in df_metrics.columns:
                    df_cond = df_metrics[
                        df_metrics["conditioning_binder_secondary_class"].notna()
                        & (df_metrics["conditioning_binder_secondary_class"] != 0)
                    ]
                    if not df_cond.empty:
                        cond_match = df_cond["generated_binder_secstruct_match"].mean()
                        logger.info(
                            f"Epoch {self.current_epoch} SS Control | "
                            f"Conditioned non-null match: {cond_match:.4f} "
                            f"(n={len(df_cond)})"
                        )
                        if "generated_binder_secstruct_match_raw" in df_cond.columns:
                            raw_match = df_cond["generated_binder_secstruct_match_raw"].mean()
                            logger.info(
                                f"Epoch {self.current_epoch} SS Control | "
                                f"Raw-label match: {raw_match:.4f} "
                                f"(n={len(df_cond)})"
                            )
                        for class_id in sorted(df_cond["conditioning_binder_secondary_class"].dropna().unique().tolist()):
                            df_cls = df_cond[df_cond["conditioning_binder_secondary_class"] == class_id]
                            if df_cls.empty:
                                continue
                            logger.info(
                                f"Epoch {self.current_epoch} SS Control "
                                f"[requested={binder_secstruct_name_from_id(class_id)}] | "
                                f"Count: {len(df_cls)} | "
                                f"Match: {df_cls['generated_binder_secstruct_match'].mean():.4f}"
                            )

                category_order = [
                    "freegen",
                    "hotspot",
                    "motif",
                    "docking-motif",
                    "docking-hotspot",
                    "docking-freegen",
                    "invfold-motif",
                    "invfold-hotspot",
                    "invfold-freegen",
                ]
                # Legacy checkpoint-only batches may still report folding-*
                # categories; they remain discoverable through extra_categories
                # but are not part of the active docking validation contract.
                extra_categories = [
                    category for category in df_metrics["category"].unique().tolist()
                    if category not in category_order
                ]
                for category in category_order + sorted(extra_categories):
                    df_category = df_metrics[df_metrics["category"] == category]
                    if df_category.empty:
                        continue
                    category_avg_rmsd = df_category["peptide_rmsd"].mean()
                    category_avg_recovery = df_category["peptide_seq_recovery"].mean()
                    logger.info(
                        f"Epoch {self.current_epoch} Summary [{category}] | "
                        f"Count: {len(df_category)} | "
                        f"Avg RMSD: {category_avg_rmsd:.4f} Å | "
                        f"Avg Recovery: {category_avg_recovery:.4f}"
                    )
                    df_category_res = df_category[df_category["conditioning_residue_count"] > 0]
                    if not df_category_res.empty:
                        logger.info(
                            f"Epoch {self.current_epoch} Condition [{category}] | "
                            f"ResidueHit: {df_category_res['conditioning_residue_hit_rate'].mean():.4f} "
                            f"(n={len(df_category_res)})"
                        )
                    df_category_pair = df_category[df_category["conditioning_pair_count"] > 0]
                    if not df_category_pair.empty:
                        logger.info(
                            f"Epoch {self.current_epoch} Condition [{category}] | "
                            f"PairHit: {df_category_pair['conditioning_pair_bin_hit_rate'].mean():.4f} | "
                            f"PairHit±1: {df_category_pair['conditioning_pair_bin_hit_rate_pm1'].mean():.4f} "
                            f"(n={len(df_category_pair)})"
                        )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

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

    def configure_inference(self, inf_cfg, nn_ag, search_cfg=None, receptor_source: Any = "decoded"):
        """Sets inference config with all sampling parameters required by the method (dt, etc)
        and autoguidance network (or None if not provided)."""
        self.inf_cfg = inf_cfg
        self.nn_ag = nn_ag
        self.search_cfg = search_cfg
        self.output_receptor_source = self._normalize_receptor_source(receptor_source)
        search_algorithm = (
            str(search_cfg.get("algorithm", "null")).lower()
            if search_cfg is not None
            else "null"
        )
        self.reward_model = None
        if search_algorithm in {"null", "none"}:
            self._search_instance = None
        elif search_algorithm == "best_of_n":
            from partiflow.sample.best_of_n import BestOfNSearch
            self._search_instance = BestOfNSearch(self, search_cfg)
        else:
            raise ValueError(f"Unsupported search.algorithm: {search_algorithm}")

    def predict_step(self, batch: Dict, batch_idx: int, ret_mode=None) -> List[Tuple[torch.tensor]]:
        """
        Makes predictions. Should call set_inf_cfg before calling this.

        Args:
            batch: data batch, contains all info for the samples to generate (nsamples, nres, dt, etc.)
                The full batch is passed through to the prediction functions.

        Returns:
            List of tuples. Each tuple represents one of the generated samples, has two elements:
                coordinates tensor of shape [n, 37, 3], and aatype tensor of shape [n].
        """

        if ret_mode is None:
            ret_mode = getattr(self, "predict_ret_mode", "coors37_n_aatype")

        noise_scale = self.inf_cfg.args.get("weight_noise_scale", 0.0)
        noise_seed = int(self.inf_cfg.args.get("weight_noise_seed", 42))
        if self.inf_cfg.args.get("weight_noise_per_batch", False):
            noise_seed += batch_idx
        if noise_scale > 0:
            self.perturb_nn_weights(seed=noise_seed, scale=noise_scale, reverse=False)
        
        try:
            self_cond = self.inf_cfg.args.self_cond
            nsteps = self.inf_cfg.args.nsteps
            guidance_w = self.inf_cfg.args.get("guidance_w", 1.0)
            ag_ratio = self.inf_cfg.args.get("ag_ratio", 0.0)
            save_trajectory_every = 0

            with torch.no_grad():
                if self._cfg_bool_arg(
                    self.inf_cfg.args,
                    "disable_chemical_bond_condition",
                    False,
                ) and "chemical_bond_type" in batch:
                    batch["chemical_bond_type"] = torch.zeros_like(batch["chemical_bond_type"])

                has_explicit_optional_features = (
                    "use_ca_coors_nm_feature" in batch
                    or "use_residue_type_feature" in batch
                )
                if has_explicit_optional_features:
                    batch_size = int(batch["coords"].shape[0])
                    device = batch["coords"].device
                    batch["use_ca_coors_nm_feature"] = self._normalize_optional_feature_flag(
                        batch.get("use_ca_coors_nm_feature", False),
                        batch_size=batch_size,
                        device=device,
                    )
                    batch["use_residue_type_feature"] = self._normalize_optional_feature_flag(
                        batch.get("use_residue_type_feature", False),
                        batch_size=batch_size,
                        device=device,
                    )

                is_inv_folding_tensor = batch.get("is_inv_folding_task", torch.tensor([False]))
                is_docking_tensor = batch.get("is_docking_task", torch.tensor([False]))
                is_folding_tensor = batch.get("is_folding_task", torch.tensor([False]))
                is_packing_tensor = batch.get("is_packing_task", torch.tensor([False]))
                is_inv_folding = bool(is_inv_folding_tensor.all()) if isinstance(is_inv_folding_tensor, torch.Tensor) else is_inv_folding_tensor
                is_docking = bool(is_docking_tensor.all()) if isinstance(is_docking_tensor, torch.Tensor) else is_docking_tensor
                is_folding = bool(is_folding_tensor.all()) if isinstance(is_folding_tensor, torch.Tensor) else is_folding_tensor
                is_packing = bool(is_packing_tensor.all()) if isinstance(is_packing_tensor, torch.Tensor) else is_packing_tensor

                if has_explicit_optional_features:
                    logger.info(
                        "Using optional input feature flags from batch"
                        #f"CA={batch['use_ca_coors_nm_feature'].tolist()}, "
                        #f"residue_type={batch['use_residue_type_feature'].tolist()}"
                    )
                elif is_inv_folding:
                    logger.info("Inverse folding task: using CA coordinates as input feature")
                    batch["use_ca_coors_nm_feature"] = True
                    batch["use_residue_type_feature"] = False
                elif is_docking or is_folding:
                    logger.info("Docking task: using residue type as input feature")
                    batch["use_ca_coors_nm_feature"] = False
                    batch["use_residue_type_feature"] = True
                elif is_packing:
                    logger.info("Packing task: using CA coordinates and residue type as input features")
                    batch["use_ca_coors_nm_feature"] = True
                    batch["use_residue_type_feature"] = True
                else:
                    batch["use_ca_coors_nm_feature"] = False
                    batch["use_residue_type_feature"] = False

                if 'gt_cond_latents' not in batch:
                    # logger.info("Computing gt_cond_latents for inference")
                    # Compute is_fix before calling _get_cond_latents
                    mask = batch["mask_dict"]["coords"][..., 0, 0]
                    # is_peptide = (batch["is_peptide"] == 1) & mask
                    is_receptor = (batch["is_peptide"] == 0) & mask

                    if self.motif_constraint_mode == "flex":
                        batch["is_fix"] = torch.zeros_like(mask, dtype=torch.bool)
                    elif self.motif_constraint_mode == "rigid":
                        latent_cond_mask = batch.get("latent_cond_mask", is_receptor)
                        batch["is_fix"] = latent_cond_mask & mask
                    elif self.motif_constraint_mode == "rigid_receptor":
                        batch["is_fix"] = is_receptor
                    else:
                        raise ValueError(f"Unknown motif_constraint_mode: {self.motif_constraint_mode}")

                    batch['gt_cond_latents'] = self._get_cond_latents(batch)  # [b, n, d]

            search_algorithm = (
                str(self.search_cfg.get("algorithm", "null")).lower()
                if self.search_cfg is not None
                else "null"
            )

            if self._search_instance is not None:
                search_result = self._search_instance.search(batch)
                if isinstance(search_result, dict) and "final" in search_result:
                    sample_prots = search_result["final"]
                    stream_specs = list(search_result.get("streams", []))
                else:
                    sample_prots = search_result
                    stream_specs = [
                        {
                            "sample_prots": sample_prots,
                            "sample_type": "final",
                            "rankable": True,
                            "stream_subdir": None,
                            "step_start": None,
                            "step_end": None,
                        }
                    ]
            else:
                fn_predict_for_sampling = partial(
                    self.predict_for_sampling, n_recycle=self.inf_cfg.get("n_recycle", 0)
                )
                gen_samples, extra_info = self.fm.full_simulation(
                    batch=batch,
                    predict_for_sampling=fn_predict_for_sampling,
                    nsteps=nsteps,
                    self_cond=self_cond,
                    sampling_model_args=self.inf_cfg.model,
                    device=self.device,
                    save_trajectory_every=save_trajectory_every,
                    guidance_w=guidance_w,
                    ag_ratio=ag_ratio,
                    motif_constraint_mode=self.motif_constraint_mode,
                )
                sample_prots = self.sample_formatting(
                    x=gen_samples,
                    extra_info=extra_info,
                    ret_mode="coors37_n_aatype" if ret_mode == "coors37_n_aatype" else "pdb_string",
                )
                stream_specs = [
                    {
                        "sample_prots": sample_prots,
                        "sample_type": "final",
                        "rankable": False,
                        "stream_subdir": None,
                        "step_start": None,
                        "step_end": None,
                    }
                ]

            final_specs = [
                spec for spec in stream_specs
                if str(spec.get("sample_type", "final")) == "final"
            ]
            if final_specs:
                sample_prots = final_specs[-1]["sample_prots"]

            if ret_mode == "coors37_n_aatype":
                generation_list = []
                if hasattr(batch, "id"):
                    nsrc = len(batch.id)
                elif "id" in batch:
                    nsrc = len(batch["id"])
                else:
                    nsrc = stream_specs[0]["sample_prots"]["coors"].shape[0]
                task_name_list = self._get_batch_metadata_list(batch, "task_name", nsrc, "validation")
                task_id_list = self._get_batch_metadata_list(batch, "id", nsrc, "validation_sample")
                contig_list = self._get_batch_metadata_list(batch, "contig", nsrc, "null")
                binder_chain_list = self._get_batch_metadata_list(batch, "binder_chain", nsrc, "L")
                condition_mode_id_list = self._get_batch_metadata_list(batch, "condition_mode_id", nsrc, [0])
                condition_mode_name_list = self._get_batch_metadata_list(batch, "condition_mode_name", nsrc, "none")
                condition_anchor_positions_list = self._get_batch_metadata_list(batch, "condition_anchor_positions", nsrc, [-1, -1, -1])
                condition_pair_positions_list = self._get_batch_metadata_list(batch, "condition_pair_positions", nsrc, [[-1, -1], [-1, -1], [-1, -1]])
                condition_pair_bin_idx_list = self._get_batch_metadata_list(batch, "condition_pair_bin_idx", nsrc, [0, 0, 0])
                condition_pair_target_angstrom_list = self._get_batch_metadata_list(batch, "condition_pair_target_angstrom", nsrc, [0.0, 0.0, 0.0])
                condition_pair_valid_mask_list = self._get_batch_metadata_list(batch, "condition_pair_valid_mask", nsrc, [False, False, False])
                condition_residue_targets_list = self._get_batch_metadata_list(batch, "condition_residue_targets", nsrc, [-1, -1, -1])
                condition_residue_target_mask_list = self._get_batch_metadata_list(batch, "condition_residue_target_mask", nsrc, [False, False, False])
                topology_entries_list = self._get_batch_metadata_list(batch, "topology_entries", nsrc, "[]")
                binder_topology_source_list = self._get_batch_metadata_list(batch, "binder_topology_source", nsrc, "")
                binder_topology_entry_id_list = self._get_batch_metadata_list(batch, "binder_topology_entry_id", nsrc, "")
                binder_topology_preset_list = self._get_batch_metadata_list(batch, "binder_topology_preset", nsrc, "")
                postprocess_enabled_list = self._get_batch_metadata_list(batch, "postprocess_enabled", nsrc, [False])
                postprocess_method_list = self._get_batch_metadata_list(batch, "postprocess_method", nsrc, "openmm")
                reference_pdb_path_list = self._get_batch_metadata_list(batch, "reference_pdb_path", nsrc, "")
                reference_pep_chain_list = self._get_batch_metadata_list(batch, "reference_pep_chain", nsrc, "L")
                reference_receptor_center_list = self._get_batch_metadata_list(batch, "reference_receptor_center", nsrc, None)
                for stream_spec in stream_specs:
                    stream_sample_prots = stream_spec["sample_prots"]
                    source_index = stream_sample_prots.get("source_index", None)
                    search_rewards = stream_sample_prots.get("search_reward", None)
                    metadata_tags = stream_sample_prots.get("metadata_tag", [None] * stream_sample_prots["coors"].shape[0])
                    conditioning_optional_types = stream_sample_prots.get("conditioning_optional_type", None)
                    conditioning_residue_counts = stream_sample_prots.get("conditioning_residue_count", None)
                    conditioning_pair_counts = stream_sample_prots.get("conditioning_pair_count", None)
                    conditioning_residue_hit_rates = stream_sample_prots.get("conditioning_residue_hit_rate", None)
                    conditioning_pair_bin_hit_rates = stream_sample_prots.get("conditioning_pair_bin_hit_rate", None)
                    conditioning_pair_bin_hit_rates_pm1 = stream_sample_prots.get("conditioning_pair_bin_hit_rate_pm1", None)
                    conditioning_pair_bin_maes = stream_sample_prots.get("conditioning_pair_bin_mae", None)
                    conditioning_joint_all_hits = stream_sample_prots.get("conditioning_joint_all_hit", None)

                    for i in range(stream_sample_prots["coors"].shape[0]):
                        src_idx = int(source_index[i]) if source_index is not None else (i % nsrc)
                        task_name = task_name_list[src_idx]
                        task_id = task_id_list[src_idx]
                        metadata_tag = metadata_tags[i] if metadata_tags is not None else None

                        has_hotspot = False
                        if hasattr(batch, "hotspot") and batch.hotspot is not None:
                            has_hotspot = bool(batch.hotspot[src_idx].squeeze().sum() > 0)

                        has_motif = False
                        if hasattr(batch, "latent_cond_mask") and batch.latent_cond_mask is not None:
                            has_motif = bool(batch.latent_cond_mask[src_idx].sum() > 0)

                        reward_value = None
                        if search_rewards is not None:
                            reward_value = float(search_rewards[i].detach().cpu().item())

                        conditioning_binder_secondary_class = self._extract_optional_class_label(
                            batch, "binder_secondary_class", src_idx
                        )
                        conditioning_binder_secondary_class_raw = self._extract_optional_class_label(
                            batch, "binder_secondary_class_raw", src_idx
                        )
                        condition_mode_id = condition_mode_id_list[src_idx]
                        if isinstance(condition_mode_id, list):
                            condition_mode_id = condition_mode_id[0] if condition_mode_id else 0
                        condition_mode_name = condition_mode_name_list[src_idx]
                        if isinstance(condition_mode_name, list):
                            condition_mode_name = condition_mode_name[0] if condition_mode_name else "none"
                        if not isinstance(condition_mode_name, str):
                            condition_mode_name = design_condition_mode_to_name(condition_mode_id)
                        postprocess_enabled = postprocess_enabled_list[src_idx]
                        if isinstance(postprocess_enabled, list):
                            postprocess_enabled = postprocess_enabled[0] if postprocess_enabled else False
                        postprocess_method = postprocess_method_list[src_idx]
                        if isinstance(postprocess_method, list):
                            postprocess_method = postprocess_method[0] if postprocess_method else "openmm"

                        generation_list.append(
                            {
                                "coors37": stream_sample_prots["coors"][i],
                                "residue_type": stream_sample_prots["residue_type"][i],
                                "chain_index": (
                                    stream_sample_prots["chain_index"][i]
                                    if "chain_index" in stream_sample_prots
                                    else None
                                ),
                                "is_peptide": stream_sample_prots["is_peptide"][i],
                                "is_receptor": stream_sample_prots["is_receptor"][i],
                                "task_name": task_name,
                                "sample_id": task_id,
                                "base_sample_id": task_id,
                                "source_index": src_idx,
                                "contig": contig_list[src_idx],
                                "metadata_tag": metadata_tag,
                                "search_algorithm": None if search_algorithm in {"null", "none"} else search_algorithm,
                                "sample_type": stream_spec.get("sample_type", "final"),
                                "rankable": bool(stream_spec.get("rankable", False)),
                                "stream_subdir": stream_spec.get("stream_subdir", None),
                                "step_start": stream_spec.get("step_start", None),
                                "step_end": stream_spec.get("step_end", None),
                                "nres": int(
                                    (
                                        stream_sample_prots["mask"][i].bool()
                                        & stream_sample_prots["is_peptide"][i].bool()
                                    ).sum().detach().cpu().item()
                                ),
                                "peptide_seq_recovery": 0.0,
                                "peptide_rmsd": 0.0,
                                "receptor_rmsd": 0.0,
                                "complex_ca_rmsd": 0.0,
                                "has_hotspot": has_hotspot,
                                "has_motif": has_motif,
                                "search_reward": reward_value,
                                "conditioning_binder_secondary_class": conditioning_binder_secondary_class,
                                "conditioning_binder_secondary_class_raw": conditioning_binder_secondary_class_raw,
                                "condition_mode_id": int(condition_mode_id or 0),
                                "condition_mode_name": str(condition_mode_name),
                                "binder_chain": str(binder_chain_list[src_idx]),
                                "condition_anchor_positions": condition_anchor_positions_list[src_idx],
                                "condition_pair_positions": condition_pair_positions_list[src_idx],
                                "condition_pair_bin_idx": condition_pair_bin_idx_list[src_idx],
                                "condition_pair_target_angstrom": condition_pair_target_angstrom_list[src_idx],
                                "condition_pair_valid_mask": condition_pair_valid_mask_list[src_idx],
                                "condition_residue_targets": condition_residue_targets_list[src_idx],
                                "condition_residue_target_mask": condition_residue_target_mask_list[src_idx],
                                "topology_entries": topology_entries_list[src_idx],
                                "binder_topology_source": binder_topology_source_list[src_idx],
                                "binder_topology_entry_id": binder_topology_entry_id_list[src_idx],
                                "binder_topology_preset": binder_topology_preset_list[src_idx],
                                "postprocess_enabled": bool(postprocess_enabled),
                                "postprocess_method": str(postprocess_method),
                                "conditioning_optional_type": (
                                    conditioning_optional_types[i]
                                    if conditioning_optional_types is not None
                                    else None
                                ),
                                "conditioning_residue_count": (
                                    int(conditioning_residue_counts[i])
                                    if conditioning_residue_counts is not None
                                    else 0
                                ),
                                "conditioning_pair_count": (
                                    int(conditioning_pair_counts[i])
                                    if conditioning_pair_counts is not None
                                    else 0
                                ),
                                "conditioning_residue_hit_rate": (
                                    conditioning_residue_hit_rates[i]
                                    if conditioning_residue_hit_rates is not None
                                    else None
                                ),
                                "conditioning_pair_bin_hit_rate": (
                                    conditioning_pair_bin_hit_rates[i]
                                    if conditioning_pair_bin_hit_rates is not None
                                    else None
                                ),
                                "conditioning_pair_bin_hit_rate_pm1": (
                                    conditioning_pair_bin_hit_rates_pm1[i]
                                    if conditioning_pair_bin_hit_rates_pm1 is not None
                                    else None
                                ),
                                "conditioning_pair_bin_mae": (
                                    conditioning_pair_bin_maes[i]
                                    if conditioning_pair_bin_maes is not None
                                    else None
                                ),
                                "conditioning_joint_all_hit": (
                                    conditioning_joint_all_hits[i]
                                    if conditioning_joint_all_hits is not None
                                    else None
                                ),
                                "residue_pdb_idx": (
                                    stream_sample_prots["residue_pdb_idx"][i]
                                    if "residue_pdb_idx" in stream_sample_prots
                                    else None
                                ),
                                "output_residue_pdb_idx": (
                                    stream_sample_prots["output_residue_pdb_idx"][i]
                                    if "output_residue_pdb_idx" in stream_sample_prots
                                    else None
                                ),
                                "reference_pdb_path": reference_pdb_path_list[src_idx],
                                "reference_pep_chain": reference_pep_chain_list[src_idx],
                                "reference_receptor_center": reference_receptor_center_list[src_idx],
                            }
                        )
                return generation_list

            if ret_mode == "pdb_string":
                if self._search_instance is not None:
                    pdb_entries = []
                    coors_np = sample_prots["coors"].detach().float().cpu().numpy()
                    residue_type_np = sample_prots["residue_type"].detach().cpu().numpy()
                    chain_index_np = sample_prots["chain_index"].detach().cpu().numpy()
                    mask_np = sample_prots["mask"].detach().cpu().numpy().astype(bool)
                    source_index = sample_prots.get("source_index", None)
                    residue_index_np = sample_prots.get("residue_pdb_idx", None)
                    for i in range(sample_prots["coors"].shape[0]):
                        src_idx = int(source_index[i]) if source_index is not None else (i % len(batch.hotspot))
                        residue_index_i = (
                            residue_index_np[i].detach().cpu().numpy()[mask_np[i]]
                            if residue_index_np is not None
                            else None
                        )
                        prot = create_full_prot(
                            atom37=coors_np[i][mask_np[i]],
                            atom37_mask=(np.abs(coors_np[i][mask_np[i]]) > 1e-7).any(axis=-1).astype(np.float32),
                            chain_index=chain_index_np[i][mask_np[i]],
                            aatype=residue_type_np[i][mask_np[i]],
                            residue_index=residue_index_i,
                        )
                        pdb_entries.append(
                            {
                                "pdb_string": to_pdb(prot=prot),
                                "nres": int((mask_np[i] & (chain_index_np[i] == 1)).sum()),
                                "peptide_seq_recovery": 0.0,
                                "peptide_rmsd": 0.0,
                                "receptor_rmsd": 0.0,
                                "complex_ca_rmsd": 0.0,
                                "has_hotspot": bool(batch.hotspot[src_idx].squeeze().sum() > 0),
                                "has_motif": bool(batch.latent_cond_mask[src_idx].sum() > 0),
                                "source_index": src_idx,
                                "metadata_tag": (
                                    sample_prots["metadata_tag"][i]
                                    if "metadata_tag" in sample_prots
                                    else None
                                ),
                                "conditioning_optional_type": None,
                                "conditioning_residue_count": 0,
                                "conditioning_pair_count": 0,
                                "conditioning_residue_hit_rate": None,
                                "conditioning_pair_bin_hit_rate": None,
                                "conditioning_pair_bin_hit_rate_pm1": None,
                                "conditioning_pair_bin_mae": None,
                                "conditioning_joint_all_hit": None,
                            }
                        )
                    sample_prots = pdb_entries
                generation_list = []
                if hasattr(batch, "id"):
                    nsrc = len(batch.id)
                elif "id" in batch:
                    nsrc = len(batch["id"])
                else:
                    nsrc = len(sample_prots)
                task_name_list = self._get_batch_metadata_list(batch, "task_name", nsrc, "validation")
                task_id_list = self._get_batch_metadata_list(
                    batch,
                    "id",
                    nsrc,
                    "validation_sample",
                )
                condition_mode_id_list = self._get_batch_metadata_list(batch, "condition_mode_id", nsrc, [0])
                condition_mode_name_list = self._get_batch_metadata_list(batch, "condition_mode_name", nsrc, "none")
                condition_anchor_positions_list = self._get_batch_metadata_list(batch, "condition_anchor_positions", nsrc, [-1, -1, -1])
                condition_pair_positions_list = self._get_batch_metadata_list(batch, "condition_pair_positions", nsrc, [[-1, -1], [-1, -1], [-1, -1]])
                condition_pair_bin_idx_list = self._get_batch_metadata_list(batch, "condition_pair_bin_idx", nsrc, [0, 0, 0])
                condition_pair_target_angstrom_list = self._get_batch_metadata_list(batch, "condition_pair_target_angstrom", nsrc, [0.0, 0.0, 0.0])
                condition_pair_valid_mask_list = self._get_batch_metadata_list(batch, "condition_pair_valid_mask", nsrc, [False, False, False])
                condition_residue_targets_list = self._get_batch_metadata_list(batch, "condition_residue_targets", nsrc, [-1, -1, -1])
                condition_residue_target_mask_list = self._get_batch_metadata_list(batch, "condition_residue_target_mask", nsrc, [False, False, False])
                topology_entries_list = self._get_batch_metadata_list(batch, "topology_entries", nsrc, "[]")
                binder_topology_source_list = self._get_batch_metadata_list(batch, "binder_topology_source", nsrc, "")
                binder_topology_entry_id_list = self._get_batch_metadata_list(batch, "binder_topology_entry_id", nsrc, "")
                binder_topology_preset_list = self._get_batch_metadata_list(batch, "binder_topology_preset", nsrc, "")
                binder_chain_list = self._get_batch_metadata_list(batch, "binder_chain", nsrc, "L")
                postprocess_enabled_list = self._get_batch_metadata_list(batch, "postprocess_enabled", nsrc, [False])
                postprocess_method_list = self._get_batch_metadata_list(batch, "postprocess_method", nsrc, "openmm")
                reference_pdb_path_list = self._get_batch_metadata_list(batch, "reference_pdb_path", nsrc, "")
                reference_pep_chain_list = self._get_batch_metadata_list(batch, "reference_pep_chain", nsrc, "L")
                reference_receptor_center_list = self._get_batch_metadata_list(batch, "reference_receptor_center", nsrc, None)
                for i in range(len(sample_prots)):
                    src_idx = int(sample_prots[i].get("source_index", i % nsrc))
                    task_id = task_id_list[src_idx]
                    metadata_tag = sample_prots[i].get("metadata_tag", None)
                    if metadata_tag is not None:
                        task_id = f"{task_id}__{metadata_tag}"
                    conditioning_binder_secondary_class = self._extract_optional_class_label(
                        batch, "binder_secondary_class", src_idx
                    )
                    conditioning_binder_secondary_class_raw = self._extract_optional_class_label(
                        batch, "binder_secondary_class_raw", src_idx
                    )
                    condition_mode_id = condition_mode_id_list[src_idx]
                    if isinstance(condition_mode_id, list):
                        condition_mode_id = condition_mode_id[0] if condition_mode_id else 0
                    condition_mode_name = condition_mode_name_list[src_idx]
                    if isinstance(condition_mode_name, list):
                        condition_mode_name = condition_mode_name[0] if condition_mode_name else "none"
                    if not isinstance(condition_mode_name, str):
                        condition_mode_name = design_condition_mode_to_name(condition_mode_id)
                    postprocess_enabled = postprocess_enabled_list[src_idx]
                    if isinstance(postprocess_enabled, list):
                        postprocess_enabled = postprocess_enabled[0] if postprocess_enabled else False
                    postprocess_method = postprocess_method_list[src_idx]
                    if isinstance(postprocess_method, list):
                        postprocess_method = postprocess_method[0] if postprocess_method else "openmm"
                    generation_list.append(
                        {
                            "pdb_string": sample_prots[i]["pdb_string"],
                            "nres": sample_prots[i]["nres"],
                            "task_name": task_name_list[src_idx],
                            "task_id": task_id,
                            "peptide_seq_recovery": sample_prots[i]["peptide_seq_recovery"],
                            "peptide_rmsd": sample_prots[i]["peptide_rmsd"],
                            "receptor_rmsd": sample_prots[i].get("receptor_rmsd", 0.0),
                            "complex_ca_rmsd": sample_prots[i].get("complex_ca_rmsd", 0.0),
                            "has_hotspot": sample_prots[i]["has_hotspot"],
                            "has_motif": sample_prots[i]["has_motif"],
                            "conditioning_binder_secondary_class": conditioning_binder_secondary_class,
                            "conditioning_binder_secondary_class_raw": conditioning_binder_secondary_class_raw,
                            "condition_mode_id": int(condition_mode_id or 0),
                            "condition_mode_name": str(condition_mode_name),
                            "binder_chain": str(binder_chain_list[src_idx]),
                            "condition_anchor_positions": condition_anchor_positions_list[src_idx],
                            "condition_pair_positions": condition_pair_positions_list[src_idx],
                            "condition_pair_bin_idx": condition_pair_bin_idx_list[src_idx],
                            "condition_pair_target_angstrom": condition_pair_target_angstrom_list[src_idx],
                            "condition_pair_valid_mask": condition_pair_valid_mask_list[src_idx],
                            "condition_residue_targets": condition_residue_targets_list[src_idx],
                            "condition_residue_target_mask": condition_residue_target_mask_list[src_idx],
                            "topology_entries": topology_entries_list[src_idx],
                            "binder_topology_source": binder_topology_source_list[src_idx],
                            "binder_topology_entry_id": binder_topology_entry_id_list[src_idx],
                            "binder_topology_preset": binder_topology_preset_list[src_idx],
                            "postprocess_enabled": bool(postprocess_enabled),
                            "postprocess_method": str(postprocess_method),
                            "conditioning_optional_type": sample_prots[i].get("conditioning_optional_type", None),
                            "conditioning_residue_count": sample_prots[i].get("conditioning_residue_count", 0),
                            "conditioning_pair_count": sample_prots[i].get("conditioning_pair_count", 0),
                            "conditioning_residue_hit_rate": sample_prots[i].get("conditioning_residue_hit_rate", None),
                            "conditioning_pair_bin_hit_rate": sample_prots[i].get("conditioning_pair_bin_hit_rate", None),
                            "conditioning_pair_bin_hit_rate_pm1": sample_prots[i].get("conditioning_pair_bin_hit_rate_pm1", None),
                            "conditioning_pair_bin_mae": sample_prots[i].get("conditioning_pair_bin_mae", None),
                            "conditioning_joint_all_hit": sample_prots[i].get("conditioning_joint_all_hit", None),
                            "binder_receptor_ca_center_distance_angstrom": sample_prots[i].get("binder_receptor_ca_center_distance_angstrom", None),
                            "binder_ca_center_norm_angstrom": sample_prots[i].get("binder_ca_center_norm_angstrom", None),
                            "receptor_ca_center_norm_angstrom": sample_prots[i].get("receptor_ca_center_norm_angstrom", None),
                            "chemical_bond_expected_count": sample_prots[i].get("chemical_bond_expected_count", 0),
                            "chemical_bond_recovered_count": sample_prots[i].get("chemical_bond_recovered_count", 0),
                            "chemical_bond_recovery_rate": sample_prots[i].get("chemical_bond_recovery_rate", None),
                            "chemical_bond_disulfide_expected_count": sample_prots[i].get("chemical_bond_disulfide_expected_count", 0),
                            "chemical_bond_disulfide_recovered_count": sample_prots[i].get("chemical_bond_disulfide_recovered_count", 0),
                            "chemical_bond_disulfide_sg_sg_distances_angstrom": sample_prots[i].get("chemical_bond_disulfide_sg_sg_distances_angstrom", None),
                            "chemical_bond_head_tail_expected_count": sample_prots[i].get("chemical_bond_head_tail_expected_count", 0),
                            "chemical_bond_head_tail_recovered_count": sample_prots[i].get("chemical_bond_head_tail_recovered_count", 0),
                            "chemical_bond_head_tail_c_n_distances_angstrom": sample_prots[i].get("chemical_bond_head_tail_c_n_distances_angstrom", None),
                            "chemical_bond_isopeptide_expected_count": sample_prots[i].get("chemical_bond_isopeptide_expected_count", 0),
                            "chemical_bond_isopeptide_recovered_count": sample_prots[i].get("chemical_bond_isopeptide_recovered_count", 0),
                            "chemical_bond_isopeptide_distances_angstrom": sample_prots[i].get("chemical_bond_isopeptide_distances_angstrom", None),
                            "chemical_bond_other_expected_count": sample_prots[i].get("chemical_bond_other_expected_count", 0),
                            "residue_pdb_idx": sample_prots[i].get("residue_pdb_idx", None),
                            "output_residue_pdb_idx": sample_prots[i].get("output_residue_pdb_idx", None),
                            "reference_pdb_path": reference_pdb_path_list[src_idx],
                            "reference_pep_chain": reference_pep_chain_list[src_idx],
                            "reference_receptor_center": reference_receptor_center_list[src_idx],
                        }
                    )
                return generation_list

            raise NotImplementedError(f"ret_mode {ret_mode} not implemented in predict_step")
        except Exception as e:
            raise e
        finally:
            if noise_scale > 0:
                self.perturb_nn_weights(seed=noise_seed, scale=noise_scale, reverse=True)

    def sample_formatting(
        self,
        x: Dict[str, Tensor],
        extra_info: Dict[str, Tensor],
        ret_mode: str,
    ):
        """
        Given a batch of b samples x produced by the flow matcher, it returns the samples in the requested format (ret_mode).

        Supports `ret_modes` for:
            - `samples` returns the original sample from the flow matcher, a dictionary[str, Tensor].
            for the data modalities, each with batch shape b.
            - `atom37` returns an Tensor of shape [b, n, 37, 3] just for coordinates.
            - `pdb_string` returns a list of dictionaries {"pdb_string": str, "nres": int}, with one dictionary per sample.
            - `coors37_n_aatype` returns a dictionary with keys `coors` (atom37), `residue_type`, and `mask`, and
            values with shapes [b, n, 37, 3] float, [b, n] int, [b, n] boolean, respectively.

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """
        data_modes = sorted([dm for dm in self.cfg_exp.product_flowmatcher])
        if data_modes == ["bb_ca"]:
            return self._format_sample_bb_ca(
                x=x, ret_mode=ret_mode, extra_info=extra_info
            )
        elif data_modes == ["bb_ca", "local_latents"]:
            return self._format_sample_local_latents(
                x=x, ret_mode=ret_mode, extra_info=extra_info
            )
        else:
            raise NotImplementedError(f"Format {ret_mode} not implemented")

    def _format_sample_bb_ca(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        extra_info: Dict
    ):
        mask = extra_info["mask"]

        if ret_mode == "samples":
            return x

        if ret_mode == "atom37":
            return trans_nm_to_atom37(x["bb_ca"].float())

        elif ret_mode == "coors37_n_aatype":
            coors = (
                trans_nm_to_atom37(x["bb_ca"].float()) * mask[..., None, None]
            )  # [b, n, 37, 3]
            residue_type = torch.zeros_like(coors)[..., 0, 0] * mask  # [b, n]
            chain_index = extra_info.get("is_peptide", mask.long()).long()
            return {
                "coors": coors,  # [b, n, 37, 3]
                "residue_type": residue_type.long(),  # [b, n]
                "mask": mask,  # [b, n]
                "chain_index": chain_index,
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            coors = (
                trans_nm_to_atom37(x["bb_ca"]).float().detach().cpu().numpy()
            )  # [b, n, 37, 3]
            residue_type = np.zeros_like(coors[:, :, 0, 0])  # [b, n]
            atom37_mask = np.zeros_like(coors[:, :, :, 0])  # [b, n, 37]
            atom37_mask[:, :, 1] = 1.0  # [b, n, 37]
            atom37_mask = atom37_mask * mask[..., None]  # [b, n, 37]
            n = coors.shape[-3]

            for i in range(coors.shape[0]):
                prot = create_full_prot(
                    atom37=coors[i, ...],
                    atom37_mask=atom37_mask[i, ...],
                    aatype=residue_type[i, ...],
                )
                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": n,
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca]` not implemented"
            )

    def _build_decode_omit_aa_mask(
        self,
        extra_features: Dict,
        mask: torch.Tensor,
        is_peptide: torch.Tensor,
        batch_size: int,
        n_res: int,
        device: torch.device,
    ):
        n_aa = len(restypes)

        base_mask = mask.bool() & is_peptide.bool()

        use_residue_type_feature = getattr(extra_features, "use_residue_type_feature", None)
        if use_residue_type_feature is not None:
            seq_fixed = self._normalize_optional_feature_flag(
                use_residue_type_feature,
                batch_size=batch_size,
                device=device,
                default=False,
            )
            base_mask = base_mask & (~seq_fixed[:, None])

        if base_mask.shape != (batch_size, n_res):
            raise ValueError(
                f"decode omit base mask must have shape {(batch_size, n_res)}, "
                f"got {tuple(base_mask.shape)}"
            )

        residue_condition_mask = getattr(extra_features, "residue_type_condition_mask", None)
        residue_condition = getattr(extra_features, "residue_type_condition", None)
        rc_mask_bool = (
            residue_condition_mask.to(device=device).bool()
            if residue_condition_mask is not None
            else None
        )

        peptide_motif_mask = getattr(extra_features, "peptide_motif_mask", None)
        peptide_motif_bool = (
            peptide_motif_mask.to(device=device).bool()
            if peptide_motif_mask is not None
            else None
        )

        # Keep the legacy latent mask exclusion for non-decoder consumers.  Decoder
        # identity clamps use the peptide-only motif mask below because the latent
        # mask may also contain receptor positions.
        latent_cond_mask = getattr(extra_features, "latent_cond_mask", None)
        latent_cond_bool = (
            latent_cond_mask.to(device=device).bool()
            if latent_cond_mask is not None
            else None
        )

        omit_pieces = []

        receptor_mask = mask.bool() & (~is_peptide.bool())
        receptor_residue_type = getattr(extra_features, "residue_type", None)
        if receptor_residue_type is not None:
            receptor_idx = receptor_residue_type.to(device=device).long()
            receptor_valid = receptor_mask & (receptor_idx >= 0) & (receptor_idx < n_aa)
            if receptor_valid.any():
                receptor_allowed = torch.nn.functional.one_hot(
                    receptor_idx.clamp(min=0, max=n_aa - 1), num_classes=n_aa
                ).bool()
                omit_pieces.append((~receptor_allowed) & receptor_valid[..., None])

        # Hard sequence targets are carried by the explicit condition mask,
        # independently of the broad ``use_residue_type_feature`` flag. The
        # inference dataset marks every peptide position for known-sequence
        # docking and only sparse motif/restype positions for other task types.
        hard_target_mask = torch.zeros_like(base_mask, dtype=torch.bool, device=device)
        hard_target_idx = torch.zeros_like(base_mask, dtype=torch.long, device=device)
        if peptide_motif_bool is not None and receptor_residue_type is not None:
            motif_idx = receptor_residue_type.to(device=device).long()
            valid_motif = (
                peptide_motif_bool
                & mask.bool()
                & is_peptide.bool()
                & (motif_idx >= 0)
                & (motif_idx < n_aa)
            )
            hard_target_mask = hard_target_mask | valid_motif
            hard_target_idx = torch.where(valid_motif, motif_idx, hard_target_idx)

        if rc_mask_bool is not None and residue_condition is not None:
            rc_idx = residue_condition.to(device=device).long()
            valid_condition = (
                rc_mask_bool
                & mask.bool()
                & is_peptide.bool()
                & (rc_idx >= 0)
                & (rc_idx < n_aa)
            )
            hard_target_mask = hard_target_mask | valid_condition
            hard_target_idx = torch.where(valid_condition, rc_idx, hard_target_idx)

        if hard_target_mask.any():
            hard_allowed = torch.nn.functional.one_hot(
                hard_target_idx.clamp(min=0, max=n_aa - 1), num_classes=n_aa
            ).bool()
            omit_pieces.append((~hard_allowed) & hard_target_mask[..., None])

        omit_aa_indices = getattr(extra_features, "omit_aa_indices", None)
        if omit_aa_indices is not None:
            omit_aa_indices = omit_aa_indices.to(device=device).long()
            if omit_aa_indices.ndim == 1:
                omit_aa_indices = omit_aa_indices.unsqueeze(0).expand(batch_size, -1)
            if omit_aa_indices.numel() > 0:
                valid_omit = (omit_aa_indices >= 0) & (omit_aa_indices < n_aa)
                if valid_omit.any():
                    omit_aa_by_sample = torch.zeros(
                        (batch_size, n_aa),
                        dtype=torch.bool,
                        device=device,
                    )
                    rows = (
                        torch.arange(batch_size, device=device)
                        .unsqueeze(-1)
                        .expand_as(omit_aa_indices)
                    )
                    omit_aa_by_sample[rows[valid_omit], omit_aa_indices[valid_omit]] = True
                    if omit_aa_by_sample.all(dim=-1).any():
                        bad = torch.nonzero(
                            omit_aa_by_sample.all(dim=-1), as_tuple=False
                        ).flatten().tolist()
                        raise ValueError(
                            f"omit_aa cannot forbid all 20 amino acids for sample indices {bad}."
                        )
                    global_pos_mask = base_mask
                    if latent_cond_bool is not None:
                        global_pos_mask = global_pos_mask & (~latent_cond_bool)
                    global_pos_mask = global_pos_mask & (~hard_target_mask)
                    omit_pieces.append(
                        global_pos_mask[..., None] & omit_aa_by_sample[:, None, :]
                    )

        if not omit_pieces:
            return None
        omit_aa_mask = omit_pieces[0]
        for piece in omit_pieces[1:]:
            omit_aa_mask = omit_aa_mask | piece
        return omit_aa_mask if omit_aa_mask.any() else None

    def _format_sample_local_latents(
        self,
        x: Dict[str, torch.Tensor],
        ret_mode: str,
        extra_info: Dict
    ):
        """
        Given a batch of b samples consisting on `bb_ca` and `local_latents` this
        returns formatted samples.

        Note: This calls the decoder from the autoencoder, since it needs to go from
        local latent variables to the actual coordinates and sequence.

        Note: The self.autoencoder.decode function (used here) returns a dictoinary like
        {
            "coors_nm": [b, n, 37, 3], already masked
            "residue_type": [b, n], already masked, careful with 0s
            "residue_mask": [b, n]
            "atom_mask": [b, n, 37]
        }

        Args:
            x: sample.
            extra_info: a dict with additional things, including:
                - "mask", whose value is boolean of shape [nsamples, n]
                - ...
            ret_mode: target format, for now only supports atom37.

        Returns:
            Sample x in the requested format.
        """

        mask = extra_info["mask"]
        is_peptide = extra_info["is_peptide"]
        is_receptor = extra_info["is_receptor"]
        # cond_mask = extra_info["cond_mask"]["local_latents"]
        extra_features = extra_info["graph_batch"]
        omit_aa_mask = self._build_decode_omit_aa_mask(
            extra_features=extra_features,
            mask=mask,
            is_peptide=is_peptide,
            batch_size=int(mask.shape[0]),
            n_res=int(mask.shape[1]),
            device=mask.device,
        )
        if omit_aa_mask is not None:
            extra_features.omit_aa_mask = omit_aa_mask
        elif hasattr(extra_features, "omit_aa_mask"):
            delattr(extra_features, "omit_aa_mask")

        output_decoder = self.autoencoder.decode(
            z_latent=x["local_latents"], ca_coors_nm=x["bb_ca"], mask=mask,
            is_peptide=is_peptide, extra_features=extra_features,
            force_fix_receptor=(
                self.output_receptor_source == "reference"
                and self.motif_constraint_mode in ["rigid", "rigid_receptor"]
            ),
        )
        receptor_mask = is_receptor.bool() & mask.bool()
        force_receptor_sequence = bool(receptor_mask.any())
        if receptor_mask.any():
            receptor_residue_type = extra_features.residue_type.to(
                device=output_decoder["residue_type"].device,
                dtype=output_decoder["residue_type"].dtype,
            )
            output_decoder["residue_type"] = torch.where(
                receptor_mask,
                receptor_residue_type,
                output_decoder["residue_type"],
            )
            restype_atom37_mask = torch.as_tensor(
                RESTYPE_ATOM37_MASK,
                device=output_decoder["atom_mask"].device,
                dtype=output_decoder["atom_mask"].dtype,
            )
            receptor_atom_mask = restype_atom37_mask[receptor_residue_type.long().clamp(min=0, max=20)]
            output_decoder["atom_mask"] = torch.where(
                receptor_mask[..., None],
                receptor_atom_mask,
                output_decoder["atom_mask"],
            )
            output_decoder["coors_nm"] = (
                output_decoder["coors_nm"]
                * output_decoder["residue_mask"][..., None, None]
                * output_decoder["atom_mask"][..., None]
            )

        output_device = output_decoder["residue_type"].device
        hard_target_mask = torch.zeros_like(mask, dtype=torch.bool, device=output_device)
        hard_target_type = torch.zeros_like(
            output_decoder["residue_type"], dtype=output_decoder["residue_type"].dtype
        )

        peptide_motif_mask = getattr(extra_features, "peptide_motif_mask", None)
        if peptide_motif_mask is not None:
            motif_target_type = extra_features.residue_type.to(
                device=output_device,
                dtype=output_decoder["residue_type"].dtype,
            )
            valid_motif = (
                peptide_motif_mask.to(device=output_device).bool()
                & mask.to(device=output_device).bool()
                & is_peptide.to(device=output_device).bool()
                & (motif_target_type >= 0)
                & (motif_target_type < len(restypes))
            )
            hard_target_mask = hard_target_mask | valid_motif
            hard_target_type = torch.where(valid_motif, motif_target_type, hard_target_type)

        residue_condition_mask = getattr(extra_features, "residue_type_condition_mask", None)
        residue_condition = getattr(extra_features, "residue_type_condition", None)
        if residue_condition_mask is not None and residue_condition is not None:
            condition_residue_type = residue_condition.to(
                device=output_device,
                dtype=output_decoder["residue_type"].dtype,
            )
            valid_condition = (
                residue_condition_mask.to(device=output_device).bool()
                & mask.to(device=output_device).bool()
                & is_peptide.to(device=output_device).bool()
                & (condition_residue_type >= 0)
                & (condition_residue_type < len(restypes))
            )
            hard_target_mask = hard_target_mask | valid_condition
            hard_target_type = torch.where(
                valid_condition, condition_residue_type, hard_target_type
            )

        if hard_target_mask.any():
            output_decoder["residue_type"] = torch.where(
                hard_target_mask,
                hard_target_type,
                output_decoder["residue_type"],
            )
            restype_atom37_mask = torch.as_tensor(
                RESTYPE_ATOM37_MASK,
                device=output_decoder["atom_mask"].device,
                dtype=output_decoder["atom_mask"].dtype,
            )
            hard_target_atom_mask = restype_atom37_mask[
                hard_target_type.long().clamp(min=0, max=20)
            ]
            output_decoder["atom_mask"] = torch.where(
                hard_target_mask[..., None],
                hard_target_atom_mask,
                output_decoder["atom_mask"],
            )
            output_decoder["coors_nm"] = (
                output_decoder["coors_nm"]
                * output_decoder["residue_mask"][..., None, None]
                * output_decoder["atom_mask"][..., None]
            )
        

        # Log statistics
        if self.motif_constraint_mode not in ["rigid", "rigid_receptor"]:
            
            # Calculate Receptor RMSD
            receptor_atom_mask = is_receptor[..., None] & extra_features.coord_mask
            diff = (extra_features.coords_nm - output_decoder["coors_nm"]) * receptor_atom_mask[..., None] 
            sq_diff = (diff ** 2).sum(dim=-1)  # [b, n, 37]
            n_atoms_per_sample = receptor_atom_mask.sum(dim=(1, 2)).clamp(min=1)  # [b]
            rmsd_per_sample = torch.sqrt((sq_diff.sum(dim=(1, 2)) / n_atoms_per_sample))
            rmsd_per_sample_ang = rmsd_per_sample * 10.0  # Convert to Angstrom
            logger.info(f"Receptor RMSD after decoding: mean={rmsd_per_sample_ang.mean().item():.4f} Å")

            # Calculate Receptor sequence recovery rate
            receptor_mask = is_receptor & mask
            gt_residue_type = extra_features.residue_type
            pred_residue_type = output_decoder["residue_type"]
            receptor_seq_match = (gt_residue_type == pred_residue_type) & receptor_mask
            receptor_seq_recovery = receptor_seq_match.sum().float() / receptor_mask.sum().clamp(min=1)
            logger.info(f"Receptor SeqRecoveryRate after decoding: {receptor_seq_recovery.item():.4f}")

        use_residue_type_feature = getattr(extra_features, "use_residue_type_feature", None)
        seq_fixed = self._normalize_optional_feature_flag(
            use_residue_type_feature,
            batch_size=int(mask.shape[0]),
            device=mask.device,
            default=False,
        )
        allow_optimize_sc = bool(seq_fixed.any())
        simple_sidechain_rebuild = force_receptor_sequence and not allow_optimize_sc
        allow_optimize_sc = allow_optimize_sc or force_receptor_sequence
        output_decoder = optimize_sidechains(
            decoder_out=output_decoder,
            allow_optimize=allow_optimize_sc,
            simple=simple_sidechain_rebuild,
        )

        if ret_mode == "samples":
            return x

        elif ret_mode == "coors37_n_aatype":
            orig_chain_ids = extra_features.orig_chain_id
            max_nres = int(output_decoder["residue_mask"].shape[1])
            chain_index = torch.zeros(
                (mask.shape[0], max_nres),
                dtype=torch.long,
                device=mask.device,
            )
            for i in range(mask.shape[0]):
                chain_ids_i = torch.tensor(
                    [
                        PDB_CHAIN_IDS.index(chain_id) if chain_id in PDB_CHAIN_IDS else 0
                        for chain_id in orig_chain_ids[i]
                    ],
                    dtype=torch.long,
                    device=mask.device,
                )
                chain_index[i, : chain_ids_i.shape[0]] = chain_ids_i
            cyclic = getattr(extra_features, "cyclic", None)
            if cyclic is None:
                cyclic = torch.zeros(mask.shape[0], dtype=torch.bool, device=mask.device)
            elif isinstance(cyclic, torch.Tensor):
                cyclic = cyclic.to(device=mask.device).bool().reshape(mask.shape[0], -1).any(dim=-1)
            else:
                cyclic = torch.full((mask.shape[0],), bool(cyclic), dtype=torch.bool, device=mask.device)
            optional_metrics = [
                self._compute_optional_condition_metrics_for_sample(
                    pred_residue_type=output_decoder["residue_type"][i],
                    pred_coors_nm=output_decoder["coors_nm"][i],
                    pred_residue_mask=output_decoder["residue_mask"][i],
                    extra_features=extra_features,
                    sample_idx=i,
                )
                for i in range(mask.shape[0])
            ]
            return {
                "coors": nm_to_ang(output_decoder["coors_nm"]),  # [b, n, 37, 3]
                "residue_type": output_decoder["residue_type"],  # [b, n]
                "mask": output_decoder["residue_mask"],  # [b, n]
                "chain_index": chain_index,  # [b, n]
                "residue_pdb_idx": getattr(extra_features, "residue_pdb_idx", None),
                "output_residue_pdb_idx": getattr(
                    extra_features,
                    "output_residue_pdb_idx",
                    getattr(extra_features, "residue_pdb_idx", None),
                ),
                "is_peptide": is_peptide,  # [b, n]
                "is_receptor": is_receptor,  # [b, n]
                "cyclic": cyclic,  # [b]
                "conditioning_optional_type": [
                    metric.get("conditioning_optional_type", None) for metric in optional_metrics
                ],
                "conditioning_residue_count": [
                    metric.get("conditioning_residue_count", 0) for metric in optional_metrics
                ],
                "conditioning_pair_count": [
                    metric.get("conditioning_pair_count", 0) for metric in optional_metrics
                ],
                "conditioning_residue_hit_rate": [
                    metric.get("conditioning_residue_hit_rate", None) for metric in optional_metrics
                ],
                "conditioning_pair_bin_hit_rate": [
                    metric.get("conditioning_pair_bin_hit_rate", None) for metric in optional_metrics
                ],
                "conditioning_pair_bin_hit_rate_pm1": [
                    metric.get("conditioning_pair_bin_hit_rate_pm1", None) for metric in optional_metrics
                ],
                "conditioning_pair_bin_mae": [
                    metric.get("conditioning_pair_bin_mae", None) for metric in optional_metrics
                ],
                "conditioning_joint_all_hit": [
                    metric.get("conditioning_joint_all_hit", None) for metric in optional_metrics
                ],
                "binder_receptor_ca_center_distance_angstrom": [
                    metric.get("binder_receptor_ca_center_distance_angstrom", None) for metric in optional_metrics
                ],
                "binder_ca_center_norm_angstrom": [
                    metric.get("binder_ca_center_norm_angstrom", None) for metric in optional_metrics
                ],
                "receptor_ca_center_norm_angstrom": [
                    metric.get("receptor_ca_center_norm_angstrom", None) for metric in optional_metrics
                ],
                "chemical_bond_expected_count": [
                    metric.get("chemical_bond_expected_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_recovered_count": [
                    metric.get("chemical_bond_recovered_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_recovery_rate": [
                    metric.get("chemical_bond_recovery_rate", None) for metric in optional_metrics
                ],
                "chemical_bond_disulfide_expected_count": [
                    metric.get("chemical_bond_disulfide_expected_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_disulfide_recovered_count": [
                    metric.get("chemical_bond_disulfide_recovered_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_disulfide_sg_sg_distances_angstrom": [
                    metric.get("chemical_bond_disulfide_sg_sg_distances_angstrom", None) for metric in optional_metrics
                ],
                "chemical_bond_head_tail_expected_count": [
                    metric.get("chemical_bond_head_tail_expected_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_head_tail_recovered_count": [
                    metric.get("chemical_bond_head_tail_recovered_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_head_tail_c_n_distances_angstrom": [
                    metric.get("chemical_bond_head_tail_c_n_distances_angstrom", None) for metric in optional_metrics
                ],
                "chemical_bond_isopeptide_expected_count": [
                    metric.get("chemical_bond_isopeptide_expected_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_isopeptide_recovered_count": [
                    metric.get("chemical_bond_isopeptide_recovered_count", 0) for metric in optional_metrics
                ],
                "chemical_bond_isopeptide_distances_angstrom": [
                    metric.get("chemical_bond_isopeptide_distances_angstrom", None) for metric in optional_metrics
                ],
                "chemical_bond_other_expected_count": [
                    metric.get("chemical_bond_other_expected_count", 0) for metric in optional_metrics
                ],
            }

        elif ret_mode == "pdb_string":
            pdb_strings = []

            ref_coords_nm = extra_features.coords_nm
            pred_coords_nm = output_decoder["coors_nm"]
            if self.motif_constraint_mode == "flex":
                valid_ca_residue_mask = (
                    extra_features.coord_mask[:, :, atom_order["CA"]]
                    & output_decoder["atom_mask"][:, :, atom_order["CA"]]
                    & output_decoder["residue_mask"]
                )
                receptor_ca_residue_mask = is_receptor & valid_ca_residue_mask
                align_ca_residue_mask = torch.where(
                    receptor_ca_residue_mask.any(dim=1, keepdim=True),
                    receptor_ca_residue_mask,
                    valid_ca_residue_mask,
                )
                complex_ca_rmsd_ang, pred_aligned_ca_nm = self._aligned_ca_rmsd_ang(
                    pred_coords_nm=pred_coords_nm,
                    ref_coords_nm=ref_coords_nm,
                    atom_mask=extra_features.coord_mask & output_decoder["atom_mask"],
                    align_residue_mask=align_ca_residue_mask,
                    region_mask=valid_ca_residue_mask,
                )
                peptide_rmsd_ang, _ = self._aligned_ca_rmsd_ang(
                    pred_coords_nm=pred_coords_nm,
                    ref_coords_nm=ref_coords_nm,
                    atom_mask=extra_features.coord_mask & output_decoder["atom_mask"],
                    align_residue_mask=align_ca_residue_mask,
                    region_mask=is_peptide & output_decoder["residue_mask"],
                )
                receptor_rmsd_ang, _ = self._aligned_ca_rmsd_ang(
                    pred_coords_nm=pred_coords_nm,
                    ref_coords_nm=ref_coords_nm,
                    atom_mask=extra_features.coord_mask & output_decoder["atom_mask"],
                    align_residue_mask=align_ca_residue_mask,
                    region_mask=is_receptor & output_decoder["residue_mask"],
                )
                ca_idx = atom_order["CA"]
                dist_sq = torch.zeros_like(ref_coords_nm[..., 0])
                dist_sq[:, :, ca_idx] = ((ref_coords_nm[:, :, ca_idx, :] - pred_aligned_ca_nm) ** 2).sum(dim=-1)
            else:
                diff_sq = (ref_coords_nm - pred_coords_nm) ** 2
                dist_sq = diff_sq.sum(dim=-1)  # [b, n, 37]
                complex_ca_rmsd_ang = None
                peptide_rmsd_ang = None
                receptor_rmsd_ang = None
            coors_atom_37 = nm_to_ang(output_decoder["coors_nm"]).float().detach().cpu().numpy()
            residue_type = output_decoder["residue_type"].detach().cpu().numpy()
            atom_mask = output_decoder["atom_mask"].detach().cpu().numpy()
            residue_mask = output_decoder["residue_mask"].detach().cpu().numpy()
            gt_residue_type = extra_features.residue_type.detach().cpu().numpy()
            is_peptide_np = is_peptide.detach().cpu().numpy()
            coord_mask_np = extra_features.coord_mask.detach().cpu().numpy()
            dist_sq_np = dist_sq.detach().cpu().numpy()

            for i in range(atom_mask.shape[0]):
                residue_mask_i = residue_mask[i]
                is_peptide_single = is_peptide_np[i]
                pep_seq_mask = is_peptide_single & residue_mask_i
                if pep_seq_mask.sum() > 0:
                    correct_res = (gt_residue_type[i] == residue_type[i]) & pep_seq_mask
                    pep_seq_recovery = float(correct_res.sum()) / float(pep_seq_mask.sum())
                else:
                    pep_seq_recovery = 0.0

                pep_atom_mask = is_peptide_single[:, None] & coord_mask_np[i] & residue_mask_i[:, None]
                n_pep_atoms = pep_atom_mask.sum()

                rec_atom_mask = (~is_peptide_single)[:, None] & coord_mask_np[i] & residue_mask_i[:, None]
                n_rec_atoms = rec_atom_mask.sum()

                if self.motif_constraint_mode == "flex":
                    pep_rmsd = (
                        float(peptide_rmsd_ang[i].detach().cpu().item())
                        if n_pep_atoms > 0
                        else 0.0
                    )
                    rec_rmsd = (
                        float(receptor_rmsd_ang[i].detach().cpu().item())
                        if n_rec_atoms > 0
                        else 0.0
                    )
                    complex_ca_rmsd = float(complex_ca_rmsd_ang[i].detach().cpu().item())
                elif n_pep_atoms > 0:
                    pep_mse = np.sum(dist_sq_np[i] * pep_atom_mask) / n_pep_atoms
                    pep_rmsd = np.sqrt(pep_mse) * 10.0
                    rec_mse = np.sum(dist_sq_np[i] * rec_atom_mask) / n_rec_atoms if n_rec_atoms > 0 else 0.0
                    rec_rmsd = np.sqrt(rec_mse) * 10.0 if n_rec_atoms > 0 else 0.0
                    complex_ca_rmsd = 0.0
                else:
                    pep_rmsd = 0.0
                    rec_rmsd = 0.0
                    complex_ca_rmsd = 0.0

                orig_chain_ids = extra_features.orig_chain_id[i]
                chain_index = np.array([
                    PDB_CHAIN_IDS.index(chain_id) if chain_id in PDB_CHAIN_IDS else 0
                    for chain_id in orig_chain_ids
                ], dtype=np.int64)

                output_residue_pdb_idx = getattr(
                    extra_features,
                    "output_residue_pdb_idx",
                    getattr(extra_features, "residue_pdb_idx", None),
                )

                prot = create_full_prot(
                    atom37=coors_atom_37[i][residue_mask_i],
                    atom37_mask=atom_mask[i][residue_mask_i],
                    chain_index=chain_index,
                    aatype=residue_type[i][residue_mask_i],
                    residue_index=output_residue_pdb_idx[i].detach().cpu().numpy()[residue_mask_i],
                )

                pdb_string = to_pdb(prot=prot)
                pdb_strings.append(
                    {
                        "pdb_string": pdb_string,
                        "nres": int(is_peptide_single.sum()),
                        "peptide_seq_recovery": pep_seq_recovery,
                        "peptide_rmsd": float(pep_rmsd),
                        "receptor_rmsd": float(rec_rmsd),
                        "complex_ca_rmsd": float(complex_ca_rmsd),
                        "has_hotspot": bool(extra_features.hotspot[i].squeeze().sum() > 1),
                        "has_motif": bool(extra_features.latent_cond_mask[i][is_peptide_single].sum() > 0),
                        **self._compute_optional_condition_metrics_for_sample(
                            pred_residue_type=output_decoder["residue_type"][i],
                            pred_coors_nm=output_decoder["coors_nm"][i],
                            pred_residue_mask=output_decoder["residue_mask"][i],
                            extra_features=extra_features,
                            sample_idx=i,
                        ),
                    }
                )
            return pdb_strings

        else:
            raise NotImplementedError(
                f"{ret_mode} format for data modes `[bb_ca, latent_locals]` not implemented"
            )

    @staticmethod
    def _matches_weight_noise_patterns(
        name: str,
        patterns: Union[None, str, List[str], Tuple[str, ...]],
    ) -> bool:
        if patterns is None:
            return False
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            if pattern is None:
                continue
            pattern = str(pattern)
            if pattern == "":
                continue
            if re.search(pattern, name):
                return True
        return False

    def _should_perturb_nn_parameter(self, name: str) -> bool:
        args = getattr(self.inf_cfg, "args", {}) if self.inf_cfg is not None else {}
        mode = str(args.get("weight_noise_mode", "all")).lower()
        include = args.get("weight_noise_include", None)
        exclude = args.get("weight_noise_exclude", None)

        if self._matches_weight_noise_patterns(name, exclude):
            return False

        if include is not None:
            return self._matches_weight_noise_patterns(name, include)

        name_lower = name.lower()

        if mode == "all":
            return True
        if mode == "linear_only":
            return (
                "weight" in name_lower
                and "norm" not in name_lower
                and "ln" not in name_lower
            )
        if mode == "attention_only":
            return any(
                key in name_lower
                for key in ("to_qkv", "to_g", "to_out_node", "to_bias", "mhba", "mha")
            )
        if mode == "qkv_only":
            return "to_qkv" in name_lower
        if mode == "no_norm":
            return "norm" not in name_lower and "ln" not in name_lower

        raise ValueError(f"Unknown weight_noise_mode: {mode}")

    @torch.no_grad()
    def perturb_nn_weights(self, seed: int, scale: float, reverse: bool = False):
        """
        Add noise to Flow Matching network
        Args:
            seed: random seed
            scale: scale
            reverse: if true, reverse the weight
        """
        sign = -1.0 if reverse else 1.0        
        for name, p in self.nn.named_parameters():
            if not p.requires_grad:
                continue
            if not self._should_perturb_nn_parameter(name):
                continue
            gen = torch.Generator(device=p.device)
            gen.manual_seed(seed)
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sign * scale * noise)
            
        logger.info(f"{'Restored' if reverse else 'Perturbed'} nn weights with seed {seed}, scale {scale}")
