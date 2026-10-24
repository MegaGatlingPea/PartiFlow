from typing import Callable, Dict, Optional, Tuple, Union

import lightning as L
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from partiflow.flow.rdn_flow_matcher import RDNFlowMatcher
from partiflow.flow.rdn_flow_matcher import mean_w_mask
from loguru import logger

FLOW_MATCHER_FACTORY = {
    "bb_ca": RDNFlowMatcher,
    "local_latents": RDNFlowMatcher,
}  # maps data modality to correspodning class


# there is some bugs if using origin_noise_center=False
# i am not sure if given multiple center in a batch can
# severely affect the training procedure, for now we just 
# assert origin_noise_center=True
def compute_masks_unified(batch: Dict, motif_constraint_mode: str, origin_noise_center: bool = True) -> Dict:
    """
    Unified mask computation for all constraint modes.

    Args:
        batch: Input batch containing mask_dict, is_peptide, ca_cond_mask, latent_cond_mask, hotspot, x_1
        motif_constraint_mode: "flex", "rigid", or "rigid_receptor"

    Returns:
        Dict with keys:
        - mask: valid residue mask [b, n]
        - is_peptide: peptide positions [b, n]
        - is_receptor: receptor positions [b, n]
        - is_motif: motif positions [b, n]
        - is_fix: positions to fix (not denoise) [b, n]
        - is_denoise: positions to denoise [b, n]
        - cond_mask_per_mode: per-modality conditioning masks
        - noise_mask_per_mode: per-modality denoising masks
        - noise_center_per_mode: per-modality noise centers
    """
    assert origin_noise_center == True, 'origin_noise_center must be True'
    mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]
    is_peptide = (batch["is_peptide"] == 1) & mask
    is_receptor = (batch["is_peptide"] == 0) & mask

    if motif_constraint_mode == "flex":
        # Flex: noise center is always (0,0,0), no hotspot centering
        noise_center_per_mode = {
            "bb_ca": None,             # no center
            "local_latents": None, 
        }
        # Flex: motif exists but not fixed, all positions denoise
        latent_cond_mask = batch.get("latent_cond_mask", torch.zeros_like(mask, dtype=torch.bool))
        is_motif = latent_cond_mask & mask
        is_fix = torch.zeros_like(mask, dtype=torch.bool)   # true
        is_denoise = mask                                   # true
        cond_mask_per_mode = {
            "bb_ca": torch.zeros_like(mask, dtype=torch.bool),
            "local_latents": torch.zeros_like(mask, dtype=torch.bool),
        }
        noise_mask_per_mode = {
            "bb_ca": mask,
            "local_latents": mask,
        }

    elif motif_constraint_mode == "rigid":
        # Rigid-Old: fix all motif
        ca_cond_mask = batch.get("ca_cond_mask", is_receptor)
        latent_cond_mask = batch.get("latent_cond_mask", is_receptor)
        is_motif = latent_cond_mask & mask
        is_fix = latent_cond_mask & mask
        is_denoise = ~latent_cond_mask & mask

        # Compute noise center with priority: peptide_motif_center > hotspot_center > (0,0,0)
        peptide_motif_mask = is_fix & is_peptide  # [b, n]
        hotspot = batch.get("hotspot", None)  # [b, n] or None

        if "x_1" in batch and "bb_ca" in batch["x_1"]:
            coords_for_center = batch["x_1"]["bb_ca"]  # [b, n, 3]
        else:
            coords_for_center = batch["coords_nm"][:, :, 1, :]  # [b, n, 3]

        # Compute peptide motif center for all samples
        peptide_motif_center = mean_w_mask(coords_for_center, peptide_motif_mask, keepdim=True)  # [b, 1, 3]

        # Compute hotspot center for all samples (if hotspot exists)
        if hotspot is not None:
            if hotspot.ndim == 3:
                hotspot = hotspot.squeeze(-1)
            hotspot_mask = hotspot > 0.5
            hotspot_center = mean_w_mask(coords_for_center, hotspot_mask, keepdim=True)  # [b, 1, 3]
        else:
            hotspot_center = torch.zeros_like(peptide_motif_center)

        # Determine which center to use per sample: [b, 1, 1]
        has_peptide_motif = peptide_motif_mask.any(dim=-1, keepdim=True).unsqueeze(-1)  # [b, 1, 1]
        has_hotspot = (hotspot_mask.any(dim=-1, keepdim=True).unsqueeze(-1) if hotspot is not None
                       else torch.zeros_like(has_peptide_motif))

        # Priority: peptide_motif > hotspot > origin
        noise_center = torch.where(has_peptide_motif, peptide_motif_center,
                                   torch.where(has_hotspot, hotspot_center,
                                              torch.zeros_like(peptide_motif_center)))

        noise_center_per_mode = {
            "bb_ca": noise_center if not origin_noise_center else None,
            "local_latents": None,
        }

        cond_mask_per_mode = {
            "bb_ca": ca_cond_mask,
            "local_latents": latent_cond_mask,
        }
        noise_mask_per_mode = {
            "bb_ca": ~ca_cond_mask & mask,
            "local_latents": ~latent_cond_mask & mask,
        }

    elif motif_constraint_mode == "rigid_receptor":
        # rigid_receptor: only fix receptor
        latent_cond_mask = batch.get("latent_cond_mask", torch.zeros_like(mask, dtype=torch.bool))
        is_motif = latent_cond_mask & mask
        is_fix = is_receptor
        is_denoise = is_peptide

        # Compute noise center with priority: peptide_motif_center > hotspot_center > (0,0,0)
        peptide_motif_mask = is_motif & is_peptide  # [b, n]
        hotspot = batch.get("hotspot", None)  # [b, n] or None

        if "x_1" in batch and "bb_ca" in batch["x_1"]:
            coords_for_center = batch["x_1"]["bb_ca"]  # [b, n, 3]
        else:
            coords_for_center = batch["coords_nm"][:, :, 1, :]  # [b, n, 3]

        # Compute peptide motif center for all samples
        peptide_motif_center = mean_w_mask(coords_for_center, peptide_motif_mask, keepdim=True)  # [b, 1, 3]

        # Compute hotspot center for all samples (if hotspot exists)
        if hotspot is not None:
            if hotspot.ndim == 3:
                hotspot = hotspot.squeeze(-1)
            hotspot_mask = hotspot > 0.5
            hotspot_center = mean_w_mask(coords_for_center, hotspot_mask, keepdim=True)  # [b, 1, 3]
        else:
            hotspot_center = torch.zeros_like(peptide_motif_center)

        # Determine which center to use per sample: [b, 1, 1]
        has_peptide_motif = peptide_motif_mask.any(dim=-1, keepdim=True).unsqueeze(-1)  # [b, 1, 1]
        has_hotspot = (hotspot_mask.any(dim=-1, keepdim=True).unsqueeze(-1) if hotspot is not None
                       else torch.zeros_like(has_peptide_motif))

        # Priority: peptide_motif > hotspot > origin
        noise_center = torch.where(has_peptide_motif, peptide_motif_center,
                                   torch.where(has_hotspot, hotspot_center,
                                              torch.zeros_like(peptide_motif_center)))

        noise_center_per_mode = {
            "bb_ca": noise_center if not origin_noise_center else None,
            "local_latents": None,
        }

        cond_mask_per_mode = {
            "bb_ca": is_receptor,
            "local_latents": is_receptor,
        }
        noise_mask_per_mode = {
            "bb_ca": is_peptide,
            "local_latents": is_peptide,
        }

    else:
        raise ValueError(f"Unknown motif_constraint_mode: {motif_constraint_mode}")

    return {
        "mask": mask,
        "is_peptide": is_peptide,
        "is_receptor": is_receptor,
        "is_motif": is_motif,
        "is_fix": is_fix,
        "is_denoise": is_denoise,
        "cond_mask_per_mode": cond_mask_per_mode,
        "noise_mask_per_mode": noise_mask_per_mode,
        "noise_center_per_mode": noise_center_per_mode,
    }


class ProductSpaceFlowMatcher(L.LightningModule):
    """
    Base class for flow matcher. Most of the methods in this class are abstract methods
    that should be implemented by classes that inheret from this one.
    """

    def __init__(self, cfg_exp: Dict):
        super().__init__()
        self.cfg_exp = cfg_exp
        self.data_modes = [m for m in self.cfg_exp.product_flowmatcher]
        self.base_flow_matchers = self.get_base_flow_matchers()
        self.motif_constraint_mode = self.cfg_exp.get("motif_constraint_mode", "rigid")

    def get_base_flow_matchers(self):
        """Constructs all necessary flow matchers."""
        return {
            m: FLOW_MATCHER_FACTORY[m](**self.cfg_exp.product_flowmatcher[m])
            for m in self.data_modes
        }

    def _apply_mask(
        self, x: Dict[str, Tensor], mask: Optional[Bool[Tensor, "* n"]] = None
    ):
        """
        Masks x.

        Args:
            x: sample to mask.
            mask (optional): binary mask, shape [*, n].

        Returns:
            sample x masked.
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode]._apply_mask(
                x=x[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x

    def sample_noise(
        self,
        n: int,
        shape: Tuple = tuple(),
        device: Optional[torch.device] = None,
        mask: Optional[Bool[Tensor, "* n"]] = None,
        noise_center: Optional[Dict[str, Float[Tensor, "* 1 d"]]] = None,
    ) -> Dict[str, Tensor]:
        """
        Samples reference distribution (possibly centered).

        Args:
            n: number of residues in a single sample (i.e. protein length), int
            shape: tuple (if empty then single sample)
            device (optional): torch device used
            mask (optional): Binary mask of shape [*, n]
            noise_center (optional): Dict of center for noise sampling per mode, shape [*, 1, d]

        Returns:
            Samples from refenrece with shape [*shaqpe, n, ...]
        """
        x = {
            data_mode: self.base_flow_matchers[data_mode].sample_noise(
                n=n,
                shape=shape,
                device=device,
                mask=mask,
                noise_center=noise_center.get(data_mode) if noise_center else None,
            )
            for data_mode in self.data_modes
        }
        return x

    def interpolate(
        self,
        x_0: Dict[str, torch.Tensor],
        x_1: Dict[str, torch.Tensor],
        t: Dict[str, Float[Tensor, "*"]],
        mask: Optional[Bool[Tensor, "* n"]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolates between x_0 (base) and x_1 (data) using t.

        Args:
            x_0: Samples from reference with batch shape * (each value in the dict)
            x_1: Samples from target with batch shape *
            t: Interpolation times
            mask (optional): Binary mask, shape [*, n]

        Returns:
            x_t: Interpolated samples, same shape as x_0 and x_1
        """
        x_t = {
            data_mode: self.base_flow_matchers[data_mode].interpolate(
                x_0=x_0[data_mode],
                x_1=x_1[data_mode],
                t=t[data_mode],
                mask=mask,
            )
            for data_mode in self.data_modes
        }
        return x_t

    def process_batch(
        self, batch: Dict
    ) -> Tuple[Tensor, Tensor, Tuple, int, torch.dtype]:
        """
        Extracts clean sample, mask, batch size, protein length n, dtype and device
        from the batch coming from the dataloader. (Do we want dtype? Should default
        work?)

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype, device)

        WARNING: For dtype, it might become a dictionary as well, depending on
        type for sequence (discrete)
        """
        coors_tensor = batch["coords"]  # [b, n, 37, 3]
        device = coors_tensor.device
        dtype = coors_tensor.dtype
        batch_shape = coors_tensor.shape[:-3]
        n = coors_tensor.shape[-3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        x_1 = self._apply_mask(x=batch["x_1"], mask=mask)
        return (x_1, mask, batch_shape, n, dtype, device)

    def corrupt_batch(
        self,
        batch: Dict,
    ) -> Dict:
        """
        Runs forward process on x_1. Essentially samples reference and
        interpolates. If there's any coupling it should be included here.

        Augmentations in principle happen in the dataloader.

        Args:
            batch: data batch

        Returns:
            The same data batch with additional entries t, x_t, x_1, x_0, all of which are
            Dict[str, Tensor].
        """
        x_1, mask, batch_shape, n, dtype, device = self.process_batch(batch)

        # Compute all masks using unified function
        masks = compute_masks_unified(batch, self.motif_constraint_mode)
        cond_mask_per_mode = masks["cond_mask_per_mode"]
        noise_mask_per_mode = masks["noise_mask_per_mode"]
        noise_center_per_mode = masks["noise_center_per_mode"]

        t = self.sample_t(shape=batch_shape, device=device)

        # Sample noise per modality with different masks and centers
        x_0 = {}
        for mode_idx, mode in enumerate(self.data_modes):
            x_0[mode] = self.base_flow_matchers[mode].sample_noise(
                n=n,
                shape=batch_shape,
                device=device,
                mask=noise_mask_per_mode[mode],
                noise_center=noise_center_per_mode[mode],
            )

        # Apply conditioning: where cond_mask is True, use x_1 values
        for mode in x_0:
            cond_mask = cond_mask_per_mode[mode][..., None]
            x_0[mode] = torch.where(cond_mask, x_1[mode], x_0[mode]) 
            # here if has noise center, fix part of x_0 changes from 0 to noise center
            # however, here we fix it by setting fix part of x_0 to ground truth x_1

        x_t = self.interpolate(x_0=x_0, x_1=x_1, t=t, mask=mask)
        batch["x_0"] = x_0
        batch["x_1"] = x_1
        batch["x_t"] = x_t
        batch["t"] = t
        batch["mask"] = mask
        batch["cond_mask"] = cond_mask_per_mode
        batch["noise_mask"] = noise_mask_per_mode
        batch["is_fix"] = masks["is_fix"]
        batch["is_denoise"] = masks["is_denoise"]
        return batch

    def sample_t(
        self, shape: tuple, device: torch.device
    ) -> Dict[str, Float[Tensor, "*shape"]]:
        """
        Samples t for each data mode. Can use different distributions for each mode,
        and different modes can share the same t. This is all controlled in the config.

        Args:
            shape: shape of the sample
            device: device of the sample

        Returns:
            Sampled t for each modality (as a dictionary), each tensor has shape [*shape].
        """
        t = {
            data_mode: _sample_t(
                cfg_t_dist=self.cfg_exp.loss.t_distribution[data_mode],
                shape=shape,
                device=device,
            )
            for data_mode in self.data_modes
        }
        if self.cfg_exp.loss.t_distribution.shared_groups is None:
            return t

        # Apply grouping for data modalities we want the same t
        for t_share_modes in self.cfg_exp.loss.t_distribution.shared_groups:
            # each t_shared_modes is a list
            base = t_share_modes[0]  # some modality, the first one in the list
            shared_t = t[base]  # [*]
            for data_mode in t_share_modes[1:]:
                t[data_mode] = shared_t
        return t

    def compute_loss(
        self, batch: Dict, nn_out: Dict[str, Dict[str, Tensor]]
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes training loss, flow matching and some auxiliary loss.

        Args:
            batch: training batch with clean x_1, x_0, x_t, t
            nn_out: output of nn
        Returns:
            Flow matching loss and auxiliary loss, all in one Dict[str, Tensor w shape [*]]. Each loss
            is identified by its name (str) and is the loss per element in the batch (i.e. before mean
            reduction).
        """
        weights = {}
        for data_mode in self.data_modes:
            weights[data_mode] = self.cfg_exp.product_flowmatcher[data_mode].get("weight", 1.0)
        fm_loss = self.compute_fm_loss(batch, nn_out)
        sample_loss_weight = batch.get("sample_loss_weight", None)
        if sample_loss_weight is not None:
            ref_loss = next(iter(fm_loss.values()))
            sample_loss_weight = sample_loss_weight.to(device=ref_loss.device).float().reshape(-1)

        def apply_sample_weight(loss_value: Tensor) -> Tensor:
            if sample_loss_weight is None:
                return loss_value
            weight = sample_loss_weight
            while weight.ndim < loss_value.ndim:
                weight = weight.unsqueeze(-1)
            return loss_value * weight

        weighted_fm_loss = {
            k: apply_sample_weight(v * weights.get(k, 1.0))
            for k, v in fm_loss.items()
        }
        fm_loss_justlog = {k + "_justlog": v for k, v in fm_loss.items()}
        aux_loss = self.compute_aux_loss(batch, nn_out)
        weighted_aux_loss = {
            k: (v if k.endswith("_justlog") else apply_sample_weight(v))
            for k, v in aux_loss.items()
        }
        loss = {**weighted_fm_loss, **weighted_aux_loss, **fm_loss_justlog}
        return loss

    def compute_fm_loss(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes flow matching loss for all modalities.

        Args:
            batch: Input batch containing x_1, x_0, t, mask, noise_mask, etc
            nn_out: Output of nn

        Returns:
            Losses for each data modality, per element in the batch.
        """
        # Use noise_mask (positions that need denoising) for loss computation
        # noise_mask[mode] = ~cond_mask[mode] & mask
        noise_mask = batch["noise_mask"]
        loss = {
            data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                x_0=batch["x_0"][data_mode],
                x_1=batch["x_1"][data_mode],
                x_t=batch["x_t"][data_mode],
                mask=noise_mask[data_mode],
                t=batch["t"][data_mode],
                nn_out=nn_out[data_mode],
            )
            for data_mode in self.data_modes
        }
        return loss

    def compute_aux_loss(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, Tensor]],
    ) -> Dict[str, Float[Tensor, "*"]]:
        """
        Computes auxiliary loss (if any). This is done here, and not in each specific flow matcher,
        since it may be a full atom loss, etc, meaning it may interact between modalities.

        Args:
            batch: Input batch containing x_1, x_0, t, mask, etc
            nn_out: Output of nn

        Returns:
            Losses for each data modality, per element in the batch.
        """
        losses = {}

        # Optional motif loss - use is_motif mask
        if "x_motif" in batch:
            is_motif = batch.get("is_motif")
            if is_motif is not None:
                motif_loss = {
                    data_mode: self.base_flow_matchers[data_mode].compute_fm_loss(
                        x_0=batch["x_0"][data_mode],
                        x_1=batch["x_1"][data_mode],
                        x_t=batch["x_t"][data_mode],
                        mask=is_motif,
                        t=batch["t"][data_mode],
                        nn_out=nn_out[data_mode],
                    )
                    for data_mode in self.data_modes
                }
                for data_mode in motif_loss:
                    losses[data_mode + f"_motif_loss_now_justlog"] = motif_loss[data_mode]

        return losses

    def simulation_step(
        self,
        x_t: Dict[str, torch.Tensor],
        nn_out: Dict[str, Dict[str, torch.Tensor]],
        t: Dict[str, Float[Tensor, "*"]],
        dt: Dict[str, float],
        gt: Dict[str, float],
        simulation_step_params: Dict[str, Dict],
        mask: Optional[Bool[Tensor, "* n"]] = None,
        denoising_mask: Optional[Union[Bool[Tensor, "* n"], Dict[str, Bool[Tensor, "* n"]]]] = None,
        ref_center: Optional[Dict[str, Float[Tensor, "* 1 d"]]] = None,
        has_ref_center: Optional[Dict[str, Bool[Tensor, "* 1"]]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Single integration step of ODE \dot{x_t} = v(x_t, t) using Euler integration scheme.

        Args:
            x_t: Current values with batch shape *
            nn_out: Output of nn
            t: Current time, shape * for each value in the dict
            dt: Step-size, float
            gt: Noise injection schedule, float
            simulation_step_params contain extra simulation steps parameters
            mask (optional): Binary mask of shape [*, n]
            denoising_mask (optional): Binary mask of shape [*, n] or Dict[str, Tensor] indicating which residues to denoise.
                Positions where denoising_mask=True will be updated, others remain fixed.
                If a dict, each mode uses its own mask.

        Returns:
            Updated x_t after integration step, same shape as input
        """
        # Support both single mask and per-modality mask dict
        def get_denoising_mask(mode):
            if denoising_mask is None:
                return None
            elif isinstance(denoising_mask, dict):
                return denoising_mask.get(mode, None)
            else:
                return denoising_mask

        x_updated = {
            data_mode: self.base_flow_matchers[data_mode].simulation_step(
                x_t=x_t[data_mode],
                nn_out=nn_out[data_mode],
                t=t[data_mode],
                dt=dt[data_mode],
                gt=gt[data_mode],
                simulation_step_params=simulation_step_params[data_mode],
                mask=mask,
                denoising_mask=get_denoising_mask(data_mode),
                ref_center=ref_center[data_mode] if ref_center is not None else None,
                has_ref_center=has_ref_center[data_mode] if has_ref_center is not None else None,
            )
            for data_mode in self.data_modes
        }
        return x_updated

    def nn_out_to_clean_sample_prediction(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """
        Generates clean sample prediction from nn output.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            Clean sample prediction {data_mode: torch.Tensor} from nn output
        """
        nn_out = self.nn_out_add_clean_sample_prediction(batch=batch, nn_out=nn_out)
        return {data_mode: nn_out[data_mode]["x_1"] for data_mode in self.data_modes}

    def nn_out_add_clean_sample_prediction(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Adds clean sample prediction to nn output.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            nn_out with clean sample prediction ("x_1" key) added to each data modality.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[
                data_mode
            ].nn_out_add_clean_sample_prediction(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_simulation_tensor(
        self,
        batch: Dict,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Adds tensor used for simulation to nn output. This is the velocity "v" for flow matching,
        or "score" for diffusion models. One could work out their equivalence and leverage that,
        but keeping each one seems like an easier implementation.

        Args:
            batch: input to nn (needed because it contains t, x_t, and mask)
            nn_out: output of nn

        Returns:
            nn_out with simulation tensor prediction ("v", "score", ... keys) added to data modalities.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[
                data_mode
            ].nn_out_add_simulation_tensor(
                x_t=batch["x_t"][data_mode],
                t=batch["t"][data_mode],
                mask=batch["mask"],
                nn_out=nn_out[data_mode],
            )
        return nn_out

    def nn_out_add_guided_simulation_tensor(
        self,
        nn_out: Dict[str, Dict[str, torch.Tensor]],
        nn_out_ag: Union[Dict[str, Dict[str, torch.Tensor]], None],
        nn_out_ucond: Union[Dict[str, Dict[str, torch.Tensor]], None],
        guidance_w: float,
        ag_ratio: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Adds guided version of tensor used for simulation to nn output 'nn_out'. Supports classifier free
        guidance and autoguidance.

        Args:
            nn_out: nn output of full model
            nn_out_ag: nn output of model used for autoguidance
            nn_out_ucond: nn output of full but unconditional model
            guidance_w: Guidance weights, float
            ag_ratio: Autoguidance ratio, float

        Returns:
            nn_out with guided simulation tensor prediction ("v_guided", "score_guided", ... keys) added to data modalities
            with guidance enabled.
        """
        for data_mode in self.data_modes:
            nn_out[data_mode] = self.base_flow_matchers[
                data_mode
            ].nn_out_add_guided_simulation_tensor(
                nn_out=nn_out[data_mode],
                nn_out_ag=nn_out_ag[data_mode] if nn_out_ag else None,
                nn_out_ucond=nn_out_ucond[data_mode] if nn_out_ucond else None,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
            )
        return nn_out

    def get_clean_pred_n_guided_vector(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        guidance_w: float,
        ag_ratio: float,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        This makes a prediction with the corresponding nn and conditioning, computes the
        corresponding clean sample prediction, and v/score, and also the guided v/score.

        When calling `predict_for_sampling` (which internally calls some nn), the output
        nn_out is like
        {
            "bb_ca": {"v": tensor}
            "frames": {"x1": tensor}
        }
        depending on the parameterization, which is indicated in the nn config file. Then,
        we call `nn_out_add_clean_sample_prediction` and `nn_out_add_simulation_tensor` to
        extend each data mode in nn_out with "v"/"score" (the simulation tensor) and "x_1".

        This function does this for the original nn, the unconditional input, and the
        autoguidance nn, and combines predictions accordingly, adding to nn_out an entry
        `v_guided`. Overall, it returns nn_out like
        {
            "bb_ca": {"v": tensor, "x_1": tensor, "v_guided": tensor}
            "frames": {"x1": tensor, "v": tensor, "v_guided": tensor}
        }
        where v could be replaced by the score in case of a diffusion.

        Args:
            batch: input to nn
            predict_for_sampling: function that takes as input batch and mode, where the latter indicated
            what model to evaluate (supports "full", "ag", "ucond").
            guidance_w: guidance weights
            ag_ratio: autoguidance ratio

        Returns:
            an nn_out dictionary with keys "x_1", "v" (or "score") and "v_guided" (or "score_guided").
        """

        def _add_clean_n_sim_tensor(batch, mode):
            """
            We will use this for each model (full, ag, ucond).
            """
            nn_out_dict = predict_for_sampling(batch, mode=mode)
            nn_out_dict = self.nn_out_add_clean_sample_prediction(batch, nn_out_dict)
            nn_out_dict = self.nn_out_add_simulation_tensor(batch, nn_out_dict)
            return nn_out_dict

        nn_out = _add_clean_n_sim_tensor(batch, mode="full")
        nn_out_ag = None
        nn_out_ucond = None
        if guidance_w != 1.0:
            if ag_ratio > 0.0:  # Use auto-guidance
                nn_out_ag = _add_clean_n_sim_tensor(batch, mode="ag")
            if ag_ratio < 1.0:  # Use CFG
                nn_out_ucond = _add_clean_n_sim_tensor(batch, mode="ucond")

        nn_out = self.nn_out_add_guided_simulation_tensor(
            nn_out=nn_out,
            nn_out_ag=nn_out_ag,
            nn_out_ucond=nn_out_ucond,
            guidance_w=guidance_w,
            ag_ratio=ag_ratio,
        )  # Adds guided tensor used for simulation ("guided_v", "guided_score", or whatever the base flow matcher uses)
        return nn_out

    def full_simulation(
        self,
        batch: Dict,
        predict_for_sampling: Callable,
        nsteps: int,
        self_cond: bool,
        sampling_model_args: Dict[str, Dict],
        device: torch.device,
        save_trajectory_every: int = 0,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
        motif_constraint_mode: str = None,
    ) -> Dict[str, Tensor]:
        """
        Generates samples by simulating the

        dx_t = v(x_t, t) dt

        or

        dx_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

        from t=0 up to t=1.

        Which equation is simulated depends on data modality and parameters chosen.

        Args:
            batch (Dict): Input batch. May contain:
                - 'nsamples': int, number of samples to generate (optional, will use nsamples param if not present)
                - 'nres' or 'n': int, protein length (optional, will use n param if not present)
                - 'mask': torch.BoolTensor, shape [nsamples, n], mask for valid residues (optional, will be generated if not present)
                - 'ca_cond_mask': per-residue conditioning mask for bb_ca (optional, defaults to is_receptor)
                - 'latent_cond_mask': per-residue conditioning mask for local_latents (optional, defaults to is_receptor)
                - Any conditioning fields required by the prediction function (e.g., cath_codes, motif info, etc.)

                The following fields will be added/updated during simulation:
                - 'x_t': torch.Tensor, current state (updated each step)
                - 't': dict of time values for each data mode (updated each step)
                - 'mask': torch.BoolTensor, shape [nsamples, n], mask for valid residues (added if not present)
                - 'x_sc': torch.Tensor, self-conditioning input (added if self_cond=True)
            predict_for_sampling (Callable): Function that calls the nn, receives the full batch dict
            nsteps (int): Number of simulation steps
            nsamples (int): Number of samples to generate
            n (int): Protein length
            self_cond (bool): Whether to use self-conditioning
            sampling_model_args (Dict): Model-specific sampling arguments
            device (torch.device): Device to run simulation on
            save_trajectory_every (int): How often to save full generation trajectory, default 0 (no saving)
            guidance_w (float): Guidance weight
            ag_ratio (float): Autoguidance ratio

        Returns:
            x (dict): Generated samples, with keys for each data mode and values with batch shape nsamples
            mask (torch.Tensor): Boolean mask of shape [nsamples, n]
        """
        # overwrite n with the correct shape of coors_nm in batch
        # for key, value in batch.items():
        #     if (
        #         isinstance(value, torch.Tensor)
        #         and value.dim() > 0
        #         and value.size(0) == 1
        #     ):
        #         batch[key] = value.squeeze(0)

        # # Use mask from batch if present, otherwise generate it
        # if "mask" in batch and batch["mask"] is not None:
        #     mask = batch["mask"]
        # else:
        #     mask = torch.ones(nsamples, n).long().bool().to(device)
        # assert mask.shape == (nsamples, n)
        sampling_state = self.prepare_sampling_state(batch, motif_constraint_mode=motif_constraint_mode)
        device = sampling_state["device"]
        batch_shape = sampling_state["batch_shape"]
        nsamples = sampling_state["nsamples"]
        n = sampling_state["n"]
        mask = sampling_state["mask"]
        is_peptide = sampling_state["is_peptide"]
        is_receptor = sampling_state["is_receptor"]
        cond_mask_per_mode = sampling_state["cond_mask_per_mode"]
        noise_mask_per_mode = sampling_state["noise_mask_per_mode"]
        ref_center_per_mode = sampling_state["ref_center_per_mode"]
        has_ref_center_per_mode = sampling_state["has_ref_center_per_mode"]

        if save_trajectory_every > 0:
            # Create a list of trajectory dictionaries, one for each sample in batch
            [{} for _ in range(nsamples)]

        ts, gt = self.sample_schedule(
            nsteps=nsteps,
            sampling_model_args=sampling_model_args,
        )

        with torch.no_grad():
            x = self.initialize_sampling_state(batch, sampling_state)

            for step in range(nsteps):
                t = {
                    data_mode: ts[data_mode][step] * torch.ones(nsamples, device=device)
                    for data_mode in self.data_modes
                }
                dt = {
                    data_mode: ts[data_mode][step + 1] - ts[data_mode][step]
                    for data_mode in self.data_modes
                }
                gt_step = {
                    data_mode: gt[data_mode][step] for data_mode in self.data_modes
                }

                # Update the batch with current x, t, mask, etc.
                batch["x_t"] = x
                batch["t"] = t
                batch["mask"] = mask

                # self conditioning
                if step > 0 and self_cond:
                    batch["x_sc"] = x_1_pred

                # get clean prediction and guided vector
                nn_out = self.get_clean_pred_n_guided_vector(
                    batch=batch,
                    predict_for_sampling=predict_for_sampling,
                    guidance_w=guidance_w,
                    ag_ratio=ag_ratio,
                )
                # Dict[data_mode,
                #   Dict[str, torch.Tensor]
                # ] where str is some prediction ("x_1", "v", "score", "guided_v", "guided_score", ...)
                # We just track all predictions this way

                x_1_pred = self.nn_out_to_clean_sample_prediction(
                    batch=batch, nn_out=nn_out
                )

                assert not ((x['bb_ca'] - x_1_pred['bb_ca']) * cond_mask_per_mode['bb_ca'][..., None]).any(), "Neural network violated conditioning constraint in bb_ca prediction"
                assert not ((x['local_latents'] - x_1_pred['local_latents']) * cond_mask_per_mode['local_latents'][..., None]).any(), "Neural network violated conditioning constraint in local_latents prediction"
                # Dict[data_mode, torch.Tensor]

                simulation_step_params = {
                    data_mode: sampling_model_args[data_mode]["simulation_step_params"]
                    for data_mode in self.data_modes
                }

                # Perform simulation step with per-modality denoising_mask
                # # Perform simulation step with denoising_mask=is_peptide
                # # This ensures SDE noise is only applied to peptide, keeping receptor fixed
                x = self.simulation_step(
                    x_t=x,
                    nn_out=nn_out,
                    t=t,
                    dt=dt,
                    gt=gt_step,
                    mask=mask,
                    simulation_step_params=simulation_step_params,
                    denoising_mask=noise_mask_per_mode,  # KEY: Per-modality denoising mask
                    ref_center=ref_center_per_mode,  # Per-modality centering
                    has_ref_center=has_ref_center_per_mode,  # Per-sample centering mask
                )

                # Flex mode: align complex to receptor center (origin) every step
                if sampling_state["motif_constraint_mode"] == "flex":
                    receptor_center = mean_w_mask(x["bb_ca"], is_receptor, keepdim=True)  # [b, 1, 3]
                    x["bb_ca"] = x["bb_ca"] - receptor_center

                if sampling_state["motif_constraint_mode"] == "rigid":
                    assert not ((x['bb_ca'] - batch['coords_nm'][:,:,1,:]) * cond_mask_per_mode['bb_ca'][..., None]).any(), "Conditioned bb_ca was modified during simulation_step"
                    assert not ((x['local_latents'] - batch['gt_cond_latents']) * cond_mask_per_mode['local_latents'][..., None]).any(), "Conditioned local_latents was modified during simulation_step"
                elif sampling_state["motif_constraint_mode"] == "rigid_receptor":
                    assert not ((x['bb_ca'] - batch['coords_nm'][:,:,1,:]) * is_receptor[..., None]).any(), "Receptor bb_ca was modified during simulation_step"
                    assert not ((x['local_latents'] - batch['gt_cond_latents']) * is_receptor[..., None]).any(), "Receptor local_latents was modified during simulation_step"

            additional_info = {
                "mask": mask,
                "is_peptide": is_peptide,
                "is_receptor": is_receptor,
                "cond_mask": cond_mask_per_mode,
                "noise_mask": noise_mask_per_mode,
                "graph_batch": batch
            }

            return x, additional_info

    def prepare_sampling_state(
        self,
        batch: Dict,
        motif_constraint_mode: str = None,
    ) -> Dict:
        if motif_constraint_mode is None:
            motif_constraint_mode = self.motif_constraint_mode

        coors_tensor = batch["coords"]
        device = coors_tensor.device
        batch_shape = coors_tensor.shape[:-3]
        nsamples = batch_shape[0]
        n = coors_tensor.shape[-3]

        masks = compute_masks_unified(batch, motif_constraint_mode)
        mask = masks["mask"]
        is_peptide = masks["is_peptide"]
        is_receptor = masks["is_receptor"]
        cond_mask_per_mode = masks["cond_mask_per_mode"]
        noise_mask_per_mode = masks["noise_mask_per_mode"]
        noise_center_per_mode = masks["noise_center_per_mode"]

        ref_center = batch.get("ref_center", None)
        has_ref_center = batch.get("has_ref_center", None)
        if motif_constraint_mode == "flex" and has_ref_center is not None and has_ref_center.any():
            logger.warning("ref_center is not null in flex mode. Ignoring ref_center.")

        ref_center_per_mode = {
            "bb_ca": None if motif_constraint_mode == "flex" else ref_center,
            "local_latents": None,
        }
        has_ref_center_per_mode = {
            "bb_ca": None if motif_constraint_mode == "flex" else has_ref_center,
            "local_latents": None,
        }

        batch["cond_mask"] = cond_mask_per_mode
        batch["noise_mask"] = noise_mask_per_mode

        return {
            "motif_constraint_mode": motif_constraint_mode,
            "device": device,
            "batch_shape": batch_shape,
            "nsamples": nsamples,
            "n": n,
            "mask": mask,
            "is_peptide": is_peptide,
            "is_receptor": is_receptor,
            "cond_mask_per_mode": cond_mask_per_mode,
            "noise_mask_per_mode": noise_mask_per_mode,
            "noise_center_per_mode": noise_center_per_mode,
            "ref_center_per_mode": ref_center_per_mode,
            "has_ref_center_per_mode": has_ref_center_per_mode,
        }

    def initialize_sampling_state(
        self,
        batch: Dict,
        sampling_state: Dict,
    ) -> Dict[str, Tensor]:
        x = {}
        for mode in self.data_modes:
            x[mode] = self.base_flow_matchers[mode].sample_noise(
                n=sampling_state["n"],
                shape=sampling_state["batch_shape"],
                device=sampling_state["device"],
                mask=sampling_state["noise_mask_per_mode"][mode],
                noise_center=sampling_state["noise_center_per_mode"][mode],
            )

        cond_mask_per_mode = sampling_state["cond_mask_per_mode"]
        x["bb_ca"] = torch.where(cond_mask_per_mode["bb_ca"][..., None], batch["coords_nm"][:, :, 1, :], x["bb_ca"])
        x["local_latents"] = torch.where(
            cond_mask_per_mode["local_latents"][..., None],
            batch["gt_cond_latents"],
            x["local_latents"],
        )

        if sampling_state["motif_constraint_mode"] in ["rigid", "rigid_receptor"]:
            assert not (
                (x["bb_ca"] - batch["coords_nm"][:, :, 1, :]) * cond_mask_per_mode["bb_ca"][..., None]
            ).any(), "Conditioned bb_ca initialization failed"
            assert not (
                (x["local_latents"] - batch["gt_cond_latents"]) * cond_mask_per_mode["local_latents"][..., None]
            ).any(), "Conditioned local_latents initialization failed"
        return x

    def sample_schedule(
        self,
        nsteps: int,
        sampling_model_args: dict[str, dict],
    ) -> tuple[dict[str, Float[Tensor, "nsteps + 1"]], dict[str, Float[Tensor, "nsteps"]]]:
        ts = {
            data_mode: get_schedule(
                mode=sampling_model_args[data_mode]["schedule"]["mode"],
                nsteps=int(nsteps),
                p1=sampling_model_args[data_mode]["schedule"]["p"],
            )
            for data_mode in self.data_modes
        }
        gt = {
            data_mode: get_gt(
                t=ts[data_mode][:-1],
                mode=sampling_model_args[data_mode]["gt"]["mode"],
                param=sampling_model_args[data_mode]["gt"]["p"],
                clamp_val=sampling_model_args[data_mode]["gt"]["clamp_val"],
            )
            for data_mode in self.data_modes
        }
        return ts, gt

    @torch.no_grad()
    def partial_simulation(
        self,
        batch: dict,
        x: dict[str, Tensor],
        x_1_pred: dict[str, Tensor] | None,
        mask: Bool[Tensor, "nsamples n"],
        predict_for_sampling: Callable,
        start_step: int,
        end_step: int,
        self_cond: bool,
        ts: dict[str, Float[Tensor, "nsteps + 1"]],
        gt: dict[str, Float[Tensor, "nsteps"]],
        simulation_step_params: dict[str, dict],
        device: torch.device,
        guidance_w: float = 1.0,
        ag_ratio: float = 0.0,
        motif_constraint_mode: str = None,
    ) -> tuple[dict[str, Tensor], dict[str, Tensor] | None]:
        sampling_state = self.prepare_sampling_state(batch, motif_constraint_mode=motif_constraint_mode)
        cond_mask_per_mode = sampling_state["cond_mask_per_mode"]
        is_receptor = sampling_state["is_receptor"]
        ref_center_per_mode = sampling_state["ref_center_per_mode"]
        has_ref_center_per_mode = sampling_state["has_ref_center_per_mode"]
        nsamples = mask.shape[0]

        for step in range(start_step, end_step):
            t = {data_mode: ts[data_mode][step] * torch.ones(nsamples, device=device) for data_mode in self.data_modes}
            dt = {data_mode: ts[data_mode][step + 1] - ts[data_mode][step] for data_mode in self.data_modes}
            gt_step = {data_mode: gt[data_mode][step] for data_mode in self.data_modes}

            batch["x_t"] = x
            batch["t"] = t
            batch["mask"] = mask
            if step > 0 and self_cond and x_1_pred is not None:
                batch["x_sc"] = x_1_pred

            nn_out = self.get_clean_pred_n_guided_vector(
                batch=batch,
                predict_for_sampling=predict_for_sampling,
                guidance_w=guidance_w,
                ag_ratio=ag_ratio,
            )
            x_1_pred = self.nn_out_to_clean_sample_prediction(batch=batch, nn_out=nn_out)

            assert not (
                (x["bb_ca"] - x_1_pred["bb_ca"]) * cond_mask_per_mode["bb_ca"][..., None]
            ).any(), "Neural network violated conditioning constraint in bb_ca prediction"
            assert not (
                (x["local_latents"] - x_1_pred["local_latents"]) * cond_mask_per_mode["local_latents"][..., None]
            ).any(), "Neural network violated conditioning constraint in local_latents prediction"

            x = self.simulation_step(
                x_t=x,
                nn_out=nn_out,
                t=t,
                dt=dt,
                gt=gt_step,
                mask=mask,
                simulation_step_params=simulation_step_params,
                denoising_mask=sampling_state["noise_mask_per_mode"],
                ref_center=ref_center_per_mode,
                has_ref_center=has_ref_center_per_mode,
            )

            if sampling_state["motif_constraint_mode"] == "flex":
                receptor_center = mean_w_mask(x["bb_ca"], is_receptor, keepdim=True)
                x["bb_ca"] = x["bb_ca"] - receptor_center
            elif sampling_state["motif_constraint_mode"] == "rigid":
                assert not (
                    (x["bb_ca"] - batch["coords_nm"][:, :, 1, :]) * cond_mask_per_mode["bb_ca"][..., None]
                ).any(), "Conditioned bb_ca was modified during simulation_step"
                assert not (
                    (x["local_latents"] - batch["gt_cond_latents"]) * cond_mask_per_mode["local_latents"][..., None]
                ).any(), "Conditioned local_latents was modified during simulation_step"
            elif sampling_state["motif_constraint_mode"] == "rigid_receptor":
                assert not (
                    (x["bb_ca"] - batch["coords_nm"][:, :, 1, :]) * is_receptor[..., None]
                ).any(), "Receptor bb_ca was modified during simulation_step"
                assert not (
                    (x["local_latents"] - batch["gt_cond_latents"]) * is_receptor[..., None]
                ).any(), "Receptor local_latents was modified during simulation_step"

        return x, x_1_pred


def get_gt(
    t: Float[Tensor, "nsteps"],
    mode: str,
    param: float,
    clamp_val: Optional[float] = None,
    eps: float = 1e-2,
) -> Float[Tensor, "nsteps"]:
    """
    Computes gt (noise injection schedule) for different modes. This is in the SDE as

    dx_t = [v(x_t, t) + g(t) s(x_t, t)] dt + \sqrt{2g(t)} dw_t

    Args:
        t: times where we'll evaluate, covers [0, 1), shape [nsteps]
        mode: "1-t/t", "tan", "1/t", ...
        param: parameterized transformation
        clamp_val: value to clamp gt, no clamping if None
        eps: small value leave as it is

    Returns:
        Noise injection schedule, shape [nsteps]
    """

    def transform_gt(
        gt: Float[Tensor, "nsteps"], f_pow: float = 1.0
    ) -> Float[Tensor, "nsteps"]:
        """Applies a transformation to the gt. f_pow=1.0 means no transformation."""
        if f_pow == 1.0:
            return gt

        # First we somewhat normalize between 0 and 1
        log_gt = torch.log(gt)
        mean_log_gt = torch.mean(log_gt)
        log_gt_centered = log_gt - mean_log_gt
        normalized = torch.nn.functional.sigmoid(log_gt_centered)
        # Transformation here
        normalized = normalized**f_pow
        # Undo normalization with the transformed variable
        log_gt_centered_rec = torch.logit(normalized, eps=1e-6)
        log_gt_rec = log_gt_centered_rec + mean_log_gt
        gt_rec = torch.exp(log_gt_rec)
        return gt_rec

    t = torch.clamp(t, 0, 1 - 1e-5)  # For numerical reasons

    if mode == "1-t/t":
        num = 1.0 - t
        den = t
        gt = num / (den + eps)
    elif mode == "tan":
        num = torch.sin((1.0 - t) * torch.pi / 2.0)
        den = torch.cos((1.0 - t) * torch.pi / 2.0)
        gt = (torch.pi / 2.0) * num / (den + eps)
    elif mode == "1/t":
        num = 1.0
        den = t
        gt = num / (den + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    gt = transform_gt(gt, f_pow=param)
    gt = torch.clamp(gt, 0, clamp_val)  # If None no clamping
    return gt  # [s]


def get_schedule(
    mode: str, nsteps: int, *, p1: float = None, eps: float = 1e-5
) -> Float[Tensor, "nsteps_p_one"]:
    """
    Gets the partition of the unit interval with points where we'll evaluate the vector field / score.
    """
    if mode == "uniform":
        t = torch.linspace(0, 1, nsteps + 1)
        return t
    elif mode == "power":
        assert p1 is not None, "p1 cannot be none for the power schedule"
        t = torch.linspace(0, 1, nsteps + 1)
        t = t**p1
        return t
    elif mode == "log":
        assert p1 is not None, "p1 cannot be none for the log schedule"
        assert p1 > 0, f"p1 must be >0 for the log schedule, got {p1}"
        t = 1.0 - torch.logspace(-p1, 0, nsteps + 1).flip(0)
        t = t - torch.min(t)
        t = t / torch.max(t)
        return t
    else:
        raise IOError(f"Schedule mode not recognized {mode}")


def _sample_t(
    cfg_t_dist: Dict, shape: Tuple, device=torch.device
) -> Float[Tensor, "*shape"]:
    """
    Samples time from different time distributions.

    Args:
        cfg_t_dist: dict specifying distribution to sample
        shape: shape of the returned tensor t
        device: torch.device

    Returns:
        Tensor with random samples t from the requested distribution and shape
    """
    if cfg_t_dist.name == "uniform":
        t_max = cfg_t_dist.p2
        return torch.rand(shape, device=device) * t_max
    elif cfg_t_dist.name == "logit-normal":
        mean = cfg_t_dist.p1
        std = cfg_t_dist.p2
        noise = torch.randn(shape, device=device) * std + mean
        return torch.nn.functional.sigmoid(noise)
    elif cfg_t_dist.name == "beta":
        p1 = cfg_t_dist.p1
        p2 = cfg_t_dist.p2
        dist = torch.distributions.beta.Beta(p1, p2)
        return dist.sample(shape).to(device)
    elif cfg_t_dist.name == "mix_unif_beta":
        p1 = cfg_t_dist.p1
        p2 = cfg_t_dist.p2
        p3 = cfg_t_dist.p3  # For mix weight of uniform
        assert 0.0 < p3 < 1.0, f"p3 value {p3} not in (0, 1)"
        dist = torch.distributions.beta.Beta(p1, p2)
        samples_beta = dist.sample(shape).to(device)
        samples_uniform = torch.rand(shape, device=device)
        u = torch.rand(shape, device=device)
        return torch.where(u < p3, samples_uniform, samples_beta)
    else:
        raise NotImplementedError(
            f"Sampling mode for t {cfg_t_dist.name} not implemented"
        )
