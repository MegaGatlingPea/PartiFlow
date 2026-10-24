import math
from typing import Dict, List, Literal

import einops
import torch
from jaxtyping import Float
from loguru import logger
from torch.nn import functional as F
from partiflow.openfold.data import data_transforms
from partiflow.openfold.np.residue_constants import atom_types
from partiflow.utils.angle_utils import bond_angles, signed_dihedral_angle
from partiflow.utils.secstruct_utils import (
    BINDER_SECSTRUCT_FOLD_NUM_CLASSES,
    binder_secstruct_fold_code_from_id,
)
from torch.nn.utils.rnn import pad_sequence


################################
# # Some auxiliary functions # #
################################


# From frameflow code
def get_index_embedding(indices, edim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of type integer, shape either [n] or [b, n].
        edim: dimension of the embeddings to create.
        max_len: maximum length.

    Returns:
        positional embedding of shape either [n, edim] or [b, n, edim]
    """
    # indices [n] of [b, n]
    K = torch.arange(edim // 2, device=indices.device)  # [edim / 2]

    if len(indices.shape) == 1:  # [n]
        K = K[None, ...]
    elif len(indices.shape) == 2:  # [b, n]
        K = K[None, None, ...]

    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K / edim))
    ).to(indices.device)
    # [n, 1] / [1, edim/2] -> [n, edim/2] or [b, n, 1] / [1, 1, edim/2] -> [b, n, edim/2]
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K / edim))
    ).to(indices.device)
    pos_embedding = torch.cat(
        [pos_embedding_sin, pos_embedding_cos], axis=-1
    )  # [n, edim]
    return pos_embedding


def get_time_embedding(
    t: Float[torch.Tensor, "b"], edim: int, max_positions: int = 2000
) -> Float[torch.Tensor, "b embdim"]:
    """
    Code from Frameflow, which got it from
    https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py

    Creates embedding for a given vector of times t.

    Args:
        t: vector of times (float) of shape [b].
        edim: dimension of the embeddings.
        max_positions: ...

    Returns:
        Embedding for the vector t of shape [b, edim]
    """
    assert len(t.shape) == 1
    t = t * max_positions
    half_dim = edim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if edim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (t.shape[0], edim)
    return emb


def _expand_optional_feature_flag(flag_value, batch_size: int, device: torch.device) -> torch.Tensor:
    if isinstance(flag_value, torch.Tensor):
        if flag_value.numel() == 0:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return flag_value.to(device=device).bool().reshape(batch_size, -1).any(dim=-1)
    return torch.full((batch_size,), bool(flag_value), dtype=torch.bool, device=device)


def _warn_once_compile_safe(module: torch.nn.Module, message: str):
    return


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    tensor = tensor.contiguous()
    bin_limits = bin_limits.contiguous()
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: Dict):
        pass  # Implemented by each class

    def extract_bs_and_n(self, batch: Dict):
        """
        Extracts batch size and n from input batch.
        Once we add more modalities just try modaliites until you hit one.
        Other option is to pass arguments to nn which modalities are used.
        """
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise IOError("Don't know how to extract batch size and n from batch...")
        bs, n = v.shape[0], v.shape[1]
        return bs, n

    def extract_device(self, batch: Dict):
        """Extracts device from input batch."""
        if "x_t" in batch:
            if "bb_ca" in batch["x_t"]:
                v = batch["x_t"]["bb_ca"]  # [b, n, 3]
        elif "coords" in batch:
            v = batch["coords"]  # [b, n, 37, 3]
        elif "z_latent" in batch:
            v = batch["z_latent"]  # [b, n, latent_dim]
        else:
            raise IOError("Don't know how to extract device from batch...")
        return v.device

    def assert_defaults_allowed(self, batch: Dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise IOError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(
        self,
        dim_feats_out=192,
        mode: Literal["seq", "pair"] = "seq",
        name=None,
        **kwargs,
    ):
        super().__init__(dim=dim_feats_out)
        self.mode = mode

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=device)
        elif self.mode == "pair":
            return torch.zeros((b, n, n, self.dim), device=device)
        else:
            raise IOError(f"Mode {self.mode} wrong for zero feature")


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, data_mode_use, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)
        self.data_mode_use = data_mode_use

    def forward(self, batch):
        t = batch["t"][self.data_mode_use]  # [b]
        _, n = self.extract_bs_and_n(batch)
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            chain_breaks = torch.zeros((b, n), device=device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscBBCASeqFeat(Feature):
    """Computes feature from backbone CA self conditining coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, mode_key="x_sc", **kwargs):
        super().__init__(dim=3)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "bb_ca" in data_modes_avail
            ), f"`bb_ca` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            return batch[self.mode_key]["bb_ca"]  # [b, n, 3]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                f"No {self.mode_key} in batch, returning zeros for XscBBCASeqFeat",
            )
            return torch.zeros(b, n, 3, device=device)


class XscLocalLatentsSeqFeat(Feature):
    """Computes feature from local latents self conditining, seq feature of shape [b, n, dim]."""

    def __init__(self, latent_dim, mode_key="x_sc", **kwargs):
        super().__init__(dim=latent_dim)
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "local_latents" in data_modes_avail
            ), f"`local_latents` sc/recycle seq feature requested but key not available in data modes {data_modes_avail}"
            return batch[self.mode_key]["local_latents"]  # [b, n, latent_dim]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                f"No {self.mode_key} in batch, returning zeros for XscLocalLatentsSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class XtBBCASeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["bb_ca"]  # [b, n, 3]


class XtLocalLatentsSeqFeat(Feature):
    """Computes feature from backbone CA x_t coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, latent_dim, **kwargs):
        super().__init__(dim=latent_dim)

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "local_latents" in data_modes_avail
        ), f"`local_latents` seq feat feature requested but key not available in data modes {data_modes_avail}"
        return batch["x_t"]["local_latents"]  # [b, n, latent_dim]


class CaCoorsNanometersSeqFeat(Feature):
    """Computes feature from ca coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), "`ca_coors_nm` nor `coords_nm` in batch, cannot compute CaCoorsNanometersSeqFeat"
        if "ca_coors_nm" in batch:
            return batch["ca_coors_nm"]  # [b, n, 3]
        else:
            return batch["coords_nm"][:, :, 1, :]  # [b, n, 3]


class TryCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `ca_coors_nm` in batch, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `ca_coors_nm` not in batch return zero feature.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        if "ca_coors_nm" in batch or "coords_nm" in batch:
            return super().forward(batch)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No ca_coors_nm or coords_nm in batch, returning zeros for TryCaCoorsNanometersSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class OptionalCaCoorsNanometersSeqFeat(CaCoorsNanometersSeqFeat):
    """
    If `use_ca_coors_nm_feature` in batch and true, returns sequence feature with CA coordinates (in nm) of shape [b, n, 3].

    If `use_ca_coors_nm_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        flag = _expand_optional_feature_flag(
            batch.get("use_ca_coors_nm_feature", False),
            batch_size=b,
            device=device,
        )
        if "ca_coors_nm" in batch or "coords_nm" in batch:
            feat = super().forward(batch)
        else:
            feat = torch.zeros(b, n, self.dim, device=device)
        return feat * flag[:, None, None].to(dtype=feat.dtype)


class ResidueTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 20].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=20)

    def forward(self, batch):
        assert (
            "residue_type" in batch
        ), "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        rtype = rtype * rpadmask  # [b, n], the -1 padding becomes 0
        rtype = torch.where(rtype > 19, torch.zeros_like(rtype), rtype) # tmp fix
        rtype_onehot = F.one_hot(rtype, num_classes=20)  # [b, n, 20]? there is a bug here maybe?
        rtype_onehot = (
            rtype_onehot * rpadmask[..., None]
        )
        return rtype_onehot * 1.0


class OptionalResidueTypeSeqFeat(ResidueTypeSeqFeat):
    """
    If `use_residue_type_feature` in batch and true, adds residue type feature of shape [b, n, 20].

    If `use_residue_type_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def _partial_condition_feat(self, batch, full_flag: torch.Tensor, device: torch.device):
        condition = batch.get("residue_type_condition", None)
        condition_mask = batch.get("residue_type_condition_mask", None)
        if condition is None or condition_mask is None:
            return None, torch.zeros_like(full_flag)

        condition = condition.to(device=device).long()
        condition_mask = condition_mask.to(device=device).bool()
        if condition_mask.ndim == 1:
            condition_mask = condition_mask.unsqueeze(0)
            condition = condition.unsqueeze(0)

        sparse_flag = condition_mask.any(dim=-1) & (~full_flag)
        if "mask_dict" in batch and "residue_type" in batch["mask_dict"]:
            padmask = batch["mask_dict"]["residue_type"].to(device=device).bool()
        else:
            padmask = batch["mask"].to(device=device).bool()

        condition = torch.where(condition < 0, torch.zeros_like(condition), condition)
        condition = torch.where(condition > 19, torch.zeros_like(condition), condition)
        sparse_mask = condition_mask & sparse_flag[:, None] & padmask
        sparse_feat = F.one_hot(condition, num_classes=20) * sparse_mask[..., None]
        return sparse_feat * 1.0, sparse_flag

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        full_flag = _expand_optional_feature_flag(
            batch.get("use_residue_type_feature", False),
            batch_size=b,
            device=device,
        )
        feat = torch.zeros(b, n, 20, device=device)
        if "residue_type" in batch:
            full_feat = super().forward(batch)
            feat = torch.where(full_flag[:, None, None], full_feat, feat)

        sparse_feat, sparse_flag = self._partial_condition_feat(batch, full_flag=full_flag, device=device)
        if sparse_feat is not None:
            feat = feat + sparse_feat.to(dtype=feat.dtype)

        return feat


class Atom37NanometersCoorsSeqFeat(Feature):
    """
    Computes feature from the atom37 representation (in Å), feature of shape [b, n, 37 * 4].

    Atom37 has shape [b, n, 37, 3], and the appropriate mask (for the residue type) has shape
    [b, n, 37]. This feature concatenates the flattened mask (shape [b, n, 37]) with the flattened coordinates (of shape
    [b, n, 37 * 3])

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, rel=False, **kwargs):
        super().__init__(dim=int(37 * 4))
        # 37 * 4, 37 * 3 for the coordinates, + 37 for the mask
        self.rel = rel  # whether to get features relative to CA or absolute

    def forward(self, batch):
        assert (
            "coords_nm" in batch
        ), "`coords_nm` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        assert (
            "coord_mask" in batch
        ), "`coord_mask` not in batch, cannot compute Atom37NanometersCoorsSeqFeat"
        coors = batch["coords_nm"]  # [b, n, 37, 3]
        coors_mask = batch[
            "coord_mask"
        ]  # [b, n, 37]
        coors = (
            coors * coors_mask[..., None]
        )  # Zero-out non-atoms

        if self.rel:
            # If relative remove CA coordinates
            ca_coors = coors[:, :, 1, :]  # [b, n, 3]
            coors = coors - ca_coors[:, :, None, :]  # [b, n, 37, 3]
            coors = coors * coors_mask[..., None]

        coors_flat = coors.flatten(start_dim=-2)  # [b, n, 37, 3] -> [b, n, 37 * 3]
        feat = torch.cat([coors_flat, coors_mask], dim=-1)  # [b, n, 37 * 4]
        return feat


class BackboneTorsionAnglesSeqFeat(Feature):
    """
    Computes torsion angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(3 * 21))

    def forward(self, batch):
        bb_torsion = self._get_bb_torsion_angles(batch)  # [b, n, 3]
        bb_torsion_feats = bin_and_one_hot(
            bb_torsion,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_torsion.device),
        )  # [b, n, 3, nbins], nbins in 20+1
        bb_torsion_feats = bb_torsion_feats.flatten(start_dim=-2)  # [b, n, 3 * nbins]
        return bb_torsion_feats

    def _get_bb_torsion_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]
        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]

        psi = signed_dihedral_angle(
            N[:, :-1, :], CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :]
        )  # [b, n-1]
        omega = signed_dihedral_angle(
            CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :]
        )  # [b, n-1]
        phi = signed_dihedral_angle(
            C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :], C[:, 1:, :]
        )  # [b, n-1]
        bb_angles = torch.stack([psi, omega, phi], dim=-1)  # [b, n-1, 3]

        # Account for chain breaks in torsion angles
        # Check both PDB index continuity AND chain break flags
        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]

        # Additionally check for chain breaks if available
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"]  # [b, n]
            # A chain break at position i means the bond between i-1 and i is broken
            # So we need to check chain_breaks[:, 1:] (breaks after residue i-1)
            no_break = chain_breaks[:, 1:] == 0  # [b, n-1], True if no break after i-1
            good_pair = good_pair & no_break

        bb_angles = bb_angles * good_pair[..., None]  # [b, n-1, 3]

        zero_pad = torch.zeros((a37.shape[0], 1, 3), device=bb_angles.device)
        bb_angles = torch.cat([bb_angles, zero_pad], dim=1)  # [b, n, 3]
        return bb_angles


class BackboneBondAnglesSeqFeat(Feature):
    """
    Computes bond angle and featurizes it, with binning and 1-hot.

    TODO: Add mask?
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(3 * 21))

    def forward(self, batch):
        bb_bond_angle = self._get_bb_bond_angles(batch)  # [b, n, 3]
        bb_bond_angle_feats = bin_and_one_hot(
            bb_bond_angle,
            torch.linspace(-torch.pi, torch.pi, 20, device=bb_bond_angle.device),
        )  # [b, n, 3, nbins]
        bb_bond_angle_feats = bb_bond_angle_feats.flatten(start_dim=-2)  # [b, n, 3 * nbins]
        return bb_bond_angle_feats

    def _get_bb_bond_angles(self, batch):
        a37 = batch["coords"]  # [b, n, 37, 3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n]

        if "residue_pdb_idx" in batch and batch["residue_pdb_idx"] is not None:
            idx = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]
        b = a37.shape[0]

        N = a37[:, :, 0, :]  # [b, n, 3]
        CA = a37[:, :, 1, :]  # [b, n, 3]
        C = a37[:, :, 2, :]  # [b, n, 3]
        theta_1 = bond_angles(N[:, :, :], CA[:, :, :], C[:, :, :])  # [b, n]
        theta_2 = bond_angles(CA[:, :-1, :], C[:, :-1, :], N[:, 1:, :])  # [b, n-1]
        theta_3 = bond_angles(C[:, :-1, :], N[:, 1:, :], CA[:, 1:, :])  # [b, n-1]

        # Account for chain breaks in theta_2 and theta_3
        # Check both PDB index continuity AND chain break flags
        good_pair = idx[:, 1:] - idx[:, :-1] == 1  # boolean [b, n-1]

        # Additionally check for chain breaks if available
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"]  # [b, n]
            # A chain break at position i means the bond between i-1 and i is broken
            # So we need to check chain_breaks[:, 1:] (breaks after residue i-1)
            no_break = chain_breaks[:, 1:] == 0  # [b, n-1], True if no break after i-1
            good_pair = good_pair & no_break

        theta_2 = theta_2 * good_pair  # [b, n-1]
        theta_3 = theta_3 * good_pair  # [b, n-1]

        # Add a zero at the end of theta_2 and theta_3 to get shape [b, n]
        zero_pad = torch.zeros((b, 1), device=theta_2.device)  # [b, 1]
        theta_2 = torch.cat([theta_2, zero_pad], dim=-1)  # [b, n]
        theta_3 = torch.cat([theta_3, zero_pad], dim=-1)  # [b, n]

        bb_angles = torch.stack([theta_1, theta_2, theta_3], dim=-1)  # [b, n, 3]
        return bb_angles


class OpenfoldSideChainAnglesSeqFeat(Feature):
    """Computes sequence features from side chain angles."""

    def __init__(self, **kwargs):
        super().__init__(dim=int(4 * 21 + 4))  # 88

    def forward(self, batch):
        _, angles, torsion_angles_mask = self._get_sidechain_angles(batch)
        # _, [b, n, 4] and [b, n, 4]
        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, 4, nbins]
        angles_feat = angles_feat * torsion_angles_mask[..., None]
        angles_feat = angles_feat.flatten(start_dim=-2)  # [b, n, 4 * nbins]
        feat = torch.cat(
            [angles_feat, torsion_angles_mask], dim=-1
        )  # [b, n, 4 * nbins + 4]
        return feat

    def _get_sidechain_angles(self, batch):
        orig_dtype = batch["coords"].dtype
        aatype = batch["residue_type"]  # [b, n]
        torsion_dtype = (
            torch.float32 if batch["coords"].device.type == "mps" else torch.float64
        )
        coords = batch["coords"].to(dtype=torsion_dtype)  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"].to(dtype=torsion_dtype)  # [b, n, 37]
        p = {
            "aatype": aatype,
            "all_atom_positions": coords,
            "all_atom_mask": atom_mask,
        }
        # Next function defined with curry1 decorator
        p = data_transforms.atom37_to_torsion_angles(prefix="")(p)
        torsion_angles_sin_cos = p["torsion_angles_sin_cos"]  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = p["alt_torsion_angles_sin_cos"]  # [b, n, 7, 2]
        # Normalize, all these vectors should have norm 1
        torsion_angles_sin_cos = torsion_angles_sin_cos / (
            torch.linalg.norm(torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos / (
            torch.linalg.norm(alt_torsion_angles_sin_cos, dim=-1, keepdim=True) + 1e-10
        )  # [b, n, 7, 2]
        torsion_angles_mask = p["torsion_angles_mask"]  # [b, n, 7]
        torsion_angles_sin_cos = torsion_angles_sin_cos * torsion_angles_mask[..., None]
        alt_torsion_angles_sin_cos = (
            alt_torsion_angles_sin_cos * torsion_angles_mask[..., None]
        )
        angles = torch.atan2(
            torsion_angles_sin_cos[..., 0], torsion_angles_sin_cos[..., 1]
        )  # [b, n, 7]
        angles = angles * torsion_angles_mask
        # Keep only sidechain
        torsion_angles_sin_cos = torsion_angles_sin_cos[..., -4:, :]  # [b, n, 4, 2]
        alt_torsion_angles_sin_cos = alt_torsion_angles_sin_cos[
            ..., -4:, :
        ]  # [b, n, 4, 2]
        angles = angles[..., -4:]  # [b, n, 4]
        torsion_angles_mask = torsion_angles_mask[..., -4:]  # [b, n, 4]
        return (
            torsion_angles_sin_cos.to(dtype=orig_dtype),
            angles.to(dtype=orig_dtype),
            torsion_angles_mask.bool(),
        )  # [b, n, 4, 2], [b, n, 4] and [b, n, 4]


class LatentVariableSeqFeat(Feature):
    """Returns sequence feature from latent variable."""

    def __init__(self, latent_z_dim, **kwargs):
        super().__init__(dim=latent_z_dim)

    def forward(self, batch):
        assert (
            "z_latent" in batch
        ), "`z_latent` not in batch, cannot compute LatentVariableSeqFeat"
        return batch["z_latent"]  # [b, n, latent_dim]


class MotifAbsoluteCoordsSeqFeat(Feature):
    """Computes absolute coordinates feature from motif coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for absolute coords
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch:
            batch_coors = {
                "coords_nm": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return Atom37NanometersCoorsSeqFeat(rel=False)(batch_coors)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No x_motif or motif_mask in batch, returning zeros for MotifAbsoluteCoordsSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class MotifRelativeCoordsSeqFeat(Feature):
    """Computes relative coordinates feature from motif coordinates."""

    def __init__(self, **kwargs):
        super().__init__(dim=148)  # 37 * 4 for relative coords
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif_mask" in batch:
            required_atoms = torch.tensor(
                [atom_types.index("CA")], device=batch["motif_mask"].device
            )  # CA
            has_required_atoms = torch.all(
                batch["motif_mask"][:, :, required_atoms], dim=-1
            )  # [batch, seq_len]
            relevant_has_required_atoms = torch.where(
                batch["seq_motif_mask"],
                has_required_atoms,
                torch.ones_like(has_required_atoms, dtype=torch.bool),
            )
            valid_batch = relevant_has_required_atoms.reshape(-1).all()
            batch_coors = {
                "coords_nm": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            feat = Atom37NanometersCoorsSeqFeat(rel=True)(batch_coors)
            return torch.where(valid_batch, feat, torch.zeros_like(feat))
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No x_motif or motif_mask in batch, returning zeros for MotifRelativeCoordsSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class MotifSequenceSeqFeat(Feature):
    """Computes sequence feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=20)  # 20 for one-hot encoded residues
        self._has_logged = False

    def forward(self, batch):
        if "seq_motif" in batch and "seq_motif_mask" in batch:
            batch_seq = {
                "residue_type": batch["seq_motif"],
                "mask_dict": {
                    "residue_type": batch["seq_motif_mask"],
                },
            }
            return ResidueTypeSeqFeat()(batch_seq)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No seq_motif or seq_motif_mask in batch, returning zeros for MotifSequenceSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class MotifSideChainAnglesSeqFeat(Feature):
    """Computes side chain angles feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=88)  # 4 * 21 + 4 for side chain angles
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif" in batch:
            batch_sc_angles = {
                "residue_type": batch["seq_motif"],
                "coords": batch["x_motif"],
                "coord_mask": batch["motif_mask"],
            }
            return OpenfoldSideChainAnglesSeqFeat()(batch_sc_angles)
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "Missing required motif data in batch, returning zeros for MotifSideChainAnglesSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class MotifTorsionAnglesSeqFeat(Feature):
    """Computes torsion angles feature from motif."""

    def __init__(self, **kwargs):
        super().__init__(dim=63)  # 3 * 21 for torsion angles
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch and "motif_mask" in batch and "seq_motif_mask" in batch:
            backbone_atoms = torch.tensor(
                [
                    atom_types.index("N"),
                    atom_types.index("CA"),
                    atom_types.index("C"),
                    atom_types.index("O"),
                ],
                device=batch["motif_mask"].device,
            )
            motif_mask_per_residue_backbone = torch.any(
                batch["motif_mask"][:, :, backbone_atoms], dim=-1
            )  # [batch, seq_len]
            relevant_motif_mask = torch.where(
                batch["seq_motif_mask"],
                motif_mask_per_residue_backbone,
                torch.ones_like(motif_mask_per_residue_backbone, dtype=torch.bool),
            )
            valid_batch = relevant_motif_mask.reshape(-1).all()

            batch_torsion_angles = {
                "coords": batch["x_motif"],
                "residue_pdb_idx": batch.get("residue_pdb_idx", None),
            }
            feat = BackboneTorsionAnglesSeqFeat()(batch_torsion_angles)
            return torch.where(valid_batch, feat, torch.zeros_like(feat))
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No x_motif or motif_mask in batch, returning zeros for MotifTorsionAnglesSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class XmotifBulkTipSeqFeat(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_seq = ResidueTypeSeqFeat()

        dim = self.const_coors_abs.dim + self.const_seq.dim + 37
        self.dim = dim  # fix dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]

            # concatenate all features
            feat = torch.cat([feat_coors_abs, feat_seq, motif_mask], dim=-1)  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            return feat

        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)


class MotifMaskSeqFeat(Feature):
    """Computes motif mask feature."""

    def __init__(self, **kwargs):
        super().__init__(dim=37)  # 37 for atom mask
        self._has_logged = False

    def forward(self, batch):
        if "motif_mask" in batch:
            return batch["motif_mask"] * 1.0  # [b, n, 37]
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No motif_mask in batch, returning zeros for MotifMaskSeqFeat",
            )
            return torch.zeros(b, n, self.dim, device=device)


class XmotifSeqFeatUnindexed(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""
    
    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_seq = ResidueTypeSeqFeat()
        
        dim = self.const_coors_abs.dim + self.const_seq.dim + 37
        self.dim = dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]
            motif_mask_residue = motif_mask.sum(-1).bool()  # [b, n]

            # concatenate all features
            feat = torch.cat([feat_coors_abs, feat_seq, motif_mask], dim=-1)  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            feats_ind = []
            masks_ind = []
            for b in range(feat.shape[0]):
                feat_local = feat[b, ...]  # [n, some # added up]
                mask_local = motif_mask_residue[b, ...]  # [n]
                feat_local = feat_local[mask_local, ...]  # [# motif residues for b element, some # added up]
                mask_local = mask_local[mask_local]  # [# motif residues for b element], should be all True
                feat_local = feat_local * mask_local[..., None]  # [# motif residues for b element, some # added up]
                feats_ind.append(feat_local)
                masks_ind.append(mask_local)
                assert torch.all(mask_local), "Mask local wrong"

            masks_motif_uidx = pad_sequence(masks_ind, batch_first=True, padding_value=False)  # [b, n]
            feats_motif_uidx = pad_sequence(feats_ind, batch_first=True, padding_value=0.0)  # [b, n, some # added up]

            return feats_motif_uidx, masks_motif_uidx
        
        else:
            raise IOError("No x_motif in batch")
            # b, n = self.extract_bs_and_n(batch)
            # device = self.extract_device(batch)
            # return torch.zeros(b, n, self.dim, device=device)



class BulkAllAtomXmotifSeqFeat(Feature):
    """Computes feature from x_motif coordinates, seq feature of shape [b, n, 3] or [b, na, 3]."""
    
    def __init__(self, **kwargs):
        super().__init__(dim=None)  # dim will be fixed
        self.const_coors_abs = Atom37NanometersCoorsSeqFeat(rel=False)
        self.const_coors_rel = Atom37NanometersCoorsSeqFeat(rel=True)
        self.const_seq = ResidueTypeSeqFeat()
        self.const_sc_angles = OpenfoldSideChainAnglesSeqFeat()
        self.const_torsion_angles = BackboneTorsionAnglesSeqFeat()
        
        dim = self.const_coors_abs.dim + self.const_coors_rel.dim + self.const_seq.dim + self.const_sc_angles.dim + self.const_torsion_angles.dim + 37
        self.dim = dim

    def forward(self, batch):
        if "x_motif" in batch:
            # Coordinates features
            batch_coors = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_coors_abs = self.const_coors_abs(batch_coors)  # [b, n, some #]
            feat_coors_rel = self.const_coors_rel(batch_coors)  # [b, n, some #]

            # Sequence features
            seq_mask = batch["motif_mask"].sum(-1).bool()  # [b, n]
            batch_seq = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "mask_dict": {
                    "residue_type": seq_mask,
                },
            }
            feat_seq = self.const_seq(batch_seq)  # [b, n, some #]

            # Side chain angles features
            batch_sc_angles = {
                "residue_type": batch["seq_motif"],  # [b, n]
                "coords": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat_sc_angles = self.const_sc_angles(batch_sc_angles)  # [b, n, some #]
            if "residue_pdb_idx" in batch:
                idx  = batch["residue_pdb_idx"]  # [b, n]
            else:
                self.assert_defaults_allowed(batch, "Relative sequence separation pair")
                b, n = self.extract_bs_and_n(batch)
                device = self.extract_device(batch)
                idx = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                    device
                )  # [b, n]
            # Torsion angle features
            batch_torsion_angles = {
                "coords": batch["x_motif"],  # [b, n, 37, 3]
                "residue_pdb_idx": idx,  # [b, n]
            }
            feat_torsion_angles = self.const_torsion_angles(batch_torsion_angles)  # [b, n, some #]

            # motif mask
            motif_mask = batch["motif_mask"] * 1.0  # [b, n, 37]

            # concatenate all features
            feat = torch.cat([feat_coors_abs, feat_coors_rel, feat_seq, feat_sc_angles, feat_torsion_angles, motif_mask], dim=-1)  # [b, n, some # added up]
            feat = feat * seq_mask[..., None]  # [b, n, some # added up]

            return feat
        
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            return torch.zeros(b, n, self.dim, device=device)


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chains" in batch:
            mask = batch["chains"].unsqueeze(-1)  # [b, n, 1]
        else:
            raise ValueError("chains")
        return mask


class SequenceSeparationPairFeat(Feature):
    """
    Computes sequence separation with cross-chain detection.

    For same-chain pairs: uses actual sequence separation
    For cross-chain pairs: unified encoding at (seq_sep_max + 10) to avoid confusion
    """

    def __init__(self, seq_sep_dim, cyclic: bool = False, **kwargs):
        super().__init__(dim=seq_sep_dim)
        self.cyclic = bool(cyclic)

    def _extract_cyclic_flag(self, batch, batch_size: int, device: torch.device) -> torch.Tensor:
        if "cyclic" in batch:
            cyclic = batch["cyclic"]
        elif "cyclic_binder" in batch:
            cyclic = batch["cyclic_binder"]
        else:
            cyclic = None

        if cyclic is None:
            return torch.full((batch_size,), self.cyclic, dtype=torch.bool, device=device)

        if isinstance(cyclic, torch.Tensor):
            cyclic = cyclic.to(device=device).bool().reshape(batch_size, -1)
            return cyclic.any(dim=-1)

        return torch.full((batch_size,), bool(cyclic), dtype=torch.bool, device=device)

    def _extract_cyclization_pair(self, batch, batch_size: int, device: torch.device):
        has_pair = batch.get("has_cyclization_pair", None)
        pair_indices = batch.get("cyclization_pair_indices", None)
        shortcut_len = batch.get("cyclization_shortcut_len", None)
        if has_pair is None or pair_indices is None:
            return (
                torch.zeros((batch_size,), dtype=torch.bool, device=device),
                torch.full((batch_size, 2), -1, dtype=torch.long, device=device),
                torch.ones((batch_size,), dtype=torch.long, device=device),
            )

        if isinstance(has_pair, torch.Tensor):
            has_pair = has_pair.to(device=device).bool().reshape(batch_size, -1).any(dim=-1)
        else:
            has_pair = torch.full((batch_size,), bool(has_pair), dtype=torch.bool, device=device)

        if isinstance(pair_indices, torch.Tensor):
            pair_indices = pair_indices.to(device=device).long().reshape(batch_size, -1)
        else:
            pair_indices = torch.as_tensor(pair_indices, dtype=torch.long, device=device).reshape(batch_size, -1)
        if pair_indices.shape[-1] != 2:
            raise ValueError(f"cyclization_pair_indices must have width 2, got shape {tuple(pair_indices.shape)}")

        if shortcut_len is None:
            shortcut_len = torch.ones((batch_size,), dtype=torch.long, device=device)
        elif isinstance(shortcut_len, torch.Tensor):
            shortcut_len = shortcut_len.to(device=device).long().reshape(batch_size, -1)[:, 0]
        else:
            shortcut_len = torch.full((batch_size,), int(shortcut_len), dtype=torch.long, device=device)

        return has_pair, pair_indices, shortcut_len

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                device
            )  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]
        seq_sep = seq_sep.float()
        cyclic_flag = self._extract_cyclic_flag(batch, batch_size=inds.shape[0], device=inds.device)
        has_cyclization_pair, cyclization_pair_indices, cyclization_shortcut_len = self._extract_cyclization_pair(
            batch,
            batch_size=inds.shape[0],
            device=inds.device,
        )

        if "is_peptide" in batch:
            is_peptide = batch["is_peptide"] == 1
            peptide_pair = is_peptide[:, :, None] & is_peptide[:, None, :]
            if "chains" in batch:
                chains = batch["chains"]
                peptide_pair = peptide_pair & (chains[:, :, None] == chains[:, None, :])

            peptide_len = is_peptide.sum(dim=1).clamp(min=1).to(dtype=inds.dtype)
            half = torch.div(peptide_len, 2, rounding_mode="floor")[:, None, None]
            period = peptide_len[:, None, None]
            wrapped_sep = torch.remainder(seq_sep.long() + half, period) - half
            cyclic_mask = cyclic_flag[:, None, None] & peptide_pair
            seq_sep = torch.where(cyclic_mask, wrapped_sep.float(), seq_sep)

            batch_idx = torch.arange(inds.shape[0], device=inds.device)
            anchor_a_idx = cyclization_pair_indices[:, 0].clamp(min=0)
            anchor_b_idx = cyclization_pair_indices[:, 1].clamp(min=0)
            anchor_a_pos = inds[batch_idx, anchor_a_idx].float()
            anchor_b_pos = inds[batch_idx, anchor_b_idx].float()
            shortcut_len = cyclization_shortcut_len.float()

            abs_sep = seq_sep.abs()
            d_short_1 = (
                (inds[:, :, None].float() - anchor_a_pos[:, None, None]).abs()
                + shortcut_len[:, None, None]
                + (inds[:, None, :].float() - anchor_b_pos[:, None, None]).abs()
            )
            d_short_2 = (
                (inds[:, :, None].float() - anchor_b_pos[:, None, None]).abs()
                + shortcut_len[:, None, None]
                + (inds[:, None, :].float() - anchor_a_pos[:, None, None]).abs()
            )
            eff_abs = torch.minimum(abs_sep, torch.minimum(d_short_1, d_short_2))
            signed_eff = torch.sign(seq_sep) * eff_abs
            cyclization_mask = has_cyclization_pair[:, None, None] & peptide_pair
            seq_sep = torch.where(cyclization_mask, signed_eff, seq_sep)

        # Cross-chain detection and unified encoding
        if "chains" in batch:
            chains = batch["chains"]  # [b, n], chain ID for each residue
            same_chain = (chains[:, :, None] == chains[:, None, :])  # [b, n, n]
        elif "is_peptide" in batch:
            is_peptide = batch["is_peptide"]  # [b, n]
            same_chain = (is_peptide[:, :, None] == is_peptide[:, None, :])  # [b, n, n]
        else:
            same_chain = None

        if same_chain is not None:
            seq_sep_max = self.dim // 2 - 1  
            cross_chain_value = seq_sep_max + 10  
            seq_sep = torch.where(
                same_chain,
                seq_sep.float(),
                torch.full_like(seq_sep, float(cross_chain_value))
            )

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert (
            self.dim % 2 == 1
        ), "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        data_modes_avail = [k for k in batch["x_t"]]
        assert (
            "bb_ca" in data_modes_avail
        ), f"`bb_ca` pair dist feature requested but key not available in data modes {data_modes_avail}"
        return bin_pairwise_distances(
            x=batch["x_t"]["bb_ca"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class CaCoorsNanometersPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=30)
        self.min_dist = 0.1
        self.max_dist = 3.0

    def forward(self, batch):
        assert (
            "ca_coors_nm" in batch or "coords_nm" in batch
        ), f"`ca_coors_nm` pair dist feature requested but key `ca_coors_nm` nor `coords_nm` not available"
        if "ca_coors_nm" in batch:
            ca_coors = batch["ca_coors_nm"]
        else:
            ca_coors = batch["coords_nm"][:, :, 1, :]
        return bin_pairwise_distances(
            x=ca_coors,
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class OptionalCaCoorsNanometersPairwiseDistancesPairFeat(
    CaCoorsNanometersPairwiseDistancesPairFeat
):
    """
    If `use_ca_coors_nm_feature` in batch and true, returns pair feature with CA pairwise distances binned, shape [b, n, n, nbins].

    If `use_ca_coors_nm_feature` not in batch, defaults to False.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._has_logged = False

    def _sparse_pair_condition_feat_from_list(
        self,
        batch,
        full_flag: torch.Tensor,
        device: torch.device,
        b: int,
        n: int,
    ):
        pair_indices = batch.get("optional_ca_pair_indices", None)
        pair_bin_idx = batch.get("optional_ca_pair_bin_idx", None)
        pair_valid_mask = batch.get("optional_ca_pair_valid_mask", None)
        if pair_indices is None or pair_bin_idx is None or pair_valid_mask is None:
            return None, torch.zeros_like(full_flag)

        pair_indices = pair_indices.to(device=device).long()
        pair_bin_idx = pair_bin_idx.to(device=device).long()
        pair_valid_mask = pair_valid_mask.to(device=device).bool()

        if pair_indices.ndim == 2:
            pair_indices = pair_indices.unsqueeze(0)
        if pair_bin_idx.ndim == 1:
            pair_bin_idx = pair_bin_idx.unsqueeze(0)
        if pair_valid_mask.ndim == 1:
            pair_valid_mask = pair_valid_mask.unsqueeze(0)

        sparse_flag = pair_valid_mask.any(dim=-1) & (~full_flag)

        feat = torch.zeros(b, n, n, self.dim, device=device)
        pair_padmask = batch["mask"][:, None, :] & batch["mask"][:, :, None]
        batch_idx = torch.arange(b, device=device)[:, None].expand_as(pair_valid_mask)

        idx_i = pair_indices[..., 0]
        idx_j = pair_indices[..., 1]
        safe_i = idx_i.clamp(min=0, max=max(n - 1, 0))
        safe_j = idx_j.clamp(min=0, max=max(n - 1, 0))
        valid = pair_valid_mask & sparse_flag[:, None]
        valid = valid & (idx_i >= 0) & (idx_i < n) & (idx_j >= 0) & (idx_j < n)
        valid = valid & pair_padmask[batch_idx, safe_i, safe_j]

        safe_bin = pair_bin_idx.clamp(min=0, max=self.dim - 1)
        valid_weight = valid.to(dtype=feat.dtype)
        flat_size = b * n * n * self.dim

        flat_feat = feat.reshape(flat_size)
        flat_forward = (((batch_idx * n + safe_i) * n + safe_j) * self.dim + safe_bin).reshape(-1)
        flat_reverse = (((batch_idx * n + safe_j) * n + safe_i) * self.dim + safe_bin).reshape(-1)
        flat_weight = valid_weight.reshape(-1)

        flat_feat = flat_feat.scatter_add(0, flat_forward, flat_weight)
        flat_feat = flat_feat.scatter_add(0, flat_reverse, flat_weight)
        feat = flat_feat.reshape(b, n, n, self.dim).clamp(max=1.0)
        return feat, sparse_flag

    def _sparse_pair_condition_feat(self, batch, full_flag: torch.Tensor, device: torch.device):
        condition = batch.get("optional_ca_pair_dist_condition", None)
        condition_mask = batch.get("optional_ca_pair_dist_mask", None)
        if condition is None or condition_mask is None:
            return None, torch.zeros_like(full_flag)

        condition = condition.to(device=device).float()
        condition_mask = condition_mask.to(device=device).bool()
        if condition.ndim == 3:
            condition = condition.unsqueeze(0)
        if condition_mask.ndim == 2:
            condition_mask = condition_mask.unsqueeze(0)

        sparse_flag = condition_mask.reshape(condition_mask.shape[0], -1).any(dim=-1) & (~full_flag)

        pair_padmask = batch["mask"][:, None, :] & batch["mask"][:, :, None]
        sparse_mask = condition_mask & pair_padmask & sparse_flag[:, None, None]
        sparse_feat = condition * sparse_mask[..., None].to(dtype=condition.dtype)
        return sparse_feat, sparse_flag

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)
        full_flag = _expand_optional_feature_flag(
            batch.get("use_ca_coors_nm_feature", False),
            batch_size=b,
            device=device,
        )

        feat = torch.zeros(b, n, n, self.dim, device=device)
        if "ca_coors_nm" in batch or "coords_nm" in batch:
            full_feat = super().forward(batch)
            feat = torch.where(full_flag[:, None, None, None], full_feat, feat)

        sparse_feat, sparse_flag = self._sparse_pair_condition_feat(
            batch,
            full_flag=full_flag,
            device=device,
        )
        list_sparse_feat, list_sparse_flag = self._sparse_pair_condition_feat_from_list(
            batch,
            full_flag=full_flag,
            device=device,
            b=b,
            n=n,
        )
        if sparse_feat is None:
            sparse_feat = list_sparse_feat
            sparse_flag = list_sparse_flag
        elif list_sparse_feat is not None:
            condition_has_sparse = sparse_flag.any()
            sparse_feat = sparse_feat + torch.where(
                condition_has_sparse,
                torch.zeros_like(list_sparse_feat),
                list_sparse_feat,
            )
            sparse_flag = sparse_flag | ((~condition_has_sparse) & list_sparse_flag)

        if sparse_feat is not None:
            feat = feat + sparse_feat.to(dtype=feat.dtype)

        return feat


class XscBBCAPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
        self,
        x_sc_pair_dist_dim,
        x_sc_pair_dist_min,
        x_sc_pair_dist_max,
        mode_key="x_sc",
        **kwargs,
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max
        self.mode_key = mode_key
        self._has_logged = False

    def forward(self, batch):
        if self.mode_key in batch:
            data_modes_avail = [k for k in batch[self.mode_key]]
            assert (
                "bb_ca" in data_modes_avail
            ), f"`bb_ca` sc/recycle pair dist feature requested but key not available in data modes {data_modes_avail}"
            return bin_pairwise_distances(
                x=batch[self.mode_key]["bb_ca"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                f"No {self.mode_key} in batch, returning zeros for XscBBCAPairwiseDistancesPairFeat",
            )
            return torch.zeros(b, n, n, self.dim, device=device)


class RelativeResidueOrientationPairFeat(Feature):
    """Computes pair feature with pairwise residue orientations.

    See paper "Improved protein structure prediction using
    predicted inter-residue orientations".

    TODO: Impute beta carbon for Glycine
    TODO: 20 as argument
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(5 * 21))  # 105

    def forward(self, batch):
        aatype = batch["residue_type"]  # [b, n]
        coords = batch["coords"]  # [b, n, 37, 3]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n] boolean
        beta_carbon_pair_mask = (
            has_cb[:, :, None] * has_cb[:, :, None]
        )  # [b, n, n] boolean
        pair_mask = pair_mask * beta_carbon_pair_mask  # [b, n, n]

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        N_p1, CA_p1, CB_p1 = map(
            lambda v: v[:, :, None, :], (N, CA, CB)
        )  # Each [b, n, 1, 3]
        N_p2, CA_p2, CB_p2 = map(
            lambda v: v[:, None, :, :], (N, CA, CB)
        )  # Each [b, 1, n, 3]

        theta_12 = signed_dihedral_angle(N_p1, CA_p1, CB_p1, CB_p2)  # [b, n, n]
        theta_21 = signed_dihedral_angle(N_p2, CA_p2, CB_p2, CB_p1)  # [b, n, n]
        phi_12 = bond_angles(CA_p1, CB_p1, CB_p2)  # [b, n, n]
        phi_21 = bond_angles(CA_p2, CB_p2, CB_p1)  # [b, n, n]
        w = signed_dihedral_angle(CA_p1, CB_p1, CB_p2, CA_p2)  # [b, n, n]
        angles = torch.stack(
            [theta_12, theta_21, phi_12, phi_21, w], dim=-1
        )  # [b, n, n, 5]

        angles_feat = bin_and_one_hot(
            angles, torch.linspace(-torch.pi, torch.pi, 20, device=angles.device)
        )  # [b, n, n, 5, nbins]
        angles_feat = angles_feat.flatten(start_dim=-2)  # [b, n, n, 5 * nbins]
        angles_feat = angles_feat * pair_mask[..., None]  # Mask padding
        return angles_feat


class BackbonePairDistancesNanometerPairFeat(Feature):
    """
    Computes pairwise distances between backbone atoms.

    Position (i, j) encodes the distance between CA_i and
    {N_j, CA_j, C_j, CB_j}.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=int(4 * 21))  # 84

    def forward(self, batch):
        assert (
            "coords_nm" in batch
        ), "`coords_nm` not in batch, cannot comptue BackbonePairDistancesNanometerPairFeat"
        coords = batch["coords_nm"]
        atom_mask = batch["coord_mask"]  # [b, n, 37]
        mask = atom_mask[:, :, 1]  # [b, n]
        pair_mask = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
        has_cb = atom_mask[:, :, 3]  # [b, n] boolean, indicates if corresponding

        N = coords[:, :, 0, :]  # [b, n, 3]
        CA = coords[:, :, 1, :]  # [b, n, 3]
        C = coords[:, :, 2, :]  # [b, n, 3]
        CB = coords[:, :, 3, :]  # [b, n, 3]

        CA_i = CA[:, :, None, :]  # [b, n, 1, 3]
        N_j, CA_j, C_j, CB_j = map(
            lambda v: v[:, None, :, :], (N, CA, C, CB)
        )  # Each [b, 1, n, 3]

        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: torch.linalg.norm(v[0] - v[1], dim=-1),
            ((CA_i, N_j), (CA_i, CA_j), (CA_i, C_j), (CA_i, CB_j)),
        )  # Each shape [b, n, n]
        # CA_X[..., i, j] has distance (nm) between CA[..., i] and X[..., j]

        # Accomodate residues without CB
        CA_CB = CA_CB * has_cb[:, None, :]  # [b, n, n]

        # Fix for mask
        CA_N, CA_CA, CA_C, CA_CB = map(
            lambda v: v * pair_mask,
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each shape [b, n, n]

        bin_limits = torch.linspace(0.1, 2, 20, device=coords.device)
        CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat = map(
            lambda v: bin_and_one_hot(v, bin_limits=bin_limits),
            (CA_N, CA_CA, CA_C, CA_CB),
        )  # Each [b, n, n, 21]

        feat = torch.cat(
            [CA_N_feat, CA_CA_feat, CA_C_feat, CA_CB_feat], dim=-1
        )  # [b, n, n, 4 * 21]
        feat = feat * pair_mask[..., None]
        return feat


class XmotifPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone motif atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, **kwargs):
        super().__init__(dim=None)
        self.const = BackbonePairDistancesNanometerPairFeat()
        self.dim = self.const.dim  # Fix dim, cannot put init here
        self._has_logged = False

    def forward(self, batch):
        if "x_motif" in batch:
            batch_bbpd = {
                "coords_nm": batch["x_motif"],  # [b, n, 37, 3]
                "coord_mask": batch["motif_mask"],  # [b, n, 37]
            }
            feat = self.const(batch_bbpd)  # [b, n, n, some #]
            return feat
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No x_motif in batch, returning zeros for XmotifPairwiseDistancesPairFeat",
            )
            return torch.zeros(b, n, n, self.dim, device=device)


class ChainIdxPairFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chains" in batch:
            seq_mask = batch["chains"]  # [b, n]
            mask = torch.einsum("bi,bj->bij", seq_mask, seq_mask).unsqueeze(
                -1
            )  # [b, n, n, 1]
        else:
            raise ValueError("chains")
        return mask


class IsPeptideSeqFeat(Feature):
    """
    Returns a binary sequence feature indicating if a residue is peptide (1) or receptor (0).
    Shape: [b, n, 1]
    """
    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        assert "is_peptide" in batch, "`is_peptide` not in batch, cannot compute IsPeptideSeqFeat"
        is_peptide = batch["is_peptide"]  # [b, n], bool or long
        is_peptide = is_peptide.float()
        is_peptide[is_peptide < 0] = 0.0  
        return is_peptide[..., None]  # [b, n, 1]

class IsConditionSeqFeat(Feature):
    """
    Returns a binary sequence feature indicating if a residue is from the condition chain (1) or not (0).
    Shape: [b, n, 1]
    """
    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        assert "latent_cond_mask" in batch, "`latent_cond_mask` not in batch, cannot compute IsConditionSeqFeat"
        is_condition = batch["latent_cond_mask"]  # [b, n], bool or long
        is_condition = is_condition.float()
        return is_condition[..., None]  # [b, n, 1]


class ReceptorHotspotSeqFeat(Feature):
    """
    Returns a sequence feature indicating receptor hotspot residues.
    Shape: [b, n, 1]
    """
    def __init__(self, **kwargs):
        super().__init__(dim=1)
        self._has_logged = False

    def forward(self, batch):
        if "hotspot" in batch:
            hotspot = batch["hotspot"]  # [b, n] or [b, n, 1]
            if hotspot.ndim == 2:
                hotspot = hotspot.unsqueeze(-1)  # [b, n, 1]
            return hotspot
        else:
            b, n = self.extract_bs_and_n(batch)
            device = self.extract_device(batch)
            _warn_once_compile_safe(
                self,
                "No hotspot in batch, returning zeros for ReceptorHotspotSeqFeat. "
                "Make sure ReceptorHotspotTransform is in your data pipeline if needed.",
            )
            return torch.zeros(b, n, 1, device=device)


class BinderSecondaryStructureSeqFeat(Feature):
    """
    Embeds a coarse binder secondary-structure label and broadcasts it along the
    sequence. The embedding is masked to binder residues when `is_peptide` is
    present in the batch.

    Class ids:
      0 -> null
      1 -> alpha
      2 -> beta
      3 -> alpha_beta
      4 -> few_ss
    """

    def __init__(self, binder_secstruct_emb_dim, binder_secstruct_num_classes: int = 5, **kwargs):
        super().__init__(dim=binder_secstruct_emb_dim)
        self.embedding = torch.nn.Embedding(binder_secstruct_num_classes, binder_secstruct_emb_dim)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        label = batch.get("binder_secondary_class", None)
        if label is None:
            label = torch.zeros(b, dtype=torch.long, device=device)
        elif isinstance(label, torch.Tensor):
            label = label.to(device=device, dtype=torch.long).reshape(b, -1)[:, 0]
        else:
            label = torch.as_tensor(label, device=device, dtype=torch.long).reshape(b, -1)[:, 0]

        feat = self.embedding(label)[:, None, :].expand(b, n, self.dim)

        if "is_peptide" in batch:
            is_peptide = batch["is_peptide"]
            if not isinstance(is_peptide, torch.Tensor):
                is_peptide = torch.as_tensor(is_peptide, device=device)
            is_peptide = is_peptide.to(device=device).reshape(b, n).float()
            feat = feat * is_peptide[..., None]

        return feat


class BinderSecondaryFoldEmbeddingSeqFeat(Feature):
    """
    FoldEmbedding-style coarse binder secondary-structure condition.

    The input is a three-level code [structuredness, family, detail]. Each level
    has its own embedding table; the embeddings are concatenated and broadcast
    along binder residues. If the new fold code is absent, it is derived from
    the legacy binder_secondary_class field so old inference configs remain
    usable with new checkpoints.
    """

    def __init__(
        self,
        binder_secondary_fold_emb_dim: int = 64,
        binder_secondary_fold_struct_num_classes: int = BINDER_SECSTRUCT_FOLD_NUM_CLASSES[0],
        binder_secondary_fold_family_num_classes: int = BINDER_SECSTRUCT_FOLD_NUM_CLASSES[1],
        binder_secondary_fold_detail_num_classes: int = BINDER_SECSTRUCT_FOLD_NUM_CLASSES[2],
        **kwargs,
    ):
        super().__init__(dim=binder_secondary_fold_emb_dim * 3)
        self.emb_dim = int(binder_secondary_fold_emb_dim)
        self.num_classes = (
            int(binder_secondary_fold_struct_num_classes),
            int(binder_secondary_fold_family_num_classes),
            int(binder_secondary_fold_detail_num_classes),
        )
        self.embedding_struct = torch.nn.Embedding(self.num_classes[0], self.emb_dim)
        self.embedding_family = torch.nn.Embedding(self.num_classes[1], self.emb_dim)
        self.embedding_detail = torch.nn.Embedding(self.num_classes[2], self.emb_dim)
        self.register_buffer(
            "legacy_class_to_fold_code",
            torch.tensor(
                [binder_secstruct_fold_code_from_id(i) for i in range(BINDER_SECSTRUCT_FOLD_NUM_CLASSES[2])],
                dtype=torch.long,
            ),
            persistent=False,
        )

    def _fold_code_from_legacy_class(self, label: torch.Tensor) -> torch.Tensor:
        mapping = self.legacy_class_to_fold_code
        label = torch.where(
            (label >= 0) & (label < mapping.shape[0]),
            label,
            torch.zeros_like(label),
        )
        return mapping[label]

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        fold_code = batch.get("binder_secondary_fold_code", None)
        if fold_code is not None:
            fold_code = fold_code.to(device=device, dtype=torch.long).reshape(b, -1, 3)[:, 0, :]
        else:
            label = batch.get("binder_secondary_class", None)
            if label is None:
                label = torch.zeros(b, dtype=torch.long, device=device)
            elif isinstance(label, torch.Tensor):
                label = label.to(device=device, dtype=torch.long).reshape(b, -1)[:, 0]
            else:
                label = torch.as_tensor(label, device=device, dtype=torch.long).reshape(b, -1)[:, 0]
            fold_code = self._fold_code_from_legacy_class(label)

        parts = []
        for level_idx, num_classes in enumerate(self.num_classes):
            value = fold_code[:, level_idx]
            value = torch.where((value >= 0) & (value < num_classes), value, torch.zeros_like(value))
            parts.append(value)
        fold_code = torch.stack(parts, dim=-1)

        feat = torch.cat(
            [
                self.embedding_struct(fold_code[:, 0]),
                self.embedding_family(fold_code[:, 1]),
                self.embedding_detail(fold_code[:, 2]),
            ],
            dim=-1,
        )[:, None, :].expand(b, n, self.dim)

        if "is_peptide" in batch:
            is_peptide = batch["is_peptide"]
            if not isinstance(is_peptide, torch.Tensor):
                is_peptide = torch.as_tensor(is_peptide, device=device)
            is_peptide = is_peptide.to(device=device).reshape(b, n).float()
            feat = feat * is_peptide[..., None]

        return feat


class BinderSecondaryStructureTokenSeqFeat(Feature):
    """
    Embeds RFdiffusion-style per-residue secondary-structure tokens.

    Token ids:
      0 -> helix
      1 -> strand
      2 -> loop
      3 -> mask/unknown
    """

    MASK_TOKEN = 3

    def __init__(self, binder_ss_token_num_classes: int = 4, **kwargs):
        super().__init__(dim=binder_ss_token_num_classes)
        self.num_classes = int(binder_ss_token_num_classes)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        token = batch.get("binder_ss_token", None)
        if token is None:
            token = torch.full((b, n), self.MASK_TOKEN, dtype=torch.long, device=device)
        else:
            token = token.to(device=device, dtype=torch.long).reshape(b, n)

        if "mask" in batch:
            valid_mask = batch["mask"].to(device=device).bool().reshape(b, n)
            token = torch.where(valid_mask, token, torch.full_like(token, self.MASK_TOKEN))
        if "is_peptide" in batch:
            binder_mask = batch["is_peptide"].to(device=device).reshape(b, n) == 1
            token = torch.where(binder_mask, token, torch.full_like(token, self.MASK_TOKEN))
        token = torch.where(token < 0, torch.full_like(token, self.MASK_TOKEN), token)
        token = torch.where(token >= self.num_classes, torch.full_like(token, self.MASK_TOKEN), token)

        feat = F.one_hot(token, num_classes=self.num_classes).to(dtype=torch.float32)
        return feat


class BinderBlockAdjacencyPairFeat(Feature):
    """
    Embeds RFdiffusion-style block-adjacency pair tokens.

    Token ids:
      0 -> not adjacent / unconditioned none
      1 -> adjacent secondary-structure blocks
      2 -> mask/unknown
    """

    MASK_TOKEN = 2

    def __init__(self, binder_block_adj_num_classes: int = 3, **kwargs):
        super().__init__(dim=binder_block_adj_num_classes)
        self.num_classes = int(binder_block_adj_num_classes)

    def forward(self, batch):
        b, n = self.extract_bs_and_n(batch)
        device = self.extract_device(batch)

        adj = batch.get("binder_block_adj", None)
        if adj is None:
            adj = torch.full((b, n, n), self.MASK_TOKEN, dtype=torch.long, device=device)
        else:
            adj = adj.to(device=device, dtype=torch.long).reshape(b, n, n)

        if "mask" in batch:
            valid_mask = batch["mask"].to(device=device).bool().reshape(b, n)
            valid_pair = valid_mask[:, :, None] & valid_mask[:, None, :]
            adj = torch.where(valid_pair, adj, torch.full_like(adj, self.MASK_TOKEN))
        if "is_peptide" in batch:
            binder_mask = batch["is_peptide"].to(device=device).reshape(b, n) == 1
            binder_pair = binder_mask[:, :, None] & binder_mask[:, None, :]
            adj = torch.where(binder_pair, adj, torch.full_like(adj, self.MASK_TOKEN))
        adj = torch.where(adj < 0, torch.full_like(adj, self.MASK_TOKEN), adj)
        adj = torch.where(adj >= self.num_classes, torch.full_like(adj, self.MASK_TOKEN), adj)

        feat = F.one_hot(adj, num_classes=self.num_classes).to(dtype=torch.float32)
        return feat


class IsPeptidePairFeat(Feature):
    """
    Returns a pair feature indicating peptide-peptide, receptor-receptor, and peptide-receptor pairs.
    Shape: [b, n, n, 1]
    """
    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        assert "is_peptide" in batch, "`is_peptide` not in batch, cannot compute IsPeptidePairFeat"
        is_peptide = batch["is_peptide"]  # [b, n], bool or long
        is_peptide = is_peptide.float()
        is_peptide[is_peptide < 0] = 0.0
        pair_flag = torch.einsum("bi,bj->bij", is_peptide, is_peptide)  # [b, n, n]
        return pair_flag.unsqueeze(-1)  # [b, n, n, 1]


class RelativeChainIndexPairFeat(Feature):
    """
    Computes relative chain index feature (inspired by AlphaFold3's a_rel_chain).

    Returns a one-hot encoded feature indicating the chain relationship:
        - [1, 0, 0]: receptor → peptide (cross-chain, R to P)
        - [0, 1, 0]: same chain (intra-chain)
        - [0, 0, 1]: peptide → receptor (cross-chain, P to R)
    """

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):

        if "chains" in batch and "is_peptide" in batch:
            chains = batch["chains"]  # [b, n]
            is_peptide = batch["is_peptide"]  # [b, n]
            same_chain = (chains[:, :, None] == chains[:, None, :])  # [b, n, n]
            is_receptor = (is_peptide == 0)  # [b, n]
            is_pep = (is_peptide == 1)  # [b, n]
            receptor_to_peptide = is_receptor[:, :, None] & is_pep[:, None, :]  # [b, n, n]
            peptide_to_receptor = is_pep[:, :, None] & is_receptor[:, None, :]  # [b, n, n]
            rel_chain = torch.zeros_like(chains[:, :, None].expand(-1, -1, chains.shape[1]), dtype=torch.long)
            rel_chain[receptor_to_peptide] = -1
            rel_chain[peptide_to_receptor] = 1

        elif "is_peptide" in batch:
            is_peptide = batch["is_peptide"]  # [b, n]
            chain_idx = torch.clamp(is_peptide, 0, 1).float()  # 0=receptor, 1=peptide
            rel_chain = chain_idx[:, :, None] - chain_idx[:, None, :]  # [b, n, n]
            rel_chain = rel_chain.long()
        
        else:
            raise AssertionError("`is_peptide` not in batch, cannot compute RelativeChainIndexPairFeat")

        rel_chain_shifted = (rel_chain + 1).long()  # [b, n, n]

        rel_chain_one_hot = torch.nn.functional.one_hot(
            rel_chain_shifted, num_classes=3
        ).float()  # [b, n, n, 3]

        return rel_chain_one_hot


class RelativeChainTypePairFeat(Feature):
    """
    Computes detailed chain type relationship feature.

    Distinguishes 4 types of residue pair relationships:
        - [1, 0, 0, 0]: receptor → peptide (cross-chain, R to P)
        - [0, 1, 0, 0]: receptor ↔ receptor (intra-receptor)
        - [0, 0, 1, 0]: peptide ↔ peptide (intra-peptide)
        - [0, 0, 0, 1]: peptide → receptor (cross-chain, P to R)

    This is more informative than RelativeChainIndexPairFeat as it distinguishes
    receptor-receptor interactions from peptide-peptide interactions, which have
    different characteristics:
        - receptor-receptor: stable, fixed structure
        - peptide-peptide: being generated, more flexible
    """

    def __init__(self, **kwargs):
        super().__init__(dim=4)

    def forward(self, batch):
        assert "is_peptide" in batch, "`is_peptide` not in batch, cannot compute RelativeChainTypePairFeat"

        is_peptide = batch["is_peptide"]  # [b, n]
        is_receptor = (is_peptide == 0)  # [b, n]
        is_pep = (is_peptide == 1)  # [b, n]

        # 4 types of relationships
        receptor_to_peptide = is_receptor[:, :, None] & is_pep[:, None, :]      # R→P [b, n, n]
        receptor_to_receptor = is_receptor[:, :, None] & is_receptor[:, None, :] # R↔R [b, n, n]
        peptide_to_peptide = is_pep[:, :, None] & is_pep[:, None, :]             # P↔P [b, n, n]
        peptide_to_receptor = is_pep[:, :, None] & is_receptor[:, None, :]       # P→R [b, n, n]

        rel_chain_type = torch.stack([
            receptor_to_peptide,
            receptor_to_receptor,
            peptide_to_peptide,
            peptide_to_receptor
        ], dim=-1).float()  # [b, n, n, 4]

        return rel_chain_type


class InvariantAtom37SeqFeat(Feature):
    """
    Computes Atom37 coordinates projected into the residue's own local rigid frame.
    Output shape: [b, n, 37 * 3 + 37] (coords + mask), this feature is 
    rotationally and translationally invariant.
    """
    def __init__(self, **kwargs):
        # 37 atoms * 3 (xyz) + 37 (mask)
        super().__init__(dim=int(37 * 3 + 37))

    def _construct_frames(self, coords, valid_mask):
        """
        Constructs local frames from N, CA, C for valid residues.
        coords: [b, n, 37, 3]
        valid_mask: [b, n] boolean mask indicating residues with valid coordinates
        Returns: R [b, n, 3, 3], t [b, n, 3]
        """
        # 0: N, 1: CA, 2: C
        N = coords[:, :, 0]
        CA = coords[:, :, 1]
        C = coords[:, :, 2]

        eps = 1e-8

        # Gram-Schmidt Orthogonalization
        v1 = C - CA
        v2 = N - CA
        e1 = F.normalize(v1, dim=-1, eps=eps)  # direction of C
        u2 = v2 - e1 * (torch.sum(e1 * v2, dim=-1, keepdim=True))
        e2 = F.normalize(u2, dim=-1, eps=eps)  # in (N, CA, C) plane
        e3 = torch.cross(e1, e2, dim=-1)  # perpendicular

        # R = [e1, e2, e3] (columns)
        R = torch.stack([e1, e2, e3], dim=-1)  # [b, n, 3, 3]
        t = CA  # [b, n, 3]

        identity = torch.eye(
            3, dtype=coords.dtype, device=coords.device
        ).view(1, 1, 3, 3)
        zero_t = torch.zeros_like(t)
        valid_mask_expanded = valid_mask[..., None, None]
        R = torch.where(valid_mask_expanded, R, identity)
        t = torch.where(valid_mask[..., None], t, zero_t)
        return R, t

    def forward(self, batch):
        assert "coords_nm" in batch, "`coords_nm` missing"
        assert "coord_mask" in batch, "`coord_mask` missing"

        coords = batch["coords_nm"]  # [b, n, 37, 3]
        coors_mask = batch["coord_mask"]  # [b, n, 37]
        if "mask" in batch:
            residue_mask = batch["mask"].bool()  # [b, n]
        else:
            residue_mask = coors_mask[:, :, 1].bool()  # rely on CA availability
        backbone_mask = (
            coors_mask[:, :, 0].bool()
            & coors_mask[:, :, 1].bool()
            & coors_mask[:, :, 2].bool()
        )
        valid_mask = residue_mask & backbone_mask
        coords = coords * coors_mask[..., None]
        R, t = self._construct_frames(coords, valid_mask)
        coords_centered = coords - t.unsqueeze(2)
        coords_local = torch.einsum('bnai,bnij->bnaj', coords_centered, R)
        coords_flat = coords_local.flatten(start_dim=-2)  # [b, n, 37 * 3]

        feat = torch.cat([coords_flat, coors_mask], dim=-1) # [b, n, 148]
        feat = feat * residue_mask[..., None]

        return feat
    
####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
        self,
        feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        mode: Literal["seq", "pair"],
        **kwargs,
    ):
        """
        Feature factory for creating sequence and pair features.

        Sequence features include:
            Time embeddings:
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Position and structure:
                - "res_seq_pdb_idx": Residue sequence position (requires ResidueSequencePositionPdbTransform)
                - "chain_break_per_res": Chain break per residue (requires ChainBreakPerResidueTransform)
                - "chain_idx_seq": Chain index sequence feature
                - "is_peptide_seq": Binary flag indicating peptide (1) vs receptor (0) residues
                - "is_condition_seq": Binary flag indicating condition chain residues
                - "receptor_hotspot": Receptor hotspot residues
                - "binder_secondary_structure": Coarse binder secondary-structure label

            Coordinates and angles:
                - "x_sc_bb_ca": Self-conditioning backbone CA coordinates
                - "x_recycle_bb_ca": Recycled backbone CA coordinates
                - "x_sc_local_latents": Self-conditioning local latents
                - "x_recycle_local_latents": Recycled local latents
                - "xt_bb_ca": Target backbone CA coordinates
                - "xt_local_latents": Target local latents
                - "ca_coors_nm": CA coordinates in nanometers
                - "ca_coors_nm_try": Try CA coordinates in nanometers
                - "optional_ca_coors_nm_seq_feat": Optional CA coordinates in nanometers
                - "x1_bb_angles": Backbone torsion angles
                - "x1_bond_angles": Backbone bond angles
                - "x1_sidechain_angles": Sidechain angles

            Residue information:
                - "x1_aatype": Residue type
                - "optional_res_type_seq_feat": Optional residue type
                - "x1_a37coors_nm": Atom37 coordinates in nanometers
                - "x1_a37coors_nm_rel": Relative atom37 coordinates in nanometers

            Motif and target:
                - "x_motif": Motif coordinates and features
                - "z_latent_seq": Latent variable sequence

        Pair features include:
            Distance features:
                - "xt_bb_ca_pair_dists": Target backbone CA pairwise distances
                - "x_sc_bb_ca_pair_dists": Self-conditioning backbone CA pairwise distances
                - "x_recycle_bb_ca_pair_dists": Recycled backbone CA pairwise distances
                - "ca_coors_nm_pair_dists": CA coordinates pairwise distances in nanometers
                - "x1_bb_pair_dists_nm": Backbone pairwise distances in nanometers
                - "optional_ca_pair_dist": Optional CA pairwise distances
                - "x_motif_pair_dists": Motif pairwise distances

            Sequence and time:
                - "rel_seq_sep": Relative sequence separation
                - "time_emb_bb_ca": Time embedding for backbone CA atoms
                - "time_emb_local_latents": Time embedding for local latents

            Structure and orientation:
                - "x1_bb_pair_orientation": Relative residue orientation
                - "chain_idx_pair": Chain index pairwise feature
                - "is_peptide_pair": Peptide-peptide interaction indicator (outer product of is_peptide flags)

        """
        super().__init__()
        self.mode = mode
        self.dim_feats_out = dim_feats_out
        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList(
            [self.get_creator(f, **kwargs) for f in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False
        )

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingSeqFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingSeqFeat(data_mode_use="local_latents", **kwargs)

            # Position and indexing
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "chain_idx_seq":
                return ChainIdxSeqFeat(**kwargs)

            # Peptide/Receptor identification
            elif f == "is_peptide_seq":
                return IsPeptideSeqFeat(**kwargs)
            elif f == "is_condition_seq":
                return IsConditionSeqFeat(**kwargs)
            elif f == "receptor_hotspot":
                return ReceptorHotspotSeqFeat(**kwargs)
            elif f == "binder_secondary_structure":
                return BinderSecondaryStructureSeqFeat(**kwargs)
            elif f == "binder_secondary_fold_emb":
                return BinderSecondaryFoldEmbeddingSeqFeat(**kwargs)
            elif f == "binder_ss_token":
                return BinderSecondaryStructureTokenSeqFeat(**kwargs)


            # Basic residue information
            elif f == "x1_aatype":
                return ResidueTypeSeqFeat(**kwargs)
            elif f == "optional_res_type_seq_feat":
                return OptionalResidueTypeSeqFeat(**kwargs)

            # Raw coordinate features
            elif f == "ca_coors_nm":
                return CaCoorsNanometersSeqFeat(**kwargs)
            elif f == "ca_coors_nm_try":
                return TryCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "optional_ca_coors_nm_seq_feat":
                return OptionalCaCoorsNanometersSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm":
                return Atom37NanometersCoorsSeqFeat(**kwargs)
            elif f == "x1_a37coors_nm_rel":
                return Atom37NanometersCoorsSeqFeat(rel=True, **kwargs)
            elif f == 'x1_a37coors_invariant':
                return InvariantAtom37SeqFeat(**kwargs)

            # Diffusion/sampling coordinates
            elif f == "xt_bb_ca":
                return XtBBCASeqFeat(**kwargs)
            elif f == "xt_local_latents":
                return XtLocalLatentsSeqFeat(**kwargs)
            elif f == "x_sc_bb_ca":
                return XscBBCASeqFeat(**kwargs)
            elif f == "x_recycle_bb_ca":
                return XscBBCASeqFeat(mode_key="x_recycle", **kwargs)
            elif f == "x_sc_local_latents":
                return XscLocalLatentsSeqFeat(**kwargs)
            elif f == "x_recycle_local_latents":
                return XscLocalLatentsSeqFeat(mode_key="x_recycle", **kwargs)

            # Structural features (angles)
            elif f == "x1_bb_angles":
                return BackboneTorsionAnglesSeqFeat(**kwargs)
            elif f == "x1_bond_angles":
                return BackboneBondAnglesSeqFeat(**kwargs)
            elif f == "x1_sidechain_angles":
                return OpenfoldSideChainAnglesSeqFeat(**kwargs)

            # Latent variables
            elif f == "z_latent_seq":
                return LatentVariableSeqFeat(**kwargs)

            # Motif features
            elif f == "motif_abs_coords":
                return MotifAbsoluteCoordsSeqFeat(**kwargs)
            elif f == "motif_rel_coords":
                return MotifRelativeCoordsSeqFeat(**kwargs)
            elif f == "motif_seq":
                return MotifSequenceSeqFeat(**kwargs)
            elif f == "motif_sc_angles":
                return MotifSideChainAnglesSeqFeat(**kwargs)
            elif f == "motif_torsion_angles":
                return MotifTorsionAnglesSeqFeat(**kwargs)
            elif f == "motif_mask":
                return MotifMaskSeqFeat(**kwargs)
            
            #######################################################
            # Some manual changes needed here for the motif task
            #######################################################

            # # Uncomment this for indexed + tip atom
            # elif f == "x_motif":
            #     return XmotifBulkTipSeqFeat(**kwargs)

            # Uncomment this for any other motif task
            elif f == "bulk_all_atom_xmotif" or f == "x_motif":
                return BulkAllAtomXmotifSeqFeat(**kwargs)
            
            #######################################################
            #######################################################
            #######################################################

            elif f == "bulk_all_atom_xmotif":
                return BulkAllAtomXmotifSeqFeat(**kwargs)

            # Special/utility features
            elif f == "zero_feat_seq":
                return ZeroFeat(dim_feats_out=self.dim_feats_out, mode=self.mode, **kwargs)
            else:
                raise IOError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            # Time embeddings
            if f == "time_emb_bb_ca":
                return TimeEmbeddingPairFeat(data_mode_use="bb_ca", **kwargs)
            elif f == "time_emb_local_latents":
                return TimeEmbeddingPairFeat(data_mode_use="local_latents", **kwargs)

            # Sequence separation
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)

            # Distance features
            elif f == "xt_bb_ca_pair_dists":
                return XtBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_sc_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_recycle_bb_ca_pair_dists":
                return XscBBCAPairwiseDistancesPairFeat(mode_key="x_recycle", **kwargs)
            elif f == "ca_coors_nm_pair_dists":
                return CaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "optional_ca_pair_dist":
                return OptionalCaCoorsNanometersPairwiseDistancesPairFeat(**kwargs)
            elif f == "x1_bb_pair_dists_nm":
                return BackbonePairDistancesNanometerPairFeat(**kwargs)
            elif f == "x_motif_pair_dists":
                return XmotifPairwiseDistancesPairFeat(**kwargs)

            # Structural and orientation features
            elif f == "x1_bb_pair_orientation":
                return RelativeResidueOrientationPairFeat(**kwargs)

            # Chain and indexing features
            elif f == "chain_idx_pair":
                return ChainIdxPairFeat(**kwargs)

            # Peptide/Receptor identification
            elif f == "is_peptide_pair":
                return IsPeptidePairFeat(**kwargs)
            elif f == "rel_chain_idx":
                return RelativeChainIndexPairFeat(**kwargs)
            elif f == "rel_chain_type":
                return RelativeChainTypePairFeat(**kwargs)
            elif f == "binder_block_adj":
                return BinderBlockAdjacencyPairFeat(**kwargs)

            else:
                raise IOError(f"Pair feature {f} not implemented.")

        else:
            raise IOError(
                f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'."
            )

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(
                fcreator(batch)
            )  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(
            feature_tensors, dim=-1
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(
            features, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(
            self.linear_out(features)
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(
            features_proc, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]


class FeatureFactoryUidxMotif(torch.nn.Module):
    def __init__(
        self,
        # feats: List[str],
        dim_feats_out: int,
        use_ln_out: bool,
        **kwargs,
    ):
        """
        Sequence features include:
            - "x_motix_uidx"

        Pair features include:
        """
        super().__init__()

        self.feat_creator = XmotifSeqFeatUnindexed(**kwargs)
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(self.feat_creator.get_dim(), dim_feats_out, bias=False)

    def forward(self, batch):
        """Returns feature and mask, shapes [b, n_motif, dim_f] and [b, n_motif]"""
        feat, feat_mask = self.feat_creator(batch)  # [b, n_motif, dim_f], [b, n_motif]
        feat_proc = self.ln_out(self.linear_out(feat))  # [b, n_motif, dim_f]
        return feat_proc * feat_mask[..., None], feat_mask
