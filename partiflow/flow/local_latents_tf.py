from typing import Dict

import torch

from partiflow.openfold.np.residue_constants import RESTYPE_ATOM37_MASK
from partiflow.networks.feature import FeatureFactory
from partiflow.networks.modules.attn_n_transition import MultiheadAttnAndTransition
from partiflow.networks.modules.pair_update import PairReprUpdate
from partiflow.networks.modules.seq_transition_af3 import Transition
from partiflow.networks.modules.pair_rep_initial import PairReprBuilder


def get_atom_mask(device: torch.device = None):
    return torch.from_numpy(RESTYPE_ATOM37_MASK).to(
        dtype=torch.bool, device=device
    )  # [21, 37]


class ChemicalBondPairAdapter(torch.nn.Module):
    """Zero-initialized pair bias for residue-level covalent bond conditioning."""

    def __init__(self, *, pair_repr_dim: int, num_classes: int = 5, zero_init: bool = True):
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding = torch.nn.Embedding(
            self.num_classes,
            int(pair_repr_dim),
            padding_idx=0,
        )
        if zero_init:
            torch.nn.init.zeros_(self.embedding.weight)

    def forward(self, input: Dict, pair_rep: torch.Tensor) -> torch.Tensor:
        bond_type = input.get("chemical_bond_type", None)
        if bond_type is None:
            return pair_rep
        bond_type = bond_type.to(device=pair_rep.device).long()
        if bond_type.ndim == 2:
            bond_type = bond_type.unsqueeze(0)
        if bond_type.shape[:3] != pair_rep.shape[:3]:
            raise ValueError(
                f"chemical_bond_type shape {tuple(bond_type.shape)} is incompatible "
                f"with pair representation shape {tuple(pair_rep.shape)}"
            )
        valid_mask = (bond_type > 0) & (bond_type < self.num_classes)
        safe_bond_type = torch.where(valid_mask, bond_type, torch.zeros_like(bond_type))
        pair_bias = self.embedding(safe_bond_type).to(dtype=pair_rep.dtype)
        pair_bias = pair_bias * valid_mask[..., None].to(dtype=pair_rep.dtype)
        return pair_rep + pair_bias


class LocalLatentsTransformer(torch.nn.Module):
    """
    Encoder part of the autoencoder. A transformer with pair-biased attention.
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(LocalLatentsTransformer, self).__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.update_pair_repr = kwargs["update_pair_repr"]
        self.update_pair_repr_every_n = kwargs["update_pair_repr_every_n"]
        self.use_tri_mult = kwargs["use_tri_mult"]
        self.use_qkln = kwargs["use_qkln"]
        self.output_param = kwargs["output_parameterization"]

        # Separate factories for peptide / receptor features
        #feats_seq_peptide = kwargs["feats_seq_peptide"]
        #feats_seq_receptor = kwargs["feats_seq_receptor"]

        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # self.receptor_repr_factory = FeatureFactory(
        #     feats=feats_seq_receptor,
        #     dim_feats_out=kwargs["token_dim"],
        #     use_ln_out=False,
        #     mode="seq",
        #     **kwargs,
        # )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )
        chemical_bond_pair_cfg = kwargs.get("chemical_bond_pair", {}) or {}
        self.chemical_bond_pair_adapter = None
        if bool(chemical_bond_pair_cfg.get("enabled", False)):
            self.chemical_bond_pair_adapter = ChemicalBondPairAdapter(
                pair_repr_dim=kwargs["pair_repr_dim"],
                num_classes=int(chemical_bond_pair_cfg.get("num_classes", 5)),
                zero_init=bool(chemical_bond_pair_cfg.get("zero_init", True)),
            )

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=self.token_dim,
                    dim_pair=self.pair_repr_dim,
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=True,
                    residual_transition=True,
                    parallel_mha_transition=False,
                    use_attn_pair_bias=True,
                    use_qkln=self.use_qkln,
                )
                for _ in range(self.nlayers)
            ]
        )

        # To update pair representations if needed
        if self.update_pair_repr:
            self.pair_update_layers = torch.nn.ModuleList(
                [
                    (
                        PairReprUpdate(
                            token_dim=kwargs["token_dim"],
                            pair_dim=kwargs["pair_repr_dim"],
                            use_tri_mult=self.use_tri_mult,
                        )
                        if i % self.update_pair_repr_every_n == 0
                        else None
                    )
                    for i in range(self.nlayers - 1)
                ]
            )

        self.local_latents_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, kwargs["latent_dim"], bias=False),
        )
        
        self.ca_linear = torch.nn.Sequential(
            torch.nn.LayerNorm(self.token_dim),
            torch.nn.Linear(self.token_dim, 3, bias=False),
        )

    # @torch.compile
    def forward(self, input: Dict) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Runs the network.

        Args:
            input: {
                # Sampling and training
                "x_t": Dict[str, torch.Tensor[b, n, dim]]
                "t": Dict[str, torch.Tensor[b]]
                "mask": boolean torch.Tensor[b, n]

                # Only training (other batch elements)
                "z_latent": torch.Tensor(b, n, latent_dim),
                "ca_coors_nm": torch.Tensor(b, n, 3),
                "residue_mask": boolean torch.Tensor(b, n)
                ...
            }

        Returns:
            Dictionary:
            {
                "coors_nm": all atom coordinates, shape [b, n, 37, 3]
                "seq_logits": logits for the residue types, shape [b, n, 20]
                "residue_mask": boolean [b, n]
                "aatype_max": residue type by taking the most likely logit, shape [b, n], with integer values {0, ..., 19}
                "atom_mask": boolean [b, n, 37], atom37 mask corresponding to aatype_max
            }
        """
        mask = input["mask"]  # [b, n] boolean

        # Conditioning variables
        c = self.cond_factory(input)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Initial sequence representation from features
        cond_mask_per_mode = input.get("cond_mask") # it must have this field!
        noise_mask_per_mode = input.get("noise_mask") # it must have this field!

        # seq_pep = self.peptide_repr_factory(input)  # [b, n, token_dim]
        # seq_rec = self.receptor_repr_factory(input)  # [b, n, token_dim]

        # seqs = seq_pep * peptide_mask_f + seq_rec * receptor_mask_f
        seqs = self.init_repr_factory(input)  # [b, n, token_dim]
        seqs = seqs * mask[..., None]  # [b, n, token_dim]

        cond_mask_per_mode_f = {
            mode: cond_mask_per_mode[mode][..., None].to(seqs.dtype)
            for mode in cond_mask_per_mode
        }   
        
        noise_mask_per_mode_f = {
            mode: noise_mask_per_mode[mode][..., None].to(seqs.dtype)
            for mode in noise_mask_per_mode
        }

        pair_rep = self.pair_repr_builder(input)  # [b, n, n, pair_dim]
        if self.chemical_bond_pair_adapter is not None:
            pair_rep = self.chemical_bond_pair_adapter(input, pair_rep)

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask
            )  # [b, n, token_dim]

            if self.update_pair_repr:
                if i < self.nlayers - 1:
                    if self.pair_update_layers[i] is not None:
                        pair_rep = self.pair_update_layers[i](
                            seqs, pair_rep, mask
                        )  # [b, n, n, pair_dim]

        # Get outputs
        local_latents_out = (
            self.local_latents_linear(seqs) * mask[..., None]
        )  # [b, n, latent_dim]
        ca_nm_out = self.ca_linear(seqs) * mask[..., None]  # [b, n, 3]

        x_t = input["x_t"]

        local_latent_tf_out = {
            "bb_ca": ca_nm_out * noise_mask_per_mode_f["bb_ca"] if self.output_param["bb_ca"] == "v" else torch.where(cond_mask_per_mode_f["bb_ca"].bool(), x_t["bb_ca"], ca_nm_out),
            "local_latents": local_latents_out * noise_mask_per_mode_f["local_latents"] if self.output_param["local_latents"] == "v" else torch.where(cond_mask_per_mode_f["local_latents"].bool(), x_t["local_latents"], local_latents_out),
        }

        nn_out = {}
        nn_out["bb_ca"] = {self.output_param["bb_ca"]: local_latent_tf_out["bb_ca"]}
        nn_out["local_latents"] = {
            self.output_param["local_latents"]: local_latent_tf_out["local_latents"]
        }

        return nn_out
