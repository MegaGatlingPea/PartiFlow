# below are openfold related utils for optimizing conformers

import copy
import torch
from typing import Dict, Optional

from partiflow.utils.coors_utils import nm_to_ang, ang_to_nm
from partiflow.openfold.utils.geometry import vector
from partiflow.openfold.utils import all_atom_multimer
from partiflow.openfold.utils.tensor_utils import masked_mean
from partiflow.openfold.np import residue_constants as rc
from partiflow.openfold.data.data_transforms import (
    atom37_to_torsion_angles
)


def compute_torsion_angles_from_atom37(
    aatype: torch.Tensor,
    all_atom_positions: torch.Tensor,
    all_atom_mask: torch.Tensor,
    residue_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute 7-way torsion sin/cos using OpenFold transform."""
    torsion_transform = atom37_to_torsion_angles(prefix="")
    torsion_feats = torsion_transform(
        {
            "aatype": aatype.long(),
            "all_atom_positions": all_atom_positions.double(),
            "all_atom_mask": all_atom_mask.double(),
        }
    )
    torsion_angles = torsion_feats["torsion_angles_sin_cos"].to(
        all_atom_positions.dtype
    )
    torsion_mask = torsion_feats["torsion_angles_mask"].to(
        all_atom_positions.dtype
    )
    identity_rot = torch.zeros_like(torsion_angles)
    identity_rot[..., 1] = 1.0
    torsion_angles = torch.where(
        torsion_mask[..., None].bool(),
        torsion_angles,
        identity_rot,
    )

    if residue_mask is not None:
        mask = residue_mask.to(torsion_angles.dtype)[..., None, None]
        torsion_angles = torsion_angles * mask + identity_rot * (1.0 - mask)
        torsion_mask = torsion_mask * residue_mask[..., None].to(torsion_mask.dtype)

    return torsion_angles, torsion_mask


def optimize_sidechains(
    decoder_out: dict,
    allow_optimize: bool = True,
    simple: bool = False,
    lr: float = 1e-2,
    max_steps: int = 200,
    convergence_delta: float = 1e-4,
    debug: bool = False,
) -> dict:

    if not allow_optimize:
        return decoder_out
    if simple:
        return repack_sidechains(decoder_out, allow_repack=allow_optimize)
    with torch.inference_mode(False):
        return _optimize_sidechains_impl(decoder_out, lr, max_steps, convergence_delta, debug)


def _optimize_sidechains_impl(
    decoder_out: dict,
    lr: float,
    max_steps: int,
    convergence_delta: float,
    debug: bool,
) -> dict:
    coords = nm_to_ang(decoder_out["coors_nm"])
    if "residue_mask" in decoder_out:
        residue_mask = decoder_out["residue_mask"].bool()
    elif "mask" in decoder_out:
        residue_mask = decoder_out["mask"].bool()
    else:
        raise ValueError("No residue mask found in decoder_out.")

    atom_mask = decoder_out["atom_mask"].float() * residue_mask[..., None]

    if "aatype" in decoder_out:
        aatype = decoder_out["aatype"].long()
    elif "aatype_max" in decoder_out:
        aatype = decoder_out["aatype_max"].long()
    elif "residue_type" in decoder_out:
        aatype = decoder_out["residue_type"].long()
    else:
        raise ValueError("No amino acid type information found in decoder_out.")

    unk_index = rc.restype_num
    unk = torch.full_like(aatype, unk_index)
    aatype = torch.where(residue_mask, aatype.clamp(max=unk_index), unk)

    all_atom_positions = coords * atom_mask[..., None]  # target coords to fit

    torsion_angles_init, torsion_mask = compute_torsion_angles_from_atom37(
        aatype,
        all_atom_positions,
        atom_mask,
        residue_mask=residue_mask,
    )
    
    sidechain_atom_mask = atom_mask.clone()
    sidechain_atom_mask[..., :4] = 0.0  # zero out backbone atoms (N, CA, C, O)

    with torch.enable_grad():
        all_atom_pos_vec = vector.Vec3Array.from_array(all_atom_positions)
        backb_to_global, _ = all_atom_multimer.make_backbone_affine(
            all_atom_pos_vec, atom_mask, aatype
        )

        chi_sincos = torsion_angles_init[..., 3:, :].clone().detach().float()  # [..., N, 4, 2], force fp32
        chi_sincos.requires_grad_(True)

        optimizer = torch.optim.Adam([chi_sincos], lr=lr)
        last_loss = torch.tensor(float("inf"))

        for step in range(max_steps):
            chi_norm = chi_sincos / (chi_sincos.norm(dim=-1, keepdim=True) + 1e-8)
            torsion_angles = torch.cat(
                [torsion_angles_init[..., :3, :].detach().float(), chi_norm], dim=-2
            )  # [..., N, 7, 2]

            all_frames = all_atom_multimer.torsion_angles_to_frames(
                aatype, backb_to_global, torsion_angles
            )
            atom14 = all_atom_multimer.frames_and_literature_positions_to_atom14_pos(
                aatype, all_frames
            )
            atom14_tensor = torch.stack([atom14.x, atom14.y, atom14.z], dim=-1)
            atom37_rebuilt, _ = all_atom_multimer.atom14_to_atom37(atom14_tensor, aatype)

            if debug and step == 0:
                print(f"[optimize_sidechains] chi_sincos.requires_grad: {chi_sincos.requires_grad}")
                print(f"[optimize_sidechains] chi_norm.requires_grad: {chi_norm.requires_grad}")
                print(f"[optimize_sidechains] torsion_angles.requires_grad: {torsion_angles.requires_grad}")
                print(f"[optimize_sidechains] atom14_tensor.requires_grad: {atom14_tensor.requires_grad}")
                print(f"[optimize_sidechains] atom37_rebuilt.requires_grad: {atom37_rebuilt.requires_grad}")

            sc_mask_f = sidechain_atom_mask.unsqueeze(-1).float()  # [..., N, 37, 1]
            diff = (atom37_rebuilt - all_atom_positions.float()) ** 2  # [..., N, 37, 3]
            loss = (diff * sc_mask_f).sum() / (sc_mask_f.sum() * 3 + 1e-8)

            if debug:
                rmsd = torch.sqrt(loss).item()
                print(f"[optimize_sidechains] step {step:4d} | sidechain RMSD: {rmsd:.4f} Å")

            if torch.abs(loss - last_loss) < convergence_delta:
                if debug:
                    print(f"[optimize_sidechains] converged at step {step}")
                break

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last_loss = loss.detach()

        # final rebuild with optimized chi
        chi_norm = chi_sincos.detach() / (
            chi_sincos.detach().norm(dim=-1, keepdim=True) + 1e-8
        )

    torsion_angles_final = torch.cat(
        [torsion_angles_init[..., :3, :].detach().float(), chi_norm], dim=-2
    )
    all_frames = all_atom_multimer.torsion_angles_to_frames(
        aatype, backb_to_global, torsion_angles_final
    )
    atom14 = all_atom_multimer.frames_and_literature_positions_to_atom14_pos(
        aatype, all_frames
    )
    atom14_tensor = torch.stack([atom14.x, atom14.y, atom14.z], dim=-1)
    atom37, _ = all_atom_multimer.atom14_to_atom37(atom14_tensor, aatype)
    atom37 = atom37 * residue_mask[..., None, None]

    opt_out = {
        key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
        for key, value in decoder_out.items()
    }
    opt_out["coors_nm"] = ang_to_nm(atom37)
    return opt_out


def repack_sidechains(decoder_out: dict, allow_repack: bool = True) -> dict:

    if allow_repack:
        coords = nm_to_ang(decoder_out["coors_nm"])
        if "residue_mask" in decoder_out:
            residue_mask = decoder_out["residue_mask"].bool()
        elif "mask" in decoder_out:
            residue_mask = decoder_out["mask"].bool()
        else:
            raise ValueError("No residue mask found in decoder_out.")
        atom_mask = decoder_out["atom_mask"].float() * residue_mask[..., None]
        if "aatype" in decoder_out:
            aatype = decoder_out["aatype"].long()
        elif "aatype_max" in decoder_out:
            aatype = decoder_out["aatype_max"].long()
        elif "residue_type" in decoder_out:
            aatype = decoder_out["residue_type"].long()
        else:
            raise ValueError("No amino acid type information found in decoder_out.")
        unk_index = rc.restype_num
        unk = torch.full_like(aatype, unk_index)
        aatype = torch.where(residue_mask, aatype.clamp(max=unk_index), unk)
        all_atom_positions = coords * atom_mask[..., None]
        all_atom_pos_vec = vector.Vec3Array.from_array(all_atom_positions)
        backb_to_global, rigid_mask = all_atom_multimer.make_backbone_affine(
            all_atom_pos_vec, atom_mask, aatype
        )
        torsion_angles, _ = compute_torsion_angles_from_atom37(
            aatype,
            all_atom_positions,
            atom_mask,
            residue_mask=residue_mask,
        )
        all_frames = all_atom_multimer.torsion_angles_to_frames(
            aatype, backb_to_global, torsion_angles
        )
        atom14 = all_atom_multimer.frames_and_literature_positions_to_atom14_pos(
            aatype, all_frames
        )
        atom14_tensor = torch.stack([atom14.x, atom14.y, atom14.z], dim=-1)
        atom37, _ = all_atom_multimer.atom14_to_atom37(atom14_tensor, aatype)
        atom37 = atom37 * residue_mask[..., None, None]
        repack_out = {
            key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value)
            for key, value in decoder_out.items()
        }
        repack_out["coors_nm"] = ang_to_nm(atom37)
        return repack_out
    return decoder_out


# modified from openfold
def supervised_chi_loss(
    pred_torsion_angles_sin_cos: torch.Tensor,  # [B, N, 7, 2]
    true_torsion_angles_sin_cos: torch.Tensor,  # [B, N, 7, 2]
    aatype: torch.Tensor,  # [B, N]
    residue_mask: torch.Tensor,  # [B, N]
    torsion_angles_mask: torch.Tensor,  # [B, N, 7]
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """Batch-capable supervised chi loss modified from OpenFold."""
    norm = torch.linalg.norm(
        pred_torsion_angles_sin_cos, dim=-1, keepdim=True
    )

    pred_angles_sin_cos = pred_torsion_angles_sin_cos / (norm + eps)
    pred_chi = pred_angles_sin_cos[..., 3:, :]
    true_chi = true_torsion_angles_sin_cos[..., 3:, :]

    valid_torsion_mask = residue_mask[..., None] * torsion_angles_mask
    valid_torsion_mask = valid_torsion_mask.to(pred_angles_sin_cos.dtype)
    valid_chi_mask = valid_torsion_mask[..., 3:]

    residue_type_one_hot = torch.nn.functional.one_hot(
        aatype,
        rc.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->...ik",
        residue_type_one_hot.type(pred_angles_sin_cos.dtype),
        pred_angles_sin_cos.new_tensor(rc.chi_pi_periodic),
    )
    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi

    sq_chi_error = torch.sum((true_chi - pred_chi) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_chi) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)
    sq_chi_loss = masked_mean(
        valid_chi_mask, sq_chi_error, dim=(-1, -2)
    )

    loss = chi_weight * sq_chi_loss

    norm_error = torch.abs(norm.squeeze(-1) - 1.0)
    angle_norm_loss = masked_mean(
        valid_torsion_mask, norm_error, dim=(-1, -2)
    )

    return loss + angle_norm_weight * angle_norm_loss


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,
    pred_atom_mask: torch.Tensor,
    residue_index: torch.Tensor,
    aatype: torch.Tensor,
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residue_index[..., 1:] - residue_index[..., :-1]) == 1.0

    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    next_is_proline = aatype[..., 1:] == rc.resname_to_idx["PRO"]
    gt_length = (
        ~next_is_proline
    ) * rc.between_res_bond_length_c_n[0] + next_is_proline * rc.between_res_bond_length_c_n[1]
    gt_stddev = (
        ~next_is_proline
    ) * rc.between_res_bond_length_stddev_c_n[0] + next_is_proline * rc.between_res_bond_length_stddev_c_n[1]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_ca_c_n[0]
    gt_stddev = rc.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = rc.between_res_cos_angles_c_n_ca[0]
    gt_stddev = rc.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask
    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residue_index: torch.Tensor,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    fp_type = atom14_pred_positions.dtype

    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, None, :, None, :]
                - atom14_pred_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    dists_mask = dists_mask * (
        residue_index[..., :, None, None, None]
        < residue_index[..., None, :, None, None]
    )

    c_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(2), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)
    n_one_hot = torch.nn.functional.one_hot(
        residue_index.new_tensor(0), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residue_index.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (
        residue_index[..., :, None, None, None] + 1
    ) == residue_index[..., None, :, None, None]
    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    cys = rc.restype_name_to_atom14_names["CYS"]
    cys_sg_idx = cys.index("SG")
    cys_sg_idx = residue_index.new_tensor(cys_sg_idx)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residue_index.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14)
    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    reduce_dims = tuple(range(1, dists_to_low_error.ndim))
    per_example_error = torch.sum(dists_to_low_error, dim=reduce_dims)
    normalizer = torch.sum(dists_mask, dim=reduce_dims)
    mean_loss = per_example_error / (1e-6 + normalizer)

    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, axis=(-3, -1)
    )

    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, axis=(-4, -2)),
        torch.amax(clash_mask, axis=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_clash_mask": per_atom_clash_mask,
    }


def within_residue_violations(
    atom14_pred_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )

    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_pred_positions[..., :, :, None, :]
                - atom14_pred_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
    }

def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append([(rc.atom_order.get(name, 0)) for name in atom_names])
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append([(atom_name_to_idx14.get(name, 0)) for name in rc.atom_types])

        restype_atom14_mask.append([(1.0 if name else 0.0) for name in atom_names])

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein["aatype"].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros([21, 37], dtype=torch.float32, device=protein["aatype"].device)
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


def convert_atom37_to_atom14_openfold(positions, aatype):
    """Convert atom37 positions to atom14 using OpenFold's approach"""

    # Step 1: Create a minimal protein dict to use make_atom14_masks
    protein_dict = {
        "aatype": aatype,  # [batch_size, n_residues]
        "all_atom_positions": positions,  # [batch_size, n_residues, 37, 3]
        "all_atom_mask": torch.ones_like(positions[..., 0]),  # [batch_size, n_residues, 37]
    }

    # Step 2: Generate the atom14 mappings dynamically
    protein_dict = make_atom14_masks(protein_dict)

    # Step 3: Use the generated mappings to convert positions
    residx_atom14_to_atom37 = protein_dict["residx_atom14_to_atom37"]
    atom14_atom_exists = protein_dict["atom14_atom_exists"]

    # Convert atom37 to atom14 format
    positions_atom14 = torch.gather(
        positions,
        dim=2,
        index=residx_atom14_to_atom37.unsqueeze(-1).expand(-1, -1, -1, 3),
    )

    # Apply the existence mask
    mask = atom14_atom_exists.unsqueeze(-1).expand(-1, -1, -1, 3)
    positions_atom14 = positions_atom14 * mask

    return positions_atom14