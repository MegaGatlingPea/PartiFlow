import io
from typing import Dict, Tuple

import numpy as np
import biotite.structure as struc
import biotite.structure.io.pdb as pdbio


BINDER_SECSTRUCT_NULL = 0
BINDER_SECSTRUCT_ALPHA = 1
BINDER_SECSTRUCT_BETA = 2
BINDER_SECSTRUCT_ALPHA_BETA = 3
BINDER_SECSTRUCT_FEW_SS = 4

BINDER_SECSTRUCT_NAME_TO_ID = {
    "null": BINDER_SECSTRUCT_NULL,
    "alpha": BINDER_SECSTRUCT_ALPHA,
    "beta": BINDER_SECSTRUCT_BETA,
    "alpha_beta": BINDER_SECSTRUCT_ALPHA_BETA,
    "few_ss": BINDER_SECSTRUCT_FEW_SS,
}

BINDER_SECSTRUCT_ID_TO_NAME = {v: k for k, v in BINDER_SECSTRUCT_NAME_TO_ID.items()}

BINDER_SECSTRUCT_FOLD_STRUCT_NULL = 0
BINDER_SECSTRUCT_FOLD_STRUCT_STRUCTURED = 1
BINDER_SECSTRUCT_FOLD_STRUCT_UNSTRUCTURED = 2

BINDER_SECSTRUCT_FOLD_FAMILY_NULL = 0
BINDER_SECSTRUCT_FOLD_FAMILY_ALPHA = 1
BINDER_SECSTRUCT_FOLD_FAMILY_BETA = 2
BINDER_SECSTRUCT_FOLD_FAMILY_ALPHA_BETA = 3
BINDER_SECSTRUCT_FOLD_FAMILY_COIL = 4

BINDER_SECSTRUCT_FOLD_NUM_CLASSES = (3, 5, 5)

BINDER_SECSTRUCT_FOLD_CODE_BY_CLASS = {
    BINDER_SECSTRUCT_NULL: (
        BINDER_SECSTRUCT_FOLD_STRUCT_NULL,
        BINDER_SECSTRUCT_FOLD_FAMILY_NULL,
        BINDER_SECSTRUCT_NULL,
    ),
    BINDER_SECSTRUCT_ALPHA: (
        BINDER_SECSTRUCT_FOLD_STRUCT_STRUCTURED,
        BINDER_SECSTRUCT_FOLD_FAMILY_ALPHA,
        BINDER_SECSTRUCT_ALPHA,
    ),
    BINDER_SECSTRUCT_BETA: (
        BINDER_SECSTRUCT_FOLD_STRUCT_STRUCTURED,
        BINDER_SECSTRUCT_FOLD_FAMILY_BETA,
        BINDER_SECSTRUCT_BETA,
    ),
    BINDER_SECSTRUCT_ALPHA_BETA: (
        BINDER_SECSTRUCT_FOLD_STRUCT_STRUCTURED,
        BINDER_SECSTRUCT_FOLD_FAMILY_ALPHA_BETA,
        BINDER_SECSTRUCT_ALPHA_BETA,
    ),
    BINDER_SECSTRUCT_FEW_SS: (
        BINDER_SECSTRUCT_FOLD_STRUCT_UNSTRUCTURED,
        BINDER_SECSTRUCT_FOLD_FAMILY_COIL,
        BINDER_SECSTRUCT_FEW_SS,
    ),
}


def binder_secstruct_name_from_id(class_id: int) -> str:
    return BINDER_SECSTRUCT_ID_TO_NAME.get(int(class_id), "unknown")


def binder_secstruct_fold_code_from_id(class_id: int) -> tuple[int, int, int]:
    return BINDER_SECSTRUCT_FOLD_CODE_BY_CLASS.get(
        int(class_id),
        BINDER_SECSTRUCT_FOLD_CODE_BY_CLASS[BINDER_SECSTRUCT_NULL],
    )


def atom_array_from_pdb_string(pdb_string: str):
    pdb_file = pdbio.PDBFile.read(io.StringIO(pdb_string))
    return pdb_file.get_structure(model=1)


def longest_label_run(labels: np.ndarray, label: str) -> int:
    longest = 0
    current = 0
    for value in labels:
        if value == label:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _empty_secondary_structure_metrics() -> Dict[str, float]:
    return {
        "helix_fraction": 0.0,
        "sheet_fraction": 0.0,
        "coil_fraction": 0.0,
        "longest_helix_run": 0.0,
        "longest_sheet_run": 0.0,
        "num_residues": 0.0,
    }


def classify_secondary_structure_metrics(
    metrics: Dict[str, float],
    *,
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
) -> Tuple[int, str, bool]:
    helix = float(metrics.get("helix_fraction", 0.0))
    beta = float(metrics.get("sheet_fraction", 0.0))
    coil = float(metrics.get("coil_fraction", 0.0))
    num_residues = float(metrics.get("num_residues", 0.0))
    structured_fraction = helix + beta
    longest_helix_count = float(metrics.get("longest_helix_run", 0.0)) * num_residues
    longest_sheet_count = float(metrics.get("longest_sheet_run", 0.0)) * num_residues

    if (
        helix >= alpha_beta_min_helix_fraction
        and beta >= alpha_beta_min_sheet_fraction
        and structured_fraction >= alpha_beta_min_structured_fraction
        and longest_helix_count >= min_longest_helix_run
        and longest_sheet_count >= min_longest_sheet_run
    ):
        return BINDER_SECSTRUCT_ALPHA_BETA, "alpha_beta", True

    if (
        helix >= alpha_threshold
        and beta <= alpha_max_sheet_fraction
        and longest_helix_count >= min_longest_helix_run
    ):
        return BINDER_SECSTRUCT_ALPHA, "alpha", True

    if (
        beta >= beta_threshold
        and helix <= beta_max_helix_fraction
        and longest_sheet_count >= min_longest_sheet_run
    ):
        return BINDER_SECSTRUCT_BETA, "beta", True

    if structured_fraction < few_ss_max_structured_fraction or coil >= few_ss_min_coil_fraction:
        return BINDER_SECSTRUCT_FEW_SS, "few_ss", True

    return BINDER_SECSTRUCT_NULL, "null", False


def secondary_structure_metrics_from_pdb_string(
    pdb_string: str,
    peptide_chain: str = "L",
) -> Dict[str, float]:
    atom_array = atom_array_from_pdb_string(pdb_string)
    peptide = atom_array[atom_array.chain_id == peptide_chain]
    if peptide.array_length() == 0:
        return _empty_secondary_structure_metrics()

    sse = struc.annotate_sse(peptide)
    valid_mask = np.isin(sse, np.array(["a", "b", "c"]))
    valid_sse = sse[valid_mask]
    n = int(valid_sse.size)
    if n == 0:
        return _empty_secondary_structure_metrics()

    helix_count = int(np.sum(valid_sse == "a"))
    sheet_count = int(np.sum(valid_sse == "b"))
    coil_count = int(np.sum(valid_sse == "c"))
    longest_helix = longest_label_run(valid_sse, "a")
    longest_sheet = longest_label_run(valid_sse, "b")
    return {
        "helix_fraction": helix_count / n,
        "sheet_fraction": sheet_count / n,
        "coil_fraction": coil_count / n,
        "longest_helix_run": longest_helix / n,
        "longest_sheet_run": longest_sheet / n,
        "num_residues": float(n),
    }


def secondary_structure_reward_from_metrics(
    metrics: Dict[str, float],
    helix_weight: float = 1.0,
    sheet_weight: float = 0.0,
    coil_weight: float = 0.0,
    longest_run_weight: float = 0.5,
    longest_sheet_run_weight: float = 0.0,
) -> float:
    return (
        helix_weight * metrics["helix_fraction"]
        + sheet_weight * metrics["sheet_fraction"]
        + coil_weight * metrics["coil_fraction"]
        + longest_run_weight * metrics["longest_helix_run"]
        + longest_sheet_run_weight * metrics["longest_sheet_run"]
    )


def secondary_structure_reward_from_pdb_string(
    pdb_string: str,
    peptide_chain: str = "L",
    helix_weight: float = 1.0,
    sheet_weight: float = 0.0,
    coil_weight: float = 0.0,
    longest_run_weight: float = 0.5,
    longest_sheet_run_weight: float = 0.0,
) -> Tuple[float, Dict[str, float]]:
    metrics = secondary_structure_metrics_from_pdb_string(
        pdb_string=pdb_string,
        peptide_chain=peptide_chain,
    )
    reward = secondary_structure_reward_from_metrics(
        metrics=metrics,
        helix_weight=helix_weight,
        sheet_weight=sheet_weight,
        coil_weight=coil_weight,
        longest_run_weight=longest_run_weight,
        longest_sheet_run_weight=longest_sheet_run_weight,
    )
    metrics["reward"] = reward
    return reward, metrics


def helix_metrics_from_pdb_string(
    pdb_string: str,
    peptide_chain: str = "L",
) -> Dict[str, float]:
    return secondary_structure_metrics_from_pdb_string(
        pdb_string=pdb_string,
        peptide_chain=peptide_chain,
    )


def helix_reward_from_pdb_string(
    pdb_string: str,
    peptide_chain: str = "L",
    helix_weight: float = 1.0,
    longest_run_weight: float = 0.5,
) -> Tuple[float, Dict[str, float]]:
    return secondary_structure_reward_from_pdb_string(
        pdb_string=pdb_string,
        peptide_chain=peptide_chain,
        helix_weight=helix_weight,
        longest_run_weight=longest_run_weight,
    )
