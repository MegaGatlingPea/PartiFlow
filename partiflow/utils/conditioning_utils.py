from __future__ import annotations

from typing import Any


PAIR_NBINS_DEFAULT = 30
PAIR_MIN_DIST_NM_DEFAULT = 0.1
PAIR_MAX_DIST_NM_DEFAULT = 3.0


DESIGN_CONDITION_NONE = 0
DESIGN_CONDITION_STAPLED = 1
DESIGN_CONDITION_HEAD_TO_TAIL = 2
DESIGN_CONDITION_DISULFIDE = 3
DESIGN_CONDITION_BICYCLE = 4
DESIGN_CONDITION_STAPLE_RTY = 5


DESIGN_CONDITION_NAME_TO_ID = {
    "none": DESIGN_CONDITION_NONE,
    "stapled": DESIGN_CONDITION_STAPLED,
    "staple": DESIGN_CONDITION_STAPLED,
    "stapled_peptide": DESIGN_CONDITION_STAPLED,
    "isopep": DESIGN_CONDITION_STAPLED,
    "isopeptide": DESIGN_CONDITION_STAPLED,
    "head_to_tail": DESIGN_CONDITION_HEAD_TO_TAIL,
    "head_tail": DESIGN_CONDITION_HEAD_TO_TAIL,
    "head-tail": DESIGN_CONDITION_HEAD_TO_TAIL,
    "disulfide": DESIGN_CONDITION_DISULFIDE,
    "cys_cys": DESIGN_CONDITION_DISULFIDE,
    "bicycle": DESIGN_CONDITION_BICYCLE,
    "bicyclic": DESIGN_CONDITION_BICYCLE,
    "staple_rty": DESIGN_CONDITION_STAPLE_RTY,
    "rty": DESIGN_CONDITION_STAPLE_RTY,
    "mk8_staple": DESIGN_CONDITION_STAPLED,
    "staple_mk8": DESIGN_CONDITION_STAPLED,
    "mk8": DESIGN_CONDITION_STAPLED,
    "nle_staple": DESIGN_CONDITION_STAPLED,
    "staple_nle": DESIGN_CONDITION_STAPLED,
    "nle": DESIGN_CONDITION_STAPLED,
    "alkene_staple": DESIGN_CONDITION_STAPLED,
    "alkene-staple": DESIGN_CONDITION_STAPLED,
    "alkane_staple": DESIGN_CONDITION_STAPLED,
    "alkane-staple": DESIGN_CONDITION_STAPLED,
}


DESIGN_CONDITION_ID_TO_NAME = {
    value: key for key, value in DESIGN_CONDITION_NAME_TO_ID.items()
}
DESIGN_CONDITION_ID_TO_NAME[DESIGN_CONDITION_STAPLED] = "stapled"
DESIGN_CONDITION_ID_TO_NAME[DESIGN_CONDITION_HEAD_TO_TAIL] = "head_to_tail"
DESIGN_CONDITION_ID_TO_NAME[DESIGN_CONDITION_DISULFIDE] = "disulfide"
DESIGN_CONDITION_ID_TO_NAME[DESIGN_CONDITION_BICYCLE] = "bicycle"
DESIGN_CONDITION_ID_TO_NAME[DESIGN_CONDITION_STAPLE_RTY] = "staple_rty"


CHEMICAL_BOND_NONE = 0
CHEMICAL_BOND_HEAD_TAIL_AMIDE = 1
CHEMICAL_BOND_DISULFIDE = 2
CHEMICAL_BOND_ISOPEPTIDE = 3
CHEMICAL_BOND_OTHER_COVALENT = 4
CHEMICAL_BOND_NUM_CLASSES = 5


CHEMICAL_BOND_NAME_TO_ID = {
    "none": CHEMICAL_BOND_NONE,
    "head_to_tail": CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    "head_tail": CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    "head-tail": CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    "headtail": CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    "head_tail_amide": CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    "head_tail_amide_cyclic": CHEMICAL_BOND_HEAD_TAIL_AMIDE,
    "disulfide": CHEMICAL_BOND_DISULFIDE,
    "disulfied": CHEMICAL_BOND_DISULFIDE,
    "cys_cys": CHEMICAL_BOND_DISULFIDE,
    "s_s": CHEMICAL_BOND_DISULFIDE,
    "sg_sg": CHEMICAL_BOND_DISULFIDE,
    "isopep": CHEMICAL_BOND_ISOPEPTIDE,
    "isopeptide": CHEMICAL_BOND_ISOPEPTIDE,
    "stapled": CHEMICAL_BOND_ISOPEPTIDE,
    "staple": CHEMICAL_BOND_ISOPEPTIDE,
    "stapled_peptide": CHEMICAL_BOND_ISOPEPTIDE,
    "mk8_staple": CHEMICAL_BOND_OTHER_COVALENT,
    "staple_mk8": CHEMICAL_BOND_OTHER_COVALENT,
    "mk8": CHEMICAL_BOND_OTHER_COVALENT,
    "nle_staple": CHEMICAL_BOND_OTHER_COVALENT,
    "staple_nle": CHEMICAL_BOND_OTHER_COVALENT,
    "nle": CHEMICAL_BOND_OTHER_COVALENT,
    "alkene_staple": CHEMICAL_BOND_OTHER_COVALENT,
    "alkene-staple": CHEMICAL_BOND_OTHER_COVALENT,
    "alkane_staple": CHEMICAL_BOND_OTHER_COVALENT,
    "alkane-staple": CHEMICAL_BOND_OTHER_COVALENT,
    "bicycle": CHEMICAL_BOND_OTHER_COVALENT,
    "bicyclic": CHEMICAL_BOND_OTHER_COVALENT,
    "staple_rty": CHEMICAL_BOND_OTHER_COVALENT,
    "rty": CHEMICAL_BOND_OTHER_COVALENT,
    "other": CHEMICAL_BOND_OTHER_COVALENT,
    "covalent": CHEMICAL_BOND_OTHER_COVALENT,
}


CHEMICAL_BOND_ID_TO_NAME = {
    CHEMICAL_BOND_NONE: "none",
    CHEMICAL_BOND_HEAD_TAIL_AMIDE: "head_tail_amide",
    CHEMICAL_BOND_DISULFIDE: "disulfide",
    CHEMICAL_BOND_ISOPEPTIDE: "isopeptide",
    CHEMICAL_BOND_OTHER_COVALENT: "other_covalent",
}


def design_condition_mode_to_id(mode: Any) -> int:
    if mode is None:
        return DESIGN_CONDITION_NONE
    if isinstance(mode, int):
        return int(mode)
    normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    return int(DESIGN_CONDITION_NAME_TO_ID.get(normalized, DESIGN_CONDITION_NONE))


def design_condition_mode_to_name(mode: Any) -> str:
    if isinstance(mode, str):
        normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in DESIGN_CONDITION_NAME_TO_ID:
            return DESIGN_CONDITION_ID_TO_NAME[DESIGN_CONDITION_NAME_TO_ID[normalized]]
    try:
        return DESIGN_CONDITION_ID_TO_NAME.get(int(mode), "none")
    except Exception:
        return "none"


def chemical_bond_type_to_id(mode: Any) -> int:
    if mode is None:
        return CHEMICAL_BOND_NONE
    if isinstance(mode, int):
        return int(mode)
    normalized = str(mode).strip().lower().replace("-", "_").replace(" ", "_")
    return int(CHEMICAL_BOND_NAME_TO_ID.get(normalized, CHEMICAL_BOND_NONE))


def chemical_bond_type_to_name(mode: Any) -> str:
    if isinstance(mode, str):
        normalized = mode.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in CHEMICAL_BOND_NAME_TO_ID:
            return CHEMICAL_BOND_ID_TO_NAME[CHEMICAL_BOND_NAME_TO_ID[normalized]]
    try:
        return CHEMICAL_BOND_ID_TO_NAME.get(int(mode), "none")
    except Exception:
        return "none"


def pair_bin_idx_from_distance_angstrom(
    distance_angstrom: float,
    *,
    pair_nbins: int = PAIR_NBINS_DEFAULT,
    pair_min_dist_nm: float = PAIR_MIN_DIST_NM_DEFAULT,
    pair_max_dist_nm: float = PAIR_MAX_DIST_NM_DEFAULT,
) -> int:
    if pair_nbins < 2:
        raise ValueError("pair_nbins must be >= 2")
    distance_nm = float(distance_angstrom) * 0.1
    if distance_nm <= pair_min_dist_nm:
        return 0
    if distance_nm >= pair_max_dist_nm:
        return int(pair_nbins - 1)

    n_boundaries = pair_nbins - 1
    step = (pair_max_dist_nm - pair_min_dist_nm) / max(n_boundaries - 1, 1)
    boundaries = [pair_min_dist_nm + step * i for i in range(n_boundaries)]
    for idx, boundary in enumerate(boundaries):
        if distance_nm <= boundary:
            return idx
    return int(pair_nbins - 1)
