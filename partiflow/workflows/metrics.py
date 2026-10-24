"""AF2 confidence units and complete-coverage structure comparisons."""

from __future__ import annotations
import re
import numpy as np
from .structure import polymer_residue_groups, read_structure, AA3_TO_1


def _points(atoms, chains, names=None, positions=None):
    result = {}
    for chain in chains:
        for ordinal, group in enumerate(polymer_residue_groups(atoms, chain), 1):
            if (
                positions is not None
                and (
                    chain,
                    int(group["locator"]["res_id"]),
                    group["locator"].get("ins_code", ""),
                )
                not in positions
            ):
                continue
            for row in group["row_indices"]:
                name = str(atoms.atom_name[row])
                if names is not None and name not in names:
                    continue
                if names is None and str(atoms.element[row]).upper() in ("H", "D", "T"):
                    continue
                key = (chain, ordinal, name)
                if key in result:
                    raise ValueError("Duplicate structure atom occurrence")
                result[key] = np.asarray(atoms.coord[row], dtype=float)
    return result


def _paired(reference, predicted, chains, names=None, positions=None):
    a = _points(reference, chains, names, positions)
    b = _points(predicted, chains, names, positions)
    if not a or set(a) != set(b):
        raise ValueError(
            "Structure comparison requires complete matching atom coverage"
        )
    if names is not None:
        expected = (
            sum((len(polymer_residue_groups(reference, chain)) for chain in chains))
            * len(names)
            if positions is None
            else len(positions) * len(names)
        )
        if len(a) != expected:
            raise ValueError("Structure comparison lacks required backbone atoms")
    keys = sorted(a)
    return (np.asarray([a[k] for k in keys]), np.asarray([b[k] for k in keys]))


def _transform(reference, predicted):
    if (
        reference.shape != predicted.shape
        or reference.ndim != 2
        or reference.shape[1] != 3
        or (len(reference) < 3)
    ):
        raise ValueError("At least three matched atoms required for alignment")
    if not np.isfinite(reference).all() or not np.isfinite(predicted).all():
        raise ValueError("Non-finite comparison coordinates")
    ref_center = reference.mean(0)
    pred_center = predicted.mean(0)
    ref = reference - ref_center
    pred = predicted - pred_center
    if np.linalg.matrix_rank(ref) < 2 or np.linalg.matrix_rank(pred) < 2:
        raise ValueError("Alignment atoms are degenerate")
    u, _, vt = np.linalg.svd(pred.T @ ref)
    fix = np.eye(3)
    fix[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ fix @ vt
    return (rotation, pred_center, ref_center)


def _measure(reference, predicted, fit_reference=None, fit_predicted=None):
    rotation, pred_center, ref_center = _transform(
        reference if fit_reference is None else fit_reference,
        predicted if fit_predicted is None else fit_predicted,
    )
    delta = (predicted - pred_center) @ rotation + ref_center - reference
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def _motif_positions(candidate):
    result = set()
    for segment in candidate.contig_string.split("/"):
        match = re.fullmatch(
            "(?:([^:]+):|([A-Za-z]))(-?\\d+)([A-Za-z]?)(?:-(-?\\d+)([A-Za-z]?))?",
            segment.strip(),
        )
        if not match:
            continue
        chain = match[1] or match[2]
        start = int(match[3])
        end = int(match[5] or start)
        if end < start:
            continue
        for residue in range(start, end + 1):
            result.add(
                (
                    chain,
                    residue,
                    match[4]
                    if residue == start
                    else match[6] or ""
                    if residue == end
                    else "",
                )
            )
    return result


def prediction_metrics(candidate, input_data, prediction):
    confidence = prediction.get("confidence", {})
    metrics = {}
    reasons = {}
    for name in (
        "ptm",
        "iptm",
        "plddt_total",
        "pae_interaction",
        "pae_angstrom",
        "pae_matrix_mean_angstrom",
        "ipae_angstrom",
    ):
        value = confidence.get(name)
        if isinstance(value, (float, int)) and np.isfinite(value):
            metrics[name] = float(value)
    if "plddt_normalized" in confidence:
        metrics["plddt"] = float(confidence["plddt_normalized"])
    elif "plddt" in confidence:
        metrics["plddt"] = float(confidence["plddt"]) / 100.0
    for source, target in [("pae", "pae_normalized"), ("ipae", "ipae_normalized")]:
        if source in confidence:
            metrics[target] = float(confidence[source])
    reference = input_data["atoms"]
    predicted = read_structure(prediction["structure_path"])
    chain = candidate.binder_chain
    for name, names in [
        ("design_rmsd", ("N", "CA", "C")),
        ("binder_scRMSD_bb3", ("N", "CA", "C")),
        ("binder_design_rmsd", ("CA",)),
    ]:
        try:
            a, b = _paired(reference, predicted, [chain], names)
            metrics[name] = _measure(a, b)
        except ValueError as exc:
            reasons[name] = str(exc)
    if prediction["protocol"] == "complex":
        try:
            ref_context, pred_context = _paired(
                reference, predicted, input_data["context_chains"], ("CA",)
            )
            a, b = _paired(reference, predicted, [chain], ("N", "CA", "C"))
            metrics["context_aligned_design_rmsd"] = _measure(
                a, b, ref_context, pred_context
            )
        except ValueError as exc:
            reasons["context_aligned_design_rmsd"] = str(exc)
        try:
            a, b = _paired(
                reference,
                predicted,
                [*input_data["context_chains"], chain],
                ("N", "CA", "C"),
            )
            metrics["complex_design_rmsd"] = _measure(a, b)
        except ValueError as exc:
            reasons["complex_design_rmsd"] = str(exc)
    motifs = {
        position for position in _motif_positions(candidate) if position[0] == chain
    }
    if motifs:
        try:
            a, b = _paired(reference, predicted, [chain], ("N", "CA", "C"), motifs)
            metrics["motif_rmsd_design"] = _measure(a, b)
        except ValueError as exc:
            reasons["motif_rmsd_design"] = str(exc)
    return (metrics, reasons)


def paired_metrics(candidate, sequence, complex_prediction, monomer_prediction):
    try:
        reference = read_structure(complex_prediction["structure_path"])
        predicted = read_structure(monomer_prediction["structure_path"])
        chain = candidate.binder_chain
        for atoms in (reference, predicted):
            actual = "".join(
                (
                    AA3_TO_1.get(g["res_name"], "X")
                    for g in polymer_residue_groups(atoms, chain)
                )
            )
            if actual != sequence:
                raise ValueError(
                    "Bound/unbound chain sequence differs from requested variant"
                )
        ca_ref, ca_pred = _paired(reference, predicted, [chain], ("CA",))
        a, b = _paired(reference, predicted, [chain])
        return (
            {
                "bound_unbound_rmsd": _measure(a, b, ca_ref, ca_pred),
                "bound_unbound_ca_rmsd": _measure(ca_ref, ca_pred),
            },
            {},
        )
    except ValueError as exc:
        return ({}, {"bound_unbound_rmsd": str(exc)})
