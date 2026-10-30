"""Full Rosetta metrics over a strict native residue Pose, without relaxation."""

from __future__ import annotations
import math
from pathlib import Path
from partiflow.postprocess.records import nonstandard_codes

CANONICAL_METRICS = {
    "ShapeComplementarity": "shape_complementarity",
    "n_InterfaceHbonds": "interface_hbonds",
    "n_InterfaceUnsatHbonds": "interface_unsat_hbonds",
    "Surface_Hydrophobicity": "surface_hydrophobicity",
}


def raw_metric_requirements(names):
    """Resolve canonical and legacy names in one place without dropping requests."""
    raw_names = {target: source for source, target in CANONICAL_METRICS.items()}
    return {raw_names.get(name, name) for name in names}


def _compact_native_load(info, pose, binder_chain, codes):
    compact = {key: value for key, value in info.items()
               if value is None or type(value) in (str, bool, int)
               or (type(value) is float and math.isfinite(value))}
    compact["source_nonstandard_codes"] = list(codes)
    if codes:
        pdb_info = pose.pdb_info()
        compact["binder_residue_types"] = sorted({
            str(pose.residue(i).name()) for i in range(1, pose.total_residue()+1)
            if str(pdb_info.chain(i)) == binder_chain})
    return compact


def finite_metrics(raw):
    return {
        key: value
        for key, value in raw.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }


def canonical_metrics(raw):
    metrics = finite_metrics(raw)
    for source, target in CANONICAL_METRICS.items():
        value = raw.get(source)
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            metrics[target] = value
    return metrics


def _legacy_metrics(
    path, *, binder_chain="L", topology_entries=None, compute_buns=True
):
    from partiflow.postprocess import native, rosetta

    pose, load_info = native.load_score_only_pose(
        path, binder_chain=binder_chain, topology_entries=topology_entries
    )
    snapshot = native.snapshot_coordinates(pose)
    ncaa = bool(nonstandard_codes(Path(path)))
    chains = rosetta._chain_ids_in_pose(pose)
    targets = "".join(chain for chain in chains if chain != binder_chain)
    raw, failures = {}, {}
    scorefxn, adjustments = native._score_function_for_pose(pose)
    try:
        raw["total_score"] = float(scorefxn(pose))
        components, weights, pathology, warnings = native._score_components(
            scorefxn, pose
        )
        if pathology:
            raise ValueError(f"Non-finite Rosetta components: {pathology}")
        raw.update(components)
        load_info.update(
            score_component_weights=weights,
            score_component_warnings=warnings,
            scorefxn_adjustments=adjustments,
        )
    except Exception as exc:
        failures["energy"] = f"{type(exc).__name__}: {exc}"
    if targets:
        try:
            raw.update(
                rosetta.score_interface(
                    str(path),
                    binder_chain=binder_chain,
                    target_chains=targets,
                    topology_entries=topology_entries,
                    compute_buns=compute_buns,
                    relax_before_scoring=False,
                    native_pose=pose,
                    native_no_pack=ncaa,
                )
            )
        except Exception as exc:
            failures["full_interface"] = f"{type(exc).__name__}: {exc}"
            # Independent fixed-pose geometry remains useful if an optional metric fails.
            try:
                iam = rosetta.InterfaceAnalyzerMover()
                rosetta._set_interface_analyzer_interface(
                    iam, f"{targets}_{binder_chain}"
                )
                iam.set_scorefunction(scorefxn)
                iam.set_pack_input(False)
                iam.set_pack_separated(False)
                iam.set_compute_packstat(False)
                iam.set_compute_interface_energy(True)
                iam.set_calc_dSASA(True)
                iam.set_calc_hbond_sasaE(True)
                iam.set_compute_interface_sc(True)
                iam.apply(pose.clone())
                raw.update(
                    ShapeComplementarity=float(iam.get_all_data().sc_value),
                    n_InterfaceHbonds=float(iam.get_all_data().interface_hbonds),
                    dG=float(iam.get_interface_dG()),
                    dSASA=float(iam.get_interface_delta_sasa()),
                )
            except Exception as geometry_exc:
                failures["interface_geometry"] = (
                    f"{type(geometry_exc).__name__}: {geometry_exc}"
                )
    if snapshot != native.snapshot_coordinates(pose):
        raise ValueError(
            "Metric computation changed input atom identity or coordinates"
        )
    failures.update(raw.get("unavailable_metric_errors", {}))
    metrics = canonical_metrics(raw)
    missing = (
        [target for target in CANONICAL_METRICS.values() if target not in metrics]
        if targets
        else []
    )
    return {
        "status": "complete" if raw else "unavailable",
        "protocol": "complex" if targets else "monomer",
        "metrics": metrics,
        "raw_metrics": raw,
        "unavailable_metrics": missing,
        "metric_errors": failures,
        "native_load": load_info,
        "packing_enabled": bool(targets) and not ncaa,
        "nonstandard_residues": nonstandard_codes(Path(path)),
        "target_chains": targets,
    }


def preset_record(report):
    key = "complex_0" if report["protocol"] == "complex" else "monomer_0"
    return {
        "method": "rosetta",
        "status": report["status"],
        "expected_predictions": [key],
        "predictions": {
            key: {
                "protocol": report["protocol"],
                "status": report["status"],
                "metrics_status": report["status"],
                "metrics": report["metrics"],
            }
        },
    }


def _head_tail_distance(source, binder_chain):
    atoms = [atom for atom in source.atoms if atom["chain"] == binder_chain]
    residues = list(dict.fromkeys((atom["res_id"], atom["ins_code"]) for atom in atoms))
    first, last = residues[0], residues[-1]
    nitrogen = next(atom["xyz"] for atom in atoms if (atom["res_id"],atom["ins_code"]) == first and atom["atom"] == "N")
    carbon = next(atom["xyz"] for atom in atoms if (atom["res_id"],atom["ins_code"]) == last and atom["atom"] == "C")
    return round(math.sqrt(sum((a-b)**2 for a,b in zip(nitrogen,carbon))), 3)


def _compute_metrics(path, *, binder_chain="L", topology_entries=None, compute_buns=True,
                     context=None, required_metrics=None):
    """Score one prepared pose; source and native objects stay within its worker."""
    if context is None and required_metrics is None:
        return _legacy_metrics(path, binder_chain=binder_chain,
                               topology_entries=topology_entries, compute_buns=compute_buns)
    from partiflow.postprocess import native, rosetta
    from partiflow.postprocess.source import PDBSource
    source = getattr(context, "output_source", None)
    if source is None:
        source = PDBSource.read(path)
        if context is not None:
            context.output_source = source
    profile = getattr(context, "sdk_profile", "native_ref2015")
    codes = list(source.nonstandard_codes)
    fixed_native = bool(codes) or (context is not None and profile == "native_ref2015")
    pose = getattr(context, "pose", None)
    load_info = dict(getattr(context, "native_load", {}) or {})
    if pose is None:
        if profile == "standard_beta":
            pose, load_info = rosetta.load_standard_pose(
                source, binder_chain=binder_chain, topology_entries=topology_entries)
        else:
            pose, load_info = native.load_score_only_pose(
                path, binder_chain=binder_chain, topology_entries=topology_entries, source=source)
        if context is not None:
            context.pose, context.native_load = pose, load_info
    if context is not None:
        rosetta.seed_metrics(context.seed)
    before = native.snapshot_coordinates(pose) if fixed_native else None
    chains = rosetta._chain_ids_in_pose(pose)
    targets = "".join(chain for chain in chains if chain != binder_chain)
    score_name = "beta_nov16" if profile == "standard_beta" else "ref2015"
    if profile == "standard_beta":
        scorefxn, adjustments = rosetta.get_scorefxn(score_name), []
    else:
        scorefxn, adjustments = native._score_function_for_pose(pose)
    requested = None if required_metrics is None else set(required_metrics)
    raw_requested = None if requested is None else raw_metric_requirements(requested)
    raw, failures = {}, {}
    component_names = {"fa_atr", "fa_rep", "fa_sol", "fa_elec", "hbond_sr_bb", "hbond_lr_bb",
                       "hbond_bb_sc", "hbond_sc", "rama", "omega", "pro_close", "total"}
    if raw_requested is None or raw_requested & (component_names | {"total_score"}):
        try:
            value = float(scorefxn(pose))
            if not math.isfinite(value):
                raise ValueError("nonfinite native energy")
            if raw_requested is None or "total_score" in raw_requested:
                raw["total_score"] = value
            if raw_requested is None or raw_requested & component_names:
                components, weights, pathology, warnings = native._score_components(scorefxn, pose)
                if pathology:
                    raise ValueError(f"Non-finite Rosetta components: {pathology}")
                raw.update({name: value for name,value in components.items()
                            if raw_requested is None or name in raw_requested})
                if raw_requested is None:
                    load_info.update(score_component_weights=weights,
                                     score_component_warnings=warnings,
                                     scorefxn_adjustments=adjustments)
        except Exception as exc:
            failures["energy"] = f"{type(exc).__name__}: {exc}"
    if targets:
        try:
            interface_requested = None if raw_requested is None else raw_requested-component_names-{"total_score", "binder_head_tail_distance_before", "binder_head_tail_distance_after"}
            if interface_requested is None or interface_requested:
                raw.update(rosetta.score_interface(
                    str(path), binder_chain=binder_chain, target_chains=targets,
                    topology_entries=topology_entries, scorefxn=score_name,
                    compute_buns=compute_buns, relax_before_scoring=False,
                    native_pose=pose, native_no_pack=fixed_native,
                    required_metrics=interface_requested))
        except Exception as exc:
            failures["interface"] = f"{type(exc).__name__}: {exc}"
    if raw_requested is not None:
        for metric, view in (("binder_head_tail_distance_before", getattr(context,"input_source",None)),
                             ("binder_head_tail_distance_after", source)):
            if metric in raw_requested:
                try:
                    if not any(rosetta._normalize_topology_type(e.get("type")) == "head_to_tail"
                               for e in rosetta._decode_topology_entries(topology_entries)):
                        raise ValueError("head-to-tail topology is not requested")
                    if view is None:
                        raise ValueError("original input view is unavailable")
                    raw[metric] = _head_tail_distance(view,binder_chain)
                except Exception as exc:
                    failures[metric] = f"{type(exc).__name__}: {exc}"
    if before is not None and before != native.snapshot_coordinates(pose):
        raise ValueError("Metric computation changed input atom identity or coordinates")
    failures.update(raw.pop("unavailable_metric_errors", {}))
    measured = canonical_metrics(raw)
    required = set(CANONICAL_METRICS.values()) if requested is None else requested
    missing = sorted(required-set(measured)) if targets else []
    for name in missing:
        original = next(iter(raw_metric_requirements([name])))
        failures.setdefault(name, failures.get(original, "required metric unavailable"))
    iam_keys = {"ShapeComplementarity", "n_InterfaceHbonds", "dG", "dSASA", "PackStat", "dG/dSASA",
                "I_sc", "pep_sc", "reweighted_sc", "rank_score", "Interface_SASA_%", "InterfaceHbondsPercentage"}
    packing = bool(targets) and not fixed_native and (raw_requested is None or bool(raw_requested & iam_keys))
    return {"status": "complete" if measured or not required else "unavailable",
            "protocol": "complex" if targets else "monomer", "metrics": measured,
            "raw_metrics": raw, "unavailable_metrics": missing, "metric_errors": failures,
            "native_load": _compact_native_load(load_info,pose,binder_chain,codes)
                           if context is not None and requested is not None else load_info,
            "packing_enabled": packing, "nonstandard_residues": codes,
            "target_chains": targets, "sdk_profile": profile}


def compute_metrics(path, *, binder_chain="L", topology_entries=None, compute_buns=True,
                    context=None, required_metrics=None):
    """Execute within the owning actor; native process failures belong to Ray."""
    return _compute_metrics(path,binder_chain=binder_chain,topology_entries=topology_entries,
                            compute_buns=compute_buns,context=context,
                            required_metrics=required_metrics)
