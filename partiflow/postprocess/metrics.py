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


def _compute_metrics(
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


def compute_metrics(
    path, *, binder_chain="L", topology_entries=None, compute_buns=True
):
    """Keep native-library failures local while preserving completed geometry."""
    if not nonstandard_codes(Path(path)):
        return _compute_metrics(
            path,
            binder_chain=binder_chain,
            topology_entries=topology_entries,
            compute_buns=compute_buns,
        )
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from partiflow.postprocess.records import pdb_atoms
    from partiflow.postprocess.runner import _json_value

    targets = {atom["chain"] for atom in pdb_atoms(path)} - {binder_chain}
    protocol = "complex" if targets else "monomer"
    with tempfile.TemporaryDirectory(prefix="partiflow-metrics-") as temporary:
        request = Path(temporary) / "input.json"
        output = Path(temporary) / "output.json"
        value = dict(
            path=str(Path(path).resolve()),
            binder_chain=binder_chain,
            topology_entries=topology_entries,
            compute_buns=compute_buns,
        )
        request.write_text(json.dumps(_json_value(value), allow_nan=False))
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "partiflow.postprocess.metric_worker",
                    str(request),
                    str(output),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                capture_output=True,
                text=True,
                timeout=3600,
            )
            if process.returncode == 0 and output.is_file():
                return json.loads(output.read_text())
            reason = (
                f"Native metric worker exited {process.returncode}: "
                + process.stderr[-2000:]
            )
        except subprocess.TimeoutExpired:
            reason = "Native metric calculation exceeded 3600 seconds"
        return {
            "status": "unavailable",
            "protocol": protocol,
            "metrics": {},
            "raw_metrics": {},
            "metric_errors": {"native_worker": reason},
        }
