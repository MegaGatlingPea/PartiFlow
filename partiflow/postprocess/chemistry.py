"""Chemistry routing and production relaxation, loaded only when executing."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any
from partiflow.postprocess.records import decode_topology as _decode_topology_entries
from partiflow.postprocess.records import normalize_type as _normalize_topology_type

def _pdb_atom_lookup(pdb_str: str) -> dict[tuple[str, str, str], tuple[str, tuple[float, float, float]]]:
    atoms: dict[tuple[str, str, str], tuple[str, tuple[float, float, float]]] = {}
    for line in pdb_str.splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        key = (line[21].strip() or "_", line[22:26].strip(), line[12:16].strip())
        atoms[key] = (
            line[17:20].strip(),
            (float(line[30:38]), float(line[38:46]), float(line[46:54])),
        )
    return atoms


def _angstrom_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# These are the explicit CPSet geometry windows used by the evaluator.  They
# intentionally describe the chemically meaningful output interval rather
# than a target distance plus a generic tolerance.  In particular, a CUDA
# minimizer result below the lower bound must be retried on CPU instead of
# being accepted as a successful postprocess.
CPSET_TOPOLOGY_DISTANCE_RANGES: dict[str, tuple[float, float]] = {
    "head_to_tail": (1.1, 1.7),
    "disulfide": (1.7, 2.3),
    "mk8_staple": (1.4, 1.7),
    "nle_staple": (1.2, 1.7),
}


def _validate_topology_bond_distances(
    pdb_str: str,
    applied_entries: list[dict[str, Any]],
    *,
    binder_chain: str,
    tolerance_angstrom: float = 0.35,
) -> list[str]:
    atoms = _pdb_atom_lookup(pdb_str)
    failures: list[str] = []

    def check(
        label: str,
        resid_a: Any,
        atom_a: str,
        resid_b: Any,
        atom_b: str,
        target: float,
        allowed_range: tuple[float, float] | None = None,
    ) -> None:
        key_a = (binder_chain, str(resid_a), atom_a)
        key_b = (binder_chain, str(resid_b), atom_b)
        item_a = atoms.get(key_a)
        item_b = atoms.get(key_b)
        if item_a is None or item_b is None:
            failures.append(f"{label}:missing_atoms:{key_a}:{key_b}")
            return
        observed = _angstrom_distance(item_a[1], item_b[1])
        if allowed_range is not None:
            lower, upper = allowed_range
            # The CPSet ranges are closed intervals: exact lower/upper
            # boundary values are valid, while anything outside must fail.
            if not lower <= observed <= upper:
                failures.append(
                    f"{label}:{observed:.3f}A_required_range_{lower:.3f}-{upper:.3f}A"
                )
        elif abs(observed - target) > tolerance_angstrom:
            failures.append(f"{label}:{observed:.3f}A_target_{target:.3f}A")

    for entry in applied_entries or []:
        entry_type = str(entry.get("type", "")).strip().lower().replace("-", "_")
        anchors = [int(v) for v in entry.get("anchors", []) if int(v) > 0]
        if entry_type == "head_to_tail" and len(anchors) >= 2:
            check(
                "head_to_tail",
                anchors[0],
                "N",
                anchors[1],
                "C",
                1.335,
                CPSET_TOPOLOGY_DISTANCE_RANGES["head_to_tail"],
            )
        elif entry_type == "stapled" and len(anchors) >= 2:
            acceptor_atom = "CD"
            if (binder_chain, str(anchors[1]), acceptor_atom) not in atoms:
                acceptor_atom = "CG"
            check("stapled", anchors[0], "NZ", anchors[1], acceptor_atom, 1.335)
        elif entry_type == "disulfide" and len(anchors) >= 2:
            check(
                "disulfide",
                anchors[0],
                "SG",
                anchors[1],
                "SG",
                2.038,
                CPSET_TOPOLOGY_DISTANCE_RANGES["disulfide"],
            )
        elif entry_type == "bicycle" and len(anchors) >= 3:
            for resid, cm_atom in zip(anchors[:3], ("CM1", "CM2", "CM3")):
                tmb_candidates = [
                    key[1] for key in atoms if key[0] == binder_chain and key[2] == cm_atom
                ]
                tmb_resid = tmb_candidates[0] if tmb_candidates else max(anchors) + 1
                check(f"bicycle_{resid}_{cm_atom}", resid, "SG", tmb_resid, cm_atom, 1.81)
        elif entry_type == "staple_rty" and len(anchors) >= 3:
            check("staple_rty_N_C2", anchors[0], "N", anchors[1], "C2", 1.34)
            check("staple_rty_C_N", anchors[0], "C", anchors[1], "N", 1.34)
            check("staple_rty_N_C6", anchors[2], "N", anchors[1], "C6", 1.46)
        elif entry_type == "mk8_staple" and len(anchors) >= 2:
            check(
                "mk8_staple",
                anchors[0],
                "CE",
                anchors[1],
                "CE",
                1.526,
                CPSET_TOPOLOGY_DISTANCE_RANGES["mk8_staple"],
            )
        elif entry_type == "nle_staple" and len(anchors) >= 2:
            check(
                "nle_staple",
                anchors[0],
                "CE",
                anchors[1],
                "CE",
                1.330,
                CPSET_TOPOLOGY_DISTANCE_RANGES["nle_staple"],
            )
    return failures


def _disulfide_anchor_pairs(topology_entries: Any) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for entry in _decode_topology_entries(topology_entries):
        if _normalize_topology_type(entry.get("type")) != "disulfide":
            continue
        anchors = entry.get("anchors", None)
        if not isinstance(anchors, (list, tuple)):
            anchors = [entry.get("begin_resid"), entry.get("end_resid")]
        parsed: list[int] = []
        for value in anchors:
            try:
                position = int(value)
            except (TypeError, ValueError):
                continue
            if position > 0:
                parsed.append(position)
        if len(parsed) >= 2:
            pairs.append((parsed[0], parsed[1]))
    return pairs


def _normalize_postprocess_method(method: Any) -> str:
    text = str(method or "auto").strip().lower() or "auto"
    aliases = {
        "auto": "auto",
        "rosetta": "rosetta_topology",
        "pyrosetta": "rosetta_topology",
        "pyrosetta_disulfide": "rosetta_topology",
        "rosetta_disulfide": "rosetta_topology",
        "rosetta_topology": "rosetta_topology",
        "rosetta_cyclic": "rosetta_topology",
        "openmm": "openmm",
    }
    return aliases.get(text, text)


def _metadata_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _topology_supported_by_rosetta(topology_entries: Any) -> bool:
    entries = _decode_topology_entries(topology_entries)
    if not entries:
        return False
    supported = {"head_to_tail", "disulfide"}
    for entry in entries:
        entry_type = _normalize_topology_type(entry.get("type", None))
        if entry_type == "residue_template":
            return False
        if entry_type and entry_type not in {"none", *supported}:
            return False
    return any(_normalize_topology_type(entry.get("type", None)) in supported for entry in entries)


def _validate_explicit_postprocess_method(
    *,
    method: str,
    topology_entries: Any,
) -> None:
    """Reject task metadata that requests an incompatible chemistry backend."""
    entries = _decode_topology_entries(topology_entries)
    if not entries:
        return
    active_types = {
        _normalize_topology_type(entry.get("type", None))
        for entry in entries
        if _normalize_topology_type(entry.get("type", None)) != "none"
    }
    if not active_types:
        return
    required_method = (
        "rosetta_topology"
        if _topology_supported_by_rosetta(entries)
        else "openmm"
    )
    normalized_method = _normalize_postprocess_method(method)
    if normalized_method != "auto" and normalized_method != required_method:
        raise ValueError(
            f"Postprocess method '{normalized_method}' is incompatible with topology "
            f"{sorted(active_types)}; use '{required_method}' or 'auto'."
        )


def _resolve_auto_postprocess_method(task: Any) -> str:
    entries = _decode_topology_entries(task.topology_entries)
    if not entries:
        return "rosetta_topology" if task.relax_linear else "openmm"
    if _topology_supported_by_rosetta(entries):
        return "rosetta_topology"
    return "openmm"


def _resolve_rosetta_relax_settings(
    *,
    relax_mode: str = "global",
    relax_subset: str | None = None,
    relax_max_iter: int | None = None,
) -> tuple[str, str, int]:
    mode = str(relax_mode or "global").strip().lower()
    if mode not in {"local", "global"}:
        raise ValueError(
            f"Unsupported relax_mode {relax_mode!r}; expected local or global."
        )
    subset = (
        str(relax_subset).strip().lower()
        if relax_subset is not None and str(relax_subset).strip()
        else ("all" if mode == "global" else "nbrs")
    )
    if subset not in {"all", "nbrs", "target"}:
        raise ValueError(
            f"Unsupported relax_subset {relax_subset!r}; expected all, nbrs, or target."
        )
    max_iter = (
        int(relax_max_iter)
        if relax_max_iter is not None
        else (200 if subset == "all" else 1000)
    )
    if max_iter < 1:
        raise ValueError("relax_max_iter must be >= 1.")
    return mode, subset, max_iter



def relax_with_topology_openmm(
    *,
    pdb_path: str,
    output_path: str,
    topology_entries: Any,
    binder_chain: str = "L",
    platform: str = "CUDA",
    context=None,
    threads: int = 1,
) -> tuple[str, dict]:
    from partiflow.postprocess.relax.topology import TopologyFixer, parse_topology_entries
    from partiflow.postprocess.relax.openmm_relaxer import run_topology_relax

    if context is not None and context.input_source is not None:
        pdb_str = context.input_source.text
    else:
        with open(pdb_path, "r") as handle:
            pdb_str = handle.read()

    entries = parse_topology_entries(_decode_topology_entries(topology_entries))
    fix_result = TopologyFixer()(pdb_str, binder_chain, entries)
    pdb_min, energy_info = run_topology_relax(fix_result, platform=platform, threads=threads)
    validation_failures = _validate_topology_bond_distances(
        pdb_min,
        fix_result.applied_entries,
        binder_chain=binder_chain,
    )
    postprocess_notes = list(fix_result.notes)
    postprocess_warnings = list(fix_result.warnings)
    if validation_failures and platform.upper() != "CPU":
        postprocess_warnings.append(
            f"{platform} topology bond validation failed; retrying on CPU: {validation_failures}"
        )
        pdb_min, energy_info = run_topology_relax(fix_result, platform="CPU", threads=threads)
        postprocess_notes.append(f"cpu_retry_after_{platform.lower()}_validation_failure")
        validation_failures = _validate_topology_bond_distances(
            pdb_min,
            fix_result.applied_entries,
            binder_chain=binder_chain,
        )
    if validation_failures:
        raise ValueError(f"Topology bond validation failed after relax: {validation_failures}")

    with open(output_path, "w") as handle:
        handle.write(pdb_min)
    if context is not None:
        from partiflow.postprocess.source import PDBSource
        context.output_source = PDBSource.from_text(pdb_min, path=output_path)
        context.pose = None
        context.native_load = {}
        context.backend = "openmm"

    info = {
        "einit": energy_info["einit"],
        "efinal": energy_info["efinal"],
        "postprocess_note": ";".join(postprocess_notes),
        "applied_topology_entries": fix_result.applied_entries,
        "skipped_topology_entries": fix_result.skipped_entries,
        "force_fixed_residues": fix_result.force_fixed_residues,
        "postprocess_warnings": postprocess_warnings,
    }
    return output_path, info



def relax_with_rosetta(*, pdb_path, output_path, topology_entries, binder_chain="L",
                       relax_mode="global", relax_subset=None, relax_max_iter=None,
                       relax_repeats=1, seed=2026, threads=1, context=None):
    from partiflow.postprocess import rosetta
    from partiflow.postprocess.source import PDBSource
    rosetta.initialize_standard_beta(seed=seed, threads=threads)
    source = context.input_source if context is not None else PDBSource.read(pdb_path)
    pose, loaded = rosetta.load_standard_pose(source, binder_chain=binder_chain,
        topology_entries=topology_entries)
    cyclization, disulfides = loaded["cyclization"], loaded["disulfides"]
    mode, subset, iterations = _resolve_rosetta_relax_settings(
        relax_mode=relax_mode, relax_subset=relax_subset, relax_max_iter=relax_max_iter)
    rosetta.fast_relax_pose2(pose, binder_chains=binder_chain, scorefxn="beta_nov16",
        repeats=relax_repeats, max_iter=iterations, subset=subset, move_bb=True,
        cyclic_first_idx=cyclization["first_pose_idx"] if cyclization else None,
        cyclic_last_idx=cyclization["last_pose_idx"] if cyclization else None)
    if cyclization:
        rosetta._refresh_head_tail_geometry_after_relax(pose, cyclization)
    if disulfides:
        disulfides = rosetta.validate_declared_disulfides(pose, disulfides)
    pose.dump_pdb(str(output_path))
    rosetta.clean_pdb(str(output_path))
    if disulfides:
        rosetta.append_disulfide_records(str(output_path), disulfide_info=disulfides)
    if cyclization:
        rosetta.append_head_tail_conect(str(output_path), cyclization_info=cyclization)
    if context is not None:
        context.pose = pose
        context.output_source = PDBSource.read(output_path)
        context.backend = "rosetta_topology"
    return str(output_path), {"applied_topology_entries": topology_entries,
        "relax_mode": mode, "relax_subset": subset, "relax_max_iter": iterations,
        "relax_repeats": relax_repeats, "scorefunction": "beta_nov16"}


def relax_candidate(candidate, output_path, *, platform="CPU", method="auto",
                    relax_mode="global", relax_subset=None, relax_max_iter=None,
                    relax_repeats=1, seed=2026, threads=1, context=None):
    from partiflow.postprocess.records import nonstandard_codes, SUPPORTED_TOPOLOGIES
    entries = _decode_topology_entries(candidate.topology_entries)
    unknown = sorted({_normalize_topology_type(e.get("type")) for e in entries} - SUPPORTED_TOPOLOGIES)
    if unknown:
        raise ValueError(f"Unsupported topology entries: {unknown}")
    requested = _normalize_postprocess_method(candidate.method or method)
    if requested not in {"auto", "openmm", "rosetta_topology"}:
        raise ValueError(f"Unsupported postprocess method: {requested}")
    ncaa = nonstandard_codes(Path(candidate.path), source=context.input_source if context is not None else None)
    needs_openmm = bool(ncaa) or any(_normalize_topology_type(e.get("type")) not in
        {"none", "head_to_tail", "disulfide"} for e in entries)
    if requested == "rosetta_topology" and needs_openmm:
        raise ValueError("Explicit Rosetta relaxation does not support these nonstandard residues/topologies")
    if requested != "auto" and not needs_openmm:
        _validate_explicit_postprocess_method(method=requested, topology_entries=entries)
    resolved = ("openmm" if needs_openmm else _resolve_auto_postprocess_method(candidate)) if requested == "auto" else requested
    def execute(backend):
        if backend == "openmm":
            return relax_with_topology_openmm(pdb_path=str(candidate.path), output_path=str(output_path),
                topology_entries=entries, binder_chain=candidate.binder_chain, platform=platform,
                context=context, threads=threads)
        return relax_with_rosetta(pdb_path=str(candidate.path), output_path=str(output_path),
            topology_entries=entries, binder_chain=candidate.binder_chain, relax_mode=relax_mode,
            relax_subset=relax_subset, relax_max_iter=relax_max_iter,
            relax_repeats=relax_repeats, seed=seed, threads=threads, context=context)
    try:
        result, info = execute(resolved)
    except Exception as exc:
        if requested != "auto" or resolved != "rosetta_topology":
            raise
        result, info = execute("openmm")
        info["rosetta_error"] = f"{type(exc).__name__}: {exc}"
        resolved = "openmm"
    return Path(result), {**info, "method": resolved}
