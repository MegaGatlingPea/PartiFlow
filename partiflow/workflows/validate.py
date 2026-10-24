"""Resumable task-local AF2 initial-guess complex/monomer validation."""

from __future__ import annotations
import csv
import json
import os
import re
from pathlib import Path
from partiflow.common.records import Candidate, CandidateError, WorkflowError
from partiflow.common.results import read_results
from partiflow.paths import config_dir, model_root
from .sequence import normalize_sources, validate_sequence_config, sequence_settings


def _merge(a, b):
    result = dict(a)
    for key, val in b.items():
        result[key] = (
            _merge(result[key], val)
            if isinstance(val, dict) and isinstance(result.get(key), dict)
            else val
        )
    return result


def _load(args):
    import yaml

    config = yaml.safe_load((config_dir() / "validate.yaml").read_text())
    path = getattr(args, "config", None)
    if path:
        raw = yaml.safe_load(Path(path).read_text())
        if not isinstance(raw, dict):
            raise ValueError("Validation configuration must be a mapping")
        config = _merge(config, raw)
    if config.get("methods") != ["af2_ig"]:
        raise ValueError("This validator supports methods: [af2_ig]")
    unknown = set(config) - {"methods", "sequence", "af2", "criteria", "design_chain"}
    if unknown:
        raise ValueError(f"Unknown validation settings: {sorted(unknown)}")
    unknown_af2 = set(config["af2"]) - {
        "models",
        "samples",
        "recycles",
        "seed",
        "mode",
        "params_dir",
        "source_path",
        "colabdesign_source",
    }
    if unknown_af2:
        raise ValueError(f"Unknown AF2 validation settings: {sorted(unknown_af2)}")
    sequence = config.get("sequence", {})
    if type(sequence.get("seed", 101)) is not int or sequence.get("seed", 101) < 0:
        raise ValueError("sequence.seed must be a nonnegative integer")
    if getattr(args, "source", None):
        entries = []
        for value in args.source:
            try:
                source, count = value.rsplit(":", 1)
                count = int(count)
            except (ValueError, AttributeError) as exc:
                raise ValueError("Use --source SOURCE:COUNT") from exc
            entries.append(dict(source=source, count=count))
        sequence = {**sequence, "sources": entries}
    sequence = validate_sequence_config(sequence, reward=False)
    config["sequence"] = sequence
    if getattr(args, "criteria", None) is not None:
        config["criteria"] = args.criteria
    if not isinstance(config.get("criteria", []), list):
        raise ValueError("criteria must be a list")
    if config["af2"].get("mode", "both") != "both":
        raise ValueError("AF2 validation requires paired complex and monomer protocols")
    return config


def prediction_keys(config):
    models = config.get("models", [1])
    samples = config.get("samples", 1)
    recycles = config.get("recycles", 3)
    if (
        not isinstance(models, list)
        or not models
        or any((type(v) is not int or v not in (1, 2) for v in models))
        or (len(set(models)) != len(models))
    ):
        raise ValueError("AF2 models must be ordered unique subset of [1,2]")
    if (
        type(samples) is not int
        or samples < 1
        or type(recycles) is not int
        or (recycles < 1)
    ):
        raise ValueError("AF2 samples/recycles must be positive integers")
    if type(config.get("seed", 101)) is not int:
        raise ValueError("AF2 seed must be an integer")
    return [
        f"{protocol}_model{model}_sample{sample}"
        for protocol in ("complex", "monomer")
        for model in models
        for sample in range(samples)
    ]


def _json_list(value, name):
    if not value:
        return []
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or any(
        (not isinstance(item, dict) for item in parsed)
    ):
        raise CandidateError(f"{name} must be JSON object array")
    return parsed


def _sample_name(path, index):
    if path:
        stem = Path(path).stem
        if (
            stem
            and stem not in (".", "..")
            and all((c.isalnum() or c in "._-" for c in stem))
        ):
            return stem
    return f"sample{index}"


def _candidate(task, row, index):
    filename = row.get("filename", "")
    path = (
        (task / filename).resolve()
        if filename
        else task / f".missing_sample{index}.pdb"
    )
    design_chain = row.get("design_chain_id") or "L"
    if row.get("design_modality") not in (None, "", "protein", "peptide"):
        raise CandidateError("AF2 requires protein design modality")
    declarations = _json_list(row.get("design_chains", ""), "design_chains")
    if declarations:
        if len(declarations) != 1 or declarations[0].get("modality", "protein") not in (
            "protein",
            "peptide",
        ):
            raise CandidateError("AF2 requires exactly one protein design chain")
        design_chain = declarations[0].get("chain", design_chain)
    sequence = row.get("design_sequence", "")
    if declarations and (not sequence):
        sequence = declarations[0].get("sequence", "")
    if (
        declarations
        and row.get("design_sequence")
        and declarations[0].get("sequence")
        and (row["design_sequence"] != declarations[0]["sequence"])
    ):
        raise CandidateError("Scalar and declared design sequences disagree")
    length = int(row.get("token_length") or len(sequence))
    base = int(row.get("base_sample_id") or index)
    name = _sample_name(filename, index)
    metadata, topology = _task_metadata(task)
    metadata["input_error"] = row.get("error", "")
    return Candidate(
        name,
        task.name,
        base,
        index,
        1,
        path,
        str(design_chain),
        length,
        sequence,
        contig_string=row.get("contig_string", ""),
        tip_atoms=row.get("tip_atoms", ""),
        fixed_restypes=_json_list(row.get("fixed_restypes", ""), "fixed_restypes"),
        bond_constraints=_json_list(
            row.get("bond_constraints", ""), "bond_constraints"
        ),
        topology_entries=topology,
        metadata=metadata,
    )


def _task_metadata(task):
    import yaml

    path = task / "resolved_config.yaml"
    saved = yaml.safe_load(path.read_text()) if path.is_file() else {}
    tasks = (saved or {}).get("tasks", {})
    value = tasks.get(task.name, {}) if isinstance(tasks, dict) else {}
    topology = value.get("topology", []) or []
    cyclic = bool(
        value.get("cyclic", False)
        or value.get("cyclic_offset", False)
        or value.get("sampler") in ("head_to_tail", "head_tail")
        or any(
            (
                isinstance(entry, dict) and entry.get("type") == "head_tail"
                for entry in topology
            )
        )
    )
    return ({"cyclic_offset": cyclic}, topology)


def _tasks(paths, selected, config):
    discovered = {}
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        files = (
            ([path] if path.name == "result.csv" else list(path.rglob("result.csv")))
            if path.is_dir() or path.name == "result.csv"
            else []
        )
        if (
            not files
            and path.is_file()
            and (path.suffix.lower() in (".pdb", ".cif", ".mmcif"))
        ):
            task = path.parent
            row = dict(
                filename=path.name,
                design_chain_id=config.get("design_chain", "L"),
                design_sequence="",
                token_length="",
            )
            discovered.setdefault(task, []).append(row)
        elif not files:
            raise ValueError(f"No result.csv or structure inputs found: {path}")
        for file in sorted(files):
            relative = file.relative_to(path if path.is_dir() else path.parent).parts
            if any(
                (
                    relative[index : index + 2] == ("validate", "af2_ig")
                    for index in range(len(relative) - 1)
                )
            ):
                continue
            if file.parent in discovered:
                continue
            _, rows = read_results(file)
            discovered[file.parent] = rows
    result = []
    for task, rows in sorted(discovered.items(), key=lambda item: str(item[0])):
        if selected and task.name not in selected:
            continue
        result.append((task, rows))
    if not result:
        raise ValueError("No matching validation tasks")
    return result


def _sequence_input_files(config, defaults, model_dir):
    from .sequence import backend_settings

    paths = {}
    for source in config["sequence"]["sources"]:
        if source["source"] == "gt":
            continue
        settings = sequence_settings(config["sequence"], source["settings"])
        effective = backend_settings(
            source["source"], {**defaults, **config["sequence"]}, model_dir, settings
        )
        for key in ("pssm_path", "tied_positions_path"):
            raw = effective.get(key)
            if not raw:
                continue
            path = Path(raw).expanduser().resolve()
            if not path.is_file():
                raise WorkflowError(f"Sequence constraint input is missing: {path}")
            stat = path.stat()
            paths[str(path)] = {
                "path": str(path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
    return [paths[key] for key in sorted(paths)]


def _structure_state(path):
    from math import isfinite
    from biotite import DeserializationError, InvalidFileError
    from .structure import read_structure

    path = Path(path)
    result = {}
    try:
        if path.suffix.lower() in (".cif", ".mmcif"):
            from biotite.structure.io import pdbx

            block = next(iter(pdbx.CIFFile.read(path).values()))
            result["tables"] = {
                category: {
                    name: block[category][name].as_array(str).tolist()
                    for name in sorted(block[category])
                }
                for category in (
                    "atom_site",
                    "struct_conn",
                    "chem_comp_atom",
                    "chem_comp_bond",
                )
                if category in block
            }
        else:
            result["records"] = [
                line
                for line in path.read_text().splitlines()
                if line.startswith(
                    (
                        "ATOM  ",
                        "HETATM",
                        "CONECT",
                        "LINK  ",
                        "SSBOND",
                        "TER",
                        "MODEL ",
                        "ENDMDL",
                    )
                )
            ]
        atoms = read_structure(path)
        result.update(
            coordinates=atoms.coord.tolist(),
            annotations={
                name: [
                    str(value)
                    if isinstance(value, float) and not isfinite(value)
                    else value
                    for value in atoms.get_annotation(name).tolist()
                ]
                for name in sorted(atoms.get_annotation_categories())
            },
            bonds=atoms.bonds.as_array().tolist() if atoms.bonds is not None else None,
        )
    except (
        OSError,
        ValueError,
        KeyError,
        IndexError,
        TypeError,
        StopIteration,
        DeserializationError,
        InvalidFileError,
    ) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _input_snapshot(candidate, row):
    path = Path(candidate.pdb_path)
    stat = path.stat() if path.is_file() else None
    return dict(
        candidate_id=candidate.candidate_id,
        base_sample_id=candidate.base_sample_id,
        candidate_ordinal=candidate.candidate_ordinal,
        path=str(path),
        size=stat.st_size if stat else None,
        mtime_ns=stat.st_mtime_ns if stat else None,
        structure=_structure_state(path) if stat else None,
        sequence=candidate.sequence,
        length=candidate.length,
        binder_chain=candidate.binder_chain,
        context_chains=list(candidate.context_chains),
        fixed_restypes=candidate.fixed_restypes,
        bond_constraints=candidate.bond_constraints,
        topology_entries=candidate.topology_entries,
        metadata=candidate.metadata,
        contig_string=candidate.contig_string,
        tip_atoms=candidate.tip_atoms,
    )


def _atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    try:
        with temporary.open("w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _valid_prediction(prediction):
    if not isinstance(prediction, dict) or prediction.get("status") != "complete":
        return False
    for field in ("structure_path", "confidence_path"):
        path = Path(prediction.get(field, ""))
        if not path.is_file() or path.stat().st_size < 1:
            return False
    return True


def _slots(sources):
    index = 0
    for source in sources:
        for source_index in range(1, source["count"] + 1):
            yield (index, source, source_index)
            index += 1


def _budget(task):
    import yaml

    path = task / "resolved_config.yaml"
    saved = yaml.safe_load(path.read_text()) if path.is_file() else {}
    saved = saved or {}
    runtime = saved.get("_runtime", {})
    return {
        key: runtime.get(key)
        for key in ("planned_base_samples", "replicas_per_base", "planned_candidates")
    }


def _write_csv(path, rows, preferred):
    rows = list(rows)
    fields = list(preferred)
    for key in sorted(set().union(*(row.keys() for row in rows)) if rows else set()):
        if key not in fields:
            fields.append(key)
    path = Path(path)
    temporary = path.with_name("." + path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _refresh_metrics(candidate, data, record):
    import numpy as np
    from .metrics import prediction_metrics, paired_metrics

    predictions = record["predictions"]
    complete = True
    for key in record["expected_predictions"]:
        prediction = predictions.get(key, {})
        if not _valid_prediction(prediction):
            complete = False
            continue
        try:
            with np.load(prediction["confidence_path"], allow_pickle=False) as arrays:
                if "pae_matrix" not in arrays:
                    raise ValueError("Persisted PAE matrix is missing")
                pae = arrays["pae_matrix"]
                if (
                    pae.ndim != 2
                    or pae.shape[0] != pae.shape[1]
                    or (not np.isfinite(pae).all())
                ):
                    raise ValueError("Invalid persisted PAE matrix")
            metrics, reasons = prediction_metrics(candidate, data, prediction)
            required = {"plddt", "ptm", "pae_normalized", "pae_angstrom"}
            if prediction.get("protocol") == "complex":
                required.update({"iptm", "ipae_normalized", "ipae_angstrom"})
            if not required.issubset(metrics) or any(
                (not np.isfinite(metrics[name]) for name in required)
            ):
                raise ValueError("Persisted AF2 confidence is incomplete or non-finite")
            if any(
                (
                    not 0.0 <= metrics[name] <= 1.0
                    for name in required
                    & {"plddt", "ptm", "iptm", "pae_normalized", "ipae_normalized"}
                )
            ):
                raise ValueError(
                    "Persisted AF2 confidence is outside native metric ranges"
                )
            prediction.update(
                metrics=metrics, metric_reasons=reasons, metrics_status="complete"
            )
        except (OSError, ValueError, KeyError) as exc:
            prediction.update(
                metrics={}, metric_reasons={"error": str(exc)}, metrics_status="failed"
            )
            complete = False
    pairs = {}
    pair_reasons = {}
    for key in record["expected_predictions"]:
        if not key.startswith("complex_"):
            continue
        suffix = key.removeprefix("complex_")
        complex_pred = predictions.get(key, {})
        monomer_pred = predictions.get("monomer_" + suffix, {})
        if _valid_prediction(complex_pred) and _valid_prediction(monomer_pred):
            value, reasons = paired_metrics(
                candidate, record["sequence"], complex_pred, monomer_pred
            )
            if value:
                pairs[suffix] = value
            else:
                complete = False
            if reasons:
                pair_reasons[suffix] = reasons
        else:
            complete = False
    record.update(
        paired_metrics=pairs,
        paired_metric_reasons=pair_reasons,
        status="complete" if complete else "incomplete",
    )
    return record


def _report(root, records, criteria, presets, budget, observed):
    from partiflow.common.presets import apply_presets

    metric_rows = []
    success_rows = []
    complete = 0
    passing = 0
    for path, record in records:
        apply_presets(record, criteria, presets)
        _atomic_json(path, record)
        complete += record.get("status") == "complete"
        identity = dict(
            sample=record["candidate_id"],
            base_sample_id=record.get("base_sample_id", ""),
            sequence_index=record["seq_index"],
            source=record["source"],
            source_label=record.get("source_label", ""),
            source_index=record["source_index"],
        )
        for key, prediction in record.get("predictions", {}).items():
            metric_rows.append(
                dict(
                    **identity,
                    prediction=key,
                    protocol=prediction.get("protocol", ""),
                    status=prediction.get("status", ""),
                    metrics_status=prediction.get("metrics_status", ""),
                    **prediction.get("metrics", {}),
                    error=prediction.get("error", ""),
                )
            )
        if not record.get("predictions"):
            metric_rows.append(
                dict(
                    **identity,
                    prediction="",
                    protocol="",
                    status=record.get("status", "incomplete"),
                    metrics_status="",
                    error=record.get("error", ""),
                )
            )
        for key, metrics in record.get("paired_metrics", {}).items():
            metric_rows.append(
                dict(
                    **identity,
                    prediction="paired_" + key,
                    protocol="paired",
                    status="complete",
                    metrics_status="complete",
                    **metrics,
                    error="",
                )
            )
        for name, decision in record.get("decisions", {}).items():
            if decision["status"] == "passed":
                passing += 1
            success_rows.append(
                dict(
                    **identity,
                    criteria=name,
                    status=decision["status"],
                    reasons="; ".join(decision.get("reasons", [])),
                )
            )
    _write_csv(
        root / "metrics.csv",
        metric_rows,
        [
            "sample",
            "base_sample_id",
            "sequence_index",
            "source",
            "source_label",
            "source_index",
            "prediction",
            "protocol",
            "status",
            "metrics_status",
        ],
    )
    _write_csv(
        root / "success.csv",
        success_rows,
        [
            "sample",
            "base_sample_id",
            "sequence_index",
            "source",
            "source_label",
            "source_index",
            "criteria",
            "status",
            "reasons",
        ],
    )
    grouped = {}
    for _, record in records:
        grouped.setdefault(record["candidate_id"], []).append(record)
    complete_candidates = sum(
        (
            all((slot.get("status") == "complete" for slot in slots))
            for slots in grouped.values()
        )
    )
    criteria_counts = {
        name: {
            status: sum(
                (
                    record.get("decisions", {}).get(name, {}).get("status") == status
                    for _, record in records
                )
            )
            for status in ("passed", "failed", "unavailable", "not_applicable")
        }
        for name in criteria
    }
    summary = dict(
        method="af2_ig",
        complete_candidates=complete_candidates,
        criteria_counts=criteria_counts,
        **budget,
        observed_result_rows=observed,
        planned_sequence_slots=budget["planned_candidates"]
        * len({record["seq_index"] for _, record in records})
        if budget.get("planned_candidates") is not None
        else None,
        requested_sequence_slots=observed
        * len({record["seq_index"] for _, record in records}),
        observed_sequence_slots=len(records),
        complete_sequence_slots=complete,
        passing_criteria_slots=passing,
        criteria=criteria,
        metrics_only=not criteria,
    )
    _atomic_json(root / "summary.json", summary)
    return summary


def run(args):
    config = _load(args)
    model_dir = model_root(getattr(args, "model_dir", None))
    criteria = list(config.get("criteria", []))
    from partiflow.common.presets import load_presets, validate_criteria

    presets = load_presets(
        getattr(args, "presets", None) or config_dir() / "preset.yaml"
    )
    validate_criteria(presets, criteria, "af2_ig")
    expected = prediction_keys(config["af2"])
    tasks = _tasks(args.paths, getattr(args, "task", None), config)
    plans = []
    limited = getattr(args, "limit", None)
    if limited is not None and (type(limited) is not int or limited < 1):
        raise ValueError("--limit must be a positive integer")
    rerun = bool(getattr(args, "rerun", False))
    refilter = bool(getattr(args, "refilter", False))
    recompute = bool(getattr(args, "recompute_metrics", False))
    if rerun and (refilter or recompute):
        raise ValueError("--rerun cannot be combined with model-free cache updates")
    runtime_config = {key: value for key, value in config.items() if key != "criteria"}
    for task, rows in tasks:
        if limited is not None:
            rows = rows[:limited]
        candidates = []
        for index, row in enumerate(rows, 1):
            try:
                candidate = _candidate(task, row, index)
            except (CandidateError, ValueError) as exc:
                candidate = Candidate(
                    f"sample{index}",
                    task.name,
                    index,
                    index,
                    1,
                    task / f".invalid_sample{index}.pdb",
                    "L",
                    0,
                    "",
                    metadata={"input_error": str(exc)},
                )
            candidates.append((candidate, row))
        if len({candidate.candidate_id for candidate, row in candidates}) != len(
            candidates
        ):
            raise ValueError("Duplicate sample filenames would share validation slots")
        root = task / "validate" / "af2_ig"
        state_path = root / ".state" / "config.json"
        defaults = __import__("yaml").safe_load(
            (config_dir() / "sequence.yaml").read_text()
        )
        state = dict(
            config=runtime_config,
            sequence_defaults=defaults,
            sequence_input_files=_sequence_input_files(config, defaults, model_dir),
            budget=_budget(task),
            model_dir=str(model_dir),
            inputs=[_input_snapshot(candidate, row) for candidate, row in candidates],
        )
        saved = _read_json(state_path)
        if saved is not None and saved != state and (not rerun):
            raise WorkflowError(
                f"Validation inputs/settings changed for {task.name}; use --rerun"
            )
        if (
            saved is None
            and root.exists()
            and any(root.glob("refolds/*/seq*.json"))
            and (not rerun)
        ):
            raise WorkflowError(
                f"Validation cache has no comparable saved settings: {task.name}; use --rerun"
            )
        if not rerun:
            for candidate, row in candidates:
                for slot, source, source_index in _slots(config["sequence"]["sources"]):
                    slot_path = (
                        root / "refolds" / candidate.candidate_id / f"seq{slot}.json"
                    )
                    old = _read_json(slot_path)
                    if slot_path.exists() and (not isinstance(old, dict)):
                        raise WorkflowError(
                            f"Invalid saved sequence state: {slot_path}; use --rerun"
                        )
                    if old is None:
                        continue
                    wanted = dict(
                        method="af2_ig",
                        candidate_id=candidate.candidate_id,
                        seq_index=slot,
                        source=source["source"],
                        source_label=source["label"],
                        source_index=source_index,
                        expected_predictions=expected,
                    )
                    if any((old.get(key) != value for key, value in wanted.items())):
                        raise WorkflowError(
                            f"Saved sequence source/settings differ: {slot_path}; use --rerun"
                        )
                    sequence = old.get("sequence", "")
                    if sequence:
                        if (
                            not isinstance(sequence, str)
                            or any(
                                (aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in sequence)
                            )
                            or (candidate.length and len(sequence) != candidate.length)
                        ):
                            raise WorkflowError(
                                f"Invalid saved sequence: {slot_path}; use --rerun"
                            )
                        if (
                            source["source"] == "gt"
                            and candidate.sequence
                            and (sequence != candidate.sequence)
                        ):
                            raise WorkflowError(
                                f"Saved self sequence differs: {slot_path}; use --rerun"
                            )
                        if (
                            source["source"] != "gt"
                            and candidate.sequence
                            and candidate.pdb_path.is_file()
                        ):
                            from .structure import candidate_input
                            from .sequence import fixed_positions

                            data = candidate_input(candidate)
                            positions = fixed_positions(
                                candidate,
                                data,
                                {
                                    **config["sequence"].get("settings", {}),
                                    **source["settings"],
                                },
                            )
                            if old.get("fixed_positions") != positions or any(
                                (
                                    sequence[p - 1] != candidate.sequence[p - 1]
                                    for p in positions
                                )
                            ):
                                raise WorkflowError(
                                    f"Saved fixed positions/identities differ: {slot_path}; use --rerun"
                                )
        plans.append((task, rows, candidates, root, state_path, state))
    plan_summary = [
        dict(
            task=str(task),
            method="af2_ig",
            inputs=len(candidates),
            requested_sequence_slots=len(candidates)
            * sum((source["count"] for source in config["sequence"]["sources"])),
            observed_sequence_slots=len(candidates)
            * sum((s["count"] for s in config["sequence"]["sources"])),
            planned_sequence_slots=_budget(task)["planned_candidates"]
            * sum((s["count"] for s in config["sequence"]["sources"]))
            if _budget(task).get("planned_candidates") is not None
            else None,
            predictions_per_slot=len(expected),
            **_budget(task),
        )
        for task, rows, candidates, root, state_path, state in plans
    ]
    if getattr(args, "plan", False):
        print(json.dumps({"tasks": plan_summary, "criteria": criteria}, indent=2))
        return 0
    from .structure import candidate_input, polymer_residue_groups, AA3_TO_1
    from .sequence import MPNNBackend, backend_settings, fixed_positions
    from .af2 import AF2Backend, validate_capability

    backend = None
    mpnn = None
    all_complete = True
    try:
        for task, rows, candidates, root, state_path, state in plans:
            root.mkdir(parents=True, exist_ok=True)
            _atomic_json(state_path, state)
            _atomic_json(
                root / ".state" / "plan.json",
                next((item for item in plan_summary if item["task"] == str(task))),
            )
            records = []
            sequence_defaults = __import__("yaml").safe_load(
                (config_dir() / "sequence.yaml").read_text()
            )
            for (candidate, row), input_state in zip(candidates, state["inputs"]):
                data = None
                candidate_error = None
                candidate_error_code = ""
                candidate_error_details = {}
                try:
                    structure_error = (input_state.get("structure") or {}).get("error")
                    if structure_error:
                        raise ValueError(structure_error)
                    if not candidate.sequence and candidate.pdb_path.is_file():
                        from .structure import read_structure

                        source_atoms = read_structure(candidate.pdb_path)
                        groups = polymer_residue_groups(
                            source_atoms, candidate.binder_chain
                        )
                        candidate.sequence = "".join(
                            (AA3_TO_1.get(g["res_name"], "X") for g in groups)
                        )
                        candidate.length = len(groups)
                    data = candidate_input(candidate)
                    validate_capability(data)
                except (CandidateError, ValueError, OSError) as exc:
                    candidate_error = str(exc)
                    if getattr(exc, "code", "") == "unsupported_chemistry":
                        candidate_error_code = exc.code
                        candidate_error_details = exc.details
                sample_dir = root / "refolds" / candidate.candidate_id
                for index, source, source_index in _slots(
                    config["sequence"]["sources"]
                ):
                    path = sample_dir / f"seq{index}.json"
                    old = None if rerun else _read_json(path)
                    record = (
                        old
                        if isinstance(old, dict)
                        else dict(
                            method="af2_ig",
                            candidate_id=candidate.candidate_id,
                            base_sample_id=candidate.base_sample_id,
                            seq_index=index,
                            source=source["source"],
                            source_label=source["label"],
                            source_index=source_index,
                            expected_predictions=expected,
                            predictions={},
                            sequence="",
                            status="incomplete",
                        )
                    )
                    records.append((path, record))
                    if candidate_error:
                        record.update(
                            status="not_applicable"
                            if candidate_error_code
                            else "failed",
                            error=candidate_error,
                            error_code=candidate_error_code,
                            error_details=candidate_error_details,
                        )
                        _atomic_json(path, record)
                        all_complete = False
                        continue
                    if refilter and (not recompute):
                        all_complete &= record.get("status") == "complete"
                        pairs = {
                            key.removeprefix("complex_")
                            for key in expected
                            if key.startswith("complex_")
                        }
                        if (
                            any(
                                (
                                    not _valid_prediction(
                                        record.get("predictions", {}).get(key)
                                    )
                                    or record["predictions"][key].get("metrics_status")
                                    != "complete"
                                    for key in expected
                                )
                            )
                            or set(record.get("paired_metrics", {})) != pairs
                        ):
                            record["status"] = "incomplete"
                            all_complete = False
                        continue
                    if recompute:
                        _refresh_metrics(candidate, data, record)
                        _atomic_json(path, record)
                        all_complete &= record["status"] == "complete"
                        continue
                    if not record.get("sequence"):
                        seed = (
                            config["sequence"].get("seed", 101)
                            + candidate.candidate_ordinal * 100003
                            + index
                        ) % 2147483647
                        try:
                            settings = sequence_settings(
                                config["sequence"], source["settings"]
                            )
                            positions = fixed_positions(candidate, data, settings)
                            if source["source"] == "gt":
                                sequence = candidate.sequence
                            else:
                                settings = backend_settings(
                                    source["source"],
                                    {**sequence_defaults, **config["sequence"]},
                                    model_dir,
                                    settings,
                                )
                                if mpnn is None:
                                    mpnn = MPNNBackend()
                                sequence = mpnn.generate(
                                    candidate, data, settings, positions, seed
                                )
                            record.update(
                                sequence=sequence,
                                sequence_seed=seed,
                                fixed_positions=positions,
                            )
                            _atomic_json(path, record)
                        except CandidateError as exc:
                            record.update(status="failed", error=str(exc))
                            _atomic_json(path, record)
                            all_complete = False
                            continue
                    completed = {
                        key
                        for key in expected
                        if _valid_prediction(record["predictions"].get(key))
                    }
                    if len(completed) < len(expected):
                        if backend is None:
                            backend = AF2Backend(model_dir, config["af2"])
                        variant = dict(
                            seq_index=index,
                            variant_id=f"seq{index}",
                            design_sequence=record["sequence"],
                            source=source["source"],
                            source_index=source_index,
                        )
                        try:
                            for prediction in backend.predict(
                                data, variant, sample_dir, completed
                            ):
                                arrays = prediction.pop("_confidence_arrays", None)
                                if prediction.get("status") == "complete":
                                    import numpy as np

                                    array_path = sample_dir / (
                                        Path(prediction["structure_path"]).stem + ".npz"
                                    )
                                    temporary = array_path.with_name(
                                        "." + array_path.name + ".tmp"
                                    )
                                    with temporary.open("wb") as handle:
                                        np.savez_compressed(handle, **arrays)
                                    os.replace(temporary, array_path)
                                    prediction["confidence_path"] = str(array_path)
                                record["predictions"][prediction["key"]] = prediction
                                _atomic_json(path, record)
                        except CandidateError as exc:
                            record.update(status="failed", error=str(exc))
                            _atomic_json(path, record)
                            all_complete = False
                            continue
                    _refresh_metrics(candidate, data, record)
                    _atomic_json(path, record)
                    all_complete &= record["status"] == "complete"
                _report(root, records, criteria, presets, _budget(task), len(rows))
            summary = _report(
                root, records, criteria, presets, _budget(task), len(rows)
            )
            summary["af2_model_cache_hits"] = (
                backend.cache_hit_count if backend is not None else 0
            )
            summary["af2_model_cache_misses"] = (
                backend.cache_miss_count if backend is not None else 0
            )
            summary["mpnn_model_cache_hits"] = mpnn.hit_count if mpnn is not None else 0
            summary["mpnn_model_cache_misses"] = (
                mpnn.load_count if mpnn is not None else 0
            )
            _atomic_json(root / "summary.json", summary)
            print(json.dumps({"task": task.name, **summary}, sort_keys=True))
    finally:
        if backend is not None:
            backend.close()
        if mpnn is not None:
            mpnn.close()
    return 0 if all_complete else 2
