"""Shared candidate and standalone postprocessing with atomic publication."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import json
import os
import shutil
import tempfile
from pathlib import Path
from partiflow.common.results import read_results, write_results
from partiflow.paths import config_dir
from partiflow.postprocess.records import (
    WorkItem,
    decode_topology,
    final_identity,
    pdb_atoms,
    refresh_locators,
    strict_bool,
)

DEFAULTS = {
    "method": "auto",
    "platform": "CPU",
    "relax_mode": "local",
    "relax_subset": None,
    "relax_max_iter": None,
    "relax_linear": True,
    "use_ray": False,
    "compute_buns": True,
    "filter": False,
    "criteria": ["rosetta_bindcraft_peptide"],
}


def _normalize_method(method):
    from partiflow.postprocess.chemistry import _normalize_postprocess_method

    method = _normalize_postprocess_method(method)
    if method not in {"auto", "openmm", "rosetta_topology"}:
        raise ValueError(f"Unsupported postprocess method: {method}")
    return method


def _requested_method(saved_method, options):
    method = options["method"]
    return _normalize_method(saved_method if method == "auto" else method)


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        import math

        return value if math.isfinite(value) else None
    return value


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name, suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(_json_value(value), handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


def _configuration(config=None, **overrides):
    value = dict(DEFAULTS)
    if config is not None:
        if isinstance(config, (str, Path)):
            import yaml

            loaded = yaml.safe_load(Path(config).read_text())
        else:
            loaded = dict(config)
        if not isinstance(loaded, dict):
            raise ValueError("Postprocess config must be a mapping")
        value.update(loaded.get("postprocess", loaded))
    value.update({key: val for key, val in overrides.items() if val is not None})
    value["method"] = _normalize_method(value["method"])
    for key in ("filter", "compute_buns", "relax_linear"):
        value[key] = strict_bool(value[key], field_name=key)
    if value.get("use_ray") is not None:
        value["use_ray"] = strict_bool(value["use_ray"], field_name="use_ray")
    platform = str(value["platform"]).upper()
    if platform not in {"CPU", "CUDA", "OPENCL"}:
        raise ValueError("platform must be CPU, CUDA, or OpenCL")
    value["platform"] = {"CPU": "CPU", "CUDA": "CUDA", "OPENCL": "OpenCL"}[platform]
    if not isinstance(value["criteria"], (list, tuple)) or any(
        not isinstance(name, str) for name in value["criteria"]
    ):
        raise ValueError("criteria must be a list of preset names")
    if value["filter"]:
        from partiflow.common.presets import load_presets, validate_criteria

        presets = load_presets(
            Path(value.get("presets") or config_dir() / "preset.yaml")
        )
        missing = set(value["criteria"]) - set(presets)
        if missing:
            raise ValueError(f"Unknown criteria: {sorted(missing)}")
        validate_criteria(presets, value["criteria"], "rosetta")
        value["_presets"] = presets
    return value


def _report_path(item):
    return Path(item.report_dir or item.path.parent / "postprocess") / (
        item.path.stem + ".json"
    )


def _decision(item, result, options):
    if not options["filter"]:
        result["metrics_status"] = "not_requested"
        result["filter_status"] = "not_requested"
        result.pop("decisions", None)
        return result
    from partiflow.postprocess.metrics import compute_metrics, preset_record
    from partiflow.common.presets import apply_presets

    if "metrics_report" not in result or options.get("rerun_metrics"):
        try:
            result["metrics_report"] = compute_metrics(
                result["pdb_path"],
                binder_chain=item.binder_chain,
                topology_entries=item.topology_entries,
                compute_buns=options["compute_buns"],
            )
        except Exception as exc:
            result["metrics_report"] = {
                "status": "unavailable",
                "protocol": "complex",
                "metrics": {},
                "metric_errors": {"loader": f"{type(exc).__name__}: {exc}"},
            }
    report = result["metrics_report"]
    result["metrics_status"] = report["status"]
    record = preset_record(report)
    names = list(options["criteria"])
    inapplicable = {
        name
        for name in names
        if report["protocol"] == "monomer"
        and any(
            rule.get("scope", "complex") in {"complex", "paired"}
            for rule in options["_presets"][name]["rules"]
        )
    }
    apply_presets(
        record,
        [name for name in names if name not in inapplicable],
        options["_presets"],
    )
    result["decisions"] = record.get("decisions", {})
    result["decisions"].update(
        {
            name: {
                "criteria": name,
                "status": "not_applicable",
                "reasons": ["monomer has no receptor interface"],
            }
            for name in inapplicable
        }
    )
    statuses = [decision["status"] for decision in result["decisions"].values()]
    result["filter_status"] = (
        "not_applicable"
        if statuses and all(s == "not_applicable" for s in statuses)
        else "unavailable"
        if not statuses or "unavailable" in statuses
        else "failed"
        if "failed" in statuses
        else "passed"
    )
    return result


def _structure_state(path):
    path = Path(path)
    return {
        "atoms": _json_value(pdb_atoms(path)),
        "links": [
            line
            for line in path.read_text().splitlines()
            if line.startswith(("CONECT", "LINK  ", "SSBOND"))
        ],
    }


def _resume_settings(item, options):
    keys = ("method", "platform", "relax_mode", "relax_subset", "relax_max_iter")
    return _json_value(
        {
            "settings": {key: options.get(key) for key in keys},
            "method": item.method,
            "relax_linear": item.relax_linear,
            "binder_chain": item.binder_chain,
            "topology_entries": item.topology_entries,
            "replace": item.replace,
            "input_path": str(item.path),
        }
    )


def _process_one(item, options):
    report_path = _report_path(item)
    cached = _read_json(report_path)
    completed = (
        cached.get("status") == "success" and Path(cached.get("pdb_path", "")).is_file()
    )
    if completed and not options.get("rerun", False):
        saved = cached.get("resume")
        current = {
            "settings": _resume_settings(item, options),
            "input": _structure_state(item.path),
            "output": _structure_state(cached["pdb_path"]),
        }
        if saved != current:
            raise ValueError(
                f"Postprocess input/settings changed; use --rerun: {item.path}"
            )
        if cached.get("compute_buns") != options["compute_buns"]:
            cached.pop("metrics_report", None)
        result = _decision(item, cached, options)
        result["resumed"] = True
        result["compute_buns"] = options["compute_buns"]
        _write_json(report_path, result)
        return result
    result = {
        "candidate_id": item.candidate_id or item.sample_id,
        "task_name": item.task_name,
        "sample_id": item.sample_id,
        "base_sample_id": item.base_sample_id,
        "input_path": str(item.path),
        "status": "failed",
        "pdb_path": str(item.path),
        "output_path": None,
        "error": "",
        "metadata": item.metadata,
        "metrics_status": "not_requested",
        "filter_status": "not_requested",
        "previous_postprocess_error": cached.get("error", ""),
    }
    try:
        if item.path.suffix.lower() != ".pdb":
            raise ValueError("Topology postprocessing requires a PDB structure")
        if not item.path.is_file():
            raise ValueError(f"Input structure missing: {item.path}")
        pdb_atoms(item.path)
        from partiflow.postprocess.chemistry import relax_candidate

        destination = (
            item.path
            if item.replace
            else Path(item.report_dir) / "structures" / item.path.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".postprocess-", dir=destination.parent
        ) as temporary:
            staged = Path(temporary) / item.path.name
            processed, info = relax_candidate(
                item,
                staged,
                platform=options["platform"],
                method=options["method"],
                relax_mode=options["relax_mode"],
                relax_subset=options["relax_subset"],
                relax_max_iter=options["relax_max_iter"],
            )
            identity = final_identity(processed, item.binder_chain, item.row)
            identity.update(refresh_locators(item.path, processed, item.row))
            pdb_atoms(processed)
            with processed.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(processed, destination)
        result.update(
            status="success",
            pdb_path=str(destination),
            output_path=str(destination),
            identity=identity,
            postprocess=info,
        )
    except OSError:
        raise
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    if result["status"] == "success":
        result["resume"] = {
            "settings": _resume_settings(item, options),
            "input": _structure_state(item.path),
            "output": _structure_state(result["pdb_path"]),
        }
        result["compute_buns"] = options["compute_buns"]
        _write_json(report_path, result)
        _decision(item, result, options)
    _write_json(report_path, result)
    return result


def _as_work(candidate, options, report_dir=None):
    if isinstance(candidate, WorkItem):
        return replace(candidate, method=_requested_method(candidate.method, options))
    metadata = dict(getattr(candidate, "metadata", {}) or {})
    path = Path(candidate.pdb_path).resolve()
    return WorkItem(
        path=path,
        task_name=str(getattr(candidate, "task_name", path.parent.name)),
        sample_id=str(
            metadata.get("sample_id", getattr(candidate, "candidate_id", path.stem))
        ),
        candidate_id=str(getattr(candidate, "candidate_id", path.stem)),
        base_sample_id=getattr(candidate, "base_sample_id", ""),
        binder_chain=str(
            getattr(candidate, "binder_chain", metadata.get("binder_chain", "L"))
        ),
        topology_entries=decode_topology(
            getattr(candidate, "topology_entries", metadata.get("topology_entries", []))
        ),
        method=_requested_method(metadata.get("postprocess_method", "auto"), options),
        relax_linear=strict_bool(
            metadata.get("postprocess_relax_linear", options["relax_linear"]),
            field_name="postprocess_relax_linear",
        ),
        metadata=metadata,
        row={
            key: json.dumps(getattr(candidate, key))
            for key in ("fixed_restypes", "bond_constraints")
            if getattr(candidate, key, None)
        },
        report_dir=Path(report_dir) if report_dir else path.parent / "postprocess",
    )


def process_candidates(
    candidates, config=None, *, enabled_only=False, report_dir=None, **options
):
    """Return same-order status/pdb_path/error records with per-candidate atomic replacement."""
    settings = _configuration(config, **options)
    items = [_as_work(candidate, settings, report_dir) for candidate in candidates]
    results, active, active_indices = [None] * len(items), [], []
    for index, item in enumerate(items):
        enabled = strict_bool(
            item.metadata.get("postprocess_enabled", True),
            field_name="postprocess_enabled",
        )
        if enabled_only and not enabled:
            results[index] = {
                "candidate_id": item.candidate_id,
                "task_name": item.task_name,
                "sample_id": item.sample_id,
                "base_sample_id": item.base_sample_id,
                "status": "skipped",
                "pdb_path": str(item.path),
                "output_path": str(item.path),
                "error": "",
            }
        elif settings.get("plan"):
            results[index] = {
                "candidate_id": item.candidate_id,
                "task_name": item.task_name,
                "sample_id": item.sample_id,
                "base_sample_id": item.base_sample_id,
                "status": "planned",
                "pdb_path": str(item.path),
                "error": "",
                "filter": settings["filter"],
                "method": item.method,
                "replace": item.replace,
            }
        else:
            active.append(item)
            active_indices.append(index)
    if settings.get("use_ray") and active:
        import ray

        started = not ray.is_initialized()
        if started:
            kwargs = {"ignore_reinit_error": True, "include_dashboard": False}
            for option, target in (
                ("ray_num_cpus", "num_cpus"),
                ("ray_num_gpus", "num_gpus"),
            ):
                if settings.get(option) is not None:
                    kwargs[target] = settings[option]
            if settings.get("ray_object_store_mb") is not None:
                kwargs["object_store_memory"] = (
                    int(settings["ray_object_store_mb"]) * 1024 * 1024
                )
            ray.init(**kwargs)
        try:
            gpu = settings.get("ray_task_gpus")
            if gpu is None:
                gpu = 1 / 16 if settings["platform"] == "CUDA" else 0
            remote = ray.remote(
                num_cpus=int(settings.get("ray_task_cpus", 1)), num_gpus=float(gpu)
            )(_process_one)
            limit = int(settings.get("max_inflight") or len(active))
            if limit < 1:
                raise ValueError("max_inflight must be positive")
            active_results = []
            for offset in range(0, len(active), limit):
                active_results.extend(
                    ray.get(
                        [
                            remote.remote(item, settings)
                            for item in active[offset : offset + limit]
                        ]
                    )
                )
        finally:
            if started:
                ray.shutdown()
    else:
        active_results = [_process_one(item, settings) for item in active]
    for index, result in zip(active_indices, active_results):
        results[index] = result
    return results


def _task_items(result_path, options):
    fields, rows = read_results(result_path)
    task_dir = result_path.parent
    samples = _read_json(task_dir / ".state" / "design.json").get("samples", [])
    items = []
    for row in rows:
        filename = row.get("filename", "")
        if not filename:
            continue
        path = (task_dir / filename).resolve()
        if not path.is_relative_to(task_dir.resolve()):
            raise ValueError(f"Result filename leaves task directory: {filename}")
        metadata = next(
            (
                sample
                for sample in samples
                if str(sample.get("filename", "")) == filename
            ),
            {},
        )
        if not metadata:
            metadata = next(
                (
                    sample
                    for sample in samples
                    if Path(str(sample.get("filename", ""))).name == path.name
                ),
                {},
            )
        chain = str(metadata.get("binder_chain") or row.get("design_chain_id") or "L")
        items.append(
            WorkItem(
                path,
                task_dir.name,
                str(metadata.get("sample_id", path.stem)),
                metadata.get("base_sample_id", row.get("base_sample_id", "")),
                chain,
                decode_topology(metadata.get("topology_entries", [])),
                _requested_method(metadata.get("postprocess_method", "auto"), options),
                strict_bool(
                    metadata.get("postprocess_relax_linear", options["relax_linear"]),
                    field_name="postprocess_relax_linear",
                ),
                row=row,
                metadata=metadata,
                result_path=result_path,
                report_dir=task_dir / "postprocess",
            )
        )
    return fields, rows, items


def _collect_paths(paths, options, task_names=None):
    result_paths, refolds = set(), []
    requested = set(task_names or [])
    for value in paths:
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.name.lower() == "af2success" and path.is_dir():
            refolds.append(path)
        elif path.is_file() and path.name == "result.csv":
            result_paths.add(path)
        elif path.is_dir():
            if (path / "result.csv").is_file():
                result_paths.add(path / "result.csv")
            else:
                result_paths.update(
                    p for p in path.rglob("result.csv") if "postprocess" not in p.parts
                )
        elif path.suffix.lower() == ".pdb":
            refolds.append(path)
        else:
            raise ValueError(f"Unsupported postprocess input: {path}")
    grouped, items = {}, []
    for result_path in sorted(result_paths):
        if requested and result_path.parent.name not in requested:
            continue
        fields, rows, task_items = _task_items(result_path, options)
        grouped[result_path] = (fields, rows, task_items)
        items.extend(task_items)
    for source in refolds:
        task_dir = next(
            (p for p in source.parents if (p / "result.csv").is_file()), source.parent
        )
        if requested and task_dir.name not in requested:
            continue
        sources = (
            [source]
            if source.is_file()
            else sorted(
                p
                for p in source.rglob("*.pdb")
                if "postprocess" not in p.relative_to(source).parts
            )
        )
        for path in sources:
            parent = source.parent if source.is_file() else source
            relative = path.relative_to(parent)
            report_dir = task_dir / "postprocess" / "refold" / relative.parent
            metadata = {
                "source_kind": "refold",
                "sequence_slot": str(relative),
                "postprocess_enabled": True,
            }
            atoms = pdb_atoms(path)
            chains = list(dict.fromkeys(atom["chain"] for atom in atoms))
            binder = str(
                options.get("binder_chain") or ("L" if "L" in chains else chains[-1])
            )
            items.append(
                WorkItem(
                    path,
                    task_dir.name,
                    str(relative.with_suffix("")),
                    binder_chain=binder,
                    method=options["method"],
                    relax_linear=options["relax_linear"],
                    metadata=metadata,
                    report_dir=report_dir,
                    replace=False,
                )
            )
    if requested:
        found = {item.task_name for item in items}
        if requested - found:
            raise ValueError(
                f"Tasks have no postprocess inputs: {sorted(requested - found)}"
            )
    if not grouped and not items:
        raise ValueError("No result.csv or explicit AF2success inputs were found")
    return grouped, items


def _publish_results(grouped, items, results):
    by_item = {id(item): result for item, result in zip(items, results)}
    for result_path, (fields, rows, task_items) in grouped.items():
        for item in task_items:
            result = by_item[id(item)]
            if result["status"] in {"skipped", "planned"}:
                continue
            row = item.row
            row["postprocess_status"] = (
                "succeeded" if result["status"] == "success" else "failed"
            )
            if result["status"] == "success":
                row.update(result.get("identity", {}))
                previous = result.get("previous_postprocess_error", "")
                current = row.get("error", "")
                if previous and current == previous:
                    row["error"] = ""
                elif previous and current.endswith("; " + previous):
                    row["error"] = current[: -(len(previous) + 2)]
            else:
                previous = row.get("error", "")
                row["error"] = "; ".join(
                    value for value in (previous, result["error"]) if value
                )
    by_directory = {}
    for item, result in zip(items, results):
        if result["status"] not in {"skipped", "planned"}:
            directory = Path(item.report_dir)
            by_directory.setdefault(directory, []).append(result)
    for directory, group in by_directory.items():
        _write_json(
            directory / "summary.json",
            {
                "samples": group,
                "relaxation_successes": sum(
                    item["status"] == "success" for item in group
                ),
                "filter_passes": sum(
                    item.get("filter_status") == "passed" for item in group
                ),
            },
        )
    for result_path, (fields, rows, _) in grouped.items():
        write_results(result_path, rows, fields=fields)


@contextmanager
def _task_publication(result_path, items):
    paths = [Path(result_path)]
    for item in items:
        destination = (
            item.path
            if item.replace
            else Path(item.report_dir) / "structures" / item.path.name
        )
        paths.extend(
            (destination, _report_path(item), Path(item.report_dir) / "summary.json")
        )
    with tempfile.TemporaryDirectory(
        prefix=".postprocess-backup-", dir=result_path.parent
    ) as temporary:
        backups = []
        for index, path in enumerate(dict.fromkeys(paths)):
            backup = None
            if path.exists():
                backup = Path(temporary) / str(index)
                try:
                    os.link(path, backup)
                except OSError:
                    shutil.copy2(path, backup)
            backups.append((path, backup))
        try:
            yield
        except BaseException:
            restore_error = None
            for path, backup in reversed(backups):
                try:
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        try:
                            os.replace(backup, path)
                        except OSError:
                            # A failed publication may also reject rollback replacements.
                            if backup.exists():
                                try:
                                    shutil.copy2(backup, path)
                                except shutil.SameFileError:
                                    pass
                except OSError as exc:
                    restore_error = restore_error or exc
            if restore_error is not None:
                raise restore_error
            raise


def _process_standalone(grouped, items, settings, *, enabled_only=False):
    if settings.get("plan"):
        return process_candidates(items, settings, enabled_only=enabled_only)
    by_item = {}
    for result_path, group in grouped.items():
        task_items = group[2]
        with _task_publication(result_path, task_items):
            task_results = process_candidates(
                task_items, settings, enabled_only=enabled_only
            )
            _publish_results({result_path: group}, task_items, task_results)
        by_item.update(
            (id(item), result) for item, result in zip(task_items, task_results)
        )
    refold_items = [item for item in items if id(item) not in by_item]
    if refold_items:
        refold_results = process_candidates(
            refold_items, settings, enabled_only=enabled_only
        )
        _publish_results({}, refold_items, refold_results)
        by_item.update(
            (id(item), result) for item, result in zip(refold_items, refold_results)
        )
    return [by_item[id(item)] for item in items]


def run_postprocess_for_run(
    root_run_dir, *, config=None, task_names=None, enabled_only=False, **options
):
    settings = _configuration(config, **options)
    grouped, items = _collect_paths([root_run_dir], settings, task_names)
    return _process_standalone(grouped, items, settings, enabled_only=enabled_only)


def has_postprocess_work(root_run_dir):
    for state_path in Path(root_run_dir).rglob("design.json"):
        if state_path.parent.name == ".state":
            if any(
                sample.get("postprocess_enabled") is True
                for sample in _read_json(state_path).get("samples", [])
            ):
                return True
    return False


def run(args):
    try:
        options = {
            "filter": getattr(args, "filter", False),
            "criteria": getattr(args, "criteria", None),
            "presets": getattr(args, "presets", None),
            "rerun": getattr(args, "rerun", False),
            "plan": getattr(args, "plan", False),
            "platform": getattr(args, "platform", None),
        }
        settings = _configuration(getattr(args, "config", None), **options)
        grouped, items = _collect_paths(
            args.paths, settings, getattr(args, "task", None)
        )
        results = _process_standalone(grouped, items, settings)
        if settings.get("plan"):
            print(json.dumps({"postprocess": results}, indent=2))
        else:
            print(
                f"Postprocess: {sum(r['status'] == 'success' for r in results)}/{len(results)} structures complete"
            )
        return int(any(result["status"] == "failed" for result in results))
    except (OSError, ValueError, KeyError) as exc:
        print(f"Postprocess failed: {exc}")
        return 1
