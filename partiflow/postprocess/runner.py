"""Shared candidate and standalone postprocessing with atomic publication."""

from __future__ import annotations
from contextlib import contextmanager
from dataclasses import replace
import sys
import time
import json
import os
import shutil
import tempfile
from pathlib import Path
from partiflow.common.results import read_results, write_results
from partiflow.paths import config_dir
from partiflow.postprocess.records import (
    WorkItem,
    SampleContext,
    decode_topology,
    final_identity,
    refresh_locators,
    strict_bool,
)



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
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path):
    try:
        value = json.loads(Path(path).read_text())
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, ValueError):
        return {}


class _Settings(dict):
    """Validated runtime settings, reusable within one command invocation."""


def _configuration(config=None, **overrides):
    import yaml
    changes = {key: value for key, value in overrides.items() if value is not None}
    if isinstance(config, _Settings) and not changes:
        return config
    value = yaml.safe_load((config_dir() / "postprocess.yaml").read_text())
    if config is not None:
        loaded = yaml.safe_load(Path(config).read_text()) if isinstance(config, (str, Path)) else dict(config)
        if not isinstance(loaded, dict):
            raise ValueError("Postprocess config must be a mapping")
        value.update(loaded.get("postprocess", loaded))
    value.update(changes)
    retired = set(value) & {"ray_num_cpus", "ray_num_gpus", "ray_task_cpus", "ray_task_gpus", "ray_object_store_mb", "max_inflight"}
    if retired:
        raise ValueError("Use workers and threads instead of legacy Ray settings: " + ", ".join(sorted(retired)))
    value["method"] = _normalize_method(value["method"])
    for key in ("filter", "compute_buns", "relax_linear"):
        value[key] = strict_bool(value[key], field_name=key)
    value["use_ray"] = True if value.get("use_ray") is None else strict_bool(value["use_ray"], field_name="use_ray")
    for key in ("threads", "timeout", "relax_repeats"):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if value["workers"] is not None and (type(value["workers"]) is not int or value["workers"] < 1):
        raise ValueError("workers must be a positive integer or null")
    if type(value["seed"]) is not int or value["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    if value["scorefunction"] != "beta_nov16":
        raise ValueError("Standard peptide postprocessing uses beta_nov16")
    from partiflow.postprocess.chemistry import _resolve_rosetta_relax_settings
    mode, subset, iterations = _resolve_rosetta_relax_settings(
        relax_mode=value["relax_mode"], relax_subset=value["relax_subset"], relax_max_iter=value["relax_max_iter"])
    value.update(relax_mode=mode, relax_subset=subset, relax_max_iter=iterations)
    platform = str(value["platform"]).upper()
    if platform not in {"CPU", "CUDA", "OPENCL"}:
        raise ValueError("platform must be CPU, CUDA, or OpenCL")
    value["platform"] = {"CPU": "CPU", "CUDA": "CUDA", "OPENCL": "OpenCL"}[platform]
    if not isinstance(value["criteria"], (list, tuple)) or any(not isinstance(name, str) for name in value["criteria"]):
        raise ValueError("criteria must be a list of preset names")
    if value["filter"]:
        from partiflow.common.presets import load_presets, validate_criteria
        presets = load_presets(Path(value.get("presets") or config_dir() / "preset.yaml"))
        missing = set(value["criteria"]) - set(presets)
        if missing:
            raise ValueError(f"Unknown criteria: {sorted(missing)}")
        validate_criteria(presets, value["criteria"], "rosetta")
        value["_presets"] = presets
        value["_required_metrics"] = sorted({rule["metric"] for name in value["criteria"] for rule in presets[name]["rules"]})
    return _Settings(value)


def _report_path(item):
    return Path(item.report_dir or item.path.parent / "postprocess") / (
        item.path.stem + ".json"
    )


def _metric_request(item, options):
    return {"required_metrics": options["_required_metrics"],
            "compute_buns": options["compute_buns"], "seed": options["seed"],
            "sdk_profile": item.sdk_profile}


def _metrics_reusable(item, result, options):
    return (result.get("metrics_request") == _metric_request(item, options)
            and result.get("metrics_report", {}).get("status") == "complete"
            and not options.get("rerun_metrics"))


def _decision(item, result, options, context=None):
    result["compute_buns"] = options["compute_buns"]
    if not options["filter"]:
        result["metrics_status"] = "not_requested"
        result["filter_status"] = "not_requested"
        result.pop("decisions", None)
        return result
    from partiflow.postprocess.metrics import compute_metrics, preset_record
    from partiflow.common.presets import apply_presets

    request = _metric_request(item, options)
    if not _metrics_reusable(item, result, options):
        try:
            result["metrics_report"] = compute_metrics(
                result["pdb_path"],
                binder_chain=item.binder_chain,
                topology_entries=item.topology_entries,
                compute_buns=options["compute_buns"],
                context=context, required_metrics=options["_required_metrics"],
            )
        except Exception as exc:
            result["metrics_report"] = {
                "status": "unavailable",
                "protocol": "complex",
                "metrics": {},
                "metric_errors": {"loader": f"{type(exc).__name__}: {exc}"},
            }
    result["metrics_request"] = request
    report = result["metrics_report"]
    result["metrics_status"] = report["status"]
    if report["status"] == "complete":
        result["error"] = ""
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


def _resume_settings(item, options):
    keys = ("method", "platform", "relax_mode", "relax_subset", "relax_max_iter", "relax_repeats", "seed")
    return _json_value({
        "settings": {key: options.get(key) for key in keys}, "method": item.method,
        "scorefunction": "ref2015" if item.sdk_profile == "native_ref2015" else options["scorefunction"],
        "sdk_profile": item.sdk_profile, "relax_linear": item.relax_linear,
        "binder_chain": item.binder_chain, "topology_entries": item.topology_entries,
        "replace": item.replace, "input_path": str(item.path),
        "task_name": item.task_name, "sample_id": item.sample_id,
        "candidate_id": item.candidate_id, "base_sample_id": item.base_sample_id,
    })


def _completed_geometry(item, options, cached):
    return (cached.get("version") == 2 and cached.get("status") == "success"
            and cached.get("geometry_status") == "complete"
            and cached.get("resume", {}).get("settings") == _resume_settings(item, options)
            and bool(cached.get("pdb_path")) and Path(cached["pdb_path"]).is_file())


def _process_one(item, options):
    import random
    import numpy as np
    random.seed(options["seed"])
    np.random.seed(options["seed"] % (2 ** 32))
    from partiflow.postprocess.source import PDBSource
    report_path = _report_path(item)
    cached = _read_json(report_path)
    if cached.get("status") == "success" and not options.get("rerun", False):
        if not _completed_geometry(item, options, cached):
            raise ValueError(f"Postprocess input/settings changed; use --rerun: {item.path}")
        context = SampleContext(sdk_profile=item.sdk_profile, seed=options["seed"])
        result = _decision(item, cached, options, context)
        result["resumed"] = True
        _write_json(report_path, result)
        return result
    result = {
        "version": 2, "candidate_id": item.candidate_id or item.sample_id,
        "task_name": item.task_name, "sample_id": item.sample_id,
        "base_sample_id": item.base_sample_id, "input_path": str(item.path),
        "status": "failed", "geometry_status": "not_run", "pdb_path": str(item.path),
        "output_path": None, "error": "", "metrics_status": "not_requested",
        "filter_status": "not_requested", "previous_postprocess_error": cached.get("error", ""),
        "sdk_profile": item.sdk_profile,
    }
    started = time.monotonic()
    context = SampleContext(sdk_profile=item.sdk_profile, seed=options["seed"])
    try:
        if item.path.suffix.lower() != ".pdb":
            raise ValueError("Topology postprocessing requires a PDB structure")
        if not item.path.is_file():
            raise ValueError(f"Input structure missing: {item.path}")
        context.input_source = PDBSource.read(item.path)
        if context.input_source.nonstandard_codes and item.sdk_profile != "native_ref2015":
            raise ValueError("Nonstandard chemistry requires its native_ref2015 worker")
        from partiflow.postprocess.chemistry import relax_candidate
        destination = item.path if item.replace else Path(item.report_dir) / "structures" / item.path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".postprocess-", dir=destination.parent) as temporary:
            staged = Path(temporary) / item.path.name
            processed, info = relax_candidate(item, staged, platform=options["platform"], method=options["method"],
                relax_mode=options["relax_mode"], relax_subset=options["relax_subset"],
                relax_max_iter=options["relax_max_iter"], relax_repeats=options["relax_repeats"],
                seed=options["seed"], threads=options["threads"], context=context)
            context.output_source = context.output_source or PDBSource.read(processed)
            identity = final_identity(processed, item.binder_chain, item.row, source=context.output_source)
            identity.update(refresh_locators(item.path, processed, item.row,
                source=context.input_source, output_source=context.output_source))
            os.replace(processed, destination)
        result.update(status="success", geometry_status="complete", pdb_path=str(destination),
            output_path=str(destination), identity=identity, postprocess=info,
            resume={"settings": _resume_settings(item, options)})
        result["timings"] = {"relaxation": time.monotonic() - started}
        _write_json(report_path, result)
        _decision(item, result, options, context)
    except OSError:
        raise
    except Exception as exc:
        if result["status"] == "success":
            result.update(metrics_status="unavailable", filter_status="unavailable")
        else:
            result["geometry_status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result.setdefault("timings", {})["total"] = time.monotonic() - started
    _write_json(report_path, result)
    return result


def recover_actor_failure(item, options, reason):
    cached = _read_json(_report_path(item))
    if _completed_geometry(item, options, cached):
        cached.update(metrics_status="unavailable", filter_status="unavailable", error=str(reason))
        cached["metrics_report"] = {"status": "unavailable", "protocol": "complex", "metrics": {},
            "metric_errors": {"native_worker": str(reason)}}
        cached["decisions"] = {name: {"status": "unavailable", "reasons": [str(reason)]}
            for name in options.get("criteria", ())}
        _write_json(_report_path(item), cached)
        return cached
    result = {"version": 2, "candidate_id": item.candidate_id or item.sample_id,
        "task_name": item.task_name, "sample_id": item.sample_id, "base_sample_id": item.base_sample_id,
        "input_path": str(item.path), "pdb_path": str(item.path), "output_path": None,
        "status": "failed", "geometry_status": "failed", "error": str(reason),
        "metrics_status": "unavailable", "filter_status": "unavailable", "sdk_profile": item.sdk_profile}
    _write_json(_report_path(item), result)
    return result


class PostprocessWorker:
    """A fixed SDK profile, initialized at most once in its owning Ray process."""

    def __init__(self, settings):
        self.settings = settings
        self.profile = settings["sdk_profile"]
        self.processed = 0
        self.pr = None
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[key] = str(settings["threads"])
        if not settings.get("needs_gpu"):
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if self.profile == "standard_beta":
            from partiflow.postprocess import rosetta
            rosetta.initialize_standard_beta(seed=settings["seed"], threads=settings["threads"])
            self.pr = rosetta.pr
        elif self.profile == "native_ref2015":
            if settings["filter"]:
                from partiflow.postprocess import native
                native.initialize_pyrosetta_score_only(seed=settings["seed"], threads=settings["threads"])
                self.pr = native.pr
        else:
            raise ValueError(f"Unknown SDK profile: {self.profile}")
        self._initial_rng = None
        if self.pr is not None:
            stream = self.pr.rosetta.std.ostringstream()
            self.pr.rosetta.numeric.random.rg().saveState(stream)
            self._initial_rng = stream.str()

    def stats(self):
        return {"pid": os.getpid(), "sdk_profile": self.profile,
            "initializations": int(self.pr is not None), "processed": self.processed}

    def ready(self):
        return self.stats()

    def process(self, item):
        if item.sdk_profile != self.profile:
            raise ValueError("A worker cannot change its native SDK profile")
        if self.pr is not None:
            stream = self.pr.rosetta.std.istringstream(self._initial_rng)
            self.pr.rosetta.numeric.random.rg().restoreState(stream)
        self.processed += 1
        return _process_one(item, self.settings)


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


def _execution_item(item, settings):
    from partiflow.postprocess.records import STANDARD, STANDARD_ALIASES, normalize_type
    special = any(normalize_type(entry.get("type")) not in {"none", "head_to_tail", "disulfide"}
        for entry in item.topology_entries)
    nonstandard = False
    if not special and item.path.is_file():
        with item.path.open() as handle:
            nonstandard = any(line[17:20].strip() not in STANDARD and line[17:20].strip() not in STANDARD_ALIASES
                for line in handle if line.startswith(("ATOM  ", "HETATM")))
    profile = "native_ref2015" if special or nonstandard else "standard_beta"
    needs_openmm = profile == "native_ref2015" or item.method in {"openmm", "auto"}
    return replace(item, sdk_profile=profile, needs_gpu=settings["platform"] == "CUDA" and needs_openmm)


def process_candidates(candidates, config=None, *, enabled_only=False, report_dir=None, on_result=None, **options):
    """Dispatch complete samples through one actor pool and preserve input order."""
    settings = _configuration(config, **options)
    items = [_as_work(candidate, settings, report_dir) for candidate in candidates]
    results, active, indices = [None] * len(items), [], []
    completed = []
    for index, item in enumerate(items):
        enabled = strict_bool(item.metadata.get("postprocess_enabled", True), field_name="postprocess_enabled")
        if enabled_only and not enabled:
            result = {"candidate_id": item.candidate_id or item.sample_id, "task_name": item.task_name,
                "sample_id": item.sample_id, "base_sample_id": item.base_sample_id, "status": "skipped",
                "pdb_path": str(item.path), "output_path": str(item.path), "error": ""}
        elif settings.get("plan"):
            result = {"candidate_id": item.candidate_id or item.sample_id, "task_name": item.task_name,
                "sample_id": item.sample_id, "base_sample_id": item.base_sample_id, "status": "planned",
                "pdb_path": str(item.path), "error": "", "filter": settings["filter"],
                "method": item.method, "replace": item.replace}
        else:
            cached = _read_json(_report_path(item))
            if cached.get("status") == "success" and not settings.get("rerun"):
                item = replace(item, sdk_profile=cached.get("sdk_profile", "standard_beta"))
                if not _completed_geometry(item, settings, cached):
                    raise ValueError(f"Postprocess input/settings changed; use --rerun: {item.path}")
                if not settings["filter"] or _metrics_reusable(item, cached, settings):
                    completed.append((index, item, cached))
                    continue
                item = replace(item, needs_gpu=False)
            else:
                item = _execution_item(item, settings)
            active.append(item)
            indices.append(index)
            continue
        results[index] = result
    # Validate every reusable record before any sample can mutate its outputs.
    if settings.get("rerun"):
        for item in active:
            _report_path(item).unlink(missing_ok=True)
    for index, item, cached in completed:
        result = _decision(item, cached, settings)
        result["resumed"] = True
        _write_json(_report_path(item), result)
        results[index] = result
    if on_result is not None:
        for index, result in enumerate(results):
            if result is not None:
                on_result(index, result)
    def receive(index, result):
        original_index = indices[index]
        results[original_index] = result
        if on_result is not None:
            on_result(original_index, result)
    if settings["use_ray"] and active:
        from partiflow.postprocess.dispatch import dispatch_items
        _, stats = dispatch_items(active, settings, on_result=receive)
        print(json.dumps({"postprocess_workers": stats}))
    else:
        profiles = {item.sdk_profile for item in active}
        if settings["filter"] and len(profiles) > 1:
            raise ValueError("Mixed SDK profiles require use_ray: true")
        for index, item in enumerate(active):
            receive(index, _process_one(item, settings))
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
            with path.open() as handle:
                chains = list(dict.fromkeys(line[21:22].strip() or " " for line in handle
                    if line.startswith(("ATOM  ", "HETATM"))))
            if not chains:
                raise ValueError(f"No structure chains found: {path}")
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
    positions = {id(item): index for index, item in enumerate(items)}
    grouped_ids = {id(item) for group in grouped.values() for item in group[2]}
    results = [None] * len(items)
    guards = {}
    order = list(grouped)
    next_group = 0
    try:
        for result_path, group in grouped.items():
            guard = _task_publication(result_path, group[2])
            guard.__enter__()
            guards[result_path] = guard
        def received(index, result):
            nonlocal next_group
            results[index] = result
            while next_group < len(order):
                result_path = order[next_group]
                group = grouped[result_path]
                selected = [results[positions[id(item)]] for item in group[2]]
                if any(value is None for value in selected):
                    break
                _publish_results({result_path: group}, group[2], selected)
                guards.pop(result_path).__exit__(None, None, None)
                next_group += 1
        process_candidates(items, settings, enabled_only=enabled_only, on_result=received)
        refold_items = [item for item in items if id(item) not in grouped_ids]
        if refold_items:
            _publish_results({}, refold_items, [results[positions[id(item)]] for item in refold_items])
        return results
    except BaseException:
        failure = sys.exc_info()
        restore_error = None
        for guard in reversed(list(guards.values())):
            try:
                guard.__exit__(*failure)
            except BaseException as exc:
                restore_error = restore_error or exc
        if restore_error is not None:
            raise restore_error
        raise


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
            "filter": getattr(args, "filter", None),
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
