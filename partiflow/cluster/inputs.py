"""Select saved successes, then resolve only current task result structures."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClusterJob:
    task_dir: Path
    method_dir: Path
    method: str = "af2_ig"
    selection_files: tuple[Path, ...] = ()

    @property
    def report_dir(self):
        return (
            self.method_dir / "clusters"
            if self.method == "postprocess"
            else self.method_dir
        )


@dataclass(frozen=True)
class ClusterSample:
    sample: str
    structure: Path
    design_chains: tuple[dict, ...]


@dataclass(frozen=True)
class ClusterInput:
    job: ClusterJob
    samples: dict[str, ClusterSample]
    groups: dict[str, tuple[str, ...]]
    sources: dict[str, dict[str, tuple[str, ...]]]


def _names(values):
    values = (values,) if isinstance(values, str) else values
    try:
        values = iter(values)
    except TypeError as exc:
        raise ValueError(
            "Task and criterion names must be an iterable of strings"
        ) from exc
    result = []
    for value in values:
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(
                "Task and criterion names must be nonempty strings without surrounding whitespace"
            )
        if value not in result:
            result.append(value)
    return tuple(result)


def _af2_job(directory):
    if directory.name != "af2_ig" or directory.parent.name != "validate":
        raise ValueError(
            f"AF2 selections must be TASK/validate/af2_ig/success.csv: {directory}"
        )
    return ClusterJob(directory.parent.parent, directory)


def _task_jobs(task):
    directory = task / "validate" / "af2_ig"
    return [_af2_job(directory)] if (directory / "success.csv").is_file() else []


def _postprocess_job(directory, files=None):
    if (
        directory.name != "postprocess"
        or not (directory.parent / "result.csv").is_file()
    ):
        raise ValueError(
            f"Postprocess selections must belong to TASK/postprocess: {directory}"
        )
    if files is None:
        files = tuple(
            sorted(
                path for path in directory.glob("*.json") if path.name != "summary.json"
            )
        )
        if not files and (directory / "summary.json").is_file():
            files = (directory / "summary.json",)
    if not files:
        raise ValueError(f"No saved postprocess records found: {directory}")
    return ClusterJob(directory.parent, directory, "postprocess", tuple(files))


def discover_jobs(paths, *, tasks=()):
    """Default to AF2 tables; postprocess records require explicit paths.

    Discovery visits a run's direct task children and the documented task,
    validation, method and success directory levels. It never scans refolds.
    """
    roots = (paths,) if isinstance(paths, (str, Path)) else paths
    selected_tasks = set(_names(tasks))
    found = {}
    for raw in roots:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"Cluster input does not exist: {path}")
        if path.is_file():
            if path.name == "result.csv":
                jobs = _task_jobs(path.parent)
            elif path.name == "success.csv":
                jobs = [_af2_job(path.parent)]
            elif path.suffix.lower() == ".json":
                jobs = [_postprocess_job(path.parent, (path,))]
            else:
                raise ValueError(f"Unsupported cluster selection: {path}")
        elif path.name == "postprocess":
            jobs = [_postprocess_job(path)]
        elif (
            path.name in {"success", "AF2success"}
            and (path.parent / "success.csv").is_file()
        ):
            jobs = [_af2_job(path.parent)]
        elif (path / "success.csv").is_file():
            jobs = [_af2_job(path)]
        elif (path / "result.csv").is_file():
            jobs = _task_jobs(path)
        elif path.name == "validate" and (path.parent / "result.csv").is_file():
            jobs = _task_jobs(path.parent)
        else:
            jobs = [
                job
                for child in sorted(path.iterdir())
                if child.is_dir()
                and not child.name.startswith(".")
                and (child / "result.csv").is_file()
                for job in _task_jobs(child)
            ]
        for job in jobs:
            job = ClusterJob(
                job.task_dir.resolve(),
                job.method_dir.resolve(),
                job.method,
                tuple(file.resolve() for file in job.selection_files),
            )
            expected = job.task_dir / (
                "postprocess" if job.method == "postprocess" else "validate/af2_ig"
            )
            if job.method_dir != expected:
                raise ValueError(
                    f"Selection directory leaves its task: {job.method_dir}"
                )
            if selected_tasks and job.task_dir.name not in selected_tasks:
                continue
            key = (job.task_dir, job.method_dir, job.method)
            previous = found.get(key)
            if previous is not None and job.method == "postprocess":
                files = tuple(
                    sorted(set(previous.selection_files) | set(job.selection_files))
                )
                job = ClusterJob(job.task_dir, job.method_dir, job.method, files)
            found[key] = job
    return [found[key] for key in sorted(found)]


def _read_csv(path):
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if not fields or len(fields) != len(set(fields)):
                raise ValueError(f"Missing or duplicate CSV columns: {path}")
            rows = []
            for index, row in enumerate(reader, 2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(
                        f"{path}:{index}: CSV row width differs from its header"
                    )
                rows.append({key: value.strip() for key, value in row.items()})
            return fields, rows
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def _read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read saved selection {path}: {exc}") from exc


def _sample_name(value, context):
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"{context}: invalid or missing sample filename stem")
    return value


def _af2_groups(job, requested):
    path = job.method_dir / "success.csv"
    fields, rows = _read_csv(path)
    if not {"sample", "criteria", "status"}.issubset(fields):
        raise ValueError(f"{path}: expected sample, criteria and status columns")
    known = set()
    summary_path = job.method_dir / "summary.json"
    if summary_path.is_file():
        summary = _read_json(summary_path)
        if not isinstance(summary, dict):
            raise ValueError(f"{summary_path}: summary must be an object")
        known.update(_names(summary.get("criteria", ())))
    for index, row in enumerate(rows, 2):
        if not row["criteria"] or row["criteria"] != row["criteria"].strip():
            raise ValueError(f"{path}:{index}: missing criterion name")
        if row["status"] not in {"passed", "failed", "unavailable", "not_applicable"}:
            raise ValueError(
                f"{path}:{index}: invalid criterion status {row['status']!r}"
            )
        known.add(row["criteria"])
    unknown = set(requested) - known
    if unknown and (rows or known):
        raise ValueError(f"{path}: unknown criteria: {', '.join(sorted(unknown))}")
    names = requested or tuple(sorted(known)) or ("",)
    groups = {name: {} for name in names}
    decisions = {}
    for index, row in enumerate(rows, 2):
        name = row["criteria"]
        if name not in groups:
            continue
        sample = _sample_name(row["sample"], f"{path}:{index}")
        slot = (
            sample,
            row.get("sequence_index", ""),
            row.get("source", ""),
            row.get("source_index", ""),
            name,
        )
        if slot in decisions and decisions[slot] != row["status"]:
            raise ValueError(
                f"{path}:{index}: conflicting decisions for one sequence slot"
            )
        decisions[slot] = row["status"]
        if row["status"] == "passed":
            source = row.get("source", "")
            if not source:
                raise ValueError(f"{path}:{index}: passing sequence slot has no source")
            label = "self" if source == "gt" else source
            groups[name].setdefault(sample, set()).add(label)
    return groups


def _postprocess_groups(job, requested):
    records = []
    for path in job.selection_files:
        value = _read_json(path)
        if not isinstance(value, dict):
            raise ValueError(f"{path}: saved postprocess record must be an object")
        items = value.get("samples") if path.name == "summary.json" else [value]
        if not isinstance(items, list) or any(
            not isinstance(item, dict) for item in items
        ):
            raise ValueError(f"{path}: summary.samples must be an object array")
        records.extend((path, item) for item in items)
    known = set()
    for path, record in records:
        decisions = record.get("decisions", {})
        if not isinstance(decisions, dict):
            raise ValueError(f"{path}: decisions must be an object")
        known.update(_names(decisions))
        for name, decision in decisions.items():
            if not isinstance(decision, dict) or decision.get("status") not in {
                "passed",
                "failed",
                "unavailable",
                "not_applicable",
            }:
                raise ValueError(f"{path}: invalid decision for {name!r}")
    unknown = set(requested) - known
    if unknown and records:
        raise ValueError(
            f"{job.method_dir}: unknown criteria: {', '.join(sorted(unknown))}"
        )
    names = requested or ("postprocess_success",)
    groups = {name: {} for name in names}
    previous = {}
    for path, record in records:
        if record.get("status") != "success":
            continue
        decisions = record.get("decisions", {})
        chosen = (
            [
                name
                for name in requested
                if decisions.get(name, {}).get("status") == "passed"
            ]
            if requested
            else names
        )
        if not chosen:
            continue
        metadata = record.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"{path}: metadata must be an object")
        if metadata.get("source_kind") == "refold":
            raise ValueError(
                f"{path}: refold postprocess records cannot select generated backbones"
            )
        input_file = record.get("input_path")
        if input_file:
            if not isinstance(input_file, str):
                raise ValueError(f"{path}: input_path must be a string")
            parts = Path(input_file).parts
            task_positions = [
                index for index, part in enumerate(parts) if part == job.task_dir.name
            ]
            relative_parts = parts[task_positions[-1] + 1 :] if task_positions else ()
            if any(
                part.lower()
                in {"validate", "refolds", "af2success", "success", "original"}
                for part in relative_parts
            ):
                raise ValueError(
                    f"{path}: derived structure records cannot select generated backbones"
                )
            sample = Path(input_file).stem
        else:
            sample = record.get("sample_id") or record.get("candidate_id")
        sample = _sample_name(sample, path)
        decision_values = {name: item["status"] for name, item in decisions.items()}
        if sample in previous and previous[sample] != decision_values:
            raise ValueError(
                f"{path}: conflicting postprocess decisions for {sample!r}"
            )
        previous[sample] = decision_values
        for name in chosen:
            groups[name].setdefault(sample, set()).add("postprocess")
    return groups


def _design_chains(row, context):
    try:
        chains = json.loads(row.get("design_chains", ""))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{context}: design_chains must be a JSON object array"
        ) from exc
    if not isinstance(chains, list) or not chains:
        raise ValueError(
            f"{context}: design_chains must be a nonempty JSON object array"
        )
    seen = set()
    for index, item in enumerate(chains):
        prefix = f"{context}: design_chains[{index}]"
        if not isinstance(item, dict) or not {
            "chain",
            "modality",
            "sequence",
            "residue_count",
        }.issubset(item):
            raise ValueError(
                f"{prefix} requires chain, modality, sequence and residue_count"
            )
        chain = item["chain"]
        if (
            not isinstance(chain, str)
            or not chain
            or chain != chain.strip()
            or chain in seen
        ):
            raise ValueError(f"{prefix}: invalid or duplicate final chain label")
        seen.add(chain)
        if not isinstance(item["modality"], str) or item["modality"] not in {
            "protein",
            "rna",
            "dna",
            "ligand",
        }:
            raise ValueError(f"{prefix}: unsupported modality")
        if not isinstance(item["sequence"], str) or not item["sequence"]:
            raise ValueError(f"{prefix}: sequence must be a nonempty string")
        if type(item["residue_count"]) is not int or item["residue_count"] < 1:
            raise ValueError(f"{prefix}: residue_count must be a positive integer")
    return tuple(chains)


def _samples(task, selected):
    path = task / "result.csv"
    fields, rows = _read_csv(path)
    if not {"filename", "design_chains"}.issubset(fields):
        raise ValueError(f"{path}: expected filename and design_chains columns")
    matched = {}
    for index, row in enumerate(rows, 2):
        sample = Path(row.get("filename", "")).stem
        if sample not in selected:
            continue
        if sample in matched and matched[sample][0] != row:
            raise ValueError(f"{path}:{index}: conflicting result rows for {sample!r}")
        matched[sample] = (row, index)
    missing = selected - matched.keys()
    if missing:
        raise ValueError(
            f"{path}: no current filename matches successful samples: {', '.join(sorted(missing))}"
        )
    samples = {}
    for sample in sorted(selected):
        row, index = matched[sample]
        context = f"{path}:{index} ({sample})"
        relative = Path(row["filename"])
        structure = (task / relative).resolve()
        if (
            relative.is_absolute()
            or not structure.is_relative_to(task)
            or any(
                part.lower()
                in {
                    "validate",
                    "postprocess",
                    "original",
                    "refolds",
                    "success",
                    "af2success",
                }
                for part in relative.parts
            )
        ):
            raise ValueError(
                f"{context}: filename must identify a generated task structure"
            )
        if not structure.is_file():
            raise ValueError(
                f"{context}: current generated structure is missing: {structure}"
            )
        samples[sample] = ClusterSample(sample, structure, _design_chains(row, context))
    return samples


def load_job(job, *, criteria=()):
    requested = _names(criteria)
    grouped = (
        _postprocess_groups(job, requested)
        if job.method == "postprocess"
        else _af2_groups(job, requested)
    )
    selected = {sample for values in grouped.values() for sample in values}
    samples = _samples(job.task_dir, selected) if selected else {}
    groups = {name: tuple(sorted(values)) for name, values in grouped.items()}
    sources = {
        name: {sample: tuple(sorted(labels)) for sample, labels in values.items()}
        for name, values in grouped.items()
    }
    return ClusterInput(job, samples, groups, sources)
