"""Prepare, run, and independently inspect fresh paired AF2 CLI work.

The ordinary case uses an unmodified ``python -m partiflow validate`` process.
Interruption/resume and constructor-blocked updates are separate cases. Model
execution is explicit; importing this module or preparing inputs loads no model.
All generated files belong in the supplied directory outside the checkout.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import json
import math
import operator
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time


CHECKOUT = Path(__file__).resolve().parents[2]
AA = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
              "ARNDCQEGHILKMFPSTWYV"))
PRESET = {
    "methods": ["af2_ig"], "reduction": "all_and_mean",
    "rules": [
        {"metric": "plddt", "op": ">=", "value": 0.8, "scope": "complex"},
        {"metric": "ptm", "op": ">=", "value": 0.55, "scope": "complex"},
        {"metric": "iptm", "op": ">=", "value": 0.4, "scope": "complex"},
        {"metric": "ipae_normalized", "op": "<=", "value": 0.3, "scope": "complex"},
        {"metric": "plddt", "op": ">=", "value": 0.8, "scope": "monomer"},
        {"metric": "bound_unbound_rmsd", "op": "<=", "value": 2.5, "scope": "paired"},
    ],
}
OPERATORS = {">=": operator.ge, "<=": operator.le}


class AcceptanceError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AcceptanceError(str(message))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def yaml_read(path):
    import yaml

    return yaml.safe_load(Path(path).read_text())


def external_root(path, *excluded):
    path = Path(path).expanduser().resolve()
    for directory in excluded:
        directory = Path(directory).expanduser().resolve()
        require(path != directory and directory not in path.parents and path not in directory.parents,
                f"Output directory must be outside {directory}: {path}")
    return path


def table(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames, f"Missing CSV header: {path}")
        return list(reader)


def pdb(path):
    """Read the saved PDB directly, keeping residue order and atom coverage."""
    result = {}
    started = False
    for line in Path(path).read_text().splitlines():
        if line.startswith("MODEL"):
            require(not started, f"Multiple structure models: {path}")
            started = True
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        chain = line[21:22]
        locator = (int(line[22:26]), line[26:27].strip())
        group = result.setdefault(chain, {}).setdefault(locator, {"name": line[17:20].strip(), "atoms": {}})
        require(group["name"] == line[17:20].strip(), f"Conflicting residue identity: {path}")
        name = line[12:16].strip()
        require(name not in group["atoms"], f"Duplicate atom {chain}/{locator}/{name}: {path}")
        xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
        require(all(math.isfinite(value) for value in xyz), f"Nonfinite coordinates: {path}")
        group["atoms"][name] = {"xyz": xyz, "element": line[76:78].strip() or name[0],
                                "bfactor": float(line[60:66])}
    require(result, f"No atoms: {path}")
    for residues in result.values():
        for group in residues.values():
            require(group["name"] in AA and "CA" in group["atoms"], f"Nonstandard or incomplete residue: {path}")
    return result


def sequence(structure, chain):
    require(chain in structure, f"Missing chain: {chain}")
    return "".join(AA[group["name"]] for group in structure[chain].values())


def points(structure, chains, names=None, selected=None):
    import numpy as np

    result = {}
    for chain in chains:
        for index, (locator, group) in enumerate(structure[chain].items(), 1):
            if selected is not None and (chain, *locator) not in selected:
                continue
            for name, atom in group["atoms"].items():
                if names is not None and name not in names:
                    continue
                if names is None and atom["element"].upper() in {"H", "D", "T"}:
                    continue
                result[(chain, index, name)] = np.asarray(atom["xyz"], dtype=float)
    if names is not None:
        count = len(selected) if selected is not None else sum(len(structure[chain]) for chain in chains)
        require(len(result) == count * len(names), "Missing required comparison atoms")
    require(result, "Empty coordinate comparison")
    return result


def paired_points(reference, moving, chains, names=None, selected=None):
    import numpy as np

    left, right = (points(item, chains, names, selected) for item in (reference, moving))
    require(left.keys() == right.keys(), "Comparison atom coverage differs")
    keys = sorted(left)
    return np.asarray([left[key] for key in keys]), np.asarray([right[key] for key in keys])


def rmsd(reference, moving, fit=None):
    import numpy as np

    left, right = fit if fit is not None else (reference, moving)
    require(left.shape == right.shape and len(left) >= 3, "Insufficient matched alignment coordinates")
    centers = left.mean(axis=0), right.mean(axis=0)
    centered_left, centered_right = left - centers[0], right - centers[1]
    require(np.linalg.matrix_rank(centered_left) >= 2 and np.linalg.matrix_rank(centered_right) >= 2,
            "Degenerate alignment coordinates")
    u, _, vt = np.linalg.svd(centered_right.T @ centered_left)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
    aligned = (moving - centers[1]) @ (u @ correction @ vt) + centers[0]
    return float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))


def close(actual, expected, label, *, atol=1e-4):
    require(isinstance(actual, (int, float)) and not isinstance(actual, bool)
            and math.isfinite(actual) and math.isfinite(expected), f"Nonfinite metric: {label}")
    require(math.isclose(float(actual), float(expected), rel_tol=1e-5, abs_tol=atol),
            f"Metric differs: {label}: {actual} vs {expected}")


def expected_keys(models):
    return [f"{protocol}_model{model}_sample0" for protocol in ("complex", "monomer") for model in models]


def validation_config(models, sources):
    return {"methods": ["af2_ig"], "sequence": {"seed": 101,
            "sources": [{"source": name, "label": "self" if name == "gt" else name, "count": 1}
                        for name in sources]},
            "af2": {"models": models, "samples": 1, "recycles": 3, "seed": 101, "mode": "both"},
            "criteria": ["af2_ig_peptide"]}


def prepare(design_root, model_root, output_root, *, python=sys.executable):
    import yaml

    source = Path(design_root).expanduser().resolve()
    root = external_root(output_root, CHECKOUT, source)
    require(not root.exists() or not any(root.iterdir()), "Choose a fresh output directory")
    presets = CHECKOUT / "configs/preset.yaml"
    require(yaml_read(presets)["presets"]["af2_ig_peptide"] == PRESET, "Default AF2 preset changed")
    inputs = []
    for name in ("denovo", "motif"):
        directory = source / name
        rows = table(directory / "result.csv")
        require(len(rows) == 1 and not rows[0].get("error") and rows[0].get("filename"),
                f"Expected one successful raw candidate: {directory}")
        row = rows[0]
        require(not row.get("reward_method"), f"Expected raw generation: {directory}")
        path = (directory / row["filename"]).resolve()
        require(path.is_relative_to(directory.resolve()) and path.suffix == ".pdb" and path.is_file(), path)
        config = yaml_read(directory / "resolved_config.yaml")
        require(config["tasks"][name]["task_type"] == name, f"Wrong task family: {directory}")
        require(all(config["_runtime"][key] == 1 for key in
                    ("planned_base_samples", "replicas_per_base", "planned_candidates")), "Unexpected raw budget")
        structure = pdb(path)
        require(sequence(structure, row["design_chain_id"]) == row["design_sequence"], "Design sequence differs")
        fixed = json.loads(row.get("fixed_restypes") or "[]")
        require(name != "motif" or fixed, "Motif input needs fixed identities")
        inputs.append({"task": name, "source_directory": str(directory), "filename": path.name,
                       "row": row, "files": ["result.csv", "resolved_config.yaml", path.name]})
    root.mkdir(parents=True, exist_ok=True)
    cases = {}
    for name, tasks, models, sources in (
        ("public", ["denovo", "motif"], [2, 1], ["gt", "proteinmpnn", "solublempnn"]),
        ("interruption", ["denovo"], [1], ["gt"]),
    ):
        config = root / "configs" / (name + ".yaml")
        config.parent.mkdir(exist_ok=True)
        config.write_text(yaml.safe_dump(validation_config(models, sources), sort_keys=False))
        cases[name] = {"tasks": tasks, "models": models, "sources": sources, "config": str(config),
                       "expected_predictions": expected_keys(models), "sequence_slots": len(tasks) * len(sources),
                       "prediction_count": len(tasks) * len(sources) * len(models) * 2}
        for item in inputs:
            if item["task"] not in tasks:
                continue
            destination = root / name / item["task"]
            destination.mkdir(parents=True)
            for filename in item["files"]:
                shutil.copy2(Path(item["source_directory"]) / filename, destination / filename)
    for item in inputs:
        destination = root / "initial_inputs" / item["task"]
        destination.mkdir(parents=True)
        for filename in item["files"]:
            shutil.copy2(Path(item["source_directory"]) / filename, destination / filename)
    shutil.copy2(presets, root / "configs/presets.yaml")
    manifest = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "checkout": str(CHECKOUT),
                "source_root": str(source), "model_root": str(Path(model_root).expanduser().resolve()),
                "output_root": str(root), "python": str(python), "inputs": inputs, "cases": cases,
                "presets": str(root / "configs/presets.yaml"), "prepare_argv": sys.argv}
    save(root / "manifest.json", manifest)
    return manifest


def manifest_at(root):
    root = external_root(root, CHECKOUT)
    manifest = read(root / "manifest.json")
    require(Path(manifest["output_root"]) == root, "Manifest output root differs")
    require(Path(manifest["checkout"]).resolve() == CHECKOUT, "Manifest checkout differs")
    return root, manifest


def public_argv(manifest, case="public", extra=()):
    root = Path(manifest["output_root"])
    config = manifest["cases"][case]
    return ["validate", *(str(root / case / name) for name in config["tasks"]),
            "--config", config["config"], "--presets", manifest["presets"],
            "--model-dir", manifest["model_root"], *extra]


def execute(manifest, label, argv, *, public=None, expected=0, cpu=False):
    root = Path(manifest["output_root"])
    directory = root / "commands"
    directory.mkdir(exist_ok=True)
    require(not (directory / (label + ".json")).exists(), f"Command already recorded: {label}")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(CHECKOUT)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if cpu:
        environment.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    record = {"argv": list(argv), "public_argv": list(public or []), "cwd": str(CHECKOUT),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
              "stdout": str(directory / (label + ".stdout")), "stderr": str(directory / (label + ".stderr")),
              "output_directories": {name: [str(root / name / task) for task in item["tasks"]]
                                     for name, item in manifest["cases"].items()},
              "environment": {key: environment.get(key) for key in
                              ("CUDA_VISIBLE_DEVICES", "XLA_PYTHON_CLIENT_PREALLOCATE", "OMP_NUM_THREADS",
                               "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH")},
              "configs": {name: Path(item["config"]).read_text() for name, item in manifest["cases"].items()},
              "presets": Path(manifest["presets"]).read_text()}
    save(directory / (label + ".json"), record)
    started = time.monotonic()
    try:
        with (directory / (label + ".stdout")).open("w") as stdout, (directory / (label + ".stderr")).open("w") as stderr:
            result = subprocess.run(argv, cwd=CHECKOUT, env=environment, stdout=stdout, stderr=stderr, check=False)
        record.update(returncode=result.returncode, status="completed" if result.returncode == expected else "failed")
    except OSError as exc:
        record.update(returncode=None, status="failed", error=str(exc))
    record["elapsed_seconds"] = time.monotonic() - started
    save(directory / (label + ".json"), record)
    require(record["returncode"] == expected, f"{label} exited {record['returncode']}, expected {expected}; see {directory}")
    return record


def input_row(manifest, task):
    return next(item for item in manifest["inputs"] if item["task"] == task)["row"]


def fixed_positions(row, structure):
    chain = row["design_chain_id"]
    locators = list(structure[chain])
    fixed = {}
    for entry in json.loads(row.get("fixed_restypes") or "[]"):
        require(entry["chain"] == chain, "Fixed identity is on another chain")
        locator = (int(entry["res_id"]), str(entry.get("ins_code", "")))
        require(locator in structure[chain], "Fixed motif locator is absent")
        identity = entry.get("restype", entry.get("res_name"))
        actual = structure[chain][locator]["name"]
        require(identity in {actual, AA[actual]}, "Fixed identity differs from structure")
        fixed[locators.index(locator) + 1] = AA[actual]
    return fixed


def prediction_values(original, folded, chain, prediction, contig=""):
    import numpy as np

    confidence, metrics = prediction["confidence"], prediction["metrics"]
    protocol = prediction["protocol"]
    require(confidence.get("protocol") == protocol, "Confidence protocol differs")
    with np.load(prediction["confidence_path"], allow_pickle=False) as arrays:
        require(set(arrays.files) == {"pae_matrix"}, "Unexpected persisted confidence arrays")
        pae = np.asarray(arrays["pae_matrix"], dtype=float)
    length = sum(len(residues) for residues in folded.values())
    require(pae.shape == (length, length) and np.isfinite(pae).all() and (pae >= 0).all(), "Invalid PAE matrix")
    close(confidence["pae_matrix_mean_angstrom"], float(pae.mean()), "native PAE matrix mean")
    close(metrics["pae_matrix_mean_angstrom"], float(pae.mean()), "PAE matrix mean")
    for source, target, scale in (("plddt_normalized", "plddt", 1), ("ptm", "ptm", 1),
                                  ("pae", "pae_normalized", 1), ("pae", "pae_angstrom", 31)):
        value = confidence[source]
        require(0 <= value <= 1, f"Confidence out of range: {source}")
        close(metrics[target], value * scale, target)
    close(confidence["plddt"], 100 * confidence["plddt_normalized"], "native pLDDT scale")
    ca_values = [group["atoms"]["CA"]["bfactor"] / 100 for group in folded[chain].values()]
    close(confidence["plddt_normalized"], sum(ca_values) / len(ca_values), "PDB pLDDT", atol=2e-4)
    all_ca = [group["atoms"]["CA"]["bfactor"] / 100 for residues in folded.values() for group in residues.values()]
    close(metrics["plddt_total"], sum(all_ca) / len(all_ca), "total PDB pLDDT", atol=2e-4)
    close(metrics["pae_angstrom"], 31 * metrics["pae_normalized"], "PAE scale")
    for name, names in (("design_rmsd", ("N", "CA", "C")),
                        ("binder_scRMSD_bb3", ("N", "CA", "C")), ("binder_design_rmsd", ("CA",))):
        close(metrics[name], rmsd(*paired_points(original, folded, [chain], names)), name)
    if protocol == "complex":
        contexts = [key for key in original if key != chain]
        require(contexts, "Complex has no receptor context")
        for source, target, scale in (("iptm", "iptm", 1), ("ipae", "ipae_normalized", 1), ("ipae", "ipae_angstrom", 31)):
            require(0 <= confidence[source] <= 1, f"Confidence out of range: {source}")
            close(metrics[target], confidence[source] * scale, target)
        count = len(folded[chain])
        boundary = length - count
        interaction = (pae[boundary:, :boundary].mean() + pae[:boundary, boundary:].mean()) / 2
        close(metrics["pae_interaction"], float(interaction), "interaction PAE")
        close(metrics["context_aligned_design_rmsd"],
              rmsd(*paired_points(original, folded, [chain], ("N", "CA", "C")),
                   fit=paired_points(original, folded, contexts, ("CA",))), "context-aligned RMSD")
        close(metrics["complex_design_rmsd"],
              rmsd(*paired_points(original, folded, [*contexts, chain], ("N", "CA", "C"))), "complex RMSD")
    if "motif_rmsd_design" in metrics:
        positions = set()
        for segment in contig.split("/"):
            match = re.fullmatch(r"(?:([^:]+):|([A-Za-z]))(-?\d+)([A-Za-z]?)(?:-(-?\d+)([A-Za-z]?))?", segment)
            if not match or (match[1] or match[2]) != chain:
                continue
            first, last = int(match[3]), int(match[5] or match[3])
            for number in range(first, last + 1):
                insertion = match[4] if number == first else (match[6] or "") if number == last else ""
                positions.add((chain, number, insertion))
        close(metrics["motif_rmsd_design"],
              rmsd(*paired_points(original, folded, [chain], ("N", "CA", "C"), positions)), "motif RMSD")
    return length


def scientific_decision(record):
    groups = {"complex": [], "monomer": [], "paired": list(record["paired_metrics"].values())}
    for prediction in record["predictions"].values():
        groups[prediction["protocol"]].append(prediction["metrics"])
    failed = False
    for rule in PRESET["rules"]:
        values = [item.get(rule["metric"]) for item in groups[rule["scope"]]]
        if not values or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in values):
            return "unavailable"
        failed |= any(not OPERATORS[rule["op"]](value, rule["value"]) for value in [*values, sum(values) / len(values)])
    return "failed" if failed else "passed"


def audit_case(manifest, case="public", *, check_command=True):
    root = Path(manifest["output_root"])
    spec = manifest["cases"][case]
    require(yaml_read(manifest["presets"])["presets"]["af2_ig_peptide"] == PRESET, "AF2 preset changed")
    require(yaml_read(spec["config"]) == validation_config(spec["models"], spec["sources"]), "Validation protocol changed")
    if check_command:
        label = "public" if case == "public" else "resume"
        command = read(root / "commands" / (label + ".json"))
        require(command["returncode"] == 0 and command["status"] == "completed", "CLI did not complete")
        require(command["public_argv"] == public_argv(manifest, case), "Public CLI arguments differ")
        if case == "public":
            require(command["argv"] == [manifest["python"], "-m", "partiflow", *public_argv(manifest)],
                    "Normal run was instrumented")
    total_predictions = total_slots = total_pairs = 0
    decisions = {"passed": 0, "failed": 0, "unavailable": 0, "not_applicable": 0}
    details = []
    for task in spec["tasks"]:
        directory = root / case / task
        row = input_row(manifest, task)
        for filename in next(item for item in manifest["inputs"] if item["task"] == task)["files"]:
            require((directory / filename).read_bytes() == (root / "initial_inputs" / task / filename).read_bytes(),
                    f"Design input changed: {directory / filename}")
        original = pdb(directory / row["filename"])
        chain = row["design_chain_id"]
        fixed = fixed_positions(row, original)
        paths = sorted((directory / "validate/af2_ig/refolds" / Path(row["filename"]).stem).glob("seq*.json"))
        require(len(paths) == len(spec["sources"]), "Sequence slot count differs")
        metric_rows = table(directory / "validate/af2_ig/metrics.csv")
        success_rows = table(directory / "validate/af2_ig/success.csv")
        require(len(metric_rows) == len(paths) * (len(spec["expected_predictions"]) + len(spec["models"])), "Metric row count differs")
        require(len(success_rows) == len(paths), "Criterion row count differs")
        state = read(directory / "validate/af2_ig/.state/config.json")
        require(state["config"]["af2"] == yaml_read(spec["config"])["af2"], "Saved AF2 protocol differs")
        local_decisions = dict.fromkeys(decisions, 0)
        for index, path in enumerate(paths):
            record = read(path)
            require(record["status"] == "complete" and record["method"] == "af2_ig", "Incomplete validation slot")
            require(record["seq_index"] == index and record["source"] == spec["sources"][index]
                    and record["source_index"] == 1 and record["candidate_id"] == Path(row["filename"]).stem
                    and int(record["base_sample_id"]) == int(row["base_sample_id"]), "Sequence branch identity differs")
            identity = {"sample": record["candidate_id"], "base_sample_id": str(record["base_sample_id"]),
                        "sequence_index": str(index), "source": record["source"],
                        "source_label": record["source_label"], "source_index": "1"}
            for csv_row in [item for item in metric_rows + success_rows if int(item["sequence_index"]) == index]:
                require(all(csv_row.get(name) == value for name, value in identity.items()), "CSV branch identity differs")
            require(record["expected_predictions"] == spec["expected_predictions"] and
                    set(record["predictions"]) == set(spec["expected_predictions"]), "Prediction keys differ")
            variant = record["sequence"]
            require(len(variant) == len(original[chain]) and all(letter in AA.values() for letter in variant), "Invalid variant sequence")
            if record["source"] == "gt":
                require(variant == row["design_sequence"], "Original sequence changed")
            require(record["fixed_positions"] == sorted(fixed), "Fixed motif position list differs")
            require(all(variant[position - 1] == identity for position, identity in fixed.items()), "Fixed motif identity changed")
            folded = {}
            for key, prediction in record["predictions"].items():
                protocol, model = key.split("_model")
                require(prediction["key"] == key and prediction["protocol"] == protocol
                        and prediction["model"] == int(model.split("_", 1)[0]) and prediction["sample"] == 0,
                        "Prediction/model/sample association differs")
                require(prediction["status"] == prediction["metrics_status"] == "complete", "Incomplete prediction/metrics")
                for field in ("structure_path", "confidence_path"):
                    artifact = Path(prediction[field]).resolve()
                    require(artifact.is_relative_to(path.parent.resolve()) and artifact.is_file() and artifact.stat().st_size,
                            f"Missing or misplaced prediction file: {artifact}")
                structure = pdb(prediction["structure_path"])
                require(set(structure) == (set(original) if protocol == "complex" else {chain}), "Prediction chain set differs")
                for identifier in structure:
                    require(list(structure[identifier]) == list(original[identifier]), "Source residue locators changed")
                    wanted = variant if identifier == chain else sequence(original, identifier)
                    require(sequence(structure, identifier) == wanted, "Prediction sequence differs")
                prediction_values(original, structure, chain, prediction, row.get("contig_string", ""))
                rows = [item for item in metric_rows if int(item["sequence_index"]) == index and item["prediction"] == key]
                require(len(rows) == 1 and rows[0]["protocol"] == protocol and rows[0]["status"] == "complete"
                        and rows[0]["metrics_status"] == "complete" and not rows[0].get("error"), "Prediction CSV row differs")
                for name, value in prediction["metrics"].items():
                    close(float(rows[0][name]), value, "CSV " + name)
                folded[key] = structure
                total_predictions += 1
            wanted_pairs = {f"model{model}_sample0" for model in spec["models"]}
            require(set(record["paired_metrics"]) == wanted_pairs, "Matched complex/monomer pairs differ")
            for suffix, metrics in record["paired_metrics"].items():
                complex_structure, monomer_structure = folded["complex_" + suffix], folded["monomer_" + suffix]
                ca = paired_points(complex_structure, monomer_structure, [chain], ("CA",))
                close(metrics["bound_unbound_ca_rmsd"], rmsd(*ca), "paired CA RMSD")
                close(metrics["bound_unbound_rmsd"],
                      rmsd(*paired_points(complex_structure, monomer_structure, [chain]), fit=ca), "paired heavy-atom RMSD")
                rows = [item for item in metric_rows if int(item["sequence_index"]) == index and item["prediction"] == "paired_" + suffix]
                require(len(rows) == 1 and rows[0]["protocol"] == "paired", "Paired CSV row differs")
                for name, value in metrics.items():
                    close(float(rows[0][name]), value, "paired CSV " + name)
                total_pairs += 1
            decision = scientific_decision(record)
            require(record["decisions"]["af2_ig_peptide"]["status"] == decision, "Scientific criterion decision differs")
            rows = [item for item in success_rows if int(item["sequence_index"]) == index]
            require(len(rows) == 1 and rows[0]["criteria"] == "af2_ig_peptide" and rows[0]["status"] == decision,
                    "Criterion CSV row differs")
            decisions[decision] += 1
            local_decisions[decision] += 1
            details.append({"task": task, "source": record["source"], "sequence": variant,
                            "fixed_positions": sorted(fixed), "criterion_status": decision})
            total_slots += 1
        summary = read(directory / "validate/af2_ig/summary.json")
        require(summary["complete_candidates"] == 1 and summary["observed_result_rows"] == 1
                and summary["complete_sequence_slots"] == summary["requested_sequence_slots"]
                == summary["observed_sequence_slots"] == summary["planned_sequence_slots"] == len(paths)
                and summary["planned_base_samples"] == summary["replicas_per_base"] == summary["planned_candidates"] == 1,
                "Summary completion differs")
        require(summary["criteria_counts"] == {"af2_ig_peptide": local_decisions}
                and summary["criteria"] == ["af2_ig_peptide"] and not summary["metrics_only"], "Summary quality decisions differ")
    require(total_slots == spec["sequence_slots"] and total_predictions == spec["prediction_count"], "Requested work count differs")
    return {"software_complete": True, "case": case, "sequence_slots": total_slots,
            "predictions": total_predictions, "matched_pairs": total_pairs, "scientific_decisions": decisions,
            "protocol": "standalone paired pTM complex/monomer; separate from Multimer v3 design reward",
            "confidence_limit": "pTM/ipTM scalars are checked against persisted native values; PAE alone cannot reconstruct their logits",
            "slots": details}


def artifact_paths(manifest, case="public"):
    root = Path(manifest["output_root"])
    result = set()
    for task in manifest["cases"][case]["tasks"]:
        for path in (root / case / task / "validate/af2_ig/refolds").rglob("seq*.json"):
            for prediction in read(path).get("predictions", {}).values():
                for field in ("structure_path", "confidence_path"):
                    if prediction.get(field):
                        result.add(Path(prediction[field]))
    return sorted(result)


def keep_files(paths, root, label):
    directory = Path(root) / "saved_files" / label
    require(not directory.exists(), f"Saved file set already exists: {label}")
    records = []
    for index, path in enumerate(paths):
        path = Path(path).resolve()
        destination = directory / f"{index:04d}{path.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append({"path": str(path), "copy": str(destination), "mtime_ns": path.stat().st_mtime_ns,
                        "size": path.stat().st_size})
    save(directory / "files.json", records)
    return records


def unchanged(records):
    for item in records:
        path = Path(item["path"])
        require(path.is_file() and path.read_bytes() == Path(item["copy"]).read_bytes()
                and path.stat().st_mtime_ns == item["mtime_ns"] and path.stat().st_size == item["size"],
                f"Completed file changed: {path}")


def child_argv(manifest, mode, label):
    return [manifest["python"], str(Path(__file__).resolve()), "_child", "--output-root", manifest["output_root"],
            "--mode", mode, "--label", label]


@contextmanager
def child_hooks(manifest, mode, label):
    from unittest.mock import patch
    from partiflow.workflows import af2, sequence as sequence_module, validate

    root = Path(manifest["output_root"])
    guarded_modes = {"refilter", "recompute", "mutation-coordinate", "mutation-identity", "mutation-config"}
    observed = {"mode": mode, "af2_constructors": 0 if mode in guarded_modes else None,
                "mpnn_constructors": 0 if mode in guarded_modes else None,
                "predict_calls": [] if mode == "resume" else None,
                "yielded_keys": [] if mode == "resume" else None}
    original_atomic, original_predict = validate._atomic_json, af2.AF2Backend.predict

    def blocked(kind):
        def constructor(*args, **kwargs):
            observed[kind + "_constructors"] += 1
            raise AcceptanceError(f"Model constructor called during model-free update: {kind}")
        return constructor

    def atomic(path, value):
        original_atomic(path, value)
        if mode != "interrupt" or Path(path).name != "seq0.json" or not isinstance(value, dict):
            return
        predictions = value.get("predictions", {})
        if set(predictions) == {"complex_model1_sample0"} and predictions["complex_model1_sample0"].get("status") == "complete":
            require(Path(path).resolve().is_relative_to((root / "interruption").resolve()), "Fault hook reached normal case")
            require(all(Path(predictions["complex_model1_sample0"][field]).is_file()
                        for field in ("structure_path", "confidence_path")), "Complex was not fully published")
            save(root / "commands" / (label + ".hook.json"), {**observed, "record": str(path), "exit_code": 86})
            os._exit(86)

    def predict(self, data, variant, output, completed_keys=()):
        observed["predict_calls"].append({"completed_keys": sorted(completed_keys), "output": str(output)})
        for item in original_predict(self, data, variant, output, completed_keys):
            observed["yielded_keys"].append(item["key"])
            yield item

    try:
        with patch.object(validate, "_atomic_json", side_effect=atomic):
            if mode in guarded_modes:
                with patch.object(af2.AF2Backend, "__init__", blocked("af2")), patch.object(sequence_module.MPNNBackend, "__init__", blocked("mpnn")):
                    yield observed
            elif mode == "resume":
                with patch.object(af2.AF2Backend, "predict", predict):
                    yield observed
            else:
                yield observed
    finally:
        destination = root / "commands" / (label + ".hook.json")
        if mode != "interrupt" or not destination.exists():
            save(destination, observed)


def run_child(root, mode, label):
    from partiflow.cli import main as public_main

    _, manifest = manifest_at(root)
    case = "interruption" if mode in {"interrupt", "resume"} else "public"
    flags = ["--refilter"] if mode == "refilter" or mode.startswith("mutation-") else ["--recompute-metrics"] if mode == "recompute" else []
    with child_hooks(manifest, mode, label):
        return public_main(public_argv(manifest, case, flags))


def run_public(root):
    root, manifest = manifest_at(root)
    argv = public_argv(manifest)
    execute(manifest, "plan", [manifest["python"], "-m", "partiflow", *argv, "--plan"], public=[*argv, "--plan"], cpu=True)
    plan = read(root / "commands/plan.stdout")
    require(sum(item["requested_sequence_slots"] for item in plan["tasks"]) == 6
            and all(item["predictions_per_slot"] == 4 for item in plan["tasks"]), "Public plan budget differs")
    execute(manifest, "public", [manifest["python"], "-m", "partiflow", *argv], public=argv)
    result = audit_case(manifest)
    save(root / "audit_public.json", result)
    return result


def interrupt(root):
    root, manifest = manifest_at(root)
    execute(manifest, "interrupt", child_argv(manifest, "interrupt", "interrupt"),
            public=public_argv(manifest, "interruption"), expected=86)
    hook = read(root / "commands/interrupt.hook.json")
    require(hook["exit_code"] == 86, "Expected the explicit interruption hook")
    record = read(hook["record"])
    require(set(record["predictions"]) == {"complex_model1_sample0"}, "Interruption did not fall between paired predictions")
    files = keep_files(artifact_paths(manifest, "interruption"), root, "interrupted_complex")
    require(len(files) == 2, "Expected exactly one completed PDB/NPZ pair")
    result = {"instrumented": True, "exit_code": 86, "completed_keys": list(record["predictions"]),
              "missing_keys": ["monomer_model1_sample0"], "saved_files": len(files)}
    save(root / "audit_interrupted.json", result)
    return result


def resume(root):
    root, manifest = manifest_at(root)
    files = read(root / "saved_files/interrupted_complex/files.json")
    unchanged(files)
    execute(manifest, "resume", child_argv(manifest, "resume", "resume"), public=public_argv(manifest, "interruption"))
    observed = read(root / "commands/resume.hook.json")
    require(observed["predict_calls"] == [{"completed_keys": ["complex_model1_sample0"],
            "output": str(root / "interruption/denovo/validate/af2_ig/refolds" / Path(input_row(manifest, "denovo")["filename"]).stem)}],
            "Resume did not request only the missing paired stage")
    require(observed["yielded_keys"] == ["monomer_model1_sample0"], "Resume recomputed a completed prediction")
    unchanged(files)
    result = {**audit_case(manifest, "interruption"), "instrumented": True, "unchanged_completed_files": len(files),
              "executed_prediction_keys": observed["yielded_keys"]}
    save(root / "audit_resume.json", result)
    return result


def guarded(manifest, mode, label, *, expected=0):
    extra = ["--recompute-metrics"] if mode == "recompute" else ["--refilter"]
    execute(manifest, label, child_argv(manifest, mode, label), public=public_argv(manifest, "public", extra), expected=expected, cpu=True)
    observed = read(Path(manifest["output_root"]) / "commands" / (label + ".hook.json"))
    require(observed["af2_constructors"] == observed["mpnn_constructors"] == 0, "Model-free operation constructed a model")
    return observed


def changed_pdb(contents, identity=False):
    lines = contents.decode().splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
            if identity:
                replacement = "GLY" if line[17:20] != "GLY" else "ALA"
                lines[index] = line[:17] + replacement + line[20:]
            else:
                value = float(line[30:38]) + 0.125
                require(len(f"{value:8.3f}") == 8, "Coordinate change would alter field width")
                lines[index] = line[:30] + f"{value:8.3f}" + line[38:]
            result = "".join(lines).encode()
            require(len(result) == len(contents) and result != contents, "Mutation must retain size")
            return result
    raise AcceptanceError("No PDB CA atom available for mutation")


def model_free(root):
    root, manifest = manifest_at(root)
    audit_case(manifest)
    prediction_files = keep_files(artifact_paths(manifest), root, "model_free_predictions")
    require(len(prediction_files) == 48, "Expected 24 completed structure/confidence pairs")
    guarded(manifest, "refilter", "refilter")
    unchanged(prediction_files)
    audit_case(manifest)
    slot = next((root / "public/motif/validate/af2_ig/refolds").rglob("seq0.json"))
    original_record, original_stat = slot.read_bytes(), slot.stat()
    value = json.loads(original_record)
    key = manifest["cases"]["public"]["expected_predictions"][0]
    value["predictions"][key]["metrics"]["binder_design_rmsd"] = 12345.0
    save(slot, value)
    try:
        guarded(manifest, "recompute", "recompute")
        audit_case(manifest)
    except BaseException:
        slot.write_bytes(original_record)
        os.utime(slot, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        raise
    unchanged(prediction_files)
    protected = sorted(path for task in (root / "public").iterdir()
                       for path in (task / "validate/af2_ig").rglob("*") if path.is_file())
    saved = keep_files(protected, root, "before_input_mutations")
    input_path = root / "public/denovo" / input_row(manifest, "denovo")["filename"]
    mutations = [("mutation-coordinate", input_path, changed_pdb(input_path.read_bytes())),
                 ("mutation-identity", input_path, changed_pdb(input_path.read_bytes(), identity=True))]
    config = Path(manifest["cases"]["public"]["config"])
    config_bytes = config.read_bytes()
    require(b"recycles: 3" in config_bytes, "Expected explicit recycle setting")
    mutations.append(("mutation-config", config, config_bytes.replace(b"recycles: 3", b"recycles: 4", 1)))
    for label, path, replacement in mutations:
        contents, stat = path.read_bytes(), path.stat()
        require(len(contents) == len(replacement), "Mutation changed size")
        try:
            path.write_bytes(replacement)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
            guarded(manifest, label, label, expected=1)
            output = (root / "commands" / (label + ".stdout")).read_text() + (root / "commands" / (label + ".stderr")).read_text()
            require("Validation inputs/settings changed" in output, f"Wrong mutation failure: {label}")
            unchanged(saved)
        finally:
            path.write_bytes(contents)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    unchanged(prediction_files)
    result = {**audit_case(manifest), "model_free": True, "af2_constructors": 0, "mpnn_constructors": 0,
              "unchanged_prediction_files": len(prediction_files), "recomputed_changed_metric": "binder_design_rmsd",
              "same_size_mtime_mutations_rejected": [item[0] for item in mutations]}
    save(root / "audit_model_free.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subcommands.add_parser("prepare")
    prepare_parser.add_argument("--design-root", type=Path, required=True)
    prepare_parser.add_argument("--model-root", type=Path, required=True)
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument("--python", default=sys.executable)
    for name in ("run", "audit", "model-free", "interrupt", "resume", "_child"):
        command = subcommands.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "_child":
            command.add_argument("--mode", choices=("interrupt", "resume", "refilter", "recompute", "mutation-coordinate", "mutation-identity", "mutation-config"), required=True)
            command.add_argument("--label", required=True)
    args = parser.parse_args(argv)
    if args.command == "_child":
        return run_child(args.output_root, args.mode, args.label)
    if args.command == "prepare":
        result = prepare(args.design_root, args.model_root, args.output_root, python=args.python)
    elif args.command == "run":
        result = run_public(args.output_root)
    elif args.command == "audit":
        root, manifest = manifest_at(args.output_root)
        result = audit_case(manifest)
        save(root / "audit_public.json", result)
    else:
        result = {"model-free": model_free, "interrupt": interrupt, "resume": resume}[args.command](args.output_root)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AcceptanceError, OSError, ValueError, KeyError) as exc:
        print(json.dumps({"software_complete": False, "error": str(exc)}), file=sys.stderr)
        raise SystemExit(1)
