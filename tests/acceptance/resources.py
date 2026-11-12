"""Run explicit public resource imports and CPU checks in an isolated directory."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

FM = ("partiflow", "partiflow-aux", "partiflow-dock", "partiflow-ss")
EXPECTED_SPLITS = {"train": 166019, "val": 3750, "test": 1250}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def compare_files(source, destination, *, block_size=4 * 1024 * 1024):
    """Read and compare every byte, including same-size changes and trailing data."""
    source, destination = Path(source), Path(destination)
    require(source.is_file() and destination.is_file(), f"Missing comparison file: {source}, {destination}")
    size = source.stat().st_size
    require(size > 0 and size == destination.stat().st_size, f"File sizes differ or are empty: {destination}")
    total = 0
    with source.open("rb") as first, destination.open("rb") as second:
        while True:
            left, right = first.read(block_size), second.read(block_size)
            require(left == right, f"File bytes differ at block beginning {total}: {destination}")
            total += len(left)
            if not left:
                break
    require(total == size, "File changed during comparison")
    return dict(source=str(source), destination=str(destination), bytes=size, all_bytes_equal=True)


def parse_output(value, *, lines):
    if lines:
        records = [json.loads(line) for line in value.splitlines() if line.strip()]
        require(records and all(isinstance(record, dict) for record in records), "CLI did not emit JSON object lines")
        return records
    record = json.loads(value)
    require(isinstance(record, dict), "CLI did not emit a JSON object")
    return record


def public_cli(output, label, arguments, *, expected=0, lines=True):
    command = [sys.executable, "-B", "-m", "partiflow.cli", *map(str, arguments)]
    logs = Path(output) / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / f"{label}.stdout.txt").open("w") as stdout, (logs / f"{label}.stderr.txt").open("w") as stderr:
        completed = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    save(logs / f"{label}.command.json", dict(command=command, returncode=completed.returncode))
    require(completed.returncode == expected, f"CLI {label} returned {completed.returncode}, expected {expected}; see {logs}")
    return parse_output((logs / f"{label}.stdout.txt").read_text(), lines=lines)


def indexed_rows(root):
    path = Path(root) / "indexes" / "split.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, "Empty dataset index")
    require(len({row["pdbid"] for row in rows}) == len(rows), "Duplicate source dataset identifiers")
    for row in rows:
        require(row["split"] in EXPECTED_SPLITS, f"Unexpected normalized split: {row['split']}")
    return rows


def inspect_full_data(root):
    root = Path(root).resolve()
    rows = indexed_rows(root)
    counts = dict(Counter(row["split"] for row in rows))
    require(counts == EXPECTED_SPLITS, f"Full raw dataset counts differ: {counts}")
    directory_counts = {split: sum(1 for _ in (root / "raw" / ("valid" if split == "val" else split)).glob("*.pdb"))
                        for split in EXPECTED_SPLITS}
    require(directory_counts == EXPECTED_SPLITS, f"Full raw directory counts differ: {directory_counts}")
    actual = Counter()
    for row in rows:
        path = (root / "indexes" / row["structure_path"]).resolve()
        require(path.is_relative_to(root / "raw"), "Indexed raw path leaves dataset")
        require(path.is_file() and path.stat().st_size > 0, f"Missing or empty indexed raw structure: {path}")
        expected = "valid" if row["split"] == "val" else row["split"]
        require(path.parent.name == expected, f"Source split mapping differs: {path}")
        actual[row["split"]] += 1
    processed = sum(1 for path in (root / "processed").rglob("*.pt") if path.is_file())
    return rows, dict(status="PASS", counts=counts, directory_counts=directory_counts, raw_files_checked=sum(actual.values()),
                      processed_graphs_present=processed, full_preprocessing=False)


def prepare_subset(source_root, rows, output):
    """Package one real source structure per split; prepare only missing graphs."""
    from partiflow.training.prepare_data import build_indexes, prepare_graphs
    source_root, output = Path(source_root), Path(output)
    fixture = output / "pepinter-subset"
    selections = []
    for split in EXPECTED_SPLITS:
        row = next(row for row in rows if row["split"] == split)
        source = (source_root / "indexes" / row["structure_path"]).resolve()
        raw_split = "valid" if split == "val" else split
        destination = fixture / "raw" / raw_split / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        processed_source = (source_root / "indexes" / row["processed_path"]).resolve()
        reused = processed_source.is_file()
        if reused:
            processed = fixture / "processed" / split / processed_source.name
            processed.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(processed_source, processed)
        selections.append(dict(pdbid=row["pdbid"], split=split, raw_source=str(source), reused_processed_graph=reused))
    split_csv, _, counts = build_indexes(fixture, verify_counts=False)
    require(counts == {"train": 1, "val": 1, "test": 1}, "Subset contains unexpected structures")
    prepare_graphs(fixture, split_csv)
    subset_rows = indexed_rows(fixture)
    for row in subset_rows:
        path = (fixture / "indexes" / row["processed_path"]).resolve()
        require(path.is_file() and path.stat().st_size > 0, f"Subset graph preparation failed: {path}")
        require(path.parent.name == row["split"], "Processed split mapping differs")
    archive = output / "pepinter-three-structure-subset.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        for folder in ("indexes", "processed"):
            handle.add(fixture / folder, arcname=f"pepinter/{folder}")
    save(output / "subset.json", dict(counts=counts, selections=selections, archive=str(archive),
        note="Three actual structures only; this archive does not contain the full PepInter dataset."))
    return fixture, archive, selections


def check_imported_data(source, destination):
    import torch
    from partiflow.datasets.pdb_data import PepLightningDataModule
    destination = Path(destination)
    compared = []
    for folder in ("indexes", "processed"):
        for path in sorted((Path(source) / folder).rglob("*")):
            if path.is_file():
                compared.append(compare_files(path, destination / path.relative_to(source)))
    module = PepLightningDataModule(split_csv=str(destination / "indexes" / "split.csv"),
        processed_dir=str(destination / "processed"), structure_dir=str(destination / "raw"),
        validation_split="valid", num_workers=0)
    loaded = []
    for split in EXPECTED_SPLITS:
        dataset = module._get_dataset(split)
        require(len(dataset) == 1, f"Imported subset split size differs: {split}")
        graph = dataset[0]
        require(isinstance(graph.coords, torch.Tensor) and graph.coords.ndim == 3, "Imported graph coordinates malformed")
        require(tuple(graph.coords.shape[1:]) == (37, 3), "Imported graph is not atom37")
        require(bool(torch.isfinite(graph.coords).all()) and bool(graph.is_peptide.any()), "Imported real graph invalid")
        require(graph.coords.device.type == "cpu", "Dataset resource check left CPU")
        loaded.append(dict(split=split, sample_id=str(graph.id), coordinate_shape=list(graph.coords.shape)))
    require(module.validation_split == "val", "valid alias did not normalize to val")
    return dict(status="PASS", compared_files=compared, loaded_splits=loaded, valid_maps_to_val=True)


def compare_mpnn(source, downloaded):
    import torch
    before = torch.load(source, map_location="cpu", weights_only=False)
    after = torch.load(downloaded, map_location="cpu", weights_only=False)
    require(before.keys() == after.keys(), "MPNN checkpoint keys differ")
    left, right = before["model_state_dict"], after["model_state_dict"]
    require(left.keys() == right.keys() and len(left) == 118, "MPNN tensor keys/count differ")
    for name, value in left.items():
        require(value.dtype == right[name].dtype and value.shape == right[name].shape, f"MPNN tensor shape/type differs: {name}")
        require(value.device.type == right[name].device.type == "cpu", "MPNN comparison left CPU")
        require(torch.equal(value, right[name]), f"MPNN tensor values differ: {name}")
    for name in before.keys() - {"model_state_dict"}:
        require(before[name] == after[name], f"MPNN checkpoint metadata differs: {name}")
    return dict(status="PASS", tensors=118, bytes=Path(downloaded).stat().st_size)


def run(source_models, source_data, output):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Hide GPUs for resource acceptance")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(variable) == "1", f"Set {variable}=1")
    from partiflow.download import TARGETS
    source_models, source_data, output = (Path(path).resolve() for path in (source_models, source_data, output))
    require(not output.exists(), "Choose a fresh resource acceptance directory")
    output.mkdir(parents=True)
    result = dict(status="PENDING", python=sys.executable, source_models=str(source_models), source_data=str(source_data), phases={})
    save(output / "result.json", result)

    def phase(name, value):
        result["phases"][name] = value
        save(output / "result.json", result)
        print(name, "PASS", flush=True)

    empty = output / "empty-models"
    inventory = public_cli(output, "empty-list", ["download", "--list", "--model-dir", empty])
    require({record["target"] for record in inventory} == set(TARGETS), "Public inventory target coverage differs")
    require(all(record["status"] == "missing" for record in inventory), "Fresh root appears populated")
    require({record["target"] for record in inventory if record["url_configured"]} == {"af2", "proteinmpnn", "solublempnn"},
            "Configured URL status differs")
    missing = public_cli(output, "empty-check", ["check", "--all", "--model-dir", empty], expected=1, lines=False)
    require(missing["mode"] == "paths" and missing["status"] == "failed" and "loads" not in missing, "Path check unexpectedly loaded models")
    require(not empty.exists(), "Public listing/path check wrote the empty root")
    phase("empty_paths", dict(status="PASS", target_count=len(inventory)))

    unconfigured = output / "unconfigured-models"
    targets = [*FM, "autoencoder", "pepinter"]
    records = public_cli(output, "unconfigured", ["download", *targets, "--model-dir", unconfigured], expected=1)
    require(len(records) == 6 and all(record["status"] == "unconfigured" for record in records), "Unconfigured URLs were concealed")
    require(not unconfigured.exists(), "Unconfigured targets created a root")
    phase("unconfigured", records)

    local = output / "local-models"
    records = public_cli(output, "import-checkpoints", ["download", *FM, "autoencoder", "--from-local", source_models, "--model-dir", local])
    require(len(records) == 5 and all(record["status"] == "imported" for record in records), "Public checkpoint imports incomplete")
    records += public_cli(output, "import-af2", ["download", "af2", "--from-local", source_models / "af2", "--model-dir", local])
    records += public_cli(output, "import-mpnn", ["download", "proteinmpnn", "solublempnn", "--from-local", source_models, "--model-dir", local])
    require(len(records) == 8 and all(record["status"] == "imported" for record in records), "Public model target imports incomplete")
    compared = [compare_files(source_models / asset.relative, local / asset.relative)
                for target in (*FM, "autoencoder", "af2", "proteinmpnn", "solublempnn") for asset in TARGETS[target]]
    require(len(compared) == 22, "Expected five checkpoints, fifteen AF2 arrays, and two MPNN files")
    phase("local_import", dict(status="PASS", targets=8, file_count=22, files=compared))

    old = local / "ckpts" / "partiflow.ckpt"
    old_state = (old.stat().st_size, old.stat().st_mtime_ns)
    empty_file = output / "empty-source.ckpt"
    empty_file.touch()
    rejected = public_cli(output, "reject-empty-forced-import", ["download", "partiflow", "--from-local", empty_file, "--force", "--model-dir", local], expected=1)
    require(len(rejected) == 1 and rejected[0]["status"] == "failed", "Empty import was accepted")
    require((old.stat().st_size, old.stat().st_mtime_ns) == old_state, "Failed forced import replaced existing checkpoint")
    compare_files(source_models / "ckpts" / "partiflow.ckpt", old)
    require(not list(local.glob(".partiflow-import-*")) and not list((local / "ckpts").glob("*.tmp")), "Failed import left staged files")
    phase("empty_import_preserves_existing", dict(status="PASS", rejected=rejected))

    network = output / "network-models"
    downloaded = public_cli(output, "download-mpnn", ["download", "proteinmpnn", "solublempnn", "--model-dir", network])
    require(len(downloaded) == 2 and all(record["status"] == "downloaded" for record in downloaded), "Configured MPNN downloads failed")
    direct = {}
    for target in ("proteinmpnn", "solublempnn"):
        relative = TARGETS[target][0].relative
        direct[target] = compare_mpnn(source_models / relative, network / relative)
    phase("network_mpnn", dict(status="PASS", records=downloaded, comparisons=direct))

    rows, full = inspect_full_data(source_data)
    phase("full_raw_data", full)
    fixture, archive, selected = prepare_subset(source_data, rows, output)
    imported_data = output / "imported-data"
    records = public_cli(output, "import-pepinter-subset", ["download", "pepinter", "--from-local", archive,
        "--model-dir", local, "--data-dir", imported_data])
    require(len(records) == 1 and records[0]["status"] == "imported", "PepInter subset archive import failed")
    data = check_imported_data(fixture, imported_data)
    data.update(subset_only=True, selections=selected)
    phase("pepinter_subset", data)

    present = public_cli(output, "populated-list", ["download", "--list", "--model-dir", local, "--data-dir", imported_data])
    require(len(present) == 9 and all(record["status"] == "present" for record in present), "Populated inventory incomplete")
    checked = public_cli(output, "populated-check", ["check", "--all", "--model-dir", local, "--data-dir", imported_data], lines=False)
    require(checked["status"] == "complete" and checked["mode"] == "paths" and "loads" not in checked, "Populated path check failed or loaded models")
    phase("populated_paths", dict(status="PASS", inventory=present, check=checked))

    selected_models = [*FM, "autoencoder", "proteinmpnn", "solublempnn"]
    arguments = ["check", "--load", "--model-dir", local]
    for model in selected_models:
        arguments.extend(["--model", model])
    loaded = public_cli(output, "cpu-load", arguments, lines=False)
    require(loaded["status"] == "complete" and loaded["mode"] == "load", "Public CPU model load check failed")
    require({record["target"] for record in loaded["loads"]} == set(selected_models), "CPU loader model coverage differs")
    require(all(record["status"] == "loaded" and record["device"] == "cpu" and record["strict"] for record in loaded["loads"]),
            "Model CPU strict loading incomplete")
    phase("cpu_load", loaded)
    result.update(status="PASS", note="AF2's fifteen files were imported and compared directly; no AF2 inference or full PepInter preprocessing was run.")
    save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model-root", type=Path, required=True)
    parser.add_argument("--source-data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_model_root, args.source_data_root, args.output)
    print(json.dumps(dict(status=result["status"], phases=len(result["phases"]), output=str(args.output)), indent=2))


if __name__ == "__main__":
    main()
