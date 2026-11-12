"""Read-only CPU audit of a fresh PepInter subset and its data pipelines.

Supply both fresh and reference dataset roots plus the actual VAE/FM training
configurations. The audit never calls prepare_data or constructs a training
model. Reports and copies of the inspected settings stay outside the checkout
and both dataset roots.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import csv
import inspect
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback


CHECKOUT = Path(__file__).resolve().parents[2]
EXPECTED_COUNTS = {"train": 166019, "val": 3750, "test": 1250}
SUBSET_COUNTS = {"train": 64, "val": 16}
RAW_SPLITS = {"train": "train", "val": "valid", "test": "test"}
ATOM37 = ("N CA C CB O CG CG1 CG2 OG OG1 SG CD CD1 CD2 ND1 ND2 OD1 OD2 SD CE CE1 CE2 CE3 "
          "NE NE1 NE2 OE1 OE2 CH2 NH1 NH2 OH CZ CZ2 CZ3 NZ OXT").split()


class AuditError(AssertionError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(str(message))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def rows(path):
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames, f"Missing CSV header: {path}")
        return list(reader)


def external_root(path, *inputs):
    path = Path(path).expanduser().resolve()
    for source in (CHECKOUT, *inputs):
        source = Path(source).expanduser().resolve()
        require(path != source and source not in path.parents and path not in source.parents,
                f"Output must be outside {source}: {path}")
    return path


def index_audit(root, *, expected_counts=None, subset_counts=None):
    expected_counts = EXPECTED_COUNTS if expected_counts is None else expected_counts
    subset_counts = SUBSET_COUNTS if subset_counts is None else subset_counts
    root = Path(root).resolve()
    full = rows(root / "indexes/split.csv")
    selected = rows(root / "indexes/acceptance.csv")
    require(Counter(row["split"] for row in full) == expected_counts, "Full split counts differ")
    identifiers = [row["pdbid"] for row in full]
    require(len(identifiers) == len(set(identifiers)), "Identifier leakage across full source splits")
    raw_ids = {}
    for split, raw_name in RAW_SPLITS.items():
        names = sorted(path.stem for path in (root / "raw" / raw_name).glob("*.pdb"))
        require(len(names) == expected_counts[split], f"Raw {raw_name} count differs")
        actual = [row["pdbid"] for row in full if row["split"] == split]
        require(actual == names, f"Full {split} IDs differ from sorted raw {raw_name} files")
        raw_ids[split] = names
    require(not any(row["split"] == "valid" for row in full), "valid was not normalized to val")
    for row in full:
        split, identifier = row["split"], row["pdbid"]
        require(row["tag"] == "pepinter", "Unexpected dataset tag")
        require(Path(row["structure_path"]).as_posix() == f"../raw/{RAW_SPLITS[split]}/{identifier}.pdb",
                f"Raw path differs for {identifier}")
        require(Path(row["processed_path"]).as_posix() == f"../processed/{split}/{identifier}.pt",
                f"Processed path differs for {identifier}")
    wanted = []
    for split, size in subset_counts.items():
        wanted.extend([row for row in full if row["split"] == split][:size])
    require(selected == wanted, "Subset is not the fixed first sorted rows from each split")
    require(Counter(row["split"] for row in selected) == subset_counts, "Subset counts differ")
    require(len({row["pdbid"] for row in selected}) == len(selected), "Subset repeats an identifier")
    require(not ({row["pdbid"] for row in selected} & set(raw_ids["test"])), "Test IDs leaked into the subset")
    return full, selected, {"full_counts": dict(Counter(row["split"] for row in full)),
                            "subset_counts": dict(Counter(row["split"] for row in selected)),
                            "valid_directory_maps_to_val": True, "cross_split_leakage": False,
                            "selected_ids": {split: [row["pdbid"] for row in selected if row["split"] == split]
                                             for split in subset_counts}}


def compare(actual, expected, label="graph"):
    """Compare every field exactly; no tensor, metadata field, or path is skipped."""
    import numpy as np
    import torch

    counts = Counter()
    if isinstance(expected, torch.Tensor):
        require(isinstance(actual, torch.Tensor), f"{label}: expected tensor")
        require(actual.device.type == expected.device.type == "cpu", f"{label}: tensor is not on CPU")
        require(actual.dtype == expected.dtype and actual.shape == expected.shape, f"{label}: shape/dtype differs")
        require(torch.equal(actual, expected), f"{label}: tensor values differ")
        counts.update(tensors=1, tensor_elements=expected.numel())
    elif isinstance(expected, np.ndarray):
        require(isinstance(actual, np.ndarray) and actual.dtype == expected.dtype and actual.shape == expected.shape,
                f"{label}: array shape/dtype differs")
        require(np.array_equal(actual, expected), f"{label}: array values differ")
        counts.update(arrays=1, array_elements=expected.size)
    elif isinstance(expected, Mapping):
        require(isinstance(actual, Mapping) and actual.keys() == expected.keys(), f"{label}: mapping fields differ")
        for name in expected:
            counts.update(compare(actual[name], expected[name], f"{label}.{name}"))
    elif isinstance(expected, (list, tuple)):
        require(type(actual) is type(expected) and len(actual) == len(expected), f"{label}: sequence differs")
        for index, value in enumerate(expected):
            counts.update(compare(actual[index], value, f"{label}[{index}]"))
    else:
        require(type(actual) is type(expected) and actual == expected, f"{label}: scalar differs: {actual!r} vs {expected!r}")
        counts.update(scalars=1)
    return counts


def finite(value, label="data"):
    import numpy as np
    import torch

    counts = Counter()
    if isinstance(value, torch.Tensor):
        require(value.device.type == "cpu", f"{label}: expected CPU tensor")
        require(bool(torch.isfinite(value).all()), f"{label}: nonfinite tensor")
        counts.update(tensors=1, tensor_elements=value.numel())
    elif isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number):
            require(bool(np.isfinite(value).all()), f"{label}: nonfinite array")
        counts.update(arrays=1, array_elements=value.size)
    elif isinstance(value, Mapping):
        for name, item in value.items():
            counts.update(finite(item, f"{label}.{name}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            counts.update(finite(item, f"{label}[{index}]"))
    elif isinstance(value, float):
        require(math.isfinite(value), f"{label}: nonfinite scalar")
    return counts


def pdb_atoms(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        name = line[12:16].strip()
        if name not in ATOM37:
            continue
        key = (line[21:22].strip(), line[17:20].strip(), int(line[22:26]), line[26:27].strip())
        point = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
        require(all(math.isfinite(value) for value in point), f"Nonfinite raw PDB atom: {path}")
        result.setdefault(key, {}).setdefault(name, []).append(point)
    return result


def graph_audit(graph, row, source):
    import torch
    from partiflow.openfold.np import residue_constants

    values = graph.to_dict()
    finite(values, row["pdbid"])
    require(graph.id == row["pdbid"], "Graph ID differs")
    length = graph.coords.shape[0]
    require(graph.coords.dtype == torch.float32 and tuple(graph.coords.shape) == (length, 37, 3), "Invalid atom37 coordinates")
    require(graph.coord_mask.dtype == torch.bool and tuple(graph.coord_mask.shape) == (length, 37), "Invalid atom37 mask")
    for field in ("residue_type", "chains", "residue_pdb_idx", "is_peptide"):
        require(graph[field].dtype == torch.int64 and tuple(graph[field].shape) == (length,), f"Invalid graph field: {field}")
    require(tuple(graph.seq_pos.shape) == (length, 1) and torch.equal(graph.seq_pos[:, 0], torch.arange(length)), "Sequence positions differ")
    require(len(graph.residues) == len(graph.residue_id) == len(graph.orig_chain_id) == length, "Residue lists differ")
    require(set(graph.is_peptide.tolist()) == {0, 1}, "Expected peptide and receptor residues")
    missing = graph.coords[~graph.coord_mask]
    require(bool((missing == graph.coords.new_tensor(1e-5)).all()), "Missing atoms do not have the configured fill coordinates")
    require(ATOM37 == residue_constants.atom_types, "Atom37 vocabulary differs")
    raw = pdb_atoms(source)
    chain_to_code, code_to_chain = {}, {}
    atom_count, maximum_delta = 0, 0.0
    for index, identifier in enumerate(graph.residue_id):
        parts = identifier.split(":")
        require(len(parts) == 4, f"Unexpected residue identifier: {identifier}")
        chain, residue, number, insertion = parts
        key = (chain.strip(), residue, int(number), insertion)
        require(key in raw, f"Graph residue absent from raw PDB: {identifier}")
        require(graph.orig_chain_id[index] == chain and graph.residues[index] == residue
                and int(graph.residue_pdb_idx[index]) == int(number), "Raw residue/chain labels differ")
        expected_type = residue_constants.resname_to_idx.get(residue, residue_constants.resname_to_idx.get("UNK", 0))
        require(int(graph.residue_type[index]) == expected_type, "Residue-type encoding differs")
        require(int(graph.is_peptide[index]) == int(chain == "L"), "Peptide chain mask differs")
        code = int(graph.chains[index])
        require(chain_to_code.setdefault(chain, code) == code and code_to_chain.setdefault(code, chain) == chain,
                "Chain encoding is not one-to-one")
        for atom_index, name in enumerate(ATOM37):
            present = bool(graph.coord_mask[index, atom_index])
            require(present == (name in raw[key]), f"Atom37 presence mask differs: {identifier}/{name}")
            if not present:
                continue
            point = graph.coords[index, atom_index]
            delta = min(float((point - point.new_tensor(candidate)).abs().max()) for candidate in raw[key][name])
            require(delta <= 1e-4, f"Raw angstrom coordinate differs: {identifier}/{name}: {delta}")
            maximum_delta = max(maximum_delta, delta)
            atom_count += 1
    return {"residues": length, "peptide_residues": int(graph.is_peptide.sum()),
            "receptor_residues": int((graph.is_peptide == 0).sum()), "present_atoms": atom_count,
            "missing_atom_slots": int((~graph.coord_mask).sum()), "chain_encoding": chain_to_code,
            "max_raw_pdb_delta_angstrom": maximum_delta, "graph_fields": sorted(values)}


def seed_cpu(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.random.default_generator.manual_seed(seed)


def transformed_audit(sample, raw):
    import torch

    length = len(sample.coords)
    count = finite(sample.to_dict(), "transformed")
    require(tuple(sample.coords.shape) == tuple(sample.coords_nm.shape) == (length, 37, 3), "Transformed coordinate shape differs")
    require(torch.allclose(sample.coords, sample.coords_nm * 10, rtol=1e-6, atol=1e-5), "Angstrom/nanometer scale differs")
    lookup = {identifier: index for index, identifier in enumerate(raw.residue_id)}
    require(len(lookup) == len(raw.residue_id), "Raw graph repeats residue IDs")
    indices = torch.tensor([lookup[identifier] for identifier in sample.residue_id])
    require(torch.equal(sample.coords_unmodified, raw.coords[indices]), "Cropping changed the stored original coordinates")
    require(torch.equal(sample.coord_mask, raw.coord_mask[indices]), "Cropping changed atomic availability")
    present_ca = sample.coord_mask[:, 1]
    source_ca, transformed_ca = sample.coords_unmodified[present_ca, 1], sample.coords_nm[present_ca, 1]
    source_dist = torch.linalg.vector_norm(source_ca[1:] - source_ca[:-1], dim=-1)
    transformed_dist = 10 * torch.linalg.vector_norm(transformed_ca[1:] - transformed_ca[:-1], dim=-1)
    require(torch.allclose(source_dist, transformed_dist, rtol=1e-5, atol=1e-3), "Rigid augmentation changed CA distances/units")
    require(sample.motif_mask.dtype == torch.bool and tuple(sample.motif_mask.shape) == (length, 37), "Invalid motif mask")
    require(not bool((sample.motif_mask & ~sample.coord_mask).any()), "Motif includes absent atoms")
    require(torch.equal(sample.x_motif, sample.coords_nm * sample.motif_mask[..., None]), "Motif coordinates differ")
    require(torch.equal(sample.seq_motif_mask, sample.motif_mask.any(-1)), "Sequence motif mask differs")
    require(torch.equal(sample.seq_motif, sample.residue_type * sample.seq_motif_mask), "Motif sequence differs")
    for name in ("ca_cond_mask", "latent_cond_mask"):
        require(sample[name].dtype == torch.bool and tuple(sample[name].shape) == (length,), f"Invalid condition mask: {name}")
    require(tuple(sample.binder_ss_token.shape) == (length,) and bool(((sample.binder_ss_token >= 0) & (sample.binder_ss_token <= 3)).all()), "Invalid SS tokens")
    require(tuple(sample.binder_block_adj.shape) == (length, length)
            and bool(((sample.binder_block_adj >= 0) & (sample.binder_block_adj <= 2)).all()), "Invalid block adjacency")
    return {"length": length, "peptide_residues": int(sample.is_peptide.sum()),
            "receptor_residues": int((sample.is_peptide == 0).sum()), "motif_atoms": int(sample.motif_mask.sum()),
            "conditioned_ca_residues": int(sample.ca_cond_mask.sum()),
            "conditioned_latent_residues": int(sample.latent_cond_mask.sum()),
            "max_rigid_ca_distance_delta_angstrom": float((source_dist - transformed_dist).abs().max()) if source_dist.numel() else 0.0,
            "finite": dict(count)}


def collated_audit(batch, sample, pad_size):
    import torch

    length = len(sample.coords)
    padded = pad_size or length
    require(tuple(batch.coords.shape) == (1, padded, 37, 3), "Dense batch shape differs")
    require(batch.id == [sample.id], "Collation changed sample ID")
    count = finite(batch.to_dict(), "batch")
    mask = batch.mask_dict["coords"]
    require(mask.dtype == torch.bool and mask.shape == batch.coords.shape, "Invalid coordinate padding mask")
    require(bool(mask[0, :length].all()) and not bool(mask[0, length:].any()), "Padding mask does not distinguish real residues")
    for name in ("coords", "coords_nm", "coords_unmodified", "coord_mask", "residue_type", "chains", "is_peptide",
                 "motif_mask", "x_motif", "seq_motif_mask", "seq_motif", "ca_cond_mask", "latent_cond_mask", "binder_ss_token"):
        compare(batch[name][0, :length], sample[name], "collated." + name)
    require(not bool(batch.coord_mask[0, length:].any()), "Padding marks atoms as present")
    for name in ("binder_block_adj", "chemical_bond_type"):
        require(tuple(batch[name].shape) == (1, padded, padded), f"Pair batch shape differs: {name}")
        compare(batch[name][0, :length, :length], sample[name], "collated." + name)
    return {"coords_shape": list(batch.coords.shape), "pair_shape": list(batch.binder_block_adj.shape),
            "coordinate_padding_mask_true": int(mask.sum()), "padded_residues": padded - length,
            "finite": dict(count)}


def config_pipeline(path, stage, fresh, reference, selected, output):
    import hydra
    from omegaconf import OmegaConf
    import torch

    config = OmegaConf.load(path)
    if "dataset" not in config:
        from partiflow.training.config import load_config
        config = load_config(Path(path), stage=stage)
    original = OmegaConf.to_container(config.dataset.datamodule, resolve=True)
    require(original["_target_"] == "partiflow.datasets.pdb_data.PepLightningDataModule", "Expected the production datamodule")
    require(original.get("batch_size") == 1 and original.get("num_workers") == 0, "Expected the 1-sample/0-worker configurations actually used for training")
    require(original.get("batch_padding") is True and not original.get("truncate_to_pad_max_size"), "Unexpected padding/truncation settings")
    modules, effective = [], {}
    for name, root in (("fresh", fresh), ("reference", reference)):
        settings = dict(original)
        settings.update(split_csv=str(root / "indexes/acceptance.csv"), processed_dir=str(root / "processed"),
                        structure_dir=str(root / "raw"))
        module = hydra.utils.instantiate(settings)
        module.setup("fit")
        require(CHECKOUT in Path(inspect.getfile(type(module))).resolve().parents, "Imported a different production checkout")
        modules.append(module)
        effective[name] = settings
    save(output / "configs" / (stage + "_datamodules.json"), effective)
    summary = {"stage": stage, "configuration": str(Path(path).resolve()),
               "transform_classes": [type(item).__module__ + "." + type(item).__name__ for item in modules[0]._iter_transforms()],
               "batch_size": 1, "num_workers": 0, "pad_max_size": original.get("pad_max_size"),
               "transform_comparison": {}, "splits": {}, "source": inspect.getfile(type(modules[0])),
               "sampling": "each dataset index once, followed by the configured production loader collate_fn"}
    totals = Counter()
    for split, size in SUBSET_COUNTS.items():
        datasets = [module.train_dataset() if split == "train" else module.val_dataset() for module in modules]
        require(all(len(dataset) == size for dataset in datasets), f"{stage}/{split}: dataset length differs")
        loader = modules[0].train_dataloader() if split == "train" else modules[0].val_dataloader()
        require(type(loader).__name__ == "DensePaddingDataLoader", "Expected production dense collation")
        details = []
        split_rows = [row for row in selected if row["split"] == split]
        for index, row in enumerate(split_rows):
            seed = int(config.get("seed", 2026)) + index + (1000003 if split == "val" else 0)
            seed_cpu(seed)
            sample = datasets[0][index]
            seed_cpu(seed)
            previous = datasets[1][index]
            require(sample.id == previous.id == row["pdbid"], "Transformed sample order differs")
            totals.update(compare(sample.to_dict(), previous.to_dict(), f"{stage}/{split}/{sample.id}"))
            raw = torch.load((fresh / "indexes" / row["processed_path"]).resolve(), map_location="cpu", weights_only=False)
            geometry = transformed_audit(sample, raw)
            batch = loader.collate_fn([sample])
            details.append({"id": sample.id, "seed": seed, **geometry,
                            "collation": collated_audit(batch, sample, original.get("pad_max_size"))})
        summary["splits"][split] = {"samples": len(details), "min_residues": min(item["length"] for item in details),
                                     "max_residues": max(item["length"] for item in details), "details": details}
    summary["transform_comparison"] = dict(totals)
    return summary


def audit(args):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for this CPU audit")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"Set {name}=1")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    fresh, reference = (Path(path).expanduser().resolve() for path in (args.fresh_root, args.reference_root))
    require(fresh != reference, "Fresh and reference roots must differ")
    output = external_root(args.output_root, fresh, reference)
    require(not output.exists() or not any(output.iterdir()), "Choose a fresh audit output directory")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "scope": "preprocessed graphs, transforms, and collation; no training/model execution",
              "argv": sys.argv, "python": sys.executable, "torch": torch.__version__, "threads": 1,
              "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "fresh_root": str(fresh),
              "reference_root": str(reference), "output_root": str(output), "graphs": [], "pipelines": []}
    save(output / "report.json", report)
    try:
        _, selected, report["fresh_indexes"] = index_audit(fresh)
        _, old_selected, report["reference_indexes"] = index_audit(reference)
        require([(row["pdbid"], row["split"]) for row in selected] ==
                [(row["pdbid"], row["split"]) for row in old_selected], "Fresh/reference selected IDs differ")
        failure_path = fresh / "indexes/preparation_failures.csv"
        require(failure_path.is_file() and rows(failure_path) == [], "Preparation reported failures")
        expected_paths = {(fresh / "indexes" / row["processed_path"]).resolve() for row in selected}
        actual_paths = set((fresh / "processed").rglob("*.pt"))
        require(actual_paths == expected_paths and len(actual_paths) == 80, "Expected exactly 80 fresh prepared graphs")
        require(not list((fresh / "processed").rglob("*.writing")), "Incomplete graph files remain")
        before = {}
        total = Counter()
        boundary = max((fresh / "indexes" / name).stat().st_mtime_ns for name in ("split.csv", "acceptance.csv"))
        for row, old_row in zip(selected, old_selected):
            current = (fresh / "indexes" / row["processed_path"]).resolve()
            previous = (reference / "indexes" / old_row["processed_path"]).resolve()
            logical_current = fresh / "indexes" / row["processed_path"]
            require(current.is_relative_to(fresh / "processed") and not logical_current.is_symlink(), "Fresh graph points outside fresh processed data")
            require(not current.samefile(previous), "Fresh graph is a reused reference inode")
            require(current.stat().st_mtime_ns >= boundary and current.stat().st_mtime_ns > previous.stat().st_mtime_ns,
                    f"Graph is not newer than indexing/reference: {current}")
            source = (fresh / "indexes" / row["structure_path"]).resolve()
            old_source = (reference / "indexes" / old_row["structure_path"]).resolve()
            require(source.samefile(old_source) or source.read_bytes() == old_source.read_bytes(), "Raw structures differ")
            for path in (current, previous, source, old_source):
                before[path] = (path.stat().st_size, path.stat().st_mtime_ns)
            graph, old_graph = (torch.load(path, map_location="cpu", weights_only=False) for path in (current, previous))
            total.update(compare(graph.to_dict(), old_graph.to_dict(), row["pdbid"]))
            report["graphs"].append({"id": row["pdbid"], "split": row["split"], "fresh_path": str(current),
                                     "reference_path": str(previous), **graph_audit(graph, row, source)})
        report["graph_comparison"] = {**dict(total), "exact_graphs": len(report["graphs"]), "ignored_fields": []}
        if args.preparation_log:
            log = Path(args.preparation_log)
            require(log.is_file() and log.stat().st_size, "Preparation log is missing")
            report["preparation_log"] = str(log.resolve())
            require("'train': 166019" in log.read_text() and "'val': 3750" in log.read_text()
                    and "'test': 1250" in log.read_text(), "Preparation log lacks the complete split summary")
        for stage, config_path in (("vae", args.vae_config), ("fm", args.fm_config)):
            destination = output / "configs" / (stage + ".yaml")
            destination.parent.mkdir(exist_ok=True)
            shutil.copy2(config_path, destination)
            report["pipelines"].append(config_pipeline(config_path, stage, fresh, reference, selected, output))
        for path, expected in before.items():
            require((path.stat().st_size, path.stat().st_mtime_ns) == expected, f"Input changed during audit: {path}")
        require(not torch.cuda.is_initialized(), "CPU audit initialized CUDA")
        report.update(status="complete", unchanged_graph_and_structure_files=len(before),
                      cuda_initialized=False, elapsed_seconds=time.monotonic() - started)
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc(),
                      elapsed_seconds=time.monotonic() - started)
        save(output / "report.json", report)
        raise
    save(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fresh-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--vae-config", required=True, type=Path)
    parser.add_argument("--fm-config", required=True, type=Path)
    parser.add_argument("--preparation-log", type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    report = audit(parser.parse_args(argv))
    print(json.dumps({"status": report["status"], "graphs": len(report["graphs"]),
                      "pipelines": [{"stage": item["stage"], "splits": {name: value["samples"] for name, value in item["splits"].items()}}
                                    for item in report["pipelines"]],
                      "report": str(Path(report["output_root"]) / "report.json")}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, ValueError, KeyError) as exc:
        print(f"Data audit failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
