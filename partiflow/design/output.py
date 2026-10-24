"""Persist decoded candidates before deferred evaluation."""

from __future__ import annotations
import json
import re
from pathlib import Path
import lightning as L
import torch
from .pdb_writer import PDBWriter
from .tasks import nonstandard_positions_from_topology, format_sequence_with_nonstandard
from partiflow.openfold.np.residue_constants import restypes


def primitive(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [primitive(v) for v in value]
    if hasattr(value, "tolist"):
        return primitive(value.tolist())
    return value.item() if hasattr(value, "item") else value


class CandidateCollector(L.Callback):
    def __init__(self, root_dir, requests, *, replicas=1, receptor_source="decoded"):
        self.root_dir = Path(root_dir)
        self.replicas = int(replicas)
        self.writer = PDBWriter(receptor_source)
        self.requests = {}
        self.emitted = set()
        groups = {}
        for index, request in enumerate(requests):
            groups.setdefault(request["task_name"], []).append((index, request))
        offset = 0
        for task, items in groups.items():
            ordered = sorted(
                items,
                key=lambda pair: int(pair[1]["sample_fn"].rsplit("_sample_", 1)[-1]),
            )
            for base, (index, request) in enumerate(ordered, start=1):
                self.requests[(task, request["sample_fn"])] = (
                    base,
                    offset + base - 1,
                    index,
                    request,
                )
            offset += len(items)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        rank = int(getattr(trainer, "global_rank", 0))
        world = int(getattr(trainer, "world_size", 1))
        for record in outputs or ():
            task = str(record["task_name"])
            base, base_order, index, request = self.requests[
                (task, str(record["base_sample_id"]))
            ]
            if world > 1 and index % world != rank:
                continue
            match = re.search(r"_r([0-9]+)$", str(record.get("metadata_tag") or ""))
            replica = int(match.group(1)) + 1 if match else 1
            cid = f"{task}:base{base}:replica{replica}"
            if cid in self.emitted:
                raise RuntimeError(f"Duplicate candidate: {cid}")
            self.emitted.add(cid)
            directory = self.root_dir / task / ".state"
            (directory / "candidates").mkdir(parents=True, exist_ok=True)
            path = directory / "candidates" / f"base{base}_replica{replica}.pdb"
            metadata = {
                k: primitive(v)
                for k, v in request.items()
                if k not in {"struc_input", "ref_center"}
            }
            metadata.update(
                {
                    k: primitive(record[k])
                    for k in (
                        "topology_entries",
                        "postprocess_enabled",
                        "postprocess_method",
                        "binder_chain",
                        "contig",
                    )
                    if k in record
                }
            )
            metadata["cyclic_offset"] = bool(request.get("cyclic", False))
            fixed = set(int(x) for x in request.get("residue_type_constraints", {}))
            atoms = request["struc_input"]
            if "is_motif" in atoms.get_annotation_categories():
                fixed.update(
                    int(x)
                    for x in atoms.res_id[(atoms.chain_id == "L") & atoms.is_motif]
                )
            if request["task_type"] in {"docking", "packing"}:
                fixed.update(int(x) for x in atoms.res_id[atoms.chain_id == "L"])
            metadata["fixed_positions"] = sorted(fixed)
            row = {
                "candidate_id": cid,
                "task_name": task,
                "base_sample_id": base,
                "candidate_ordinal": base_order * self.replicas + replica - 1,
                "replica_id": replica,
                "pdb_path": str(path),
                "binder_chain": str(record.get("binder_chain", "L")),
                "length": int(record["nres"]),
                "contig_string": ""
                if record.get("contig") in (None, "null")
                else str(record["contig"]),
                "metadata": metadata,
                "topology_entries": primitive(record.get("topology_entries", [])),
                "error": "",
                "sequence": "",
            }
            try:
                pdb = self.writer.coors37_to_pdb_string(
                    coors37=record["coors37"],
                    residue_type=record["residue_type"],
                    is_peptide=record["is_peptide"],
                    batch=batch,
                    sample_idx=int(record["source_index"]),
                    chain_index_override=record.get("chain_index"),
                    residue_index_override=self.writer._select_output_residue_index(
                        record
                    ),
                    reference_pdb_path=record.get("reference_pdb_path"),
                    reference_pep_chain=record.get("reference_pep_chain", "L"),
                    reference_receptor_center=record.get("reference_receptor_center"),
                    binder_chain=row["binder_chain"],
                )
                if not pdb:
                    raise ValueError(
                        "Decoded structure cannot be written as a finite PDB"
                    )
                temporary = path.with_suffix(".tmp")
                temporary.write_text(pdb)
                temporary.replace(path)
                types = record["residue_type"].detach().cpu().tolist()
                mask = record["is_peptide"].detach().bool().cpu().tolist()
                sequence = "".join(
                    restypes[int(aa)] for aa, yes in zip(types, mask) if yes
                )
                row["sequence"] = format_sequence_with_nonstandard(
                    sequence,
                    nonstandard_positions_from_topology(row["topology_entries"]),
                )
            except (ValueError, ArithmeticError) as exc:
                row["error"] = f"generation: {exc}"
            with (directory / f"generation_rank{rank}.jsonl").open("a") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
