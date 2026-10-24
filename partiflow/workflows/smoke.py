"""Small real structure-model checks for the public dependency smoke command."""

from __future__ import annotations
import os
from pathlib import Path
from partiflow.common.records import Candidate, CandidateError, WorkflowError
from partiflow.paths import config_dir, model_root


def _requested_models(models):
    aliases = {
        "protein_mpnn": "proteinmpnn",
        "proteinmpnn": "proteinmpnn",
        "soluble_mpnn": "solublempnn",
        "solublempnn": "solublempnn",
        "af2": "af2",
        "af2_ig": "af2",
    }
    if models is None:
        values = ["proteinmpnn", "solublempnn", "af2"]
    elif isinstance(models, str):
        values = models.split(",")
    else:
        values = list(models)
    selected = set()
    for value in values:
        name = str(value).strip().lower()
        if name not in aliases:
            raise ValueError(f"Unsupported structure smoke model: {value}")
        selected.add(aliases[name])
    if not selected:
        raise ValueError("Select at least one structure smoke model")
    return [name for name in ("proteinmpnn", "solublempnn", "af2") if name in selected]


def run_structure_smoke(path, binder_chain, model_dir, models, out):
    """Sample selected native MPNNs and execute one requested AF2 IG pair.

    A complete result means actual execution and readable paired artifacts.
    Model/device failures are persisted as failed and remain WorkflowError.
    The caller controls accelerator visibility and generation-model offload.
    """
    requested = _requested_models(models)
    path = Path(path).expanduser().resolve()
    out = Path(out).expanduser().resolve()
    selected_root = model_root(model_dir)
    from .validate import _atomic_json, _refresh_metrics
    from .structure import (
        read_structure,
        candidate_input,
        polymer_residue_groups,
        AA3_TO_1,
    )
    from .sequence import MPNNBackend, backend_settings, fixed_positions

    out.mkdir(parents=True, exist_ok=True)
    result_path = out / "smoke.json"
    report = {
        "status": "running",
        "result_path": str(result_path),
        "input_path": str(path),
        "binder_chain": str(binder_chain),
        "model_dir": str(selected_root),
        "requested_models": requested,
        "models": {name: {"status": "not_run"} for name in requested},
    }
    _atomic_json(result_path, report)
    mpnn = None
    af2 = None
    current = "input"
    try:
        try:
            atoms = read_structure(path)
            groups = polymer_residue_groups(atoms, str(binder_chain))
            sequence = "".join(
                (AA3_TO_1.get(group["res_name"], "X") for group in groups)
            )
            if not groups or "X" in sequence:
                raise CandidateError(
                    "Structure smoke requires a canonical protein binder chain"
                )
            candidate = Candidate(
                "smoke",
                "smoke",
                1,
                1,
                1,
                path,
                str(binder_chain),
                len(groups),
                sequence,
            )
            data = candidate_input(candidate)
        except (OSError, ValueError) as exc:
            raise CandidateError(f"Structure smoke input is invalid: {exc}") from exc
        report.update(
            input_sequence=sequence,
            binder_length=len(sequence),
            context_chains=list(data["context_chains"]),
        )
        import yaml

        defaults = yaml.safe_load((config_dir() / "sequence.yaml").read_text())
        positions = fixed_positions(candidate, data, {})
        sequence_by_model = {}
        for name in requested:
            if name == "af2":
                continue
            current = name
            report["models"][name] = {"status": "running"}
            _atomic_json(result_path, report)
            if mpnn is None:
                mpnn = MPNNBackend()
            settings = backend_settings(name, defaults, selected_root)
            sampled = mpnn.generate(candidate, data, settings, positions, 101)
            model_out = out / name
            model_out.mkdir(parents=True, exist_ok=True)
            fasta = model_out / "sequence.fasta"
            fasta.write_text(f">smoke_{name}\n{sampled}\n")
            model_result = {
                "status": "complete",
                "sequence": sampled,
                "sequence_path": str(fasta),
                "checkpoint": settings["checkpoint"],
                "fixed_positions": positions,
                "temperature": settings["temperature"],
            }
            _atomic_json(model_out / "sequence.json", model_result)
            report["models"][name] = model_result
            sequence_by_model[name] = sampled
            _atomic_json(result_path, report)
        if "af2" in requested:
            current = "af2"
            from .af2 import AF2Backend

            source = (
                "solublempnn"
                if "solublempnn" in sequence_by_model
                else "proteinmpnn"
                if "proteinmpnn" in sequence_by_model
                else "gt"
            )
            folded_sequence = sequence_by_model.get(source, sequence)
            af2_out = out / "af2"
            af2_out.mkdir(parents=True, exist_ok=True)
            configuration = {
                "models": [1],
                "samples": 1,
                "recycles": 1,
                "seed": 101,
                "mode": "both",
            }
            af2 = AF2Backend(
                selected_root,
                configuration,
                device="cpu" if os.environ.get("JAX_PLATFORMS") == "cpu" else "cuda:0",
            )
            record = {
                "method": "af2_ig",
                "candidate_id": "smoke",
                "base_sample_id": 1,
                "seq_index": 0,
                "source": source,
                "source_label": "self" if source == "gt" else source,
                "source_index": 1,
                "sequence": folded_sequence,
                "expected_predictions": af2.expected_keys(),
                "predictions": {},
                "status": "incomplete",
            }
            report["models"]["af2"] = {
                "status": "running",
                "record_path": str(af2_out / "seq0.json"),
            }
            _atomic_json(result_path, report)
            _atomic_json(af2_out / "seq0.json", record)
            variant = {
                "seq_index": 0,
                "variant_id": "seq0",
                "design_sequence": folded_sequence,
                "source": source,
                "source_index": 1,
            }
            for prediction in af2.predict(data, variant, af2_out):
                arrays = prediction.pop("_confidence_arrays", None)
                if prediction.get("status") == "complete":
                    import numpy as np

                    confidence_path = af2_out / (
                        Path(prediction["structure_path"]).stem + ".npz"
                    )
                    temporary = confidence_path.with_name(
                        "." + confidence_path.name + ".tmp"
                    )
                    with temporary.open("wb") as handle:
                        np.savez_compressed(handle, **arrays)
                    os.replace(temporary, confidence_path)
                    prediction["confidence_path"] = str(confidence_path)
                record["predictions"][prediction["key"]] = prediction
                _atomic_json(af2_out / "seq0.json", record)
            _refresh_metrics(candidate, data, record)
            _atomic_json(af2_out / "seq0.json", record)
            report["models"]["af2"] = {
                "status": record["status"],
                "record_path": str(af2_out / "seq0.json"),
                "sequence_source": source,
                "sequence": folded_sequence,
                "configuration": configuration,
                "predictions": record["predictions"],
                "paired_metrics": record["paired_metrics"],
                "model_cache_hits": af2.cache_hit_count,
                "model_cache_misses": af2.cache_miss_count,
            }
        report["status"] = (
            "complete"
            if all(
                (item.get("status") == "complete" for item in report["models"].values())
            )
            else "failed"
        )
        if report["status"] != "complete":
            report["error"] = "Requested structure smoke stages are incomplete"
        if mpnn is not None:
            report["mpnn_model_cache_hits"] = mpnn.hit_count
            report["mpnn_model_cache_misses"] = mpnn.load_count
        _atomic_json(result_path, report)
        return report
    except CandidateError as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if current in report["models"]:
            report["models"][current] = {"status": "failed", "error": str(exc)}
        _atomic_json(result_path, report)
        return report
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if current in report["models"]:
            report["models"][current] = {"status": "failed", "error": str(exc)}
        _atomic_json(result_path, report)
        if isinstance(exc, WorkflowError):
            raise
        raise WorkflowError(f"Structure smoke failed: {exc}") from exc
    finally:
        if af2 is not None:
            af2.close()
        if mpnn is not None:
            mpnn.close()
