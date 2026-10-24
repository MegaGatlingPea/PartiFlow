"""Offline AF2 multimer initial-guess rewards after generation model offload."""

from __future__ import annotations
import json
import math
import tempfile
import signal
import threading
import time
import sys
from contextlib import contextmanager
from pathlib import Path
from partiflow.common.records import CandidateError, Evaluation, WorkflowError
from partiflow.paths import config_dir

_MPNN = None


@contextmanager
def _deadline(seconds):
    """Interrupt the owner thread; other callers check elapsed time on return."""
    started = time.monotonic()
    main_thread = threading.current_thread() is threading.main_thread()
    previous_handler = None
    previous_timer = (0.0, 0.0)

    def expired(_signal, _frame):
        raise WorkflowError(f"AF2 reward exceeded {seconds} seconds")

    if main_thread:
        previous_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, expired)
        previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))
    try:
        yield
        if time.monotonic() - started > seconds:
            raise WorkflowError(f"AF2 reward exceeded {seconds} seconds")
    finally:
        if main_thread:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous_handler)
            remaining = (
                max(1e-06, previous_timer[0] - (time.monotonic() - started))
                if previous_timer[0] > 0
                else 0.0
            )
            signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def _config(value):
    import yaml
    from .sequence import validate_sequence_config

    default = yaml.safe_load((config_dir() / "reward" / "af2.yaml").read_text())
    sequence = yaml.safe_load((config_dir() / "sequence.yaml").read_text())

    def merge(a, b):
        result = dict(a)
        for key, val in b.items():
            result[key] = (
                merge(result[key], val)
                if isinstance(val, dict) and isinstance(result.get(key), dict)
                else val
            )
        return result

    default["sequence"] = merge(sequence, default.get("sequence", {}))
    config = merge(default, dict(value))
    config["sequence"] = validate_sequence_config(config["sequence"], reward=True)
    unknown = set(config) - {"structure_source", "sequence", "af2", "score"}
    if unknown:
        raise ValueError(f"Unknown reward settings: {sorted(unknown)}")
    if config.get("structure_source", "fold") not in ("fold", "design"):
        raise ValueError("structure_source must be fold or design")
    unknown_af2 = set(config["af2"]) - {
        "model",
        "model_num",
        "num_recycles",
        "use_initial_guess",
        "msa_policy",
        "timeout_seconds",
        "params_dir",
    }
    if unknown_af2:
        raise ValueError(f"Unknown AF2 reward settings: {sorted(unknown_af2)}")
    if set(config["score"]) - {"complex_plddt_weight", "complex_iptm_weight"}:
        raise ValueError("AF2 reward score supports complex pLDDT and ipTM weights")
    if config["af2"].get("model", "multimer") != "multimer":
        raise ValueError("AF2 reward supports the multimer initial-guess protocol")
    from .sequence import ALIASES, normalize_fixed_interface

    if (
        config["structure_source"] == "design"
        and config["sequence"].get("source") != "design"
    ):
        raise ValueError("structure_source=design requires sequence.source=design")

    normalize_fixed_interface(config["sequence"].get("fixed_interface"))
    fixed = config["sequence"].get("fixed_positions", [])
    if not isinstance(fixed, list) or any(
        (not isinstance(item, str) or ":" not in item for item in fixed)
    ):
        raise ValueError("sequence.fixed_positions must be CHAIN:POSITION entries")
    for item in fixed:
        int(item.rsplit(":", 1)[1])
    if config["sequence"].get("source", "design") not in {*ALIASES, "auto"}:
        raise ValueError("Unsupported reward sequence source")
    if config["af2"].get("msa_policy", "none") != "none":
        raise ValueError("AF2 reward requires offline no-MSA policy")
    if config["af2"].get("use_initial_guess") is not True:
        raise ValueError("AF2 rewards require initial-guess prediction")
    if (
        type(config["sequence"].get("seed", 101)) is not int
        or config["sequence"].get("seed", 101) < 0
    ):
        raise ValueError("sequence.seed must be a nonnegative integer")
    timeout = config["af2"].get("timeout_seconds", 7200)
    if type(timeout) is not int or timeout < 1:
        raise ValueError("af2.timeout_seconds must be a positive integer")
    for key, minimum in (("model_num", 1), ("num_recycles", 0)):
        v = config["af2"].get(key)
        if type(v) is not int or v < minimum:
            raise ValueError(f"af2.{key} must be an integer >= {minimum}")
    if config["af2"]["model_num"] not in (1, 2, 3, 4, 5):
        raise ValueError("AF2 multimer model_num must be 1..5")
    for key in ("complex_plddt_weight", "complex_iptm_weight"):
        v = config["score"].get(key)
        if (
            isinstance(v, bool)
            or not isinstance(v, (int, float))
            or (not math.isfinite(v))
            or v < 0
        ):
            raise ValueError("AF2 score weights must be finite and nonnegative")
    if sum(config["score"].values()) <= 0:
        raise ValueError("AF2 complex reward weights must be positive in total")
    return config


def evaluate_candidates(candidates, config, model_dir, work_dir):
    global _MPNN
    if not candidates:
        return []
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("Reward candidate IDs must be unique")
    from .structure import candidate_input, read_structure, write_structure_atomic
    from .af2 import validate_capability, AF2CandidateError
    from .af2_mapping import (
        af2_source_ledger,
        af2_preparation_ledger,
        private_chain_id,
        write_private_af2_pdb,
        restore_af2_residue_identity,
        af2_output_suffix,
        preserve_native_pdb_confidence,
    )
    from .sequence import ALIASES, backend_settings, fixed_positions, MPNNBackend
    from . import reward_backend as native
    from .sequence import sequence_settings

    value = _config(config)
    af2 = value["af2"]
    seq = value["sequence"]
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if len({c.candidate_id for c in candidates}) != len(candidates):
        raise ValueError("Reward candidate IDs must be unique")
    raw_params = Path(af2.get("params_dir") or Path(model_dir) / "af2")
    params = (
        raw_params if raw_params.is_absolute() else Path(model_dir) / raw_params
    ).resolve()
    selected = params / f"params_model_{af2['model_num']}_multimer_v3.npz"
    model_config = {**af2, "model": "multimer", "params_dir": str(params)}
    source = str(seq.get("source", "design"))
    source = source if source == "auto" else ALIASES.get(source, source)
    fixed_settings = sequence_settings(seq)
    settings = (
        backend_settings(source, seq, model_dir, fixed_settings)
        if source != "gt"
        else None
    )
    batch_dir = Path(tempfile.mkdtemp(prefix="batch_", dir=work_dir))
    results = []
    for index, candidate in enumerate(candidates):
        root = batch_dir / f"candidate{index + 1}"
        root.mkdir(parents=True, exist_ok=True)
        model = None
        try:
            data = candidate_input(candidate)
            validate_capability(data)
            sequence = candidate.sequence
            positions = fixed_positions(candidate, data, fixed_settings)
            if source != "gt":
                if _MPNN is None:
                    _MPNN = MPNNBackend()
                sequence = _MPNN.generate(
                    candidate, data, settings, positions, int(seq.get("seed", 101))
                )
            if (
                value.get("structure_source", "fold") == "design"
                and sequence != candidate.sequence
            ):
                raise CandidateError(
                    "structure_source=design cannot publish redesigned sequence on unchanged sidechains"
                )
            order = [*data["context_chains"], candidate.binder_chain]
            ledger = af2_source_ledger(data["atoms"], order, candidate.binder_chain)
            private = root / "input.pdb"
            try:
                write_private_af2_pdb(data["atoms"], private, ledger=ledger)
            except OSError as exc:
                raise CandidateError(f"AF2 template write failed: {exc}") from exc
            request = dict(
                candidate_id=candidate.candidate_id,
                design_sequence=sequence,
                input_pdb=str(private),
                output_pdb=str(root / "native.pdb"),
                design_chain=private_chain_id(ledger, candidate.binder_chain),
                context_chains=[
                    private_chain_id(ledger, c) for c in data["context_chains"]
                ],
                source_design_chain=candidate.binder_chain,
                source_context_chains=data["context_chains"],
                source_ledger=af2_preparation_ledger(ledger),
                cyclic_offset=bool(data.get("cyclic_offset_chains")),
            )
            from .af2 import _assert_protein_input

            request["cyclic_offset"] = (
                candidate.binder_chain
                in _assert_protein_input(
                    data,
                    design_chain=candidate.binder_chain,
                    context=data["context_chains"],
                )["cyclic_chains"]
            )
            if not selected.is_file() or selected.stat().st_size == 0:
                raise WorkflowError(f"AF2 multimer checkpoint missing: {selected}")
            try:
                with _deadline(af2.get("timeout_seconds", 7200)):
                    model, _hit = native._model(model_config)
                    raw = native._predict_one(model, request, model_config)
                raw["model_cache_hit"] = bool(_hit)
            except native.AF2CandidateError as exc:
                native._MODEL_CACHE.pop(native._cache_key(model_config), None)
                raise CandidateError(str(exc)) from exc
            except Exception as exc:
                native._MODEL_CACHE.pop(native._cache_key(model_config), None)
                raise WorkflowError(f"AF2 reward execution failed: {exc}") from exc
            score = (
                float(value["score"]["complex_plddt_weight"])
                * (raw["design_plddt"] / 100.0)
                + float(value["score"]["complex_iptm_weight"]) * raw["interface_iptm"]
            )
            import numpy as np

            score = float(np.float32(score))
            if not math.isfinite(score):
                raise WorkflowError("AF2 returned non-finite reward")
            try:
                folded = restore_af2_residue_identity(
                    read_structure(raw["fold_pdb"]), ledger, order
                )
            except (KeyError, OSError, ValueError, TypeError, IndexError) as exc:
                raise CandidateError(
                    f"AF2 output identity restoration failed: {exc}"
                ) from exc
            output = root / ("fold" + af2_output_suffix(ledger, order))
            try:
                write_structure_atomic(folded, output, entry_id="fold")
                if output.suffix == ".pdb":
                    preserve_native_pdb_confidence(Path(raw["fold_pdb"]), output)
            except OSError as exc:
                raise CandidateError(f"AF2 output write failed: {exc}") from exc
            metrics = dict(
                af2multimer_score=score,
                af2multimer_plddt=float(raw["design_plddt"]),
                af2multimer_ptm=float(raw["ptm"]),
                af2multimer_iptm=float(raw["interface_iptm"]),
                af2multimer_ipae=float(raw["interface_ipae"]),
            )
            try:
                (root / "confidence.json").write_text(
                    json.dumps(raw, allow_nan=False, indent=2) + "\n"
                )
            except (OSError, ValueError, TypeError) as exc:
                raise CandidateError(f"AF2 confidence write failed: {exc}") from exc
            results.append(
                Evaluation(
                    candidate.candidate_id,
                    score,
                    output
                    if value.get("structure_source", "fold") == "fold"
                    else candidate.pdb_path,
                    sequence,
                    metrics,
                )
            )
        except (CandidateError, ValueError, OSError) as exc:
            if isinstance(exc, WorkflowError):
                raise
            results.append(
                Evaluation(
                    candidate.candidate_id,
                    sequence=candidate.sequence,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
        finally:
            native._clear_state(model)
    return results


def close():
    global _MPNN
    if _MPNN is not None:
        _MPNN.close()
        _MPNN = None
    reward_backend = sys.modules.get("partiflow.workflows.reward_backend")
    if reward_backend is None:
        return
    for model in reward_backend._MODEL_CACHE.values():
        reward_backend._clear_state(model)
    reward_backend._MODEL_CACHE.clear()
