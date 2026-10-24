"""Shared standard ProteinMPNN/SolubleMPNN source and identity policy."""

from __future__ import annotations
import copy
import json
import math
import re
from pathlib import PurePosixPath
from pathlib import Path
from partiflow.common.records import CandidateError, WorkflowError
from typing import Any, Mapping

_AUTO_SOLUBLE_BACKEND_KEYS = ("solublempnn", "soluble_mpnn", "solublempnn_defaults")


def _auto_soluble_backend(
    sequence: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Return the named SolubleMPNN options carried by a reward config.

    Reward sources intentionally keep their public namespace small.  The
    selected backend therefore lives under an internal metadata key until it
    is copied to the native proteinmpnn slot.  solublempnn_defaults is accepted
    for the shared sequence policy file; a caller-specific solublempnn block
    takes precedence over those stock defaults.
    """
    for key in _AUTO_SOLUBLE_BACKEND_KEYS:
        value = sequence.get(key)
        if isinstance(value, Mapping):
            return (value, key)
    auto = sequence.get("auto")
    if isinstance(auto, Mapping):
        for key in ("protein", "cyclic"):
            value = auto.get(key)
            if isinstance(value, Mapping):
                return (value, f"auto.{key}")
    return (None, None)


def _looks_like_stock_vanilla_backend(backend: Mapping[str, Any]) -> bool:
    """Recognize only the shipped vanilla checkpoint/temperature pair.

    This check is deliberately conservative.  It is used to identify an
    inherited stock block when a separate SolubleMPNN metadata block is
    present; custom checkpoint or temperature values must not be silently
    replaced by the automatic route.
    """
    checkpoint = str(
        backend.get("checkpoint") or "proteinmpnn/vanilla/v_48_020.pt"
    ).replace("\\", "/")
    checkpoint_name = PurePosixPath(checkpoint).name
    try:
        temperature = float(backend.get("temperature", float("nan")))
    except (TypeError, ValueError):
        temperature = float("nan")
    return (
        str(backend.get("model_type", "")) == "protein_mpnn"
        and checkpoint_name == "v_48_020.pt"
        and ("/vanilla/" in f"/{checkpoint.strip('/')}/")
        and (temperature == 0.0001)
    )


def normalize_reward_sequence_config(
    reward_config: Mapping[str, Any], *, requested: str, require_soluble: bool = False
) -> Mapping[str, Any]:
    """Resolve the internal SolubleMPNN backend for reward auto.

    proteinmpnn remains the native generation key, so this function only
    changes the effective backend options.  Explicit proteinmpnn requests
    return the original mapping untouched.  A separate solublempnn block
    (or shared solublempnn_defaults block) is required before an automatic
    protein/cyclic route can safely replace the vanilla block.  When callers
    provide a non-stock vanilla checkpoint or temperature while relying only
    on stock metadata, the choice is ambiguous and an actionable error is
    raised instead of overwriting the user value.

    The returned mapping is a deep copy only when a Soluble backend is
    selected; callers can therefore pass the original config through for
    explicit sources without changing ownership or cache behavior.
    """
    if requested != "auto":
        return reward_config
    sequence = reward_config.get("sequence")
    if not isinstance(sequence, Mapping):
        if require_soluble:
            raise ValueError("reward.sequence is required for automatic MPNN routing")
        return reward_config
    automatic = sequence.get("auto_proteinmpnn")
    if isinstance(automatic, Mapping):
        effective = copy.deepcopy(dict(reward_config))
        effective["sequence"]["proteinmpnn"] = copy.deepcopy(dict(automatic))
        return effective
    protein = sequence.get("proteinmpnn")
    if not isinstance(protein, Mapping):
        if require_soluble:
            raise ValueError(
                "reward.sequence.source=auto requires a proteinmpnn backend block"
            )
        return reward_config
    if str(protein.get("model_type", "")) == "soluble_mpnn" and bool(
        protein.get("use_soluble_model", True)
    ):
        return reward_config
    soluble, metadata_key = _auto_soluble_backend(sequence)
    if soluble is None:
        if require_soluble:
            raise ValueError(
                "reward.sequence.source=auto selected a protein/cyclic route, but no SolubleMPNN options were provided; add reward.sequence.solublempnn (or solublempnn_defaults)"
            )
        return reward_config
    model_type = str(soluble.get("model_type", ""))
    if model_type != "soluble_mpnn" or soluble.get("use_soluble_model") is False:
        raise ValueError(
            f"automatic SolubleMPNN metadata {metadata_key!r} must use model_type=soluble_mpnn and use_soluble_model=true"
        )
    if metadata_key == "solublempnn_defaults" and (
        not _looks_like_stock_vanilla_backend(protein)
    ):
        raise ValueError(
            "reward.sequence.source=auto has explicit proteinmpnn checkpoint or temperature values but only stock SolubleMPNN metadata is available; put the intended values in reward.sequence.solublempnn"
        )
    effective_sequence = copy.deepcopy(dict(sequence))
    effective_sequence["proteinmpnn"] = copy.deepcopy(dict(soluble))
    effective_config = copy.deepcopy(dict(reward_config))
    effective_config["sequence"] = effective_sequence
    return effective_config


ALIASES = {
    "design": "gt",
    "self": "gt",
    "gt": "gt",
    "protein_mpnn": "proteinmpnn",
    "proteinmpnn": "proteinmpnn",
    "soluble_mpnn": "solublempnn",
    "solublempnn": "solublempnn",
}
ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


_MPNN_BACKEND_KEYS = frozenset(
    {
        "model_type",
        "checkpoint",
        "temperature",
        "backbone_noise",
        "omit_aas",
        "rm_aa",
        "bias_aa",
        "pssm_path",
        "pssm_multi",
        "pssm_threshold",
        "pssm_log_odds",
        "pssm_bias",
        "tied_positions_path",
        "ca_only",
        "use_soluble_model",
    }
)
_SEQUENCE_KEYS = frozenset(
    {
        "source",
        "sources",
        "seed",
        "settings",
        "fixed_positions",
        "fixed_interface",
        "auto",
        "proteinmpnn",
        "auto_proteinmpnn",
        "solublempnn_defaults",
        "solublempnn",
        "soluble_mpnn",
    }
)
_BACKEND_CONTRACTS = {
    "protein_mpnn": ("vanilla", False, False),
    "soluble_mpnn": ("soluble", False, True),
    "ca_protein_mpnn": ("ca", True, False),
}


def _closed_mapping(value, keys, where):
    if not isinstance(value, Mapping):
        raise ValueError(f"{where} must be a mapping")
    unknown = sorted(set(value) - keys, key=str)
    if unknown:
        raise ValueError(f"Unknown {where} settings: {unknown}")
    return copy.deepcopy(dict(value))


def _normalize_fixed_positions(value):
    if not isinstance(value, list):
        raise ValueError("fixed_positions must be a list of CHAIN:POSITION strings")
    result = []
    seen = set()
    for item in value:
        if not isinstance(item, str) or item.count(":") != 1:
            raise ValueError("fixed_positions must use CHAIN:POSITION")
        chain, raw_position = item.split(":", 1)
        if not chain or re.fullmatch(r"[0-9]+", raw_position) is None:
            raise ValueError(f"Invalid fixed position: {item!r}")
        position = int(raw_position)
        if position < 1:
            raise ValueError(f"Invalid fixed position: {item!r}")
        key = (chain, position)
        if key in seen:
            raise ValueError(f"Duplicate fixed position: {item!r}")
        seen.add(key)
        result.append(f"{chain}:{position}")
    return result


def _backend_defaults(source):
    soluble = source in ("solublempnn", "soluble_mpnn")
    return dict(
        model_type="soluble_mpnn" if soluble else "protein_mpnn",
        checkpoint=None,
        temperature=0.1 if soluble else 0.0001,
        backbone_noise=0.0,
        rm_aa="C",
        omit_aas="",
        bias_aa={},
        pssm_path=None,
        pssm_multi=0.0,
        pssm_threshold=0.0,
        pssm_log_odds=False,
        pssm_bias=False,
        tied_positions_path=None,
        ca_only=False,
        use_soluble_model=soluble,
    )


def _validate_backend_values(config, where, *, complete=False):
    model_type = config.get("model_type")
    if "model_type" in config and (
        not isinstance(model_type, str) or model_type not in _BACKEND_CONTRACTS
    ):
        raise ValueError(f"{where}.model_type is an unsupported ProteinMPNN family")
    for flag in ("ca_only", "use_soluble_model", "pssm_bias", "pssm_log_odds"):
        if flag in config and type(config[flag]) is not bool:
            raise ValueError(f"{where}.{flag} must be a boolean")
    for key in ("checkpoint", "pssm_path", "tied_positions_path"):
        value = config.get(key)
        if value is not None and (
            not isinstance(value, str) or not value or "://" in value
        ):
            raise ValueError(f"{where}.{key} must be a non-empty local path or null")
    for key in ("temperature", "backbone_noise", "pssm_multi", "pssm_threshold"):
        if key not in config:
            continue
        value = config[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{where}.{key} must be finite numeric")
        if (
            (key == "temperature" and value <= 0)
            or (key == "backbone_noise" and value < 0)
            or (key == "pssm_multi" and not 0 <= value <= 1)
        ):
            raise ValueError(f"{where}.{key} is outside its supported range")
    if "bias_aa" in config:
        bias = config["bias_aa"]
        if not isinstance(bias, Mapping):
            raise ValueError(f"{where}.bias_aa must be a mapping")
        for amino_acid, value in bias.items():
            if (
                not isinstance(amino_acid, str)
                or len(amino_acid) != 1
                or amino_acid not in ALPHABET[:-1]
                or isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
            ):
                raise ValueError(
                    f"{where}.bias_aa requires canonical letters and finite values"
                )
        config["bias_aa"] = dict(bias)
    if "rm_aa" in config and config["rm_aa"] is not None:
        omit = config["rm_aa"]
        key = "rm_aa"
    else:
        omit = config.get("omit_aas") or ""
        key = "omit_aas"
        if config.get("omit_aas") is not None:
            omit = config["omit_aas"]
    if not isinstance(omit, str) or any(aa not in ALPHABET[:-1] for aa in omit):
        raise ValueError(f"{where}.{key} must contain canonical amino-acid letters")
    if set(omit) == set(ALPHABET[:-1]):
        raise ValueError(f"{where}.{key} cannot omit every amino acid")
    if not complete:
        return
    config["omit_aas"] = omit
    _, ca_only, soluble = _BACKEND_CONTRACTS[model_type]
    if config["ca_only"] != ca_only or config["use_soluble_model"] != soluble:
        raise ValueError(f"{where} model_type and ca_only/soluble flags disagree")
    path = config["pssm_path"]
    if path is None and (
        config["pssm_log_odds"]
        or config["pssm_bias"]
        or config["pssm_multi"] > 0
        or config["pssm_threshold"] != 0
    ):
        raise ValueError(f"{where} PSSM controls require pssm_path")
    if path is not None and not (config["pssm_log_odds"] or config["pssm_bias"]):
        raise ValueError(f"{where}.pssm_path requires pssm_log_odds or pssm_bias")
    if config["pssm_threshold"] != 0 and not config["pssm_log_odds"]:
        raise ValueError(f"{where}.pssm_threshold requires pssm_log_odds")
    if config["pssm_multi"] > 0 and not config["pssm_bias"]:
        raise ValueError(f"{where}.pssm_multi requires pssm_bias")


def _normalize_sequence_settings(value, where):
    settings = _closed_mapping(
        value, _MPNN_BACKEND_KEYS | {"fixed_positions", "fixed_interface"}, where
    )
    if "fixed_positions" in settings:
        settings["fixed_positions"] = _normalize_fixed_positions(
            settings["fixed_positions"]
        )
    if "fixed_interface" in settings:
        settings["fixed_interface"] = normalize_fixed_interface(
            settings["fixed_interface"]
        )
    _validate_backend_values(settings, where)
    return settings


def validate_sequence_config(value, *, reward: bool = False):
    """Validate the closed shared sequence schema without importing model libraries.

    Structural selector resolution remains candidate-local in fixed_positions().
    Paths are validated as local strings here; no constraint or checkpoint asset
    is read. Returned mappings and lists are detached from caller-owned values.
    """
    sequence = _closed_mapping(value, _SEQUENCE_KEYS, "sequence")
    seed = sequence.get("seed", 101)
    if type(seed) is not int or seed < 0:
        raise ValueError("sequence.seed must be a nonnegative integer")
    sequence["seed"] = seed
    sequence["fixed_positions"] = _normalize_fixed_positions(
        sequence.get("fixed_positions", [])
    )
    if "fixed_interface" in sequence:
        sequence["fixed_interface"] = normalize_fixed_interface(
            sequence["fixed_interface"]
        )
    if "settings" in sequence:
        sequence["settings"] = _normalize_sequence_settings(
            sequence["settings"], "sequence.settings"
        )
    if reward or "source" in sequence:
        source = sequence.get("source", "design")
        if not isinstance(source, str) or source not in {*ALIASES, "auto"}:
            raise ValueError("Unsupported reward sequence source")
        sequence["source"] = source
    if not reward or "sources" in sequence:
        sequence["sources"] = normalize_sources(
            sequence.get(
                "sources",
                [
                    {"source": "gt", "label": "self", "count": 1},
                    {
                        "source": "soluble_mpnn",
                        "label": "soluble",
                        "count": 1,
                        "settings": {"temperature": 0.1},
                    },
                ],
            )
        )
    for name in ("proteinmpnn", "auto_proteinmpnn", *_AUTO_SOLUBLE_BACKEND_KEYS):
        if name not in sequence:
            continue
        options = _closed_mapping(
            sequence[name], _MPNN_BACKEND_KEYS, f"sequence.{name}"
        )
        _validate_backend_values(options, f"sequence.{name}")
        config = _backend_defaults(
            "solublempnn" if name in _AUTO_SOLUBLE_BACKEND_KEYS else "proteinmpnn"
        )
        config.update(options)
        if not reward and "omit_aas" in options and "rm_aa" not in options:
            config["rm_aa"] = None
        _validate_backend_values(config, f"sequence.{name}", complete=True)
        if (
            name in _AUTO_SOLUBLE_BACKEND_KEYS
            and config["model_type"] != "soluble_mpnn"
        ):
            raise ValueError(f"sequence.{name} must select the SolubleMPNN contract")
        sequence[name] = config
    if "auto" in sequence:
        automatic = _closed_mapping(
            sequence["auto"], {"protein", "cyclic"}, "sequence.auto"
        )
        for name, options in automatic.items():
            if isinstance(options, str):
                if options not in ("soluble_mpnn", "solublempnn"):
                    raise ValueError(f"sequence.auto.{name} must select SolubleMPNN")
                continue
            options = _closed_mapping(
                options, _MPNN_BACKEND_KEYS, f"sequence.auto.{name}"
            )
            config = _backend_defaults("solublempnn")
            config.update(options)
            _validate_backend_values(config, f"sequence.auto.{name}", complete=True)
            if config["model_type"] != "soluble_mpnn":
                raise ValueError(f"sequence.auto.{name} must select SolubleMPNN")
            automatic[name] = config
        sequence["auto"] = automatic
    if reward and sequence["source"] == "auto":
        sequence.setdefault("proteinmpnn", _backend_defaults("proteinmpnn"))
        if _auto_soluble_backend(sequence)[0] is None:
            sequence["solublempnn_defaults"] = _backend_defaults("solublempnn")
        sequence = normalize_reward_sequence_config(
            {"sequence": sequence}, requested="auto", require_soluble=True
        )["sequence"]
    settings = sequence.get("settings", {})
    if reward and sequence["source"] != "auto":
        selected = ALIASES[sequence["source"]]
        if selected != "gt":
            backend_settings(selected, sequence, Path.cwd(), settings)
    elif reward:
        backend_settings("auto", sequence, Path.cwd(), settings)
    for branch in sequence.get("sources", []):
        if branch["source"] != "gt":
            backend_settings(
                branch["source"],
                sequence,
                Path.cwd(),
                {**settings, **branch["settings"]},
            )
    return sequence


def normalize_sources(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("sequence.sources must be nonempty")
    result = []
    labels = set()
    seen_gt = False
    for value in values:
        if not isinstance(value, dict) or set(value) - {
            "source",
            "count",
            "label",
            "settings",
        }:
            raise ValueError("Source entries require source/count/label/settings")
        name = str(value.get("source", "")).lower()
        fi = name.endswith("-fi")
        name = name.removesuffix("-fi")
        source = ALIASES.get(name)
        if source is None:
            raise ValueError(f"Unsupported sequence source: {name}")
        count = value.get("count", 1)
        if type(count) is not int or count < 1:
            raise ValueError("Source count must be positive integer")
        settings = copy.deepcopy(value.get("settings", {}))
        if not isinstance(settings, dict):
            raise ValueError("Source settings must be a mapping")
        if fi:
            settings.setdefault("fixed_interface", {"enabled": True})
            if not isinstance(settings["fixed_interface"], dict) or not settings[
                "fixed_interface"
            ].get("enabled"):
                raise ValueError("-fi requires fixed interface enabled")
        settings = _normalize_sequence_settings(settings, "Source settings")
        fixed_interface = settings.get("fixed_interface")
        label = value.get(
            "label",
            "self"
            if source == "gt"
            else source
            + ("_fi" if fixed_interface and fixed_interface["enabled"] else ""),
        )
        if (
            not isinstance(label, str)
            or not label
            or any(
                (
                    c
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                    for c in label
                )
            )
        ):
            raise ValueError("Invalid sequence branch label")
        if source == "gt":
            if count != 1 or seen_gt or label != "self" or settings:
                raise ValueError("GT requires one self slot with no redesign settings")
            seen_gt = True
        elif label == "self":
            raise ValueError("self is reserved for GT")
        if label in labels:
            raise ValueError("Duplicate sequence branch label")
        labels.add(label)
        result.append(dict(source=source, count=count, label=label, settings=settings))
    return result


def normalize_fixed_interface(value):
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or set(value) - {"enabled", "cutoff", "target_chains"}
        or type(value.get("enabled")) is not bool
    ):
        raise ValueError(
            "fixed_interface requires enabled boolean, optional cutoff/target_chains"
        )
    cutoff = value.get("cutoff", 6.0)
    if (
        isinstance(cutoff, bool)
        or not isinstance(cutoff, (float, int))
        or (not math.isfinite(cutoff))
        or (cutoff <= 0)
    ):
        raise ValueError("fixed_interface.cutoff must be positive finite")
    targets = value.get("target_chains")
    if targets is not None and (
        not isinstance(targets, list)
        or any(
            (
                not isinstance(c, str)
                or not c
                or c != c.strip()
                or any((character.isspace() for character in c))
                or (c in (".", "?"))
                or (":" in c)
                for c in targets
            )
        )
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("fixed_interface.target_chains must be unique chain IDs")
    return dict(enabled=value["enabled"], cutoff=float(cutoff), target_chains=targets)


def backend_settings(source, sequence_config, model_dir, settings=None):
    requested = source
    if requested == "auto":
        sequence_config = normalize_reward_sequence_config(
            {"sequence": sequence_config}, requested="auto", require_soluble=True
        )["sequence"]
        source = "proteinmpnn"
    if source in ("soluble_mpnn", "solublempnn"):
        source = "solublempnn"
        key = "solublempnn_defaults"
    elif source in ("proteinmpnn", "protein_mpnn"):
        source = "proteinmpnn"
        key = "proteinmpnn"
    else:
        raise ValueError(f"Unsupported MPNN source: {source}")
    _closed_mapping(sequence_config, _SEQUENCE_KEYS, "sequence")
    config = _backend_defaults(source)
    selected = sequence_config.get(key, {})
    if source == "solublempnn":
        selected = sequence_config.get(
            "solublempnn", sequence_config.get("soluble_mpnn", selected)
        )
    selected = _closed_mapping(selected, _MPNN_BACKEND_KEYS, f"sequence.{key}")
    options = _normalize_sequence_settings(
        {} if settings is None else settings, "MPNN settings"
    )
    config.update(selected)
    if "omit_aas" in selected and "rm_aa" not in selected:
        config["rm_aa"] = None
    if "omit_aas" in options and "rm_aa" not in options:
        config["rm_aa"] = None
    config.update(
        {
            key: value
            for key, value in options.items()
            if key not in ("fixed_interface", "fixed_positions")
        }
    )
    _validate_backend_values(config, "MPNN", complete=True)
    folder, _, soluble = _BACKEND_CONTRACTS[config["model_type"]]
    if source == "solublempnn" and not soluble:
        raise ValueError("SolubleMPNN source requires soluble model family")
    checkpoint = config.get("checkpoint")
    if checkpoint is None:
        checkpoint = Path(model_dir) / "proteinmpnn" / folder / "v_48_020.pt"
    else:
        checkpoint = Path(str(checkpoint)).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(model_dir) / checkpoint
    config["checkpoint"] = str(checkpoint.resolve())
    return config


def sequence_settings(sequence, source_settings=None):
    """Combine global identity requirements with source-specific model settings."""
    global_settings = dict(sequence.get("settings", {}))
    local_settings = dict(source_settings or {})
    result = {**global_settings, **local_settings}
    fixed = []
    for values in (
        sequence.get("fixed_positions", []),
        global_settings.get("fixed_positions", []),
        local_settings.get("fixed_positions", []),
    ):
        for position in values:
            if position not in fixed:
                fixed.append(position)
    result["fixed_positions"] = fixed
    if "fixed_interface" not in result and "fixed_interface" in sequence:
        result["fixed_interface"] = sequence["fixed_interface"]
    return result


def fixed_positions(candidate, input_data, settings):
    import numpy as np
    from .structure import polymer_residue_groups

    atoms = input_data["atoms"]
    chain = candidate.binder_chain
    groups = polymer_residue_groups(atoms, chain)
    fixed = set()
    for entry in candidate.fixed_restypes:
        if not isinstance(entry, dict):
            raise CandidateError("Malformed fixed identity entry")
        target = str(
            entry.get("chain", entry.get("chain_id", entry.get("label_asym_id", "")))
        )
        if target != chain:
            continue
        res_id = entry.get("res_id", entry.get("label_seq_id"))
        ins = str(entry.get("ins_code", ""))
        matches = [
            i + 1
            for i, g in enumerate(groups)
            if str(g["locator"]["res_id"]) == str(res_id)
            and g["locator"].get("ins_code", "") == ins
        ]
        if len(matches) != 1:
            raise CandidateError("Fixed identity does not resolve uniquely")
        position = matches[0]
        identity = entry.get(
            "res_name", entry.get("restype", entry.get("component_id"))
        )
        if identity is not None and str(identity) not in (
            groups[position - 1]["res_name"],
            candidate.sequence[position - 1],
        ):
            raise CandidateError("Fixed identity differs from input structure")
        fixed.add(position)
    # contig_string and tip_atoms describe geometry. Identity constraints are
    # carried by fixed_restypes; geometry-only motifs remain sequence-free.
    known_chains = {chain, *input_data["context_chains"]}
    groups_by_chain = {chain: groups}
    for item in _normalize_fixed_positions(settings.get("fixed_positions", [])):
        target, raw_position = item.split(":", 1)
        position = int(raw_position)
        if target not in known_chains:
            raise CandidateError(f"Fixed position does not resolve: {item!r}")
        if target not in groups_by_chain:
            groups_by_chain[target] = polymer_residue_groups(atoms, target)
        if not 1 <= position <= len(groups_by_chain[target]):
            raise CandidateError(f"Fixed position does not resolve: {item!r}")
        if target == chain:
            fixed.add(position)
    fi = normalize_fixed_interface(settings.get("fixed_interface"))
    if fi and fi["enabled"]:
        targets = (
            fi["target_chains"]
            if fi["target_chains"] is not None
            else list(input_data["context_chains"])
        )
        if any(
            (
                target == chain or target not in input_data["context_chains"]
                for target in targets
            )
        ):
            raise CandidateError("FI target must be an existing context chain")
        heavy = ~np.isin(np.char.upper(atoms.element.astype(str)), ("H", "D", "T"))
        target_rows = np.flatnonzero(np.isin(atoms.chain_id, targets) & heavy)
        if not len(target_rows):
            raise CandidateError("FI has no target heavy atoms")
        for i, group in enumerate(groups, 1):
            rows = np.asarray(group["row_indices"])
            rows = rows[heavy[rows]]
            if len(rows) and np.any(
                np.sum(
                    (atoms.coord[rows, None, :] - atoms.coord[target_rows][None, :, :])
                    ** 2,
                    axis=-1,
                )
                <= fi["cutoff"] ** 2
            ):
                fixed.add(i)
    if atoms.bonds is not None:
        position_by_row = {
            row: i for i, group in enumerate(groups, 1) for row in group["row_indices"]
        }
        for left, right, order in atoms.bonds.as_array():
            if (
                atoms.atom_name[left] == atoms.atom_name[right] == "SG"
                and atoms.res_name[left] == atoms.res_name[right] == "CYS"
                and (int(order) == 1)
            ):
                for row in (left, right):
                    if int(row) in position_by_row:
                        fixed.add(position_by_row[int(row)])
    return sorted(fixed)


class MPNNBackend:
    """One resident official model per exact model settings."""

    def __init__(self, device=None):
        self.device = device
        self.models = {}
        self.load_count = 0
        self.hit_count = 0

    def close(self):
        self.models.clear()

    def _model(self, config):
        import torch
        from partiflow._vendor.proteinmpnn.protein_mpnn_utils import ProteinMPNN

        path = Path(config["checkpoint"])
        key = (
            str(path.resolve()),
            bool(config.get("ca_only")),
            float(config["backbone_noise"]),
            str(self.device),
        )
        if key in self.models:
            self.hit_count += 1
            return self.models[key]
        if not path.is_file() or path.stat().st_size == 0:
            raise WorkflowError(f"MPNN checkpoint missing: {path}")
        device = torch.device(
            self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
            model = ProteinMPNN(
                ca_only=bool(config.get("ca_only")),
                num_letters=21,
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                augment_eps=float(config["backbone_noise"]),
                k_neighbors=checkpoint["num_edges"],
            ).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
        except Exception as exc:
            raise WorkflowError(f"MPNN model load failed: {exc}") from exc
        self.models[key] = (model, device)
        self.load_count += 1
        return (model, device)

    def generate(self, candidate, input_data, config, positions, seed):
        if type(seed) is not int or seed < 0:
            raise ValueError("MPNN seed must be a nonnegative integer")
        seed %= 2147483647
        import random
        import numpy as np
        import torch
        from partiflow._vendor.proteinmpnn.protein_mpnn_utils import (
            tied_featurize,
            _S_to_seq,
        )
        from .structure import polymer_residue_groups, polymer_break_before, AA3_TO_1

        atoms = input_data["atoms"]
        record = {"name": "candidate", "num_of_chains": 0, "seq": ""}
        aliases = {}
        design_alias = None
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        for chain in [candidate.binder_chain, *input_data["context_chains"]]:
            groups = polymer_residue_groups(atoms, chain)
            breaks = (
                set()
                if chain == candidate.binder_chain
                else polymer_break_before(atoms, groups)
            )
            segments = []
            for i, group in enumerate(groups):
                if i == 0 or i in breaks:
                    segments.append([])
                segments[-1].append(group)
            for segment in segments:
                if len(aliases) >= len(alphabet):
                    raise CandidateError("MPNN private input exceeds 62 chains")
                alias = alphabet[len(aliases)]
                aliases[alias] = (chain, segment)
                if chain == candidate.binder_chain:
                    design_alias = alias
                sequence = "".join((AA3_TO_1.get(g["res_name"], "X") for g in segment))
                if "X" in sequence:
                    raise CandidateError("MPNN requires standard protein residues")
                coords = {}
                for name in ("CA",) if config.get("ca_only") else ("N", "CA", "C", "O"):
                    values = []
                    for group in segment:
                        rows = [
                            row
                            for row in group["row_indices"]
                            if atoms.atom_name[row] == name
                        ]
                        if len(rows) != 1:
                            raise CandidateError(
                                f"MPNN backbone atom {name} missing or ambiguous"
                            )
                        values.append(atoms.coord[rows[0]].tolist())
                    coords[f"{name}_chain_{alias}"] = values
                record[f"seq_chain_{alias}"] = sequence
                record[f"coords_chain_{alias}"] = coords
                record["seq"] += sequence
        record["num_of_chains"] = len(aliases)
        fixed = {
            "candidate": {
                alias: positions if alias == design_alias else [] for alias in aliases
            }
        }
        omit = (
            config.get("rm_aa")
            if config.get("rm_aa") is not None
            else config.get("omit_aas", "")
        )
        if not isinstance(omit, str) or any((c not in ALPHABET[:-1] for c in omit)):
            raise ValueError("rm_aa must use canonical amino acids")
        free = [i for i in range(1, candidate.length + 1) if i not in positions]
        if free and len(set(omit)) == 20:
            raise CandidateError("MPNN omission excludes every free residue identity")
        if not free:
            return candidate.sequence
        omit_payload = {"candidate": {design_alias: [(free, omit)]}} if omit else None
        pssm = _pssm_payload(
            config, candidate.binder_chain, design_alias, candidate.length
        )
        tied = _tied_payload(
            config, candidate.binder_chain, design_alias, candidate.length, positions
        )
        model, device = self._model(config)
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        try:
            with torch.no_grad():
                features = tied_featurize(
                    [record],
                    device,
                    {
                        "candidate": (
                            [design_alias],
                            [a for a in aliases if a != design_alias],
                        )
                    },
                    fixed,
                    omit_payload,
                    tied,
                    pssm,
                    None,
                    ca_only=bool(config.get("ca_only")),
                )
                (
                    X,
                    S,
                    mask,
                    lengths,
                    chain_M,
                    chain_encoding,
                    chain_lists,
                    visible,
                    masked,
                    masked_lengths,
                    chain_M_pos,
                    omit_mask,
                    residue_idx,
                    dihedral,
                    tied_lists,
                    coef,
                    bias,
                    log_odds,
                    bias_by_res,
                    tied_beta,
                ) = features
                kwargs = dict(
                    mask=mask,
                    temperature=float(config["temperature"]),
                    omit_AAs_np=np.zeros(21, dtype=np.float32),
                    bias_AAs_np=np.asarray(
                        [
                            float(config.get("bias_aa", {}).get(aa, 0.0))
                            for aa in ALPHABET
                        ],
                        dtype=np.float32,
                    ),
                    chain_M_pos=chain_M_pos,
                    omit_AA_mask=omit_mask,
                    pssm_coef=coef,
                    pssm_bias=bias,
                    pssm_multi=float(config["pssm_multi"]),
                    pssm_log_odds_flag=bool(config["pssm_log_odds"]),
                    pssm_log_odds_mask=(log_odds > config["pssm_threshold"]).float(),
                    pssm_bias_flag=bool(config["pssm_bias"]),
                    bias_by_res=bias_by_res,
                )
                if tied:
                    kwargs.update(tied_pos=tied_lists[0], tied_beta=tied_beta)
                result = (model.tied_sample if tied else model.sample)(
                    X,
                    torch.randn(chain_M.shape, device=device),
                    S,
                    chain_M,
                    chain_encoding,
                    residue_idx,
                    **kwargs,
                )
                sequence = _S_to_seq(result["S"][0], chain_M[0])
        except Exception as exc:
            raise WorkflowError(f"MPNN model execution failed: {exc}") from exc
        if len(sequence) != candidate.length or any(
            (aa not in ALPHABET[:-1] for aa in sequence)
        ):
            raise WorkflowError("MPNN returned malformed sequence")
        if any((sequence[p - 1] != candidate.sequence[p - 1] for p in positions)):
            raise WorkflowError("MPNN changed a fixed identity")
        if any((sequence[p - 1] in omit for p in free)):
            raise WorkflowError("MPNN sampled an omitted free identity")
        return sequence


def _json_asset(path):
    try:
        return json.loads(Path(path).expanduser().read_text())
    except (OSError, ValueError) as exc:
        raise WorkflowError(
            f"MPNN constraint asset could not be read: {path}: {exc}"
        ) from exc


def _pssm_payload(config, chain, alias, length):
    import numpy as np

    path = config.get("pssm_path")
    if not path:
        return None
    raw = _json_asset(path)
    if (
        not isinstance(raw, dict)
        or len(raw) != 1
        or (not isinstance(raw.get(chain, raw.get(alias)), dict))
    ):
        raise WorkflowError("PSSM must contain exactly the canonical design chain")
    value = raw.get(chain, raw.get(alias))
    alphabet = value.get("alphabet", ALPHABET)
    if len(alphabet) != 21 or set(alphabet) != set(ALPHABET):
        raise WorkflowError("Invalid PSSM alphabet")
    order = [alphabet.index(c) for c in ALPHABET]
    result = {}
    for field, default, shape in [
        ("pssm_coef", 1.0, (length,)),
        ("pssm_bias", 0.0, (length, 21)),
        ("pssm_log_odds", 10000.0, (length, 21)),
    ]:
        matrix = np.asarray(value.get(field, np.full(shape, default)), dtype=float)
        if matrix.shape != shape or not np.isfinite(matrix).all():
            raise WorkflowError(f"Invalid PSSM {field}")
        if matrix.ndim == 2:
            matrix = matrix[:, order]
        result[field] = matrix.tolist()
    coef = np.asarray(result["pssm_coef"])
    bias = np.asarray(result["pssm_bias"])
    if np.any((coef < 0) | (coef > 1)):
        raise WorkflowError("PSSM coefficient must be in [0,1]")
    if config["pssm_bias"] and (
        np.any(bias < 0) or not np.allclose(bias.sum(1), 1.0, rtol=1e-06, atol=1e-06)
    ):
        raise WorkflowError("PSSM bias rows must be probability distributions")
    if config["pssm_log_odds"] and "pssm_log_odds" not in value:
        raise WorkflowError("PSSM log odds missing")
    return {"candidate": {alias: result}}


def _tied_payload(config, chain, alias, length, fixed):
    path = config.get("tied_positions_path")
    if not path:
        return None
    raw = _json_asset(path)
    groups = []
    used = set()
    if not isinstance(raw, list):
        raise WorkflowError("Tied positions must be a list")
    for group in raw:
        if not isinstance(group, list) or len(group) < 2:
            raise WorkflowError("Tied groups need at least two positions")
        positions = []
        for item in group:
            if not isinstance(item, str) or not item.startswith(chain + ":"):
                raise WorkflowError("Tied positions must target the design chain")
            try:
                position = int(item[len(chain) + 1 :])
            except ValueError as exc:
                raise WorkflowError("Invalid tied-position index") from exc
            if not 1 <= position <= length or position in fixed or position in used:
                raise WorkflowError(
                    "Tied position is fixed, repeated or outside design"
                )
            positions.append(position)
            used.add(position)
        groups.append({alias: positions})
    return {"candidate": groups}
