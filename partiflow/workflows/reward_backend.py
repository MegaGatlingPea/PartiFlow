"""Resident offline multimer initial-guess reward calculation."""
from __future__ import annotations
import os
import sys
from pathlib import Path
from typing import Any, Mapping
import numpy as np
from .af2_mapping import validate_prepared_af2
_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_PATCH_ABI = 'af2-offset-v1'
from partiflow.common.records import CandidateError, WorkflowError

class AF2CandidateError(CandidateError):
    """One validated request is incompatible with the AF2 candidate contract."""
_SYSTEMIC_ERROR_MARKERS = ('jax', 'xla', 'cuda', 'cublas', 'cudnn', 'out of memory', 'illegal memory', 'device-side assert', 'worker protocol')

def _is_systemic_error(error: Exception) -> bool:
    if isinstance(error, (ImportError, ModuleNotFoundError, MemoryError, NotImplementedError)):
        return True
    text = f'{type(error).__name__}: {error}'.lower()
    return any((marker in text for marker in _SYSTEMIC_ERROR_MARKERS))

def signed_circular_offset(length: int) -> np.ndarray:
    if isinstance(length, bool) or not isinstance(length, int) or length < 2:
        raise AF2CandidateError('cyclic AF2 offset length must be an integer >= 2')
    positions = np.arange(length, dtype=np.int32)
    linear = positions[:, None] - positions[None, :]
    absolute = np.abs(linear)
    wrapped = length - absolute
    magnitude = np.where(wrapped < absolute, -wrapped, absolute)
    return (magnitude * np.sign(linear)).astype(np.int32, copy=False)

def _cache_key(config: Mapping[str, Any]) -> tuple[Any, ...]:
    if config.get('model') != 'multimer':
        raise WorkflowError('AF2 reward backend requires the multimer initial-guess protocol')
    return (str(Path(config['params_dir']).resolve()), str(config['model']), int(config['model_num']), int(config['num_recycles']), bool(config['use_initial_guess']), False, _PATCH_ABI)

def _clear_state(model: Any) -> None:
    for name in ('_inputs', 'aux', '_tmp'):
        value = getattr(model, name, None)
        if isinstance(value, dict):
            value.clear()

def _model(config: Mapping[str, Any]) -> tuple[Any, bool]:
    key = _cache_key(config)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return (cached, True)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    source = Path(__file__).resolve().parents[1] / '_vendor' / 'af2'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    from colabdesign import mk_afdesign_model
    import colabdesign
    if source not in Path(colabdesign.__file__).resolve().parents:
        raise RuntimeError('Ambient ColabDesign package is not the packaged AF2 source')
    multimer = True
    model_num = int(config['model_num'])
    model_name = f'model_{model_num}_multimer_v3'
    built = mk_afdesign_model(protocol='binder', num_recycles=int(config['num_recycles']), data_dir=str(config['params_dir']), use_multimer=multimer, use_templates=True, use_initial_guess=bool(config['use_initial_guess']), use_initial_atom_pos=False, model_names=[model_name])
    _MODEL_CACHE[key] = built
    return (built, False)

def _prepare(model: Any, request: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[int, int]:
    design_sequence = str(request['design_sequence'])
    design_length = len(design_sequence)
    context = ','.join((str(value) for value in request['context_chains']))
    try:
        model.prep_inputs(pdb_filename=str(request['input_pdb']), chain=context, binder_chain=str(request['design_chain']), binder_len=design_length, use_binder_template=True, rm_target_seq=False, rm_target_sc=False, rm_template_ic=True)
    except (OSError, ValueError) as error:
        if _is_systemic_error(error):
            raise
        raise AF2CandidateError(f'AF2 candidate input preparation failed: {error}') from error
    ledger = request.get('source_ledger')
    if not isinstance(ledger, Mapping):
        raise AF2CandidateError('AF2 request has no authoritative source occurrence ledger')
    source_design = str(request['source_design_chain'])
    source_order = [*request['source_context_chains'], source_design]
    try:
        validate_prepared_af2(model, ledger, source_order)
    except ValueError as error:
        raise AF2CandidateError(str(error)) from error
    total = int(np.asarray(model._inputs['residue_index']).shape[0])
    start = total - design_length
    if start < 0 or start + design_length > total:
        raise AF2CandidateError('AF2 design-chain token slice is inconsistent')
    if bool(request.get('cyclic_offset', False)):
        residues = np.asarray(model._inputs['residue_index'], dtype=np.int32)
        offset = residues[:, None] - residues[None, :]
        offset[start:start + design_length, start:start + design_length] = signed_circular_offset(design_length)
        model._inputs['offset'] = offset
    else:
        model._inputs.pop('offset', None)
    return (start, design_length)

def _interface_ipae(pae: np.ndarray, asym_id: np.ndarray, *, design_start: int, design_length: int) -> float:
    """Equal-weight bidirectional PAE over design-touching interfaces."""
    pae = np.asarray(pae, dtype=np.float64)
    asym_id = np.asarray(asym_id).reshape(-1)
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1] or pae.shape[0] != len(asym_id):
        raise AF2CandidateError('AF2 PAE/asym_id tensors have inconsistent shapes')
    design = np.arange(design_start, design_start + design_length, dtype=np.int64)
    if len(design) < 1 or int(design[-1]) >= len(asym_id):
        raise AF2CandidateError('AF2 design slice is outside the PAE tensor')
    design_asym = np.unique(asym_id[design])
    if len(design_asym) != 1:
        raise AF2CandidateError('AF2 design chain does not map to one asym_id')
    context = np.arange(0, design_start, dtype=np.int64)
    interfaces: list[float] = []
    for context_asym in np.unique(asym_id[context]):
        rows = context[asym_id[context] == context_asym]
        if len(rows) < 1:
            continue
        bidirectional = np.concatenate([pae[np.ix_(rows, design)].reshape(-1), pae[np.ix_(design, rows)].reshape(-1)])
        if not np.isfinite(bidirectional).all():
            raise AF2CandidateError('AF2 interface PAE contains non-finite values')
        interfaces.append(float(bidirectional.mean()))
    if not interfaces:
        raise AF2CandidateError('AF2 Multimer output lacks a context interface')
    return float(np.mean(np.asarray(interfaces, dtype=np.float64)))

def _predict_one(model: Any, request: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    start, length = _prepare(model, request, config)
    model.predict(seq=str(request['design_sequence']), models=[0], num_recycles=int(config['num_recycles']), verbose=False)
    output_pdb = Path(str(request['output_pdb']))
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    model.save_pdb(str(output_pdb))
    try:
        plddt = np.asarray(model.aux['plddt'], dtype=np.float64).reshape(-1)
        pae = np.asarray(model.aux['pae'], dtype=np.float64)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise AF2CandidateError('AF2 confidence payload lacks pLDDT or PAE') from error
    if plddt.shape[0] < start + length or pae.shape != (plddt.shape[0], plddt.shape[0]):
        raise AF2CandidateError('AF2 confidence tensors have inconsistent shapes')
    design_values = plddt[start:start + length]
    if not np.isfinite(design_values).all() or not np.isfinite(pae).all():
        raise AF2CandidateError('AF2 confidence tensors contain non-finite values')
    if np.any(design_values < 0) or np.any(design_values > 1) or np.any(pae < 0):
        raise AF2CandidateError('AF2 confidence tensors are outside declared ranges')
    design_plddt = float(design_values.mean() * 100.0)
    try:
        ptm = float(model.aux['log']['ptm'])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise AF2CandidateError('AF2 pTM is missing or malformed') from error
    if not np.isfinite(ptm) or not 0 <= ptm <= 1:
        raise AF2CandidateError('AF2 pTM is missing or outside [0,1]')
    pae_mean = float(pae.mean())
    result = {'status': 'ok', 'candidate_id': str(request['candidate_id']), 'fold_pdb': str(output_pdb), 'design_plddt': design_plddt, 'ptm': ptm, 'pae': pae_mean}
    if start == 0:
        raise AF2CandidateError('AF2 Multimer output lacks a context block')
    try:
        interface_iptm = float(model.aux['log']['i_ptm'])
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise AF2CandidateError('AF2 Multimer ipTM is missing or malformed') from error
    if not np.isfinite(interface_iptm) or not 0 <= interface_iptm <= 1:
        raise AF2CandidateError('AF2 Multimer ipTM is missing or outside [0,1]')
    result['interface_iptm'] = interface_iptm
    result['interface_ipae'] = _interface_ipae(pae, np.asarray(model._inputs.get('asym_id')), design_start=start, design_length=length)
    return result
