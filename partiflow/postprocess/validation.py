"""Temporary CPU checks for all declared native score-only residue types."""
from __future__ import annotations
import argparse
import json
import math
import tempfile
from pathlib import Path


def _atom_lines(path):
    return [line for line in Path(path).read_text().splitlines(keepends=True) if line.startswith(('ATOM  ', 'HETATM'))]


def _key(line):
    return line[21:22], int(line[22:26]), line[26:27]


def _xyz(line):
    return tuple(float(line[start:start+8]) for start in (30, 38, 46))


def substitute_fixture(base, code, output, *, binder_chain='L'):
    """Place one exact passive residue at an internal scaffold position."""
    import numpy as np
    base = Path(base)
    base_lines = _atom_lines(base)
    residues = list(dict.fromkeys(_key(line) for line in base_lines if line[21:22] == binder_chain))
    if len(residues) < 3:
        raise ValueError('Native checks require a binder scaffold with at least three residues')
    target = residues[len(residues)//2]
    target_atoms = {line[12:16].strip(): _xyz(line) for line in base_lines if _key(line) == target}
    source = Path(__file__).parent.parent / 'utils' / 'relax_assets' / f'passive_{code}_with_H.pdb'
    source_lines = _atom_lines(source)
    source_atoms = {line[12:16].strip(): _xyz(line) for line in source_lines}
    anchors = ('N', 'CB', 'C') if code == 'BAL' else ('N', 'CA', 'C')
    left = np.asarray([source_atoms[name] for name in anchors])
    right = np.asarray([target_atoms[name] for name in ('N', 'CA', 'C')])
    left_center, right_center = left.mean(axis=0), right.mean(axis=0)
    u, _, vt = np.linalg.svd((left-left_center).T @ (right-right_center))
    if np.linalg.det(u @ vt) < 0:
        vt[-1, :] *= -1
    rotation = u @ vt
    serial = max(int(line[6:11]) for line in base_lines)
    replacement = []
    for offset, line in enumerate(source_lines, 1):
        xyz = (np.asarray(_xyz(line))-left_center) @ rotation + right_center
        replacement.append(line[:6] + f'{serial+offset:5d}' + line[11:17] + f'{code:>3}' + line[20:21]
                           + target[0] + f'{target[1]:4d}' + target[2] + line[27:30]
                           + ''.join(f'{value:8.3f}' for value in xyz) + line[54:].rstrip('\n') + '\n')
    result, inserted = [], False
    for line in base.read_text().splitlines(keepends=True):
        if line.startswith(('ATOM  ', 'HETATM')) and _key(line) == target:
            if not inserted:
                result.extend(replacement)
                inserted = True
        else:
            result.append(line)
    Path(output).write_text(''.join(result))
    return {'residue_code': code, 'binder_chain': binder_chain, 'binder_position': len(residues)//2+1,
            'source_residue': list(target)}


def validate_native_support(base_pdb, *, codes=None, binder_chain='L', full_metrics=False, relax=False):
    from partiflow.postprocess.native import NCAA_SCORE_ONLY_RESIDUES, score_fixed_pose
    from partiflow.postprocess.metrics import compute_metrics
    from partiflow.postprocess.records import WorkItem
    selected = list(codes or NCAA_SCORE_ONLY_RESIDUES)
    unknown = set(selected) - set(NCAA_SCORE_ONLY_RESIDUES)
    if unknown:
        raise ValueError(f'Unknown native residue codes: {sorted(unknown)}')
    results = {}
    with tempfile.TemporaryDirectory(prefix='partiflow-native-') as directory:
        for code in selected:
            source = Path(directory) / f'{code}.pdb'
            info = substitute_fixture(base_pdb, code, source, binder_chain=binder_chain)
            try:
                target = source
                if relax:
                    from partiflow.postprocess.chemistry import relax_candidate
                    target, relaxation = relax_candidate(WorkItem(source, 'native', code,
                        binder_chain=binder_chain, method='openmm'), source.with_name(code+'_relaxed.pdb'), platform='CPU')
                    info['postprocess'] = relaxation
                score = score_fixed_pose(target, binder_chain=binder_chain)
                required = ('total_score', 'pep_internal_sc', 'I_sc', 'reweighted_sc')
                if not all(math.isfinite(float(score[name])) for name in required):
                    raise ValueError('Native fixed-pose scoring returned a non-finite energy')
                if score['pdb_fallback_residue_types'] or score['score_component_pathology']:
                    raise ValueError('Native residue handling or component scoring failed')
                rows = [row for row in score['runtime_residue_audit'] if row['source_code'] == code]
                if len(rows) != 1:
                    raise ValueError(f'Expected one native residue audit row, found {len(rows)}')
                audit = score['input_heavy_atom_audit']
                if audit['source_only_count'] or audit['max_delta_angstrom'] > audit['tolerance_angstrom']:
                    raise ValueError('Native load did not preserve the source heavy atoms')
                info.update(status='success', energies={name: score[name] for name in required},
                            runtime_residue=rows[0], input_heavy_atoms=audit,
                            pack_input=score['pack_input'], pack_separated=score['pack_separated'],
                            relax=score['relax'])
                if full_metrics:
                    info['metrics_report'] = compute_metrics(target, binder_chain=binder_chain)
            except Exception as exc:
                info.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            results[code] = info
    passed = sum(item['status'] == 'success' for item in results.values())
    return {'status': 'success' if passed == len(selected) else 'failed', 'computability_only': True,
            'base_pdb': str(Path(base_pdb).resolve()), 'summary': {'requested': len(selected), 'passed': passed,
            'failed': len(selected)-passed}, 'results': results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--code', action='append', dest='codes')
    parser.add_argument('--binder-chain', default='L')
    parser.add_argument('--full-metrics', action='store_true')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    report = validate_native_support(args.base, codes=args.codes, binder_chain=args.binder_chain,
                                    full_metrics=args.full_metrics, relax=args.relax)
    if args.output:
        from partiflow.postprocess.runner import _write_json
        _write_json(args.output, report)
    else:
        print(json.dumps(report, indent=2))
    return int(report['status'] != 'success')


if __name__ == '__main__':
    raise SystemExit(main())
