"""PDB reuse and native import contracts without SDK initialization."""

from __future__ import annotations

from contextlib import ExitStack
import importlib.util
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import partiflow.postprocess
from partiflow.postprocess.source import PDBSource


def atom(serial, name='CA', code='ALA', chain='L', number=1, x=1.25,
         *, altloc='', occupancy=1.0, element='C'):
    return (f'ATOM  {serial:5d} {name:^4s}{altloc or " "}{code:3s} {chain}{number:4d}    '
            f'{x:8.3f}{2.5:8.3f}{3.75:8.3f}{occupancy:6.2f}{0.0:6.2f}          {element:>2s}  \n')


class PDBSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(partiflow.postprocess.__file__).parent/'native.py'
        spec = importlib.util.spec_from_file_location('native_source_reuse_tests', path)
        cls.native = importlib.util.module_from_spec(spec)
        cls.pr = types.ModuleType('pyrosetta')
        cls.pr.get_init_options = Mock(return_value='')
        cls.pr.init = Mock()
        cls.pr.rosetta = types.SimpleNamespace(
            numeric=types.SimpleNamespace(xyzVector_double_t=lambda *xyz: tuple(xyz)),
            basic=types.SimpleNamespace(options=types.SimpleNamespace(get_boolean_option=Mock(return_value=False))),
        )
        cls.pr.io = types.SimpleNamespace(pose_from_pdbstring=Mock())
        with patch.dict(sys.modules, {'pyrosetta': cls.pr}):
            spec.loader.exec_module(cls.native)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root/'input.pdb'

    def tearDown(self):
        self.temp.cleanup()

    def source(self, text=None):
        return PDBSource.from_text(text or atom(1), self.path)

    def test_parsed_atom_schema_and_alias_classification(self):
        text = atom(1, code='HIE') + atom(2, name='H', code='HIE', element='H')
        text += atom(3, code='BAL', number=2) + 'CONECT    1    3\nEND\n'
        source = self.source(text)
        self.assertEqual(source.text, text)
        self.assertEqual(source.atoms[0], {
            'chain':'L', 'res_id':'1', 'ins_code':'', 'resname':'HIE', 'atom':'CA',
            'atom_id':1, 'xyz':(1.25,2.5,3.75), 'element':'C',
        })
        self.assertEqual(source.nonstandard_codes, ['BAL'])
        self.assertEqual(source.links, ['CONECT    1    3'])
        self.assertEqual(len(source.residue_occurrences), 2)

    def test_source_read_once_and_views_survive_source_removal(self):
        self.path.write_text(atom(1)+'END\n')
        original = Path.read_text
        reads = []
        def read(path, *args, **kwargs):
            reads.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            source = PDBSource.read(self.path)
        self.path.unlink()
        with patch.object(self.native, '_load_matrix', return_value={'residues': {}}), \
             patch.object(Path, 'read_text', side_effect=AssertionError('source reopened')), \
             patch.object(Path, 'open', side_effect=AssertionError('source reopened')):
            self.assertEqual(self.native._source_atom_names(self.path, source=source), {('L',1,' '): {'CA'}})
            self.assertEqual(self.native._source_residue_keys(self.path, source=source), {})
            self.assertEqual(self.native._source_heavy_atoms(self.path, binder_chain='L', source=source),
                             {('L',1,' ',0,'CA'): (1.25,2.5,3.75)})
            self.assertEqual(self.native._source_heavy_and_hydrogen_atoms(self.path, ('L',1,' '), source=source),
                             {'CA': (1.25,2.5,3.75)})
            self.assertEqual(self.native._normalized_pdb_text(self.path, binder_chain='L', source=source), source.text)
        self.assertEqual(reads, [self.path])

    def test_legacy_and_prepared_helpers_agree(self):
        self.path.write_text(atom(1, name='N', element='N') + atom(2, name='CA') + atom(3, name='H', element='H'))
        source = PDBSource.read(self.path)
        for function, kwargs in ((self.native._source_atom_names, {}),
                                 (self.native._source_residue_keys, {}),
                                 (self.native._source_heavy_atoms, {'binder_chain':'L'})):
            with self.subTest(function=function.__name__):
                self.assertEqual(function(self.path, **kwargs), function(self.path, source=source, **kwargs))

    def test_source_can_follow_atomic_publication_without_reloading(self):
        source = self.source()
        destination = self.root/'published.pdb'
        source.path = destination
        with patch.object(Path, 'read_text', side_effect=AssertionError('source reopened')):
            self.assertEqual(self.native._source_atom_names(destination, source=source),
                             {('L',1,' '): {'CA'}})
        self.assertEqual(source.path, destination)
        self.assertFalse(destination.exists())

    def test_legacy_normalized_filename_adapter_matches_memory_text(self):
        source = self.source(atom(1)+'END\n')
        with patch.object(self.native, '_load_matrix', return_value={'residues': {}}):
            temporary = Path(self.native._normalized_pdb(self.path, binder_chain='L', source=source))
            try:
                self.assertEqual(temporary.read_text(), source.text)
            finally:
                temporary.unlink()

    def test_ter_keeps_duplicate_target_occurrences(self):
        text = atom(1, name='N', code='GLY', chain='A', element='N')
        text += atom(2, code='GLY', chain='A')+'TER\n'
        text += atom(3, name='N', code='GLY', chain='A', x=7.0, element='N')
        text += atom(4, code='GLY', chain='A', x=8.0)+'TER\n'+atom(5, code='VAL')
        source = self.source(text)
        atoms, duplicates = self.native._source_heavy_atom_identity(self.path, binder_chain='L', source=source)
        self.assertEqual(atoms[('A',1,' ',0,'N')], (1.25,2.5,3.75))
        self.assertEqual(atoms[('A',1,' ',1,'N')], (7.0,2.5,3.75))
        self.assertEqual(duplicates, [{'chain':'A','residue_number':1,'insertion_code':'',
            'occurrences':[{'occurrence':0,'residue_name':'GLY'}, {'occurrence':1,'residue_name':'GLY'}]}])
        self.assertEqual([r['occurrence'] for r in source.residue_occurrences], [0,1,0])
        self.assertEqual([r['segment'] for r in source.residue_occurrences], [0,1,2])
        with self.assertRaisesRegex(ValueError, 'duplicate binder'):
            self.native._source_heavy_atom_identity(self.path, binder_chain='A', source=source)

    def test_duplicate_ncaa_target_and_duplicate_atom_are_rejected(self):
        source = self.source(atom(1, code='BAL', chain='A')+'TER\n'+atom(2, code='BAL', chain='A'))
        with self.assertRaisesRegex(ValueError, 'duplicate NCAA/noncanonical'):
            self.native._source_heavy_atom_identity(self.path, binder_chain='L', source=source)
        source = self.source(atom(1)+atom(2))
        with self.assertRaisesRegex(ValueError, 'duplicate source heavy atom'):
            self.native._source_heavy_atom_identity(self.path, binder_chain='L', source=source)

    def test_envelope_checks_use_the_retained_records(self):
        cases = [
            ('MODEL        1\n'+atom(1)+'ENDMDL\nMODEL        2\n'+atom(2)+'ENDMDL\n', 'multi-MODEL'),
            (atom(1, altloc='A'), 'alternate-location'),
            (atom(1, occupancy=0), 'occupancy'),
            ('MODEL        1\n'+atom(1), 'malformed MODEL'),
        ]
        for text, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.native._validate_coordinate_record_envelope(self.path, source=self.source(text))
        self.native._validate_coordinate_record_envelope(self.path, source=self.source('MODEL        1\n'+atom(1)+'ENDMDL\n'))

    def test_normalization_retains_disulfide_records_and_standard_alias(self):
        text = 'SSBOND   retained\n'+atom(1, code='HIE')+'LINK      removed\nCONECT    1    2\nEND\n'
        with patch.object(self.native, '_load_matrix', return_value={'residues': {}}):
            normalized = self.native._normalized_pdb_text(self.path, binder_chain='L', source=self.source(text))
        self.assertIn('SSBOND   retained', normalized)
        self.assertIn('HIS', normalized)
        self.assertNotIn('HIE', normalized)
        self.assertNotIn('LINK', normalized)
        self.assertNotIn('CONECT', normalized)

    def test_runtime_params_and_alias_maps_are_read_only_cached(self):
        params = self.root/'type.params'
        params.write_text('ATOM N Nbb NH1 -0.5\nATOM CA CAbb CT1 0.1\n')
        entry = {'params':params,'runtime_name':'TEST','handling':'custom_alias','residue_code':'TEST'}
        original = Path.read_text
        reads=[]
        def read(path, *args, **kwargs):
            reads.append(path)
            return original(path, *args, **kwargs)
        with patch.object(Path, 'read_text', read):
            for _ in range(5):
                self.assertEqual(self.native._runtime_atom_names(entry), {'N','CA'})
        self.assertEqual(reads, [params])
        names = self.native._runtime_atom_names(entry)
        with self.assertRaises(AttributeError):
            names.add('O')
        entry.update(handling='native_patch_alias', mapping={'atom_rename':{'HN1':'1HN','CA':'CA'}})
        forward, inverse = self.native._atom_name_maps(entry)
        self.assertEqual(dict(forward), {'HN1':'HN1','CA':'CA'})
        self.assertEqual(dict(inverse), {'HN1':'HN1','CA':'CA'})
        with self.assertRaises(TypeError):
            forward['CA'] = 'changed'

    def test_cached_native_init_refuses_beta_before_early_return(self):
        for options in ('-corrections::beta_nov16 true', '-beta_nov16', '-corrections:beta_nov16=1'):
            with self.subTest(options=options), patch.object(self.native, '_INITIALIZED', True), \
                 patch.object(self.native, '_runtime_option_text', return_value=options), \
                 patch.object(self.native, 'validate_generated_params_sources') as validate:
                with self.assertRaisesRegex(RuntimeError, 'own process'):
                    self.native.initialize_pyrosetta_score_only()
                validate.assert_not_called()
        with patch.object(self.native, '_INITIALIZED', True), \
             patch.object(self.native, '_runtime_option_text', return_value=''), \
             patch.object(self.pr.rosetta.basic.options, 'get_boolean_option', return_value=True):
            with self.assertRaisesRegex(RuntimeError, 'own process'):
                self.native.initialize_pyrosetta_score_only()

    def test_native_init_options_accept_actor_seed_and_threads(self):
        options=['']
        initialize=Mock(side_effect=lambda value, **kwargs: options.__setitem__(0,value))
        with patch.object(self.native, '_INITIALIZED', False), \
             patch.object(self.native, '_runtime_option_text', side_effect=lambda: options[0]), \
             patch.object(self.native, 'validate_generated_params_sources'), \
             patch.object(self.native, '_source_paths', return_value=([self.root/'type.params'], [])), \
             patch.object(self.pr, 'init', initialize):
            self.native.initialize_pyrosetta_score_only(seed=17, threads=2, silent=False)
            self.native.initialize_pyrosetta_score_only()
        initialize.assert_called_once()
        tokens=shlex.split(options[0])
        self.assertIn('-constant_seed',tokens)
        self.assertEqual(tokens[tokens.index('-jran')+1], '17')
        self.assertEqual(tokens[tokens.index('-multithreading:total_threads')+1], '2')
        self.assertNotIn('-corrections::beta_nov16',tokens)

    def test_invalid_actor_seed_or_threads_does_not_initialize(self):
        for options in ({'seed': True}, {'seed': -1}, {'threads':0}, {'threads':False}):
            with self.subTest(options=options), patch.object(self.pr,'init') as initialize:
                with self.assertRaises(ValueError):
                    self.native.initialize_pyrosetta_score_only(**options)
                initialize.assert_not_called()

    def test_loader_uses_one_source_and_in_memory_pose_import(self):
        self.path.write_text(atom(1)+'END\n')
        pose=object()
        imported=Mock(return_value=pose)
        audit={'source_count':1,'loaded_count':1,'max_delta_angstrom':0.0}
        with ExitStack() as stack:
            stack.enter_context(patch.object(self.native,'initialize_pyrosetta_score_only'))
            stack.enter_context(patch.object(self.native,'_load_matrix',return_value={'residues':{}}))
            stack.enter_context(patch.object(self.pr.io,'pose_from_pdbstring',imported))
            stack.enter_context(patch.object(self.native,'_strip_generated_terminal_variants'))
            runtime=stack.enter_context(patch.object(self.native,'_apply_native_runtime_types',return_value={}))
            stack.enter_context(patch.object(self.native,'_runtime_residue_audit',return_value=[]))
            stack.enter_context(patch.object(self.native,'_binder_indices',return_value=[1]))
            stack.enter_context(patch.object(self.native,'_prepare_head_tail_terminal_variants',return_value=[]))
            stack.enter_context(patch.object(self.native,'_reject_pdb_residue_types'))
            audits=stack.enter_context(patch.object(self.native,'audit_input_heavy_atoms',return_value=audit))
            stack.enter_context(patch.object(self.native,'restore_polymer_and_staple_bonds',return_value=([],[],[])))
            stack.enter_context(patch.object(self.native,'audit_residue_connection_reciprocity',return_value=[]))
            stack.enter_context(patch.object(self.native,'_topology_pose_identity',return_value=[]))
            stack.enter_context(patch.object(self.native.tempfile,'mkstemp',side_effect=AssertionError('temporary PDB created')))
            original=PDBSource.read
            reads=[]
            def read(path):
                reads.append(path)
                return original(path)
            stack.enter_context(patch.object(PDBSource,'read',side_effect=read))
            loaded, info = self.native.load_score_only_pose(self.path)
        self.assertIs(loaded,pose)
        self.assertEqual(reads,[self.path])
        imported.assert_called_once_with(self.path.read_text())
        prepared=runtime.call_args.kwargs['source']
        self.assertIsInstance(prepared,PDBSource)
        self.assertTrue(all(call.kwargs['source'] is prepared for call in audits.call_args_list))
        self.assertEqual(info['input_heavy_atom_count'],1)


if __name__ == '__main__':
    unittest.main()
