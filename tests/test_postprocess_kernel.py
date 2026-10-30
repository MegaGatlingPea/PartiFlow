"""CPU contracts for fixed SDK profiles, protocol setup and requested metrics."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import shlex
import sys
import threading
from types import SimpleNamespace as NS
from typing import Any, Iterable, Literal
import unittest
from unittest.mock import Mock, patch

import partiflow.postprocess as package
from partiflow.postprocess import metrics


ROOT = Path(__file__).resolve().parents[1]


def functions(*names, **values):
    """Compile the selected pure setup functions without importing the native SDK."""
    source = ROOT / "partiflow/postprocess/rosetta.py"
    parsed = ast.parse(source.read_text())
    selected = [node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(selected) == len(names)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias("annotations")], level=0), *selected], type_ignores=[])
    scope = dict(Any=Any, Iterable=Iterable, Literal=Literal, Path=Path, shlex=shlex,
                 threading=threading, __file__=str(source), __name__="partiflow.postprocess.rosetta",
                 __package__="partiflow.postprocess", _STANDARD_BETA_INITIALIZED=False,
                 _STANDARD_BETA_LOCK=threading.Lock(), **values)
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
    return scope


class FixedSdkTests(unittest.TestCase):
    def sdk(self, initialized=False, beta=False):
        state = dict(initialized=initialized, beta=beta, options=[])
        def initialize(options, silent):
            state["options"].append(options)
            state.update(initialized=True,beta=True)
        factory = Mock(return_value="weights clone")
        pr = NS(init=Mock(side_effect=initialize), create_score_function=factory,
                rosetta=NS(basic=NS(was_init_called=lambda:state["initialized"],
                    options=NS(get_boolean_option=lambda name:state["beta"],
                               set_boolean_option=Mock(side_effect=AssertionError("global profile switch"))))))
        return pr,state

    def test_beta_initializes_once_with_seed_and_threads(self):
        pr,state = self.sdk()
        scope = functions("_beta_profile_enabled","initialize_standard_beta",pr=pr)
        scope["initialize_standard_beta"](seed=1227,threads=2)
        scope["initialize_standard_beta"](seed=1227,threads=2)
        self.assertEqual(pr.init.call_count,1)
        options = state["options"][0]
        self.assertIn("-corrections::beta_nov16 true",options)
        self.assertIn("-jran 1227",options)
        self.assertIn("-multithreading:total_threads 2",options)
        self.assertIn("-relax:default_repeats 1",options)
        self.assertNotIn("detect_disulf false",options)

    def test_ref_process_cannot_be_reinitialized_as_beta(self):
        pr,_ = self.sdk(initialized=True,beta=False)
        scope = functions("_beta_profile_enabled","initialize_standard_beta",pr=pr)
        with self.assertRaisesRegex(RuntimeError,"separate Beta"):
            scope["initialize_standard_beta"]()
        pr.init.assert_not_called()

    def test_score_weights_never_switch_chemical_profile(self):
        for active,name in ((False,"beta_nov16"),(True,"ref2015")):
            pr,_ = self.sdk(initialized=True,beta=active)
            scope = functions("_beta_profile_enabled","get_scorefxn",pr=pr)
            with self.assertRaisesRegex(RuntimeError,"immutable SDK"):
                scope["get_scorefxn"](name)
            pr.create_score_function.assert_not_called()
            pr.rosetta.basic.options.set_boolean_option.assert_not_called()

    def test_standard_loader_keeps_disulfide_and_cycle_builders(self):
        pose = object()
        entries = [{"type":"head_to_tail"},{"type":"disulfide","anchors":[1,2]}]
        cycle,disulfides = {"first_pose_idx":1,"last_pose_idx":2},[{"anchors":"1-2"}]
        payload = Mock(return_value=pose)
        build_cycle,build_ss = Mock(return_value=cycle),Mock(return_value=disulfides)
        source = NS(atoms=[{"resname":"HIE"},{"resname":"CYX"}],lines=[
            "SSBOND   1 CYS L    1    CYS L    2",
            "ATOM      1  CA  HIE L   1       0.000   0.000   0.000",
            "TER", "ATOM      2  CA  CYX L   2       4.000   0.000   0.000",
            "CONECT    1    2"])
        scope = functions("load_standard_pose",pr=NS(io=NS(pose_from_pdbstring=payload)),
            _beta_profile_enabled=lambda:True,should_cyclize_binder=Mock(return_value=True),
            cyclize_head_tail_binder_pose=build_cycle,apply_disulfide_topology_to_pose=build_ss)
        result,info = scope["load_standard_pose"](source,binder_chain="L",topology_entries=entries)
        self.assertIs(result,pose)
        self.assertIs(info["cyclization"],cycle)
        self.assertIs(info["disulfides"],disulfides)
        build_ss.assert_called_once_with(pose,binder_chain="L",topology_entries=entries)
        text = payload.call_args.args[0]
        self.assertIn("SSBOND",text)
        self.assertIn("TER",text)
        self.assertIn("HIS L",text)
        self.assertIn("CYS L",text)
        self.assertNotIn("CONECT",text)
        self.assertIn("HIE",source.lines[1])


class ProtocolTests(unittest.TestCase):
    def setup(self, *, fail=False):
        class Constraints:
            def __init__(self,items=()):self.items=list(items)
            def clone(self):return Constraints(self.items)
            def get_all_constraints(self):return self.items
        class Pose:
            def __init__(self):self.constraints=Constraints(["disulfide"])
            def clone(self):return Pose()
            def constraint_set(self,value=None):
                if value is not None:self.constraints=value
                return self.constraints
        class Score:
            def __init__(self):self.weights={}
            def clone(self):return Score()
            def set_weight(self,name,value):self.weights[name]=value
        class Map:
            def __init__(self):self.values={}
            def set_bb(self,value):self.values["bb"]=value
            def set_chi(self,value):self.values["chi"]=value
            def set_jump(self,value):self.values["jump"]=value
        class Factory:
            def __init__(self):self.operations=[]
            def push_back(self,op):self.operations.append(op)
        relaxes=[]
        class Relax:
            def __init__(self,score,repeats):
                self.score,self.repeats,self.values=score,repeats,{}
                relaxes.append(self)
            def max_iter(self,value):self.values["max_iter"]=value
            def min_type(self,value):self.values["min_type"]=value
            def constrain_relax_to_start_coords(self,value):self.values["coordinate_restraints"]=value
            def set_movemap(self,value):self.mm=value
            def set_task_factory(self,value):self.tf=value
            def apply(self,pose):
                self.constraints_during=list(pose.constraints.items)
                if fail:raise ValueError("native relax failed")
        align=Mock()
        base=Score()
        scope=functions("fast_relax_pose2",FastRelax=Relax,MoveMap=Map,TaskFactory=Factory,
            operation=NS(RestrictToRepacking=lambda:"repack"),
            get_scorefxn=Mock(return_value=base),
            add_cyclic_peptide_bond_constraints=lambda pose,**kw:pose.constraints.items.append("head-tail"),
            pr=NS(rosetta=NS(core=NS(scoring=NS(atom_pair_constraint="pair")),
                            protocols=NS(simple_moves=NS(AlignChainMover=lambda:align)))))
        return scope,Pose(),base,relaxes,align

    def test_global_protocol_and_relax_only_closure_weights(self):
        scope,pose,base,relaxes,align=self.setup()
        scope["fast_relax_pose2"](pose,cyclic_first_idx=1,cyclic_last_idx=2)
        mover=relaxes[0]
        scope["get_scorefxn"].assert_called_once_with("beta_nov16")
        self.assertEqual(mover.repeats,1)
        self.assertEqual(mover.values["max_iter"],200)
        self.assertFalse(mover.values["coordinate_restraints"])
        self.assertEqual(mover.mm.values,{"bb":True,"chi":True,"jump":False})
        self.assertEqual(mover.tf.operations,["repack"])
        self.assertEqual(mover.score.weights,{"pair":1.0})
        self.assertEqual(base.weights,{})
        self.assertEqual(mover.constraints_during,["disulfide","head-tail"])
        self.assertEqual(pose.constraints.items,["disulfide"])
        align.apply.assert_called_once_with(pose)

    def test_failed_relax_restores_original_constraint_set(self):
        scope,pose,_,_,align=self.setup(fail=True)
        with self.assertRaisesRegex(ValueError,"native relax"):
            scope["fast_relax_pose2"](pose,cyclic_first_idx=1,cyclic_last_idx=2)
        self.assertEqual(pose.constraints.items,["disulfide"])
        align.apply.assert_not_called()

    def test_repeat_override_is_configured_and_invalid_repeat_rejected(self):
        scope,pose,_,relaxes,_=self.setup()
        scope["fast_relax_pose2"](pose,repeats=2)
        self.assertEqual(relaxes[0].repeats,2)
        for value in (0,True,-1):
            with self.assertRaisesRegex(ValueError,"repeats"):
                scope["fast_relax_pose2"](pose,repeats=value)


class RequestedMetricsTests(unittest.TestCase):
    def test_iam_only_requests_dependencies_and_disables_packstat(self):
        flags={}
        class Mover:
            def __getattr__(self,name):
                if name.startswith("set_"):return lambda value:flags.__setitem__(name,value)
                raise AttributeError(name)
            def apply(self,pose):pass
            def get_all_data(self):return NS(sc_value=.677,interface_hbonds=3)
        scope=functions("_required_iam_metrics",InterfaceAnalyzerMover=Mover,
                        _set_interface_analyzer_interface=lambda m,p:None)
        result=scope["_required_iam_metrics"](NS(clone=lambda:object()),"A_L",object(),
            {"ShapeComplementarity"},native_no_pack=False)
        self.assertEqual(result,{"ShapeComplementarity":.677})
        self.assertFalse(flags["set_compute_packstat"])
        self.assertFalse(flags["set_calc_dSASA"])
        self.assertFalse(flags["set_compute_interface_energy"])
        self.assertFalse(flags["set_calc_hbond_sasaE"])
        self.assertTrue(flags["set_compute_interface_sc"])

    def test_custom_aliases_and_derived_rounding_are_retained(self):
        pose=NS(kind="full",clone=lambda:pose)
        binder=NS(kind="binder")
        score=lambda value:12.12345 if value.kind=="full" else 1.55555
        scope=functions("score_required_interface",np=importlib.import_module("numpy"),
            _required_iam_metrics=Mock(return_value={"dG":-2.12345}),
            _metric_binder_pose=Mock(return_value=binder))
        result=scope["score_required_interface"](pose,binder_chain="L",target_chains="AB",
            scorefxn=score,scorefxn_name="beta_nov16",required_metrics={"I_sc","rank_score"},
            native_no_pack=False,compute_buns=False)
        self.assertEqual(result,{"I_sc":-2.123,"rank_score":9.432})

    def test_bal_surface_stops_before_unsafe_selector(self):
        selector=Mock(side_effect=AssertionError("unsafe LayerSelector"))
        pose=NS(total_residue=lambda:1,residue_type=lambda i:NS(is_alpha_aa=lambda:False,name=lambda:"B3G"))
        scope=functions("_metric_surface_hydrophobicity",
            pr=NS(rosetta=NS(core=NS(select=NS(residue_selector=NS(LayerSelector=selector))))))
        with self.assertRaisesRegex(ValueError,"alpha-amino-acid"):
            scope["_metric_surface_hydrophobicity"](pose)
        selector.assert_not_called()

    def test_surface_preparation_error_keeps_other_required_metrics(self):
        pose=NS(clone=lambda:pose)
        scope=functions("score_required_interface",np=importlib.import_module("numpy"),
            _required_iam_metrics=Mock(return_value={"ShapeComplementarity":.7,"n_InterfaceHbonds":3}),
            _metric_binder_pose=Mock(side_effect=ValueError("binder view unavailable")),
            _required_buns=Mock(return_value=1))
        result=scope["score_required_interface"](pose,binder_chain="L",target_chains="A",
            scorefxn=Mock(),scorefxn_name="ref2015",required_metrics=set(metrics.CANONICAL_METRICS.values()),
            native_no_pack=True,compute_buns=True)
        self.assertEqual(result["ShapeComplementarity"],.7)
        self.assertEqual(result["n_InterfaceHbonds"],3)
        self.assertEqual(result["n_InterfaceUnsatHbonds"],1)
        self.assertIn("Surface_Hydrophobicity",result["unavailable_metric_errors"])

    def context_case(self, *, profile="standard_beta", codes=(), pose_present=True, raw=None):
        source=NS(nonstandard_codes=list(codes),atoms=[])
        pose=NS(total_residue=lambda:1,pdb_info=lambda:NS(chain=lambda i:"L"),
                residue=lambda i:NS(name=lambda:"B3G" if codes else "ALA"))
        context=NS(input_source=source,output_source=source,pose=pose if pose_present else None,
                   native_load={"count":1,"large":[{"atoms":[1,2,3]}]},sdk_profile=profile,seed=1227)
        native=NS(load_score_only_pose=Mock(return_value=(pose,{"loaded_heavy_atom_count":5})),
                  snapshot_coordinates=Mock(return_value=(1,2,3)),
                  _score_function_for_pose=Mock(return_value=(Mock(),[])),
                  _score_components=Mock(side_effect=AssertionError("unrequested energy components")))
        rosetta=NS(seed_metrics=Mock(),get_scorefxn=Mock(return_value=Mock()),
                   _chain_ids_in_pose=lambda pose:["A","L"],
                   score_interface=Mock(return_value=raw or {"ShapeComplementarity":.7}))
        return context,native,rosetta,pose

    def test_standard_relaxed_pose_reused_without_reload_or_energy(self):
        context,native,rosetta,pose=self.context_case()
        with patch.object(package,"native",native,create=True),patch.object(package,"rosetta",rosetta,create=True):
            report=metrics.compute_metrics("unused.pdb",context=context,required_metrics={"shape_complementarity"})
        native.load_score_only_pose.assert_not_called()
        native._score_components.assert_not_called()
        native.snapshot_coordinates.assert_not_called()
        self.assertIs(rosetta.score_interface.call_args.kwargs["native_pose"],pose)
        self.assertEqual(rosetta.score_interface.call_args.kwargs["required_metrics"],{"ShapeComplementarity"})
        rosetta.seed_metrics.assert_called_once_with(1227)
        self.assertEqual(report["metrics"]["shape_complementarity"],.7)
        self.assertNotIn("large",report["native_load"])

    def test_native_pose_loaded_once_and_kept_fixed_with_missing_surface(self):
        context,native,rosetta,pose=self.context_case(profile="native_ref2015",codes=["BAL"],pose_present=False,
            raw={"ShapeComplementarity":.7,"n_InterfaceHbonds":3,"n_InterfaceUnsatHbonds":1,
                 "unavailable_metric_errors":{"Surface_Hydrophobicity":"beta backbone"}})
        required=set(metrics.CANONICAL_METRICS.values())
        with patch.object(package,"native",native,create=True),patch.object(package,"rosetta",rosetta,create=True):
            report=metrics.compute_metrics("unused.pdb",context=context,required_metrics=required)
        native.load_score_only_pose.assert_called_once_with("unused.pdb",binder_chain="L",topology_entries=None,source=context.output_source)
        self.assertIs(context.pose,pose)
        self.assertEqual(native.snapshot_coordinates.call_count,2)
        call=rosetta.score_interface.call_args.kwargs
        self.assertTrue(call["native_no_pack"])
        self.assertFalse(call["relax_before_scoring"])
        self.assertEqual(report["unavailable_metrics"],["surface_hydrophobicity"])
        self.assertEqual(report["native_load"]["binder_residue_types"],["B3G"])


if __name__ == "__main__":
    unittest.main()
