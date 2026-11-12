"""CPU scaffold selection, chemistry propagation, and quality preset checks."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
import tempfile
import unittest

import biotite.structure.io as strucio
import torch
import yaml

from partiflow.common.presets import apply_presets, evaluate, load_presets, validate_criteria
from partiflow.datasets.gen_dataset import InferenceDataset
from partiflow.design import tasks
from partiflow.openfold.np.residue_constants import restype_order
from partiflow.utils.conditioning_utils import chemical_bond_type_to_id
from partiflow.utils.dense_padding_data_loader import DensePaddingCollater


ROOT = Path(__file__).resolve().parents[1]


class ScaffoldFixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="partiflow-scaffold-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        tasks.BINDER_TOPOLOGY_LIB_CACHE.clear()
        self.addCleanup(tasks.BINDER_TOPOLOGY_LIB_CACHE.clear)
        self.addCleanup(random.setstate, random.getstate())

    def tensor(self, name, value):
        path = self.directory / name
        torch.save(torch.as_tensor(value), path)
        return str(path)

    def library(self, entries, wrapped=False):
        path = self.directory / ("wrapped.pt" if wrapped else "library.pt")
        torch.save({"entries": entries} if wrapped else entries, path)
        tasks.BINDER_TOPOLOGY_LIB_CACHE.clear()
        return str(path)

    def entry(self, name="hairpin", length=4, category="linear_peptide", **extra):
        return {
            "name": name, "length": length, "category": category,
            "ss_token": torch.arange(length) % 4,
            "block_adj": (torch.arange(length)[:, None] + torch.arange(length)[None, :]) % 3,
            **extra,
        }

    def resolve(self, config, length=4, index=0, **task_fields):
        return tasks.resolve_binder_topology(
            {"binder_topology": config, **task_fields},
            total_len=length, task_name="tiny", sample_index=index,
        )


class ScaffoldModeTests(ScaffoldFixtures):
    def test_coarse_and_inline_tokens_supply_geometry_without_identity(self):
        for name, token in (("alpha", 0), ("beta", 1), ("coil", 2)):
            with self.subTest(name=name):
                result = self.resolve(name)
                self.assertEqual(result["binder_ss_token"].tolist(), [token] * 4)
                self.assertTrue((result["binder_block_adj"] == 2).all())
                self.assertNotIn("_scaffold_condition", result)
        original = [0, 1, 2, 3]
        result = self.resolve({"mode": "ss_token", "ss_token": original})
        result["binder_ss_token"][0] = 3
        self.assertEqual(original, [0, 1, 2, 3])
        self.assertNotIn("_scaffold_condition", result)

    def test_scaffold_pt_and_file_tokens_preserve_both_tensors(self):
        ss, adj = [0, 1, 2, 3], self.entry()["block_adj"]
        paths = {"ss_path": self.tensor("one_ss.pt", ss), "adj_path": self.tensor("one_adj.pt", adj)}
        for mode in ("scaffold_pt", "ss_token"):
            with self.subTest(mode=mode):
                result = self.resolve({"mode": mode, **paths})
                self.assertEqual(result["binder_ss_token"].tolist(), ss)
                torch.testing.assert_close(result["binder_block_adj"], adj)
                self.assertNotIn("_scaffold_condition", result)
        for key in ("adj", "block_adj"):
            result = self.resolve({"mode": "ss_tokens", "ss_tokens": ss, key: adj.tolist()})
            torch.testing.assert_close(result["binder_block_adj"], adj)

    def test_bad_tensor_lengths_shapes_and_ranges_fail(self):
        for extra, message in (
            ({"ss_token": [0, 1, 2]}, "ss length 3"),
            ({"ss_token": [0, 1, 2, 4]}, "ss tokens outside"),
            ({"ss_token": [0, 1, 2, -1]}, "ss tokens outside"),
            ({"ss_token": [0] * 4, "adj": [[0] * 3] * 4}, "adj shape"),
            ({"ss_token": [0] * 4, "adj": [[3] * 4] * 4}, "adjacency tokens"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.resolve({"mode": "ss_token", **extra})

    def test_missing_paths_and_non_tensor_files_fail(self):
        with self.assertRaisesRegex(ValueError, "requires ss_path and adj_path"):
            self.resolve({"mode": "scaffold_pt"})
        with self.assertRaisesRegex(FileNotFoundError, "file not found"):
            self.resolve({"mode": "ss_token", "ss_path": str(self.directory / "absent.pt")})
        path = self.directory / "mapping.pt"
        torch.save({"ss_token": [0] * 4}, path)
        with self.assertRaisesRegex(ValueError, "must load to a torch.Tensor"):
            self.resolve({"mode": "ss_token", "ss_path": str(path)})

    def test_scaffold_dir_filters_lengths_and_cycles_sorted_candidates(self):
        for name, length in (("z", 4), ("a", 4), ("other", 6)):
            self.tensor(name + "_ss.pt", [0] * length)
            self.tensor(name + "_adj.pt", torch.zeros(length, length))
        config = {"mode": "scaffold_dir", "scaffold_dir": str(self.directory), "systematic": True}
        sources = [self.resolve(config, index=i)["binder_topology_source"] for i in range(3)]
        self.assertEqual([Path(source).name for source in sources], ["a", "z", "a"])
        with self.assertRaisesRegex(ValueError, "Observed scaffold lengths/counts"):
            self.resolve(config, length=5)

    def test_scaffold_list_preserves_order_and_requires_adjacency(self):
        for name in ("z", "a"):
            self.tensor(name + "_ss.pt", [0] * 4)
            self.tensor(name + "_adj.pt", torch.zeros(4, 4))
        names = self.directory / "names.txt"
        names.write_text("z_ss.pt\n\na\n")
        for requested in (["z_ss.pt", "a"], str(names)):
            config = {
                "mode": "scaffold_dir", "scaffold_dir": str(self.directory),
                "scaffold_list": requested, "systematic": True,
            }
            self.assertTrue(self.resolve(config)["binder_topology_source"].endswith("/z"))
            self.assertTrue(self.resolve(config, index=1)["binder_topology_source"].endswith("/a"))
        (self.directory / "z_adj.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "scaffold adj file"):
            self.resolve(config)


class ScaffoldLibraryTests(ScaffoldFixtures):
    def test_list_and_mapping_libraries_normalize_category_and_adj_alias(self):
        entry = self.entry(category="linear")
        entry["adj"] = entry.pop("block_adj").tolist()
        for wrapped in (False, True):
            result = self.resolve({"mode": "sample", "lib_path": self.library([entry], wrapped)})
            self.assertEqual(result["binder_topology_category"], "linear_peptide")
            torch.testing.assert_close(result["binder_block_adj"], torch.as_tensor(entry["adj"]))
            self.assertEqual(result["_scaffold_condition"]["topology_entries"], [])

    def test_category_and_all_named_preset_aliases_filter_before_sampling(self):
        path = self.library([
            self.entry("first", category="linear"),
            self.entry("Beta Hairpin", category="cycpep", id="cycle-id", preset="cycle"),
        ])
        for key, selector in (
            ("preset", "beta-hairpin"), ("presets", ["cycle"]),
            ("entry", "cycle-id"), ("entries", ["entry_1"]),
            ("name", "Beta Hairpin"), ("names", ["beta_hairpin"]),
            ("entry_name", "cycle"), ("entry_names", ["cycle-id"]),
        ):
            with self.subTest(key=key):
                result = self.resolve({
                    "mode": "sample", "lib_path": path, "category": "cyclic",
                    key: selector, "systematic": True,
                })
                self.assertEqual(result["binder_topology_entry_id"], "cycle-id")
                self.assertEqual(result["binder_topology_category"], "head_tail_amide_cyclic_peptide")

    def test_cyclic_auto_filter_is_only_applied_when_explicit(self):
        path = self.library([self.entry("linear"), self.entry("cycle", category="cycpep")])
        config = {"mode": "sample", "lib_path": path, "systematic": True}
        self.assertEqual(self.resolve(config, index=1)["binder_topology_entry_id"], "cycle")
        self.assertEqual(self.resolve({**config, "cyclic": "auto"}, index=1)["binder_topology_entry_id"], "linear")
        self.assertEqual(self.resolve({**config, "cyclic": "auto"}, sampler="head_to_tail")["binder_topology_entry_id"], "cycle")

    def test_length_score_filters_and_unmatched_selectors_fail_without_fallback(self):
        path = self.library([
            self.entry("short", length=3, score=1.0),
            self.entry("low", score=0.2), self.entry("high", score=0.9),
        ])
        result = self.resolve({"mode": "sample", "lib_path": path, "min_score": 0.8, "systematic": True})
        self.assertEqual(result["binder_topology_entry_id"], "high")
        for extra, message in (
            ({"preset": "absent"}, "preset/name mismatches"),
            ({"category": "cyclic"}, "category mismatches"),
            ({"cyclic": True}, "cyclic mismatches"),
            ({"min_score": 1.0}, "score-filtered: 2"),
        ):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, message):
                self.resolve({"mode": "sample", "lib_path": path, "fallback_to_mask": True, **extra})
        with self.assertRaisesRegex(ValueError, "Observed lengths/counts"):
            self.resolve({"mode": "sample", "lib_path": path}, length=5)

    def test_systematic_wrap_and_seeded_random_selection(self):
        path = self.library([self.entry("a"), self.entry("b")])
        config = {"mode": "sample", "lib_path": path, "systematic": True}
        self.assertEqual([self.resolve(config, index=i)["binder_topology_entry_id"] for i in range(5)], ["a", "b", "a", "b", "a"])
        config["systematic"] = False
        random.seed(17)
        first = [self.resolve(config)["binder_topology_entry_id"] for _ in range(8)]
        random.seed(17)
        self.assertEqual(first, [self.resolve(config)["binder_topology_entry_id"] for _ in range(8)])


    def test_amide_shift_rolls_both_axes_without_fixing_sequence(self):
        entry = self.entry(category="cycpep")
        path = self.library([entry])
        result = self.resolve({"mode": "sample", "lib_path": path, "systematic": True}, index=1)
        self.assertEqual(result["binder_topology_circular_shift"], 1)
        torch.testing.assert_close(result["binder_ss_token"], torch.roll(entry["ss_token"], 1))
        torch.testing.assert_close(result["binder_block_adj"], torch.roll(entry["block_adj"], shifts=(1, 1), dims=(0, 1)))
        condition = result["_scaffold_condition"]
        self.assertEqual(condition["residue_type_constraints"], {})
        self.assertEqual(condition["topology_entries"], [{"type": "head_to_tail", "anchors": [1, 4]}])
        self.assertEqual(condition["cyclization"]["positions"], [1, 4])
        torch.testing.assert_close(tasks._load_topology_library(path, task_name="tiny")[0]["ss_token"], entry["ss_token"])

    def test_terminal_disulfide_has_no_amide_bond_and_does_not_rotate(self):
        entry = self.entry(
            category="head_tail_disulfied_cyclic_peptide",
            cys_pair_constraints=[{"pos_i": 1, "pos_j": 4, "pair_bin_idx": 4, "ca_ca_distance_angstrom": 5.13}],
        )
        result = self.resolve({"mode": "sample", "lib_path": self.library([entry]), "systematic": True}, index=3)
        condition = result["_scaffold_condition"]
        self.assertEqual(result["binder_topology_circular_shift"], 0)
        self.assertIsNone(condition["cyclization"])
        self.assertEqual(condition["residue_type_constraints"], {1: restype_order["C"], 4: restype_order["C"]})
        self.assertEqual(condition["topology_entries"], [{"type": "disulfide", "anchors": [1, 4]}])
        self.assertEqual(condition["optional_ca_pair_constraints"][0]["bin_idx"], 4)
        self.assertEqual(condition["optional_ca_pair_constraints"][0]["target_distance_angstrom"], 5.13)

    def test_combined_amide_disulfide_shift_retains_both_bonds(self):
        entry = self.entry(
            category="cycpep_ss",
            cys_pair_constraints=[{"pos_i": 1, "pos_j": 3, "pair_bin_idx": 5}],
        )
        result = self.resolve({"mode": "sample", "lib_path": self.library([entry]), "systematic": True}, index=1)
        condition = result["_scaffold_condition"]
        self.assertEqual(condition["cys_positions"], [2, 4])
        self.assertEqual(condition["topology_entries"], [
            {"type": "disulfide", "anchors": [2, 4]},
            {"type": "head_to_tail", "anchors": [1, 4]},
        ])
        pair = condition["cys_pair_constraints"][0]
        self.assertEqual((pair["resid_i"], pair["resid_j"]), (2, 4))

    def test_bad_library_chemistry_lengths_anchors_and_bins_fail(self):
        for entries, length, message in (
            ([], 4, "contains no entries"),
            ([{}], 4, "missing ss_token"),
            ([self.entry(category="unknown")], 4, "does not declare classifiable"),
            ([self.entry(category="disulfide")], 4, "must provide cys_pair_constraints"),
            ([self.entry(cys_pair_constraints=[{"pos_i": 1}])], 4, "must include pos_i and pos_j"),
            ([self.entry(length=5, ss_token=torch.zeros(4))], 5, "ss length 4"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                self.resolve({"mode": "sample", "lib_path": self.library(entries)}, length=length)
        for pair, message in (
            ({"pos_i": 1, "pos_j": 1, "pair_bin_idx": 4}, "identical CYS anchors"),
            ({"pos_i": 1, "pos_j": 5, "pair_bin_idx": 4}, "outside"),
            ({"pos_i": 1, "pos_j": 4, "pair_bin_idx": 30}, "outside supported range"),
        ):
            with self.subTest(pair=pair), self.assertRaisesRegex(ValueError, message):
                path = self.library([self.entry(category="disulfide", cys_pair_constraints=[pair])])
                self.resolve({"mode": "sample", "lib_path": path})


class ScaffoldConditionBatchTests(ScaffoldFixtures):
    def setUp(self):
        super().setUp()
        self.receptor = self.directory / "receptor.pdb"
        strucio.save_structure(
            str(self.receptor),
            tasks.create_std_ala_residue("A", 1, ca_coord=[10.0, 0.0, 0.0]),
        )

    def task(self, config, length=4, **extra):
        return {
            "task_type": "denovo", "pdb_path": str(self.receptor), "is_complex": False,
            "pep_chain": "L", "hotspots": None, "lens_cfg": {"nres": [length]},
            "sample_num": 1, "binder_topology": config, **extra,
        }

    def graph(self, task):
        requests = tasks.build_atom_array(task, "tiny")
        self.assertEqual(len(requests), 1)
        dataset = InferenceDataset(requests, transform=tasks.INFERENCE_TRANSFORMS)
        return requests[0], dataset[0]

    def test_real_transform_and_dense_batch_preserve_identity_pairs_and_bonds(self):
        entry = self.entry(
            category="head_tail_disulfied_cyclic_peptide",
            cys_pair_constraints=[{"pos_i": 1, "pos_j": 4, "pair_bin_idx": 4, "ca_ca_distance_angstrom": 5.13}],
        )
        request, graph = self.graph(self.task({"mode": "sample", "lib_path": self.library([entry])}, omit_aa=["CYS"]))
        linear_request, linear = self.graph(self.task({"mode": "ss_token", "ss_token": [2] * 6}, length=6))
        self.assertEqual(request["residue_type_constraints"], {1: restype_order["C"], 4: restype_order["C"]})
        self.assertFalse(request["cyclic"])
        self.assertIsNone(request["cyclization"])
        self.assertEqual(graph.residue_type_condition_mask.tolist(), [False, True, False, False, True])
        self.assertEqual(graph.residue_type_condition[[1, 4]].tolist(), [restype_order["C"]] * 2)
        self.assertEqual(graph.omit_aa_indices.tolist(), [restype_order["C"]])
        self.assertEqual(graph.optional_ca_pair_dist_condition[1, 4, 4].item(), 1.0)
        self.assertTrue(graph.optional_ca_pair_dist_mask[1, 4])
        self.assertEqual(graph.chemical_bond_type[1, 4].item(), chemical_bond_type_to_id("disulfide"))
        self.assertEqual(json.loads(graph.topology_entries), request["topology_entries"])
        self.assertFalse(linear.residue_type_condition_mask.any())
        self.assertEqual(linear_request["residue_type_constraints"], {})
        graphs = [graph, linear]
        batch = DensePaddingCollater(dataset=graphs)(graphs)
        self.assertEqual(tuple(batch.coords.shape), (2, 7, 37, 3))
        self.assertTrue(torch.isfinite(batch.coords).all())
        self.assertEqual(tuple(batch.binder_block_adj.shape), (2, 7, 7))
        self.assertEqual(batch.binder_ss_token[0, :5].tolist(), [3, 0, 1, 2, 3])
        self.assertEqual(batch.optional_ca_pair_dist_condition[0, 1, 4, 4].item(), 1.0)
        self.assertEqual(batch.chemical_bond_type[0, 1, 4].item(), chemical_bond_type_to_id("disulfide"))
        self.assertFalse(batch.residue_type_condition_mask[1].any())
        self.assertFalse(batch.optional_ca_pair_dist_mask[0, 5:, :].any())

    def test_amide_geometry_does_not_fix_turn_or_other_identities(self):
        entry = self.entry(category="cycpep", ss_token=torch.tensor([2, 1, 1, 2]))
        request, graph = self.graph(self.task({"mode": "sample", "lib_path": self.library([entry])}))
        self.assertTrue(request["cyclic"])
        self.assertEqual(request["residue_type_constraints"], {})
        self.assertFalse(graph.residue_type_condition_mask.any())
        self.assertTrue(graph.has_cyclization_pair.item())
        self.assertEqual(graph.cyclization_pair_indices.tolist(), [[1, 4]])
        self.assertEqual(graph.chemical_bond_type[1, 4].item(), chemical_bond_type_to_id("head_to_tail"))
        self.assertTrue(graph.optional_ca_pair_dist_mask[1, 4])

    def test_nonstandard_identity_is_kept_and_cys_conflicts_fail(self):
        entry = self.entry(
            category="disulfide",
            cys_pair_constraints=[{"pos_i": 1, "pos_j": 4, "pair_bin_idx": 4}],
        )
        config = {"mode": "sample", "lib_path": self.library([entry])}
        request, graph = self.graph(self.task(config, constraint={"restype": [{"resid": 2, "aa": "ORN"}]}))
        templates = [entry for entry in request["topology_entries"] if entry["type"] == "residue_template"]
        self.assertEqual(len(templates), 1)
        self.assertEqual((templates[0]["resname"], templates[0]["resid"]), ("ORN", 2))
        self.assertEqual(request["postprocess_resolved_method"], "openmm")
        self.assertEqual(json.loads(graph.topology_entries), request["topology_entries"])
        with self.assertRaisesRegex(ValueError, "conflicting residue_type_constraints"):
            tasks.build_atom_array(self.task(config, constraint={"restype": [{"resid": 1, "aa": "A"}]}), "tiny")

    def test_preflight_checks_all_reachable_choices_without_random_selection(self):
        path = self.library([
            self.entry("linear"),
            self.entry("disulfide", category="disulfide", cys_pair_constraints=[{"pos_i": 1, "pos_j": 4, "pair_bin_idx": 4}]),
        ])
        state = random.getstate()
        with self.assertRaisesRegex(ValueError, "mandatory special chemistry"):
            tasks.validate_task_postprocess_config(self.task({"mode": "sample", "lib_path": path}, postprocess=False), "tiny")
        self.assertEqual(random.getstate(), state)

    def test_preflight_ignores_unreachable_denovo_library_lengths(self):
        path = self.library([
            self.entry("linear"),
            self.entry("disulfide", length=6, category="disulfide", cys_pair_constraints=[{"pos_i": 1, "pos_j": 6, "pair_bin_idx": 4}]),
        ])
        task = self.task({"mode": "sample", "lib_path": path}, postprocess=False)
        self.assertFalse(tasks.validate_task_postprocess_config(task, "tiny")["mandatory_chemistry"])
        task["lens_cfg"] = {"nres": None, "min_len": 4, "max_len": 4, "step_len": 1}
        self.assertFalse(tasks.validate_task_postprocess_config(task, "tiny")["mandatory_chemistry"])
        task["lens_cfg"]["max_len"] = 6
        with self.assertRaisesRegex(ValueError, "mandatory special chemistry"):
            tasks.validate_task_postprocess_config(task, "tiny")


    def test_real_condition_features_consume_tokens_pairs_and_identity(self):
        from partiflow.networks.feature import (
            BinderBlockAdjacencyPairFeat,
            BinderSecondaryStructureTokenSeqFeat,
            OptionalCaCoorsNanometersPairwiseDistancesPairFeat,
            OptionalResidueTypeSeqFeat,
        )

        entry = self.entry(
            category="disulfide",
            cys_pair_constraints=[{"pos_i": 1, "pos_j": 4, "pair_bin_idx": 4}],
        )
        _, graph = self.graph(self.task({"mode": "sample", "lib_path": self.library([entry])}))
        batch = DensePaddingCollater(dataset=[graph])([graph]).to_dict()
        batch["mask"] = batch["residue_type"] >= 0
        ss_feature = BinderSecondaryStructureTokenSeqFeat()(batch)
        adj_feature = BinderBlockAdjacencyPairFeat()(batch)
        self.assertEqual(ss_feature.argmax(-1).tolist(), [[3, 0, 1, 2, 3]])
        torch.testing.assert_close(adj_feature.argmax(-1)[0, 1:, 1:], entry["block_adj"])
        self.assertTrue((adj_feature.argmax(-1)[0, 0, :] == 2).all())
        changed = dict(batch)
        changed["binder_ss_token"] = batch["binder_ss_token"].clone()
        changed["binder_ss_token"][0, 1] = 1
        self.assertFalse(torch.equal(ss_feature, BinderSecondaryStructureTokenSeqFeat()(changed)))
        pair_feature = OptionalCaCoorsNanometersPairwiseDistancesPairFeat()(batch)
        self.assertEqual(pair_feature[0, 1, 4, 4].item(), 1.0)
        identity_feature = OptionalResidueTypeSeqFeat()(batch)
        self.assertGreater(identity_feature[0, 1, restype_order["C"]].item(), 0.0)
        self.assertEqual(identity_feature[0, 2, :].sum().item(), 0.0)


class QualityPresetTests(unittest.TestCase):
    def setUp(self):
        self.presets = load_presets(ROOT / "configs" / "preset.yaml")

    def record(self, count=2):
        record = {
            "method": "af2_ig", "status": "complete",
            "expected_predictions": [], "predictions": {}, "paired_metrics": {},
        }
        for number in range(1, count + 1):
            suffix = f"model{number}_sample1"
            for protocol in ("complex", "monomer"):
                key = f"{protocol}_{suffix}"
                record["expected_predictions"].append(key)
                record["predictions"][key] = {
                    "protocol": protocol, "status": "complete", "metrics_status": "complete",
                    "metrics": {"plddt": 0.9, "ptm": 0.7, "iptm": 0.6, "ipae_normalized": 0.2},
                }
            record["paired_metrics"][suffix] = {"bound_unbound_rmsd": 1.5}
        return record

    def decide(self, record, name="af2_ig_peptide"):
        return evaluate(record, name, self.presets[name])

    def test_packaged_default_thresholds_scopes_and_operators_are_exact(self):
        af2 = self.presets["af2_ig_peptide"]
        self.assertEqual(af2["reduction"], "all_and_mean")
        self.assertEqual(
            [(r["scope"], r["metric"], r["op"], r["value"]) for r in af2["rules"]],
            [
                ("complex", "plddt", ">=", 0.8), ("complex", "ptm", ">=", 0.55),
                ("complex", "iptm", ">=", 0.4), ("complex", "ipae_normalized", "<=", 0.3),
                ("monomer", "plddt", ">=", 0.8), ("paired", "bound_unbound_rmsd", "<=", 2.5),
            ],
        )
        rosetta = self.presets["rosetta_bindcraft_peptide"]
        self.assertEqual(rosetta["reduction"], "any_prediction")
        self.assertEqual(
            [(r["metric"], r["op"], r["value"]) for r in rosetta["rules"]],
            [
                ("shape_complementarity", ">", 0.55), ("interface_hbonds", ">", 1.0),
                ("interface_unsat_hbonds", "<", 3.0), ("surface_hydrophobicity", "<", 0.5),
            ],
        )
        validate_criteria(self.presets, ["af2_ig_peptide"], "af2_ig")
        validate_criteria(self.presets, ["rosetta_bindcraft_peptide"], "rosetta")

    def test_all_and_mean_fails_individual_prediction_even_if_mean_passes(self):
        record = self.record()
        self.assertEqual(self.decide(record)["status"], "passed")
        record["predictions"]["complex_model1_sample1"]["metrics"]["plddt"] = 0.75
        record["predictions"]["complex_model2_sample1"]["metrics"]["plddt"] = 0.95
        result = self.decide(record)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any("complex_model1_sample1" in reason for reason in result["reasons"]))

    def test_complete_pairs_and_finite_metrics_are_required(self):
        record = self.record()
        del record["paired_metrics"]["model2_sample1"]
        self.assertEqual(self.decide(record)["status"], "unavailable")
        for value in (None, float("nan"), float("inf"), True):
            with self.subTest(value=value):
                record = self.record()
                record["predictions"]["monomer_model1_sample1"]["metrics"]["plddt"] = value
                self.assertEqual(self.decide(record)["status"], "unavailable")
        record = self.record()
        record["predictions"]["complex_model1_sample1"]["metrics_status"] = "failed"
        self.assertEqual(self.decide(record)["status"], "unavailable")


    def test_exact_af2_boundary_values_pass(self):
        record = self.record(1)
        record["predictions"]["complex_model1_sample1"]["metrics"] = {
            "plddt": 0.8, "ptm": 0.55, "iptm": 0.4, "ipae_normalized": 0.3,
        }
        record["predictions"]["monomer_model1_sample1"]["metrics"]["plddt"] = 0.8
        record["paired_metrics"]["model1_sample1"]["bound_unbound_rmsd"] = 2.5
        self.assertEqual(self.decide(record)["status"], "passed")

    def test_any_prediction_cannot_mix_metrics_across_structures(self):
        record = {
            "method": "rosetta", "status": "complete",
            "expected_predictions": ["complex_1", "complex_2"],
            "predictions": {
                "complex_1": {
                    "protocol": "complex", "status": "complete", "metrics_status": "complete",
                    "metrics": {"shape_complementarity": 0.7, "interface_hbonds": 0.0, "interface_unsat_hbonds": 2.0, "surface_hydrophobicity": 0.4},
                },
                "complex_2": {
                    "protocol": "complex", "status": "complete", "metrics_status": "complete",
                    "metrics": {"shape_complementarity": 0.5, "interface_hbonds": 2.0, "interface_unsat_hbonds": 2.0, "surface_hydrophobicity": 0.4},
                },
            },
        }
        self.assertEqual(self.decide(record, "rosetta_bindcraft_peptide")["status"], "failed")
        record["predictions"]["complex_2"]["metrics"]["shape_complementarity"] = 0.6
        result = self.decide(record, "rosetta_bindcraft_peptide")
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["passing_prediction_keys"], ["complex_2"])
        del record["predictions"]["complex_2"]["metrics"]["surface_hydrophobicity"]
        self.assertEqual(self.decide(record, "rosetta_bindcraft_peptide")["status"], "unavailable")

    def test_method_mismatch_and_unsupported_chemistry_are_not_applicable(self):
        record = self.record()
        self.assertEqual(self.decide(record, "rosetta_bindcraft_peptide")["status"], "not_applicable")
        record["status"], record["error"] = "not_applicable", "unsupported_chemistry"
        result = self.decide(record)
        self.assertEqual(result["status"], "not_applicable")
        self.assertEqual(result["reasons"], ["unsupported_chemistry"])
        with self.assertRaisesRegex(ValueError, "does not apply"):
            validate_criteria(self.presets, ["af2_ig_peptide"], "rosetta")

    def test_invalid_named_presets_are_rejected_before_execution(self):
        valid = self.presets["af2_ig_peptide"]
        for change in (
            {"methods": ["unknown"]}, {"reduction": "unknown"}, {"rules": []},
            {"rules": [{**valid["rules"][0], "op": "!="}]},
            {"rules": [{**valid["rules"][0], "value": True}]},
            {"rules": [{**valid["rules"][0], "value": float("nan")}]},
            {"rules": [{**valid["rules"][0], "scope": "unknown"}]},
        ):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "presets.yaml"
                path.write_text(yaml.safe_dump({"presets": {"custom": {**valid, **change}}}))
                with self.assertRaises(ValueError):
                    load_presets(path)
        modified = copy.deepcopy(self.presets)
        modified["af2_ig_peptide"]["rules"][0]["metric"] = "invented"
        with self.assertRaisesRegex(ValueError, "Unknown af2_ig metric"):
            validate_criteria(modified, ["af2_ig_peptide"], "af2_ig")
        with self.assertRaisesRegex(ValueError, "Unknown criteria"):
            apply_presets(self.record(), ["absent"], self.presets)


class BundledScaffoldTests(ScaffoldFixtures):
    def test_all_original_library_entries_have_valid_geometry_and_chemistry(self):
        libraries = (
            ("linpep_archetype_scaffold_lib.pt", 49, 6, "linear_peptide"),
            ("cycpep_archetype_scaffold_lib.pt", 36, 6, "head_tail_amide_cyclic_peptide"),
            ("head_tail_disulfied_cyclic_archetype_scaffold_lib.pt", 31, 9, "head_tail_disulfied_cyclic_peptide"),
        )
        for filename, count, minimum, category in libraries:
            with self.subTest(filename=filename):
                entries = tasks._load_topology_library(ROOT / "configs/scaffolds" / filename, task_name="bundled")
                self.assertEqual(len(entries), count)
                self.assertEqual({entry["length"] for entry in entries}, set(range(minimum, 21)))
                self.assertEqual({entry["category"] for entry in entries}, {category})
                for entry in entries:
                    length = entry["length"]
                    tasks._normalize_topology_tokens(
                        entry["ss_token"], entry["block_adj"], total_len=length,
                        task_name="bundled", source=filename,
                    )
                    payload = tasks._scaffold_condition_payload(
                        entry, total_len=length, task_name="bundled", source=filename,
                    )
                    if category == "head_tail_disulfied_cyclic_peptide":
                        self.assertIsNone(payload["cyclization"])
                        self.assertEqual(payload["cys_positions"], [1, length])
                    elif category == "head_tail_amide_cyclic_peptide":
                        self.assertEqual(payload["residue_type_constraints"], {})
                        self.assertEqual(payload["cyclization"]["positions"], [1, length])

    def test_bundled_examples_compile_and_keep_distinct_topology(self):
        from partiflow.cli import parser
        from partiflow.design.config import load

        path = ROOT / "configs/tasks/scaffold.yaml"
        cfg = load(parser().parse_args([
            "design", str(path), "--ckpt", "./ckpts/partiflow-ss.ckpt",
        ]))
        self.assertTrue(str(cfg.checkpoint.model_ckpt).endswith("/ckpts/partiflow-ss.ckpt"))
        topology_types = {}
        for name, raw in cfg.tasks.items():
            from omegaconf import OmegaConf
            task = OmegaConf.to_container(raw, resolve=True)
            requests = tasks.build_atom_array(task, name)
            self.assertEqual(len(requests), 1)
            topology_types[name] = {entry["type"] for entry in requests[0]["topology_entries"]}
        self.assertEqual(topology_types, {
            "klk2_scaffold_linear": set(),
            "klk2_scaffold_amide_cycle": {"head_to_tail"},
            "klk2_scaffold_terminal_disulfide": {"disulfide"},
        })


    def test_available_ss_checkpoint_consumes_real_scaffold_features(self):
        from omegaconf import OmegaConf
        from partiflow.cli import parser
        from partiflow.design.config import load
        from partiflow.flow.local_latents_tf import ChemicalBondPairAdapter
        from partiflow.networks.feature import (
            BinderBlockAdjacencyPairFeat,
            BinderSecondaryStructureTokenSeqFeat,
        )
        from partiflow.paths import model_root

        checkpoint = model_root() / "ckpts/partiflow-ss.ckpt"
        if not checkpoint.is_file():
            self.skipTest("Select an installed model directory containing partiflow-ss.ckpt")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False, mmap=True)
        network = ck["hyper_parameters"]["cfg_exp"].network
        self.assertIn("binder_ss_token", network.feats_seq)
        self.assertIn("binder_block_adj", network.feats_pair_repr)
        self.assertTrue(network.chemical_bond_pair.enabled)
        cfg = load(parser().parse_args([
            "design", str(ROOT / "configs/tasks/scaffold.yaml"),
            "--task", "klk2_scaffold_terminal_disulfide",
            "--ckpt", str(checkpoint),
        ]))
        task = OmegaConf.to_container(cfg.tasks.klk2_scaffold_terminal_disulfide, resolve=True)
        request = tasks.build_atom_array(task, "klk2_scaffold_terminal_disulfide")[0]
        graph = InferenceDataset([request], transform=tasks.INFERENCE_TRANSFORMS)[0]
        batch = DensePaddingCollater(dataset=[graph])([graph]).to_dict()
        batch["mask"] = batch["residue_type"] >= 0
        ss = BinderSecondaryStructureTokenSeqFeat(
            binder_ss_token_num_classes=network.binder_ss_token_num_classes,
        )(batch)
        adjacency = BinderBlockAdjacencyPairFeat(
            binder_block_adj_num_classes=network.binder_block_adj_num_classes,
        )(batch)
        binder = torch.where(graph.is_peptide == 1)[0]
        torch.testing.assert_close(ss.argmax(-1)[0, binder], request["binder_ss_token"])
        torch.testing.assert_close(
            adjacency.argmax(-1)[0, binder[:, None], binder[None, :]],
            request["binder_block_adj"],
        )
        weights = ck["state_dict"]["nn.chemical_bond_pair_adapter.embedding.weight"]
        adapter = ChemicalBondPairAdapter(
            pair_repr_dim=network.pair_repr_dim,
            num_classes=network.chemical_bond_pair.num_classes,
        )
        adapter.load_state_dict({"embedding.weight": weights})
        pair = torch.zeros(1, len(graph.residue_type), len(graph.residue_type), network.pair_repr_dim)
        conditioned = adapter(batch, pair)
        first, last = int(binder[0]), int(binder[-1])
        bond = chemical_bond_type_to_id("disulfide")
        torch.testing.assert_close(conditioned[0, first, last], weights[bond])
        self.assertEqual(torch.count_nonzero(conditioned[0, 0, 0]).item(), 0)
        self.assertEqual(sorted(request["residue_type_constraints"]), [1, 12])
        self.assertIsNone(request["cyclization"])


if __name__ == "__main__":
    unittest.main()

