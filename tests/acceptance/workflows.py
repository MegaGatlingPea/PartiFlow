"""Inspect explicitly selected fresh CLI outputs without loading a model."""

import argparse
import csv
import json
import math
from pathlib import Path

FIELDS = ("filename", "design_chains", "base_sample_id", "design_modality",
          "design_chain_id", "token_length", "reward_method", "global_rank",
          "design_sequence", "contig_string", "tip_atoms", "fixed_restypes", "bond_constraints")
TAIL = ("postprocess_status", "error")
AA = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
              "ARNDCQEGHILKMFPSTWYV"))


def atoms(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        xyz = tuple(float(line[start:start+8]) for start in (30, 38, 46))
        assert all(math.isfinite(value) for value in xyz), path
        result.append(dict(serial=int(line[6:11]), atom=line[12:16].strip(),
            restype=line[17:20].strip(), chain=line[21:22], res_id=int(line[22:26]),
            ins_code=line[26:27].strip(), xyz=xyz))
    assert result, path
    return result


def table(path):
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, rows = tuple(reader.fieldnames or ()), list(reader)
    assert fields[:len(FIELDS)] == FIELDS and fields[-2:] == TAIL, (path, fields)
    assert fields[len(FIELDS):-2] == tuple(sorted(fields[len(FIELDS):-2])), path
    return fields, rows


def inspect_design(root, replicas, *, expected_tasks=None):
    root = Path(root).resolve()
    paths = sorted(root.glob("*/result.csv"))
    assert paths, f"No new design outputs in {root}"
    if expected_tasks is not None:
        assert {p.parent.name for p in paths} == set(expected_tasks), paths
    summary = []
    for path in paths:
        task = path.parent
        fields, rows = table(path)
        state = json.loads((task / ".state/design.json").read_text())
        budget = state["generation"]
        assert budget["replicas_per_base"] == replicas, (path, budget)
        assert len(rows) == budget["planned_candidates"], (path, len(rows), budget)
        assert not state["failures"], (path, state["failures"])
        counts = {}
        by_length = {}
        for row in rows:
            assert row["error"] == "" and row["filename"], row
            pdb = (task / row["filename"]).resolve()
            assert pdb.is_relative_to(task.resolve()) and pdb.is_file(), pdb
            actual = atoms(pdb)
            chain = row["design_chain_id"]
            residues = {atom["res_id"]: atom for atom in actual
                        if atom["chain"] == chain and atom["atom"] == "CA"}
            sequence = "".join(AA.get(atom["restype"], "["+atom["restype"]+"]") for atom in residues.values())
            assert sequence == row["design_sequence"], (path, sequence, row)
            assert len(residues) == int(row["token_length"]), row
            design_chains = json.loads(row["design_chains"])
            assert len(design_chains) == 1 and design_chains[0]["chain"] == chain, row
            assert design_chains[0]["sequence"] == sequence, row
            assert design_chains[0]["residue_count"] == len(residues), row
            for fixed in json.loads(row["fixed_restypes"]):
                assert fixed["chain"] == chain, fixed
                assert residues[int(fixed["res_id"])]["restype"] == fixed["restype"], (path, fixed)
            serials = {atom["serial"]: atom for atom in actual}
            def locator(value):
                if isinstance(value, list):
                    for child in value: locator(child)
                elif isinstance(value, dict):
                    if "atom_id" in value:
                        atom = serials[int(value["atom_id"])]
                        for key in ("chain", "res_id", "ins_code"):
                            if key in value: assert str(atom[key]) == str(value[key]), (path, value)
                    for child in value.values(): locator(child)
            for field in ("fixed_restypes", "bond_constraints"):
                locator(json.loads(row[field]))
            counts[row["base_sample_id"]] = counts.get(row["base_sample_id"], 0) + 1
            by_length.setdefault(len(residues), []).append(row)
            for metric in fields[len(FIELDS):-2]:
                assert math.isfinite(float(row[metric])), (path, metric)
        assert len(counts) == budget["planned_base_samples"], (path, counts, budget)
        assert set(counts.values()) == {replicas}, (path, counts)
        if any(row["reward_method"] for row in rows):
            import numpy as np
            for length_rows in by_length.values():
                ordered = sorted(length_rows, key=lambda row: int(row["global_rank"]))
                assert [int(row["global_rank"]) for row in ordered] == list(range(1,len(ordered)+1))
                scores = [float(row["af2multimer_score"]) for row in ordered]
                assert scores == sorted(scores, reverse=True), scores
                for row in ordered:
                    expected = float(np.float32(0.25*float(row["af2multimer_plddt"])/100 + 0.75*float(row["af2multimer_iptm"])))
                    assert float(row["af2multimer_score"]) == expected, (path, row["af2multimer_score"], expected)
        else:
            assert fields == FIELDS + TAIL, (path, fields)
        summary.append(dict(task=task.name, rows=len(rows), bases=len(counts), replicas=replicas))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--replicas", required=True, type=int)
    parser.add_argument("--tasks", nargs="+")
    args = parser.parse_args()
    print(json.dumps(inspect_design(args.root, args.replicas, expected_tasks=args.tasks), indent=2))


if __name__ == "__main__":
    main()
