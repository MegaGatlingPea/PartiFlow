"""Explicit CPU chemistry acceptance runs; every input/output path is supplied."""

from __future__ import annotations

import argparse
import json
import math
import operator
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SEED = 2026
ENERGIES = ("total_score", "pep_internal_sc", "I_sc", "reweighted_sc")
WINDOWS = {"head_to_tail": (1.1, 1.7), "disulfide": (1.7, 2.3),
           "mk8_staple": (1.4, 1.7), "nle_staple": (1.2, 1.7)}
AA = dict(zip("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split(),
              "ARNDCQEGHILKMFPSTWYV"))


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def cpu_only():
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Set CUDA_VISIBLE_DEVICES='' for CPU acceptance")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"Set {name}=1")


def atoms(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith(("ATOM  ", "HETATM")):
            continue
        xyz = tuple(float(line[start:start + 8]) for start in (30, 38, 46))
        require(all(math.isfinite(x) for x in xyz), f"Nonfinite coordinates: {path}")
        result.append(dict(serial=int(line[6:11]), atom=line[12:16].strip(),
            restype=line[17:20].strip(), chain=line[21:22], number=int(line[22:26]),
            icode=line[26:27].strip(), element=line[76:78].strip(), xyz=xyz))
    require(result, f"No atoms: {path}")
    return result


def residue_identity(path, chain):
    residues = {}
    for atom in atoms(path):
        if atom["chain"] == chain and atom["atom"] == "CA":
            key = (atom["number"], atom["icode"])
            require(key not in residues, f"Duplicate binder residue: {key}")
            residues[key] = atom["restype"]
    require(residues, f"Binder absent: {chain}")
    return residues


def physical_connections(path, chain, entries):
    lookup = {(a["chain"], a["number"], a["atom"]): a for a in atoms(path)}
    result = []
    for entry in entries:
        kind = entry["type"]
        if kind not in WINDOWS:
            continue
        first, second = entry["anchors"][:2]
        names = {"head_to_tail": ("N", "C"), "disulfide": ("SG", "SG"),
                 "mk8_staple": ("CE", "CE"), "nle_staple": ("CE", "CE")}[kind]
        left, right = (lookup[(chain, position, name)] for position, name in zip((first, second), names))
        distance = math.dist(left["xyz"], right["xyz"])
        lower, upper = WINDOWS[kind]
        require(lower <= distance <= upper, f"{kind}: {distance} outside {lower}..{upper} A")
        result.append(dict(kind=kind, anchors=[first, second], atoms=list(names), distance=distance))
    return result


def rule_status(metrics, rules):
    operations = {">": operator.gt, ">=": operator.ge, "<": operator.lt,
                  "<=": operator.le, "==": operator.eq}
    failed, missing = [], []
    for rule in rules:
        name = rule["metric"]
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            missing.append(name)
        elif not operations[rule["op"]](value, rule["value"]):
            failed.append(name)
    return "unavailable" if missing else "failed" if failed else "passed", failed, missing


def prepare_native(index, base, output):
    """Rebuild static fixtures, with no assertion about historical file equality."""
    from partiflow.postprocess.validation import substitute_fixture
    from partiflow.postprocess.native import NCAA_SCORE_ONLY_RESIDUES
    data = json.loads(Path(index).read_text())
    cases = data["cases"]
    require(len(cases) == 45, "Expected 45 source cases")
    require({c["residue_code"] for c in cases} == set(NCAA_SCORE_ONLY_RESIDUES), "Support matrix coverage differs")
    output = Path(output).resolve()
    require(not output.exists(), "Choose a fresh fixture directory")
    output.mkdir(parents=True)
    prepared = []
    for case in cases:
        code = case["residue_code"]
        target = output / code / "input.pdb"
        target.parent.mkdir()
        placement = substitute_fixture(base, code, target, binder_chain=case["binder_chain"])
        require(placement["binder_position"] == case["topology_entries"][0]["resid"], "Scaffold position differs")
        require(code in residue_identity(target, case["binder_chain"]).values(), "Source chemical identity absent")
        prepared.append({**case, "path": str(target), "preparation": placement,
                         "input_kind": "rebuilt_static_passive_template_complex"})
    result = dict(cases=prepared, base=str(Path(base).resolve()), source_index=str(Path(index).resolve()),
                  note="Rebuilt deterministic static inputs. Historical temporary files are unavailable; byte equality is unverified.")
    save(output / "cases.json", result)
    return result


def actual_pose(pose):
    info, residues = pose.pdb_info(), []
    for i in range(1, pose.total_residue() + 1):
        residue, heavy, bonds, connections = pose.residue(i), [], [], []
        for j in range(1, residue.nheavyatoms() + 1):
            xyz = residue.xyz(j)
            heavy.append([residue.atom_name(j).strip(), str(residue.atom_type(j).element()),
                          float(residue.atomic_charge(j)), float(xyz.x), float(xyz.y), float(xyz.z)])
            for k in residue.bonded_neighbor(j):
                if j < int(k) <= residue.nheavyatoms():
                    bonds.append([residue.atom_name(j).strip(), residue.atom_name(k).strip(),
                                  str(residue.type().bond_type(j, int(k)))])
        for connection in range(1, residue.n_possible_residue_connections() + 1):
            partner = int(residue.connected_residue_at_resconn(connection))
            if partner:
                connections.append([connection, residue.atom_name(residue.residue_connect_atom_index(connection)).strip(),
                                    partner, int(residue.residue_connection_conn_id(connection))])
        charges = [[residue.atom_name(j).strip(), float(residue.atomic_charge(j))]
                   for j in range(1, residue.natoms() + 1)]
        residues.append(dict(index=i, chain=str(info.chain(i)), number=int(info.number(i)),
            icode=str(info.icode(i)), type=str(residue.type().name()), name3=str(residue.name3()),
            formal_charge=float(residue.type().net_formal_charge()), atom_charges=charges,
            partial_charge_sum=sum(value for _, value in charges), heavy_atoms=heavy,
            heavy_bonds=bonds, connections=connections))
    return dict(residues=residues, heavy_atom_count=sum(len(r["heavy_atoms"]) for r in residues))


def run_native(index, output):
    cpu_only()
    from partiflow.postprocess import native
    native.initialize_pyrosetta_score_only(seed=SEED, threads=1, silent=True)
    pr = native.pr
    require(not pr.rosetta.basic.options.get_boolean_option("corrections:beta_nov16"), "Wrong native SDK profile")
    cases = json.loads(Path(index).read_text())["cases"]
    require(len(cases) == 45, "Expected 45 frozen inputs")
    result = dict(status="PENDING", sdk=pr.version(), seed=SEED, threads=1,
                  sdk_profile="native_ref2015", source=str(Path(native.__file__).resolve()),
                  fixture_index=str(Path(index).resolve()), cases=[])
    save(output, result)
    for case in cases:
        code = case["residue_code"]
        row = dict(residue_code=code, path=case["path"], status="PENDING")
        result["cases"].append(row)
        save(output, result)
        try:
            pr.rosetta.numeric.random.rg().set_seed(SEED)
            score = native.score_fixed_pose(case["path"], binder_chain=case["binder_chain"],
                topology_entries=case["topology_entries"], silent=True, compute_noncore_diagnostics=False)
            pr.rosetta.numeric.random.rg().set_seed(SEED)
            pose, loaded = native.load_score_only_pose(case["path"], binder_chain=case["binder_chain"],
                topology_entries=case["topology_entries"], silent=True)
            require(score["status"] == "success", f"Native score failed: {code}")
            require(not any(score[k] for k in ("relax", "pack_input", "pack_separated")), "NCAA relaxed or packed")
            require(not score["pdb_fallback_residue_types"], "Unexpected fallback residue")
            require(not score["score_component_pathology"], "Energy component pathology")
            audit = loaded["input_heavy_atom_audit"]
            require(audit["status"] == "success" and not audit["source_only_count"] and audit["max_delta_angstrom"] <= 0.001,
                    "Source heavy atoms changed")
            runtime = [r for r in loaded["runtime_residue_audit"] if r["source_code"] == code]
            require(len(runtime) == 1 and runtime[0]["name3_matches_contract"], "Native residue identity mismatch")
            require(runtime[0]["actual_formal_charge"] == runtime[0]["expected_formal_charge"], "Formal charge mismatch")
            require(loaded["residue_connection_audit"]["status"] == "success", "Asymmetric residue connections")
            require(all(b["status"] == "present" for b in loaded["validated_bonds"]), "Missing native bond")
            energies = {key: score[key] for key in ENERGIES}
            components = {key: score[key] for key in score["score_component_weights"] if key in score}
            require(components and all(math.isfinite(value) for value in [*energies.values(), *components.values()]),
                    "Missing or nonfinite energy")
            row.update(status="PASS", energies=energies, energy_components=components,
                score_component_weights=score["score_component_weights"],
                scorefxn_adjustments=score["scorefxn_adjustments"], identity=actual_pose(pose), load=loaded)
            print(code, "PASS", flush=True)
        except BaseException as error:
            row.update(status="FAIL", error=f"{type(error).__name__}: {error}")
            result["status"] = "FAIL"
            save(output, result)
            raise
        save(output, result)
    result.update(status="PASS", count=len(cases))
    save(output, result)
    return result


def compare_native(baseline, current):
    first, second = (json.loads(Path(path).read_text()) for path in (baseline, current))
    require(first["status"] == second["status"] == "PASS", "Both actual native runs must pass")
    for field in ("sdk", "seed", "threads", "sdk_profile", "fixture_index"):
        require(first[field] == second[field], f"Native run controls differ: {field}")
    bmap, cmap = ({c["residue_code"]: c for c in data["cases"]} for data in (first, second))
    require(len(bmap) == len(cmap) == 45 and set(bmap) == set(cmap), "Incomplete native coverage")
    result = dict(status="PASS", count=45, maximum_energy_delta=0.0, cases=[])
    for code, before in bmap.items():
        after = cmap[code]
        require(before["status"] == after["status"] == "PASS", f"{code}: native case did not pass")
        for field in ("path", "identity", "scorefxn_adjustments"):
            require(before[field] == after[field], f"{code}: {field} changed")
        for field in ("runtime_residue_audit", "native_runtime", "residue_connection_audit", "validated_bonds", "input_heavy_atom_audit"):
            require(before["load"][field] == after["load"][field], f"{code}: {field} changed")
        for field in ("energies", "energy_components", "score_component_weights"):
            require(set(before[field]) == set(after[field]), f"{code}: {field} keys changed")
            for name, value in before[field].items():
                actual = after[field][name]
                require(math.isfinite(value) and math.isfinite(actual), f"{code}: nonfinite {name}")
                require(math.isclose(value, actual, rel_tol=1e-10, abs_tol=1e-6), f"{code}: {name} differs")
                result["maximum_energy_delta"] = max(result["maximum_energy_delta"], abs(value - actual))
        result["cases"].append(dict(residue_code=code, status="PASS", heavy_atoms=after["identity"]["heavy_atom_count"]))
    return result


def locator_meaning(value):
    if isinstance(value, list):
        return [locator_meaning(item) for item in value]
    if isinstance(value, dict):
        return {key: locator_meaning(item) for key, item in value.items() if key != "atom_id"}
    return value


def locator_check(row, path):
    actual = atoms(path)
    serials = {atom["serial"]: atom for atom in actual}
    all_residues = {(a["chain"], a["number"], a["icode"]): a["restype"] for a in actual}
    residues = residue_identity(path, row["design_chain_id"])
    sequence = "".join(AA.get(name, "[" + name + "]") for name in residues.values())
    require(row["design_sequence"] == sequence, "CSV sequence does not match final PDB")
    require(int(row["token_length"]) == len(residues), "CSV length does not match final PDB")
    declared = json.loads(row["design_chains"])
    require(len(declared) == 1 and declared[0]["sequence"] == sequence, "CSV chain sequence mismatch")
    require(declared[0]["chain"] == row["design_chain_id"] and declared[0]["residue_count"] == len(residues),
            "CSV chain identity mismatch")

    def visit(value):
        if isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, dict):
            if "restype" in value and "chain" in value and "res_id" in value:
                key = (value["chain"], int(value["res_id"]), value.get("ins_code", ""))
                require(all_residues.get(key) == value["restype"], "Declared fixed residue is absent or changed")
            if "atom_id" in value:
                atom = serials[int(value["atom_id"])]
                for key, field in (("chain", "chain"), ("res_id", "number"), ("ins_code", "icode"), ("atom", "atom")):
                    if key in value:
                        require(str(value[key]) == str(atom[field]), f"Invalid CSV locator: {value}")
                if "restype" in value:
                    require(value["restype"] == atom["restype"], "Fixed residue identity changed")
            for child in value.values():
                visit(child)
    for field in ("fixed_restypes", "bond_constraints"):
        visit(json.loads(row[field]))


def cli_call(paths, config, filtering, output):
    command = [sys.executable, "-B", "-m", "partiflow.cli", "postprocess", *map(str, paths),
               "--config", str(config), "--filter", str(filtering).lower(), "--platform", "CPU"]
    with Path(output).open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    require(completed.returncode == 0, f"Public CLI returned {completed.returncode}; see {output}")
    return command


def inspect_panel_topology(cases):
    from partiflow.postprocess import native
    native.initialize_pyrosetta_score_only(seed=SEED, threads=1, silent=True)
    inspected = []
    for case in cases:
        record = case["record"]
        pose, loaded = native.load_score_only_pose(record["pdb_path"], binder_chain=case["binder_chain"],
            topology_entries=case["topology_entries"], silent=True)
        require(loaded["input_heavy_atom_audit"]["status"] == "success", "Final PDB atom load failed")
        require(loaded["residue_connection_audit"]["status"] == "success", "Final PDB connection symmetry failed")
        info = pose.pdb_info()
        positions = {int(info.number(i)): i for i in range(1, pose.total_residue() + 1)
                     if info.chain(i) == case["binder_chain"]}
        connections = []
        for entry in case["topology_entries"]:
            kind = entry["type"]
            if kind not in WINDOWS:
                continue
            left_number, right_number = entry["anchors"][:2]
            left, right = (positions[number] for number in (left_number, right_number))
            left_name, right_name = {"head_to_tail": ("N", "C"), "disulfide": ("SG", "SG"),
                "mk8_staple": ("CE", "CE"), "nle_staple": ("CE", "CE")}[kind]
            residue = pose.residue(left)
            found = False
            for connection in range(1, residue.n_possible_residue_connections() + 1):
                if int(residue.connected_residue_at_resconn(connection)) != right:
                    continue
                partner_connection = int(residue.residue_connection_conn_id(connection))
                partner = pose.residue(right)
                found = (residue.atom_name(residue.residue_connect_atom_index(connection)).strip() == left_name
                    and partner.atom_name(partner.residue_connect_atom_index(partner_connection)).strip() == right_name
                    and int(partner.connected_residue_at_resconn(partner_connection)) == left)
                if found:
                    break
            require(found, f"Native final connection absent: {case['name']} {kind}")
            connections.append(dict(kind=kind, atoms=[left_name, right_name], anchors=[left_number, right_number], status="present"))
        inspected.append(dict(name=case["name"], connections=connections, load=loaded,
                              native_residues=loaded["runtime_residue_audit"]))
    return inspected


def run_panel(index, output, presets, workers=2):
    """Exercise public task/refold paths; fixture reward columns test preservation."""
    cpu_only()
    require(1 <= workers <= 2, "Acceptance permits at most two CPU workers")
    import yaml
    from partiflow.common.results import BASE_FIELDS, TAIL_FIELDS, write_results, read_results
    output = Path(output).resolve()
    require(not output.exists(), "Choose a fresh panel directory")
    output.mkdir(parents=True)
    cases = json.loads(Path(index).read_text())["cases"]
    require(len(cases) == 9, "Expected nine persistent fixture cases")
    rules = yaml.safe_load(Path(presets).read_text())["presets"]["rosetta_bindcraft_peptide"]["rules"]
    preset_bytes = Path(presets).read_bytes()
    config = output / "postprocess.yaml"
    config.write_text(yaml.safe_dump(dict(method="auto", platform="CPU", workers=workers, threads=1,
        seed=SEED, timeout=1800, relax_repeats=1, presets=str(Path(presets).resolve()))))
    prepared, original_rows = [], {}
    for case in cases:
        task = output / "tasks" / case["name"]
        task.mkdir(parents=True)
        source = task / "sample1.pdb"
        shutil.copyfile(case["path"], source)
        residues = residue_identity(source, case["binder_chain"])
        sequence = "".join(AA.get(name, "[" + name + "]") for name in residues.values())
        selected = [a for a in atoms(source) if a["chain"] == case["binder_chain"] and a["atom"] in {"N", "CA"}][:2]

        def locator(atom):
            return dict(chain=atom["chain"], res_id=atom["number"], ins_code=atom["icode"],
                        atom=atom["atom"], atom_id=atom["serial"])
        fixed = {**locator(selected[1]), "restype": selected[1]["restype"]}
        row = {field: "" for field in (*BASE_FIELDS, *TAIL_FIELDS)}
        row.update(filename=source.name, design_chains=json.dumps([dict(chain=case["binder_chain"],
            modality="protein", sequence=sequence, residue_count=len(residues))]),
            base_sample_id="3", design_modality="protein", design_chain_id=case["binder_chain"],
            token_length=len(residues), reward_method="af2multimer", global_rank="2", design_sequence=sequence,
            contig_string=f"{case['binder_chain']}{next(iter(residues))[0]}-{next(reversed(residues))[0]}",
            tip_atoms=json.dumps([locator(selected[1])]), fixed_restypes=json.dumps([fixed]),
            bond_constraints=json.dumps([dict(atoms=[locator(a) for a in selected])]),
            af2multimer_iptm=0.75, af2multimer_plddt=88.0, af2multimer_score=0.7825,
            postprocess_status="not_run", error="")
        write_results(task / "result.csv", [row])
        _, stored = read_results(task / "result.csv")
        original_rows[case["name"]] = stored[0]
        save(task / ".state" / "design.json", dict(samples=[dict(filename=source.name, sample_id="sample1",
            binder_chain=case["binder_chain"], topology_entries=case["topology_entries"],
            postprocess_method=case["historical_backend"], postprocess_enabled=True)]))
        prepared.append({**case, "output": str(source), "report": str(task / "postprocess" / "sample1.json")})
    refold_case = next(c for c in prepared if c["name"] == "standard_rosetta")
    refold = Path(refold_case["output"]).parent / "AF2success" / "original" / "complex_model_1.pdb"
    refold.parent.mkdir(parents=True)
    shutil.copyfile(refold_case["path"], refold)
    refold_bytes = refold.read_bytes()
    paths = [output / "tasks", refold.parent.parent]
    commands = [cli_call(paths, config, False, output / "filter-false.log")]
    records = [(case, json.loads(Path(case["report"]).read_text())) for case in prepared]
    geometry = {}
    for case, record in records:
        require(record["status"] == "success" and record["geometry_status"] == "complete", f"Geometry failed: {case['name']}")
        require(record["metrics_status"] == "not_requested", "filter=false computed metrics")
        path = Path(record["pdb_path"])
        geometry[case["name"]] = (path.read_bytes(), path.stat().st_mtime_ns)
    commands.append(cli_call(paths, config, True, output / "filter-true.log"))
    final = []
    for case in prepared:
        record = json.loads(Path(case["report"]).read_text())
        require(record["status"] == "success" and record["geometry_status"] == "complete", f"Postprocess failed: {case['name']}")
        path = Path(record["pdb_path"])
        require((path.read_bytes(), path.stat().st_mtime_ns) == geometry[case["name"]], "Metrics-only run changed geometry")
        require(residue_identity(path, case["binder_chain"]) == residue_identity(case["path"], case["binder_chain"]),
                f"Final residue identity changed: {case['name']}")
        distances = physical_connections(path, case["binder_chain"], case["topology_entries"])
        _, rows = read_results(path.parent / "result.csv")
        require(len(rows) == 1, "Task row count changed")
        row, original = rows[0], original_rows[case["name"]]
        locator_check(row, path)
        for field in ("fixed_restypes", "bond_constraints"):
            require(locator_meaning(json.loads(row[field])) == locator_meaning(json.loads(original[field])),
                    f"CSV locator meaning changed: {field}")
        for field in ("base_sample_id", "reward_method", "global_rank", "contig_string", "tip_atoms",
                      "af2multimer_iptm", "af2multimer_plddt", "af2multimer_score"):
            require(row[field] == original[field] and row[field] != "", f"CSV metadata changed: {field}")
        metrics = record["metrics_report"]["metrics"]
        expected, failed, missing = rule_status(metrics, rules)
        require(record["filter_status"] == expected, f"Independent rule result differs: {case['name']}")
        if case["name"] == "bal_headtail_openmm":
            require("surface_hydrophobicity" in missing, "BAL surface must remain unavailable")
            require("surface_hydrophobicity" not in metrics, "BAL surface value was fabricated")
        else:
            require(record["metrics_status"] == "complete", f"Metrics incomplete: {case['name']}")
        final.append(dict(name=case["name"], binder_chain=case["binder_chain"], topology_entries=case["topology_entries"],
                          software_status="PASS", filter_status=expected,
                          failed_rules=failed, missing_metrics=missing, physical_connections=distances, record=record))
    require(refold.read_bytes() == refold_bytes, "AF2success source was modified")
    refold_report = refold.parents[2] / "postprocess" / "refold" / "original" / "complex_model_1.json"
    refold_record = json.loads(refold_report.read_text())
    require(refold_record["status"] == "success" and Path(refold_record["output_path"]).is_file(), "Refold output failed")
    require(Path(refold_record["output_path"]) != refold, "Refold output overwrote source")
    metrics_before = [case["record"]["metrics_report"] for case in final]
    commands.append(cli_call(paths, config, True, output / "cached-rejudge.log"))
    require("postprocess_workers" not in (output / "cached-rejudge.log").read_text(), "Cached rejudge started workers")
    for case, expected_metrics in zip(prepared, metrics_before):
        record = json.loads(Path(case["report"]).read_text())
        path = Path(record["pdb_path"])
        require((path.read_bytes(), path.stat().st_mtime_ns) == geometry[case["name"]], "Rejudge changed geometry")
        require(record["metrics_report"] == expected_metrics, "Rejudge changed cached metrics")
    require(refold.read_bytes() == refold_bytes and Path(presets).read_bytes() == preset_bytes, "Original input/config changed")
    result = dict(status="PASS", normal_case_count=9, refold_case_count=1, commands=commands,
        note="Nonempty reward/rank columns are fixture metadata for CSV preservation; no AF2 prediction is claimed.",
        cases=final, refold_record=refold_record, final_topology=inspect_panel_topology(final))
    save(output / "result.json", result)
    return result


def run_ranked_task(task, output, presets):
    """Postprocess a copy of a selected real ranked task, preserving its originals."""
    cpu_only()
    import yaml
    from partiflow.common.results import read_results
    task, output = Path(task).resolve(), Path(output).resolve()
    require(not output.exists(), "Choose a fresh directory for the real ranked task")
    fields, before = read_results(task / "result.csv")
    require(before and all(row["global_rank"] and row["reward_method"] for row in before), "Select actual ranked reward rows")
    require(any(json.loads(row["fixed_restypes"]) or json.loads(row["bond_constraints"]) for row in before),
            "Select actual rows with declared residue/atom locators")
    output.mkdir(parents=True)
    copied = output / task.name
    copied.mkdir()
    sources = [task / "result.csv", task / ".state" / "design.json"]
    sources.extend(task / row["filename"] for row in before)
    originals = {}
    for source in sources:
        source = source.resolve()
        require(source.is_relative_to(task), "Task file leaves selected task directory")
        originals[source] = source.read_bytes()
        destination = copied / source.relative_to(task)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config = output / "postprocess.yaml"
    config.write_text(yaml.safe_dump(dict(method="auto", platform="CPU", workers=1, threads=1,
        seed=SEED, timeout=1800, relax_repeats=1, presets=str(Path(presets).resolve()))))
    commands = [cli_call([copied], config, False, output / "filter-false.log")]
    geometry = {copied / row["filename"]: ((copied / row["filename"]).read_bytes(),
                (copied / row["filename"]).stat().st_mtime_ns) for row in before}
    commands.append(cli_call([copied], config, True, output / "filter-true.log"))
    fields_after, after = read_results(copied / "result.csv")
    require(fields == fields_after and len(before) == len(after), "Actual design CSV shape changed")
    rules = yaml.safe_load(Path(presets).read_text())["presets"]["rosetta_bindcraft_peptide"]["rules"]

    def reports():
        return [json.loads(p.read_text()) for p in (copied / "postprocess").glob("*.json") if p.name != "summary.json"]
    records = reports()
    for old, new in zip(before, after):
        path = copied / new["filename"]
        require((path.read_bytes(), path.stat().st_mtime_ns) == geometry[path], "Metrics-only run changed actual design geometry")
        locator_check(new, path)
        for field in ("fixed_restypes", "bond_constraints"):
            require(locator_meaning(json.loads(new[field])) == locator_meaning(json.loads(old[field])),
                    f"Actual design locator meaning changed: {field}")
        for field in fields:
            if field not in {"design_chains", "design_chain_id", "design_sequence", "token_length",
                             "fixed_restypes", "bond_constraints", "postprocess_status", "error"}:
                require(new[field] == old[field], f"Actual design metadata changed: {field}")
        matches = [record for record in records if record.get("input_path") == str(path)]
        require(len(matches) == 1, "Actual design report missing")
        record = matches[0]
        require(record["status"] == "success" and record["geometry_status"] == "complete", "Actual design geometry failed")
        require(record["metrics_status"] == "complete", "Actual design metrics incomplete")
        expected = rule_status(record["metrics_report"]["metrics"], rules)[0]
        require(record["filter_status"] == expected, "Actual design independent filter differs")
    commands.append(cli_call([copied], config, True, output / "cached-rejudge.log"))
    require("postprocess_workers" not in (output / "cached-rejudge.log").read_text(), "Actual cached rejudge started workers")
    require({r["input_path"]: r["metrics_report"] for r in reports()} ==
            {r["input_path"]: r["metrics_report"] for r in records}, "Actual cached metrics changed")
    for path, state in geometry.items():
        require((path.read_bytes(), path.stat().st_mtime_ns) == state, "Actual cached rejudge changed geometry")
    require(all(path.read_bytes() == data for path, data in originals.items()), "Selected real task originals changed")
    result = dict(status="PASS", count=len(after), source_task=str(task), commands=commands, cases=records,
                  source_role="AF2Multimer reward fold; no standalone validation outcome is asserted")
    save(output / "result.json", result)
    return result


def run_faults(fixture, output, timeout=30):
    """Inject process exit and delayed metrics after real geometry publication."""
    cpu_only()
    from unittest.mock import patch
    from partiflow.postprocess import runner
    from partiflow.postprocess.records import WorkItem
    require(15 <= timeout <= 60, "Use a short timeout that still permits SDK startup")
    output = Path(output).resolve()
    require(not output.exists(), "Choose a fresh fault directory")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    event_path = output / "actor-events.jsonl"
    late_path = output / "late-write.txt"
    delay = timeout + 10
    items = []
    for name in ("crash", "next"):
        source = inputs / f"{name}.pdb"
        shutil.copyfile(fixture, source)
        items.append(WorkItem(source, "fault_injection", name, binder_chain="L",
            topology_entries=[dict(type="head_to_tail", anchors=[1, 8])],
            method="rosetta_topology", report_dir=output / "reports", replace=False))
    originals = [item.path.read_bytes() for item in items]

    def event(kind, **values):
        with event_path.open("a") as handle:
            handle.write(json.dumps(dict(event=kind, pid=os.getpid(), **values)) + "\n")

    def worker_class(mode):
        class FaultWorker(runner.PostprocessWorker):
            def __init__(self, settings):
                super().__init__(settings)
                event("initialized", mode=mode, stats=self.stats())

            def process(self, item):
                event("sample", mode=mode, sample_id=item.sample_id)
                if item.sample_id == "crash" and mode in {"exit", "timeout"}:
                    def injected(path, **kwargs):
                        saved = json.loads(runner._report_path(item).read_text())
                        event("metrics_entered", mode=mode, status=saved["status"],
                              geometry_status=saved["geometry_status"], output_exists=Path(path).is_file())
                        if mode == "exit":
                            os._exit(91)
                        time.sleep(delay)
                        late_path.write_text("Delayed actor was still alive\n")
                        event("late_write", mode=mode)
                        raise RuntimeError("Injected delayed metrics completed")
                    with patch("partiflow.postprocess.metrics.compute_metrics", side_effect=injected):
                        return super().process(item)
                if mode == "metrics_only":
                    def forbidden_relax(*args, **kwargs):
                        event("unexpected_relax", sample_id=item.sample_id)
                        raise AssertionError("Metrics-only recovery repeated geometry")
                    with patch("partiflow.postprocess.chemistry.relax_candidate", side_effect=forbidden_relax):
                        return super().process(item)
                return super().process(item)
        return FaultWorker

    settings = dict(method="rosetta_topology", platform="CPU", workers=1, threads=1,
                    timeout=180, seed=SEED, filter=True)
    result = dict(status="PENDING", fault_injection=True, settings=settings)
    save(output / "result.json", result)

    def completed(record):
        require(record["status"] == "success" and record["geometry_status"] == "complete", "Real geometry failed")
        require(record["metrics_status"] == "complete" and record["filter_status"] in {"passed", "failed"},
                "Real metrics did not complete")
        require(not record.get("error"), "Successful metrics retained a failure")

    def retained(record):
        require(record["status"] == "success" and record["geometry_status"] == "complete", "Completed geometry was lost")
        require(record["metrics_status"] == record["filter_status"] == "unavailable", "Fault metrics were accepted")
        require(record.get("error"), "Worker failure was omitted")
        physical_connections(record["pdb_path"], "L", items[0].topology_entries)

    with patch("partiflow.postprocess.runner.PostprocessWorker", worker_class("exit")):
        first = runner.process_candidates(items, **settings)
    result["exit_results"] = first
    save(output / "result.json", result)
    retained(first[0])
    completed(first[1])
    geometry = Path(first[0]["pdb_path"])
    geometry_state = (geometry.read_bytes(), geometry.stat().st_mtime_ns)
    with patch("partiflow.postprocess.runner.PostprocessWorker", worker_class("metrics_only")):
        resumed = runner.process_candidates([items[0]], **settings)[0]
    completed(resumed)
    require(resumed.get("resumed"), "Recovery did not reuse geometry")
    require((geometry.read_bytes(), geometry.stat().st_mtime_ns) == geometry_state, "Recovery changed geometry")
    with patch("partiflow.postprocess.dispatch.dispatch_items", side_effect=AssertionError("Cached rejudge dispatched a worker")):
        cached = runner.process_candidates([items[0]], **settings)[0]
    completed(cached)
    require(cached["metrics_report"] == resumed["metrics_report"], "Cached rejudge changed real metrics")
    timeout_settings = {**settings, "timeout": timeout, "rerun_metrics": True}
    with patch("partiflow.postprocess.runner.PostprocessWorker", worker_class("timeout")):
        timed = runner.process_candidates(items, **timeout_settings)
    result.update(resumed=resumed, cached=cached, timeout_results=timed, timeout_seconds=timeout,
                  injected_delay_seconds=delay)
    save(output / "result.json", result)
    retained(timed[0])
    completed(timed[1])
    require("timeout" in timed[0]["error"], "Short timeout was not exercised")
    # The delayed callback would write after this interval if termination failed.
    time.sleep(delay + 2)
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    require(not late_path.exists() and not any(e["event"] == "late_write" for e in events), "Terminated actor wrote late")
    require(not any(e["event"] == "unexpected_relax" for e in events), "Metrics recovery repeated geometry")
    for mode in ("exit", "timeout"):
        starts = [e for e in events if e["event"] == "initialized" and e["mode"] == mode]
        require(len(starts) == 2 and len({e["pid"] for e in starts}) == 2, f"Actor replacement failed: {mode}")
        require(all(e["stats"]["initializations"] == 1 for e in starts), "SDK initialized repeatedly in actor")
        entered = [e for e in events if e["event"] == "metrics_entered" and e["mode"] == mode]
        require(len(entered) == 1 and entered[0]["geometry_status"] == "complete" and entered[0]["output_exists"],
                "Fault injection did not follow real geometry")
    require((geometry.read_bytes(), geometry.stat().st_mtime_ns) == geometry_state, "Actor failure changed published geometry")
    require([item.path.read_bytes() for item in items] == originals, "Fault run changed source inputs")
    import ray
    require(not ray.is_initialized(), "Owned Ray runtime remains active")
    require("pyrosetta" not in sys.modules, "Driver loaded native SDK")
    result.update(status="PASS", events=events, no_late_writes=True)
    save(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-native")
    prepare.add_argument("--index", type=Path, required=True)
    prepare.add_argument("--base", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    native = sub.add_parser("native")
    native.add_argument("--index", type=Path, required=True)
    native.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compare-native")
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--current", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    panel = sub.add_parser("panel")
    panel.add_argument("--index", type=Path, required=True)
    panel.add_argument("--output", type=Path, required=True)
    panel.add_argument("--presets", type=Path, required=True)
    panel.add_argument("--workers", type=int, default=2)
    fault = sub.add_parser("faults")
    fault.add_argument("--fixture", type=Path, required=True)
    fault.add_argument("--output", type=Path, required=True)
    fault.add_argument("--timeout", type=int, default=30)
    ranked = sub.add_parser("ranked-task")
    ranked.add_argument("--task", type=Path, required=True)
    ranked.add_argument("--output", type=Path, required=True)
    ranked.add_argument("--presets", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare-native":
        result = prepare_native(args.index, args.base, args.output)
    elif args.command == "native":
        result = run_native(args.index, args.output)
    elif args.command == "panel":
        result = run_panel(args.index, args.output, args.presets, args.workers)
    elif args.command == "faults":
        result = run_faults(args.fixture, args.output, args.timeout)
    elif args.command == "ranked-task":
        result = run_ranked_task(args.task, args.output, args.presets)
    else:
        result = compare_native(args.baseline, args.current)
        save(args.output, result)
    fields = ("status", "count", "maximum_energy_delta", "normal_case_count", "refold_case_count",
              "fault_injection", "no_late_writes", "timeout_seconds")
    summary = {key: result[key] for key in fields if key in result}
    summary.update(output=str(args.output))
    if "count" not in summary and "cases" in result:
        summary["count"] = len(result["cases"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
