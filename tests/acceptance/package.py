"""Build and check a wheel outside the source checkout.

Subcommands keep generated distributions, logs and isolated environments in an
explicit external directory. ``build`` copies the current tracked and untracked
source files before invoking the normal build backend. ``inspect`` compares
packaged source and small resources by their actual bytes, and delegates wheel
RECORD validation to the installed ``wheel`` command. ``install`` creates a
separate environment without changing its base environment. ``probe`` must run
with that environment's interpreter from outside the checkout.

This helper never uploads a distribution or runs GPU computations.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile


class AcceptanceError(AssertionError):
    """A required distribution check did not hold."""


def require(condition, message):
    if not condition:
        raise AcceptanceError(message)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def external_root(path, checkout):
    path, checkout = Path(path).resolve(), Path(checkout).resolve()
    require(path != checkout and checkout not in path.parents and path not in checkout.parents,
            "Output must be outside the source checkout")
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_environment():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.update(CUDA_VISIBLE_DEVICES="", JAX_PLATFORMS="cpu", OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4",
               PYTHONDONTWRITEBYTECODE="1", NO_COLOR="1", TERM="dumb")
    return env


def command(root, name, argv, *, cwd=None, expected=0, env=None, input_text=None):
    root = Path(root)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    selected_env = clean_environment() if env is None else env
    selected_cwd = Path(cwd or root).resolve()
    started = time.monotonic()
    result = subprocess.run([str(item) for item in argv], cwd=selected_cwd, env=selected_env,
                            text=True, input=input_text, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (root / "logs" / f"{name}.log").write_text(result.stdout)
    write_json(root / "commands" / f"{name}.json", {
        "argv": [str(item) for item in argv], "cwd": str(selected_cwd), "returncode": result.returncode,
        "seconds": round(time.monotonic() - started, 4),
        "environment": {key: selected_env.get(key) for key in
                        ("PYTHONPATH", "CUDA_VISIBLE_DEVICES", "JAX_PLATFORMS", "OMP_NUM_THREADS",
                         "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    })
    if expected is not None:
        require(result.returncode == expected, f"{name} exited {result.returncode}; see logs/{name}.log")
    return result


EXCLUDED_DIRS = {"__pycache__", ".git", ".cache", ".agents", ".pytest_cache", "build", "dist",
                 "ckpts", "checkpoint", "checkpoints", "train_data", "inference", "runs", "weights"}
EXCLUDED_SUFFIXES = {".pyc", ".ckpt", ".pt", ".pth", ".npz", ".safetensors", ".bin", ".msgpack",
                     ".h5", ".hdf5", ".npy", ".pkl"}


def allowed(path):
    path = Path(path)
    scaffold = path.parts[:2] == ("configs", "scaffolds") or path.parts[:4] == (
        "partiflow", "_resources", "configs", "scaffolds")
    return (not any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in path.parts)
            and (path.suffix not in EXCLUDED_SUFFIXES or (path.suffix == ".pt" and scaffold)))


def allowed_sdist(path):
    # Source archives contain backend-generated metadata as well as source files.
    return allowed(Path(*(part for part in Path(path).parts if not part.endswith(".egg-info"))))


def build(args, root, checkout):
    source = root / "source"
    require(not source.exists(), "Choose a fresh package output directory for a new build")
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
                            cwd=checkout, check=True, stdout=subprocess.PIPE).stdout.decode().split("\0")
    copied = []
    for item in sorted(set(filter(None, listed))):
        relative = Path(item)
        actual = checkout / relative
        if not allowed(relative) or not actual.is_file():
            continue
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(actual, target)
        require(target.read_bytes() == actual.read_bytes(), f"Source changed while copying {relative}")
        copied.append(relative.as_posix())
    write_json(root / "source_files.json", copied)
    command(root, "build", [args.python, "-B", "-m", "build", "--no-isolation", "--sdist", "--wheel",
                            "--outdir", root / "dist", source])
    wheels = list((root / "dist").glob("*.whl"))
    sdists = list((root / "dist").glob("*.tar.gz"))
    require(len(wheels) == len(sdists) == 1, "Build did not produce one wheel and one sdist")
    result = {"wheel": str(wheels[0]), "sdist": str(sdists[0]), "source_files": len(copied)}
    write_json(root / "build.json", result)
    return result


def check_record(archive):
    names = [item.filename for item in archive.infolist() if not item.is_dir()]
    records = [name for name in names if name.endswith(".dist-info/RECORD")]
    require(len(records) == 1, "Wheel needs exactly one RECORD")
    record = records[0]
    rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
    require(all(len(row) == 3 for row in rows), "RECORD row has invalid width")
    require(len({row[0] for row in rows}) == len(rows), "RECORD repeats a path")
    require({row[0] for row in rows} == set(names), "RECORD paths differ from wheel members")
    for name, check, size in rows:
        if name == record:
            require(not check and not size, "RECORD must omit its own check and size")
        else:
            require(bool(check), f"Missing standard RECORD check for {name}")
            require(int(size) == len(archive.read(name)), f"RECORD size differs for {name}")
    return record, len(rows)


def check_sdist_helpers(archive, members, checkout):
    helpers = sorted((Path(checkout) / "tests/acceptance").glob("*.py"))
    require(bool(helpers), "Source acceptance helpers are unavailable")
    names = []
    for path in helpers:
        relative = path.relative_to(checkout).as_posix()
        require(relative in members, f"Sdist is missing acceptance helper: {relative}")
        require(archive.extractfile(members[relative]).read() == path.read_bytes(),
                f"Sdist acceptance helper differs from current source: {relative}")
        names.append(relative)
    return names


def inspect_distribution(args, root, checkout):
    wheel = Path(args.wheel).resolve()
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        require(len(names) == len(archive.namelist()), "Wheel repeats a member")
        record, record_rows = check_record(archive)
        metadata = archive.read(record.replace("RECORD", "WHEEL")).decode()
        require("Root-Is-Purelib: false" in metadata, "Native wheel claims purelib")
        tags = [line.removeprefix("Tag: ") for line in metadata.splitlines() if line.startswith("Tag: ")]
        require(tags and all(tag.startswith("py3-none-") and tag.endswith("x86_64") for tag in tags),
                f"Unexpected native wheel tags: {tags}")
        forbidden = sorted(name for name in names if not allowed(Path(name)))
        require(not forbidden, f"Distribution contains excluded files: {forbidden}")
        required = ["_resources/configs/design.yaml", "_resources/configs/tasks/scaffold.yaml",
                    "utils/relax_assets/topology.xml", "postprocess/rosetta_ncaa/support_matrix.csv",
                    "postprocess/rosetta_ncaa/params/Z28.params", "postprocess/bin/DAlphaBall.gcc",
                    "tools/dssp", "tools/sc", "_vendor/proteinmpnn/LICENSE",
                    "_vendor/af2/LICENSE-AlphaFold-Code-Apache-2.0.txt"]
        require(all("partiflow/" + item in names for item in required), "A required runtime asset is missing")
        source_count = resource_count = license_count = asset_count = 0
        runtime_suffixes = {".py", ".json", ".txt", ".csv", ".yaml", ".yml", ".params", ".xml",
                            ".pdb", ".gcc", ".cpp", ".cc", ".c", ".h", ".cu", ".cuh"}
        for path in (checkout / "partiflow").rglob("*"):
            if not path.is_file() or not allowed(path.relative_to(checkout)):
                continue
            is_license = "license" in path.name.lower() or path.name.upper() in {"NOTICE", "COPYING"}
            if path.suffix not in runtime_suffixes and not is_license and path.parent != checkout / "partiflow/tools":
                continue
            relative = path.relative_to(checkout).as_posix()
            require(relative in names, f"Missing source or copyright file: {relative}")
            repaired_binary = relative == "partiflow/postprocess/bin/DAlphaBall.gcc" and any("manylinux" in tag for tag in tags)
            if not repaired_binary:
                require(archive.read(relative) == path.read_bytes(), f"Wheel differs from current source: {relative}")
            source_count += path.suffix == ".py"
            license_count += is_license
            asset_count += path.suffix != ".py" and not is_license and not repaired_binary
        for directory in ("configs", "pdbs"):
            for path in (checkout / directory).rglob("*"):
                if not path.is_file() or not allowed(path.relative_to(checkout)):
                    continue
                relative = "partiflow/_resources/" + path.relative_to(checkout).as_posix()
                require(relative in names, f"Missing runtime resource: {relative}")
                require(archive.read(relative) == path.read_bytes(), f"Runtime resource differs: {relative}")
                resource_count += 1
        scaffolds = sorted(name for name in names if "/configs/scaffolds/" in name and name.endswith(".pt"))
        require(len(scaffolds) == 3, "Wheel must contain exactly three small scaffold libraries")
        require(all(len(archive.read(name)) < 1_000_000 for name in scaffolds), "Unexpectedly large scaffold library")
        import yaml

        task_files = sorted(name for name in names if "/_resources/configs/tasks/" in name and name.endswith(".yaml"))
        for name in task_files:
            for task in yaml.safe_load(archive.read(name)).values():
                if not isinstance(task, dict):
                    continue
                for value in (task.get("pdb_path"), (task.get("binder_topology") or {}).get("lib_path")):
                    if value:
                        require(not Path(value).is_absolute(), f"Packaged task contains an absolute path: {name}")
                        require("partiflow/_resources/" + Path(value).as_posix() in names,
                                f"Packaged task resource is absent: {name}: {value}")
        executables = {}
        for name in ("partiflow/postprocess/bin/DAlphaBall.gcc", "partiflow/tools/dssp", "partiflow/tools/sc"):
            data = archive.read(name)
            require(data[:4] == b"\x7fELF" and data[4] == 2 and int.from_bytes(data[18:20], "little") == 62,
                    f"Expected an ELF64 x86_64 executable: {name}")
            require((archive.getinfo(name).external_attr >> 16) & 0o111, f"Executable mode is missing: {name}")
            executables[name] = len(data)
    command(root, "wheel-unpack", [args.python, "-B", "-m", "wheel", "unpack", wheel, "-d", root / "unpacked"])
    broken = root / "tampered" / wheel.name
    broken.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(broken, "w") as target:
        for item in source.infolist():
            data = source.read(item.filename)
            if item.filename == "partiflow/__init__.py":
                data += b"\n# Changed after packaging.\n"
            target.writestr(item, data)
    rejected = command(root, "wheel-unpack-tampered", [args.python, "-B", "-m", "wheel", "unpack", broken,
                                                        "-d", root / "tampered/unpacked"], expected=None)
    require(rejected.returncode != 0, "Standard wheel tooling accepted altered package bytes")
    result = {"wheel": str(wheel), "tags": tags, "record_rows": record_rows,
              "python_files_equal_current_source": source_count, "runtime_resources_equal": resource_count,
              "copyright_files_equal": license_count, "other_runtime_assets_equal": asset_count,
              "scaffold_libraries": scaffolds,
              "executables": executables, "altered_wheel_rejected": True}
    if args.sdist:
        with tarfile.open(args.sdist) as archive:
            files = [item for item in archive.getmembers() if item.isfile()]
            relative = {Path(*Path(item.name).parts[1:]).as_posix(): item for item in files}
            excluded = [name for name in relative if not allowed_sdist(name)]
            require(not excluded, f"Sdist contains excluded assets: {excluded}")
            helpers = check_sdist_helpers(archive, relative, checkout)
            guides = sorted((checkout / "docs").rglob("*.md"))
            for guide in guides:
                name = guide.relative_to(checkout).as_posix()
                require(name in relative, f"Sdist is missing a documentation guide: {name}")
                require(archive.extractfile(relative[name]).read() == guide.read_bytes(),
                        f"Sdist documentation guide differs from current source: {name}")
            for scaffold in (checkout / "configs/scaffolds").glob("*.pt"):
                name = scaffold.relative_to(checkout).as_posix()
                require(name in relative, f"Sdist is missing {name}")
                require(archive.extractfile(relative[name]).read() == scaffold.read_bytes(), f"Sdist differs: {name}")
            result["sdist_files"] = len(files)
            result["sdist_acceptance_helper_count"] = len(helpers)
            result["sdist_acceptance_helpers"] = helpers
            result["sdist_guide_count"] = len(guides)
            result["sdist_guides"] = [guide.relative_to(checkout).as_posix() for guide in guides]
    write_json(root / "inspection.json", result)
    return result


def install(args, root, checkout):
    environment = root / "wheel-env"
    require(not environment.exists(), "Wheel environment already exists")
    command(root, "create-wheel-env", [args.base_python, "-B", "-m", "venv", "--system-site-packages", environment])
    python = environment / "bin/python"
    command(root, "install-wheel", [python, "-I", "-m", "pip", "install", "--no-deps", "--ignore-installed", args.wheel])
    command(root, "wheel-env-pip-check", [python, "-I", "-m", "pip", "check"])
    result = {"python": str(python), "cli": str(environment / "bin/partiflow"), "wheel": str(Path(args.wheel).resolve()),
              "base_python": str(Path(args.base_python).absolute()), "system_site_packages": True}
    write_json(root / "installation.json", result)
    return result


def sdist_wheel(args, root, checkout):
    source = root / "sdist-source"
    require(not source.exists(), "Choose a fresh output directory for an sdist rebuild")
    with tarfile.open(args.sdist) as archive:
        archive.extractall(source, filter="data")
    directories = list(source.iterdir())
    require(len(directories) == 1 and directories[0].is_dir(), "Sdist must have one source root")
    command(root, "sdist-wheel-build", [args.python, "-B", "-m", "build", "--no-isolation", "--wheel",
                                       "--outdir", root / "sdist-wheel-dist", directories[0]])
    wheels = list((root / "sdist-wheel-dist").glob("*.whl"))
    require(len(wheels) == 1, "Sdist rebuild must produce exactly one wheel")
    selected = argparse.Namespace(**{**vars(args), "wheel": str(wheels[0])})
    inspection = inspect_distribution(selected, root / "sdist-wheel-inspection", checkout)
    result = {"wheel": str(wheels[0]), "sdist": str(Path(args.sdist).resolve()), "inspection": inspection}
    write_json(root / "sdist_rebuild.json", result)
    return result


def add_library_notices(destination, system_root=Path("/usr/share")):
    destination, system_root = Path(destination), Path(system_root)
    sources = [system_root / "doc" / name / "copyright" for name in ("libgmp10", "libgfortran5")]
    selected = {f"{path.parent.name}-copyright.txt": path for path in sources}
    for path in sources:
        require(path.is_file(), f"Installed library copyright file is unavailable: {path}")
        references = re.findall(r"/usr/share/common-licenses/([A-Za-z0-9_.-]+)", path.read_text())
        for name in references:
            name = name.rstrip(".")
            selected["common-licenses/" + name] = system_root / "common-licenses" / name
    records = []
    for relative, source in sorted(selected.items()):
        require(source.is_file(), f"Referenced library license text is unavailable: {source}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        require(target.read_bytes() == source.read_bytes(), f"Copied notice differs: {source}")
        records.append({"source": str(source), "packaged": str(target), "bytes": source.stat().st_size})
    return records


def audit(args, root, checkout):
    require(args.tools_python and args.repair_platform, "Audit requires --tools-python and --repair-platform")
    tools_bin = Path(args.tools_python).absolute().parent
    env = clean_environment()
    env["PATH"] = str(tools_bin) + os.pathsep + env.get("PATH", "")
    executable = tools_bin / "auditwheel"
    command(root, "auditwheel-show", [executable, "show", args.wheel], env=env)
    command(root, "auditwheel-repair", [executable, "repair", "--plat", args.repair_platform,
                                        "--wheel-dir", root / "repaired", args.wheel], env=env)
    wheels = list((root / "repaired").glob("*.whl"))
    require(len(wheels) == 1, "Repair must produce exactly one wheel")
    with zipfile.ZipFile(wheels[0]) as archive:
        libraries = [item.filename for item in archive.infolist() if not item.is_dir() and ".libs/" in item.filename]
    require(len(libraries) == 2 and {Path(name).name.split("-")[0] for name in libraries} == {"libgmp", "libgfortran"},
            f"Review copyright inputs for the actual bundled library set: {libraries}")
    command(root, "release-unpack", [args.python, "-B", "-m", "wheel", "unpack", wheels[0],
                                      "-d", root / "release-unpacked"])
    distributions = list((root / "release-unpacked").iterdir())
    require(len(distributions) == 1, "Expected one unpacked distribution")
    infos = list(distributions[0].glob("*.dist-info"))
    require(len(infos) == 1, "Expected one distribution metadata directory")
    notices = add_library_notices(infos[0] / "licenses" / "bundled-libraries")
    write_json(root / "library_notices.json", notices)
    (root / "release").mkdir(parents=True, exist_ok=True)
    command(root, "release-pack", [args.python, "-B", "-m", "wheel", "pack", distributions[0],
                                    "-d", root / "release"])
    released = list((root / "release").glob("*.whl"))
    require(len(released) == 1, "Release packing must produce exactly one wheel")
    command(root, "auditwheel-show-release", [executable, "show", released[0]], env=env)
    binary = distributions[0] / "partiflow/postprocess/bin/DAlphaBall.gcc"
    dependencies = command(root, "release-binary-ldd", ["ldd", binary])
    require("not found" not in dependencies.stdout, "A repaired executable dependency is unresolved")
    resolved = {Path(value).resolve() for value in re.findall(r"=> (\S+)", dependencies.stdout)}
    require(all((distributions[0] / name).resolve() in resolved for name in libraries),
            "Repaired executable did not resolve both bundled libraries")
    command(root, "release-binary-headers", ["readelf", "-h", "-d", binary])
    result = {"wheel": str(released[0]), "platform": args.repair_platform,
              "bundled_libraries": libraries, "copied_library_notices": notices,
              "notice_inclusion": "Installed system copyright and referenced license texts copied after repair"}
    write_json(root / "audit.json", result)
    return result


def probe(args, root, checkout):
    require(Path(sys.prefix).resolve() == root / "wheel-env", "Run probe with the isolated wheel interpreter")
    require("PYTHONPATH" not in os.environ, "Unset PYTHONPATH for installed-wheel checks")
    require(checkout not in Path.cwd().resolve().parents and Path.cwd().resolve() != checkout,
            "Installed-wheel checks require an external working directory")
    import partiflow
    from partiflow.paths import checkout_root, resource_root

    require(Path(sys.prefix).resolve() in Path(partiflow.__file__).resolve().parents, "Imported a different package")
    require(checkout_root() is None, "Installed package resolved a source checkout")
    resource = resource_root()
    python, cli = Path(sys.executable), root / "wheel-env/bin/partiflow"
    summary = {"package": partiflow.__file__, "python": str(python), "checkout_root": None,
               "resource_root": str(resource), "help": [], "config_plans": []}
    command(root, "help-root", [cli, "--help"])
    for name in ("train", "design", "validate", "postprocess", "cluster", "check", "download"):
        command(root, f"help-{name}", [cli, name, "--help"])
        summary["help"].append(name)
    for stage, config in (("vae", "training_ae.yaml"), ("fm", "training.yaml")):
        name = "train-config-" + stage
        command(root, name, [cli, "train", stage, "--config", resource / "configs" / config,
                             "--model-dir", args.model_root, "--show-config"])
        summary["config_plans"].append(name)
    for task_file in sorted((resource / "configs/tasks").glob("*.yaml")):
        name = "design-config-" + task_file.stem
        command(root, name, [cli, "design", task_file, "--model-dir", args.model_root, "--show-config"])
        summary["config_plans"].append(name)
    import torch
    from partiflow.design.tasks import _load_topology_library

    torch.set_num_threads(4)
    libraries = []
    for path in sorted((resource / "configs/scaffolds").glob("*.pt")):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        entries = _load_topology_library(str(path), task_name="installed_wheel_acceptance")
        tensor_count = 0

        def check_tensors(value):
            nonlocal tensor_count
            if isinstance(value, torch.Tensor):
                require(value.device.type == "cpu" and bool(torch.isfinite(value).all()), f"Invalid library tensor: {path}")
                tensor_count += 1
            elif isinstance(value, dict):
                for item in value.values():
                    check_tensors(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    check_tensors(item)

        check_tensors(raw)
        require(entries and tensor_count, f"Empty scaffold library: {path}")
        libraries.append({"path": str(path), "entries": len(entries), "tensors": tensor_count, "bytes": path.stat().st_size})
    require(len(libraries) == 3, "Expected three packaged scaffold libraries")
    summary["scaffold_libraries"] = libraries
    package_dir = Path(partiflow.__file__).parent
    spheres = "NPOINTS\n4\nCOORDS\n0 0 0 1\n3 0 0 1\n0 3 0 1\n0 0 3 1\nEND\n"
    native = command(root, "native-dalphaball", [package_dir / "postprocess/bin/DAlphaBall.gcc", "surf_vol"],
                     input_text=spheres)
    values = [[float(value) for value in line.split()] for line in native.stdout.splitlines() if line.strip()]
    require(len(values) == 4 and all(len(row) == 3 for row in values), "Unexpected native sphere output")
    for index, area, volume in values:
        require(math.isclose(area, 4 * math.pi, abs_tol=1e-12) and
                math.isclose(volume, 4 * math.pi / 3, abs_tol=1e-12), "Native sphere calculation differs")
    command(root, "native-dssp-version", [package_dir / "tools/dssp", "--version"])
    summary["native_execution"] = {"dalphaball_spheres": 4, "surface_area": 4 * math.pi,
                                   "volume": 4 * math.pi / 3, "dssp_version_executed": True}
    command(root, "check-load-cpu", [cli, "check", "--model", "partiflow", "--model", "autoencoder",
                                     "--model", "proteinmpnn", "--model", "solublempnn", "--load",
                                     "--model-dir", args.model_root])
    if args.input_file:
        for name, extra in (("validate", ["--model-dir", args.model_root]),
                            ("postprocess", ["--platform", "CPU", "--filter", "false"]),
                            ("cluster", ["--workers", "1"])):
            selected_input = Path(args.input_file)
            if name == "cluster":
                if not args.cluster_input:
                    continue
                selected_input = Path(args.cluster_input)
            command(root, "plan-" + name, [cli, name, selected_input, "--plan", *extra])
            summary["config_plans"].append("plan-" + name)
        shape = command(root, "native-shape-complementarity", [package_dir / "tools/sc", args.input_file,
                                                               "A", "L", "--json", "--no-parallel"])
        shape_values = json.loads(shape.stdout)
        require(math.isfinite(float(shape_values["sc"])), "Native shape calculation is not finite")
        summary["native_execution"]["shape_complementarity"] = shape_values
        command(root, "native-dssp-structure", [package_dir / "tools/dssp", "-i", args.input_file,
                                                "-o", root / "native-input.dssp"])
        require((root / "native-input.dssp").stat().st_size > 0, "Native DSSP produced an empty output")
        summary["native_execution"]["dssp_output_bytes"] = (root / "native-input.dssp").stat().st_size
    module_files = {}
    for name, module in list(sys.modules.items()):
        if name == "partiflow" or name.startswith("partiflow."):
            location = getattr(module, "__file__", None)
            if location is not None:
                require(Path(sys.prefix).resolve() in Path(location).resolve().parents,
                        f"Imported another checkout module: {name}: {location}")
                module_files[name] = location
    summary["module_files"] = module_files
    write_json(root / "runtime.json", summary)
    return summary


def config_lookup(args, root, checkout):
    require(Path(sys.prefix).resolve() == root / "wheel-env" and "PYTHONPATH" not in os.environ,
            "Use the isolated wheel interpreter with PYTHONPATH unset")
    require(Path.cwd().resolve() != checkout and checkout not in Path.cwd().resolve().parents,
            "Run installed configuration checks outside the checkout")
    import partiflow
    from partiflow.cli import main as cli_main
    from partiflow.download import data_root
    from partiflow.paths import checkout_root, resource_root
    from partiflow.training.config import load_config

    require(Path(sys.prefix).resolve() in Path(partiflow.__file__).resolve().parents and checkout_root() is None,
            "Configuration check imported another package")
    records = []
    for stage, filename in (("vae", "training_ae.yaml"), ("fm", "training.yaml")):
        path = resource_root() / "configs" / filename
        cfg = load_config(path, stage=stage, model_directory=args.model_root)
        dataset = cfg.get("dataset") or {}
        datamodule = dataset.get("datamodule") or cfg.get("datamodule") or {}
        expected = Path(datamodule["processed_dir"]).parent.resolve()
        require(Path(datamodule["split_csv"]).parent.resolve() == expected / "indexes", "Training layout differs")
        require(expected == Path.cwd() / "train_data/pepinter", "Bundled training paths must follow the external invocation directory")
        actual = data_root(Path(args.model_root), config=path)
        require(actual == expected, "Resource and training configuration paths differ")
        require(data_root(root / "different-model-root", config=path) == expected,
                "Model directory incorrectly relocated training data")
        explicit = root / "explicit-training-data"
        require(data_root(Path(args.model_root), config=path, data_dir=explicit) == explicit,
                "Explicit data directory was not preserved")
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            status = cli_main(["check", "--model", "partiflow", "--model-dir", args.model_root,
                               "--config", str(path)])
        report = json.loads(output.getvalue())
        require(Path(report["pepinter"]["path"]) == expected, "Installed check reported a different data directory")
        require(status == int(report["status"] != "complete"), "Resource check exit status differs from its report")
        records.append({"stage": stage, "config": str(path), "training_data_root": str(expected),
                        "reported_data_root": report["pepinter"]["path"],
                        "resource_status": report["pepinter"]["status"], "cli_returncode": status})
    heavy = sorted({"torch", "jax", "numpy", "pyrosetta", "openmm"} & set(sys.modules))
    require(not heavy, f"Configuration-only check imported model libraries: {heavy}")
    result = {"package": partiflow.__file__, "cwd": str(Path.cwd()), "checkout_root": None,
              "lookups": records, "model_libraries_imported": heavy}
    write_json(root / "config_lookup.json", result)
    return result


def postprocess(args, root, checkout):
    require(Path(sys.prefix).resolve() == root / "wheel-env" and "PYTHONPATH" not in os.environ,
            "Use the installed wheel interpreter with PYTHONPATH unset")
    import partiflow
    from partiflow.paths import checkout_root
    import yaml

    require(Path(sys.prefix).resolve() in Path(partiflow.__file__).resolve().parents and checkout_root() is None,
            "Postprocessing must import the isolated wheel")
    source = Path(args.input_file).resolve()
    task = root / "postprocess-headtail-full/task"
    require(not task.exists(), "Use a fresh postprocessing task directory")
    (task / ".state").mkdir(parents=True)
    for relative in ("sample1.pdb", "result.csv", ".state/design.json"):
        require((source / relative).is_file(), f"Known head-to-tail fixture is missing: {source / relative}")
        shutil.copy2(source / relative, task / relative)
    with (task / "result.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    require(len(rows) == 1, "Expected one known head-to-tail sample")
    rows[0].update(postprocess_status="", error="")
    with (task / "result.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    configuration = root / "postprocess-headtail-full/config.yaml"
    settings = {"method": "auto", "platform": "CPU", "use_ray": False, "workers": 1, "threads": 1,
                "relax_repeats": 1, "relax_max_iter": 200, "compute_buns": False, "criteria": [], "seed": 2026}
    configuration.write_text(yaml.safe_dump(settings, sort_keys=True))
    command(root, "postprocess-headtail-full", [root / "wheel-env/bin/partiflow", "postprocess", task,
                                          "--config", configuration, "--platform", "CPU", "--filter", "false"])
    report = json.loads((task / "postprocess/sample1.json").read_text())
    require(report["status"] == "success" and report["geometry_status"] == "complete" and not report["error"],
            "Installed-wheel postprocessing did not complete")
    resumed = bool(report.get("resumed", False))
    require(not resumed, "CPU postprocessing reused a previous result")
    require(report["postprocess"]["method"] == "rosetta_topology" and
            report["postprocess"]["relax_max_iter"] == 200 and
            report["postprocess"]["relax_repeats"] == 1 and
            any(entry["type"] == "head_to_tail" for entry in report["postprocess"]["applied_topology_entries"]),
            "Native head-to-tail topology was not applied")
    result = {"package": partiflow.__file__, "fixture": str(source), "task": str(task), "settings": settings,
              "status": report["status"], "geometry_status": report["geometry_status"],
              "sdk_profile": report["sdk_profile"], "postprocess": report["postprocess"], "resumed": resumed}
    write_json(root / "postprocess-full.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("operation", choices=("build", "inspect", "sdist-wheel", "audit", "install", "config-lookup", "probe", "postprocess"))
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--checkout", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--base-python", default=sys.executable)
    parser.add_argument("--wheel")
    parser.add_argument("--sdist")
    parser.add_argument("--model-root")
    parser.add_argument("--input-file")
    parser.add_argument("--cluster-input", help="Saved AF2 table or explicit postprocess records for a cluster plan")
    parser.add_argument("--tools-python")
    parser.add_argument("--repair-platform")
    args = parser.parse_args()
    checkout = Path(args.checkout).resolve()
    root = external_root(args.output_root, checkout)
    try:
        result = {"build": build, "inspect": inspect_distribution, "sdist-wheel": sdist_wheel, "audit": audit,
                  "install": install, "config-lookup": config_lookup,
                  "probe": probe, "postprocess": postprocess}[args.operation](args, root, checkout)
        print(json.dumps({"operation": args.operation, "result": "PASS", "output_root": str(root), **result}, default=str))
        return 0
    except Exception as exc:
        write_json(root / f"{args.operation}-failure.json", {"result": "FAIL", "error": str(exc)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
