"""Package source code and small runtime resources independently of model files."""

from pathlib import Path
import shutil
import csv
import io
import os
import tempfile
import zipfile
from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.bdist_wheel import bdist_wheel

ROOT = Path(__file__).parent
EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "ckpts",
    "checkpoint",
    "checkpoints",
    "train_data",
    "inference",
    "runs",
    "weights",
}
EXCLUDED_SUFFIXES = {
    ".pyc",
    ".ckpt",
    ".pt",
    ".pth",
    ".npz",
    ".safetensors",
    ".bin",
    ".msgpack",
    ".h5",
    ".hdf5",
    ".npy",
    ".pkl",
}


def allowed(path):
    return (
        not any(part in EXCLUDED_DIRS for part in path.parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    )


class BuildResources(build_py):
    def run(self):
        super().run()
        destination = Path(self.build_lib) / "partiflow" / "_resources"
        for name in ("configs", "pdbs"):
            source = ROOT / name
            if source.is_dir():
                for path in source.rglob("*"):
                    if path.is_file() and allowed(path.relative_to(ROOT)):
                        output = destination / name / path.relative_to(source)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(path, output)
        # Keep generated distributions independent of any local model installations.
        package = Path(self.build_lib) / "partiflow"
        for path in sorted(package.rglob("*"), reverse=True):
            if path.is_file() and not allowed(path.relative_to(package)):
                path.unlink()
        executable = package / "postprocess" / "bin" / "DAlphaBall.gcc"
        if executable.is_file():
            executable.chmod(executable.stat().st_mode | 0o111)


class DistributionWheel(bdist_wheel):
    """Keep distribution records limited to filenames and actual sizes."""

    def run(self):
        super().run()
        for command, _, filename in self.distribution.dist_files:
            if command != "bdist_wheel":
                continue
            wheel = Path(filename)
            fd, name = tempfile.mkstemp(dir=wheel.parent, suffix=".whl.tmp")
            os.close(fd)
            temporary = Path(name)
            try:
                with (
                    zipfile.ZipFile(wheel) as source,
                    zipfile.ZipFile(temporary, "w") as output,
                ):
                    for info in source.infolist():
                        data = source.read(info.filename)
                        if info.filename.endswith(".dist-info/RECORD"):
                            rows = list(csv.reader(io.StringIO(data.decode())))
                            for row in rows:
                                row[1] = ""
                            text = io.StringIO()
                            csv.writer(text, lineterminator="\n").writerows(rows)
                            data = text.getvalue().encode()
                        output.writestr(info, data)
                temporary.chmod(wheel.stat().st_mode & 0o777)
                os.replace(temporary, wheel)
            finally:
                temporary.unlink(missing_ok=True)


setup(
    cmdclass={"build_py": BuildResources, "bdist_wheel": DistributionWheel},
    package_data={
        "partiflow": [
            "utils/relax_assets/*",
            "postprocess/rosetta_ncaa/*.csv",
            "postprocess/rosetta_ncaa/*.yaml",
            "postprocess/rosetta_ncaa/params/*.params",
            "postprocess/bin/*",
            "tools/*",
            "_vendor/**/*",
            "_resources/**/*",
        ]
    },
)
