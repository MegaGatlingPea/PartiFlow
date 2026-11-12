"""Package source code and small runtime resources independently of model files."""

from pathlib import Path
import shutil
import platform
from setuptools import Distribution, setup
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
    scaffold = path.suffix == ".pt" and (
        path.parts[:2] == ("configs", "scaffolds")
        or path.parts[:3] == ("_resources", "configs", "scaffolds")
    )
    return (
        not any(part in EXCLUDED_DIRS for part in path.parts)
        and (path.suffix not in EXCLUDED_SUFFIXES or scaffold)
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


class NativeDistribution(Distribution):
    def has_ext_modules(self):
        return True


class PlatformWheel(bdist_wheel):
    """The bundled executable makes this a platform-specific distribution."""

    def finalize_options(self):
        if platform.system() != "Linux" or platform.machine() not in {"x86_64", "AMD64"}:
            raise RuntimeError("The bundled DAlphaBall executable requires Linux x86_64")
        super().finalize_options()
        self.root_is_pure = False

    def get_tag(self):
        _, _, platform = super().get_tag()
        return "py3", "none", platform


setup(
    distclass=NativeDistribution,
    cmdclass={"build_py": BuildResources, "bdist_wheel": PlatformWheel},
    package_data={
        "partiflow": [
            "openfold/License",
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
