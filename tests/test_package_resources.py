import importlib.util
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import partiflow

ROOT = Path(partiflow.__file__).resolve().parent.parent


class PackageResourceTests(unittest.TestCase):
    def test_document_links_resolve(self):
        documents = [ROOT / "README.md", *ROOT.glob("docs/**/*.md")]
        missing = []
        for document in documents:
            for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
                if target.startswith(("http:", "https:", "#", "mailto:")):
                    continue
                target = target.split("#", 1)[0]
                if target and not (document.parent / target).exists():
                    missing.append(f"{document.relative_to(ROOT)} -> {target}")
        self.assertEqual(missing, [])

    @unittest.skipUnless(
        importlib.util.find_spec("setuptools"),
        "setuptools build tooling is not installed",
    )
    def test_build_layout_contains_configs_chemistry_and_executable(self):
        with tempfile.TemporaryDirectory(prefix="partiflow-package-test-") as temporary:
            checkout = Path(temporary) / "checkout"
            checkout.mkdir()
            for name in ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md"):
                shutil.copy2(ROOT / name, checkout / name)
            for name in ("partiflow", "configs", "pdbs"):
                shutil.copytree(
                    ROOT / name,
                    checkout / name,
                    ignore=shutil.ignore_patterns(
                        "__pycache__", "*.pyc", "*.ckpt", "*.pt", "*.pth", "*.npz"
                    ),
                )
            output = Path(temporary) / "lib"
            completed = subprocess.run(
                [sys.executable, "setup.py", "build_py", "--build-lib", str(output)],
                cwd=checkout,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr[-4000:])
            package = output / "partiflow"
            for relative in (
                "_resources/configs/design.yaml",
                "_resources/configs/tasks/example.yaml",
                "utils/relax_assets/topology.xml",
                "postprocess/rosetta_ncaa/support_matrix.csv",
                "postprocess/rosetta_ncaa/params/Z28.params",
                "postprocess/bin/DAlphaBall.gcc",
                "_vendor/proteinmpnn/protein_mpnn_utils.py",
            ):
                self.assertTrue((package / relative).is_file(), relative)
            for source in (ROOT / "pdbs").rglob("*.pdb"):
                self.assertTrue(
                    (
                        package
                        / "_resources"
                        / "pdbs"
                        / source.relative_to(ROOT / "pdbs")
                    ).is_file()
                )
            self.assertTrue(
                (package / "postprocess/bin/DAlphaBall.gcc").stat().st_mode
                & stat.S_IXUSR
            )
            forbidden = {".ckpt", ".pt", ".pth", ".npz", ".pyc"}
            self.assertFalse(
                [
                    str(path)
                    for path in output.rglob("*")
                    if path.is_file() and path.suffix in forbidden
                ]
            )
            self.assertEqual(
                [path.name for path in output.iterdir() if path.is_dir()], ["partiflow"]
            )


if __name__ == "__main__":
    unittest.main()
