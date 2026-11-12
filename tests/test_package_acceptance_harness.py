"""Focused guards for the distribution acceptance helper."""

import csv
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from acceptance import package


class PackageAcceptanceHarnessTests(unittest.TestCase):
    def test_only_scaffold_tensor_libraries_are_allowed(self):
        for path in ("configs/scaffolds/a.pt", "partiflow/_resources/configs/scaffolds/a.pt",
                     "partiflow/postprocess/bin/DAlphaBall.gcc", "partiflow/tools/dssp"):
            self.assertTrue(package.allowed(path), path)
        for path in ("weights/a.pt", "configs/other/a.pt", "other/scaffolds/a.pt", "runs/a.json",
                     "partiflow/__pycache__/a.pyc", "partiflow/a.ckpt", "partiflow.egg-info/PKG-INFO"):
            self.assertFalse(package.allowed(path), path)

    def test_runtime_environment_removes_import_path_and_hides_gpu(self):
        from unittest.mock import patch
        with patch.dict(package.os.environ, {"PYTHONPATH": "/some/checkout", "CUDA_VISIBLE_DEVICES": "0"}):
            env = package.clean_environment()
        self.assertNotIn("PYTHONPATH", env)
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(env["JAX_PLATFORMS"], "cpu")
        self.assertEqual(env["OMP_NUM_THREADS"], "4")

    def test_sdist_backend_metadata_is_allowed_without_allowing_model_files(self):
        self.assertTrue(package.allowed_sdist("partiflow.egg-info/SOURCES.txt"))
        self.assertTrue(package.allowed_sdist("partiflow.egg-info/PKG-INFO"))
        self.assertFalse(package.allowed_sdist("partiflow.egg-info/weights.pt"))
        self.assertFalse(package.allowed_sdist("ckpts/partiflow.ckpt"))

    def test_sdist_requires_current_acceptance_helpers_beside_top_level_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary)
            helper = checkout / "tests/acceptance/one.py"
            helper.parent.mkdir(parents=True)
            helper.write_bytes(b"value = 1\n")
            for payload in (None, b"value = 2\n", helper.read_bytes()):
                data = io.BytesIO()
                with tarfile.open(fileobj=data, mode="w") as archive:
                    files = {"tests/test_one.py": b"from acceptance import one\n"}
                    if payload is not None:
                        files["tests/acceptance/one.py"] = payload
                    for name, content in files.items():
                        item = tarfile.TarInfo(name)
                        item.size = len(content)
                        archive.addfile(item, io.BytesIO(content))
                data.seek(0)
                with tarfile.open(fileobj=data) as archive:
                    members = {item.name: item for item in archive.getmembers()}
                    if payload == helper.read_bytes():
                        self.assertEqual(package.check_sdist_helpers(archive, members, checkout),
                                         ["tests/acceptance/one.py"])
                    else:
                        with self.assertRaises(package.AcceptanceError):
                            package.check_sdist_helpers(archive, members, checkout)

    def test_failed_subprocess_is_recorded_and_never_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(package.AcceptanceError):
                package.command(root, "failure", [sys.executable, "-c", "raise SystemExit(7)"])
            row = json.loads((root / "commands/failure.json").read_text())
            self.assertEqual(row["returncode"], 7)
            self.assertIsNone(row["environment"]["PYTHONPATH"])

    def test_record_requires_complete_members_and_correct_sizes(self):
        def make(rows, directory=False):
            data = io.BytesIO()
            stream = io.StringIO()
            csv.writer(stream).writerows(rows)
            with zipfile.ZipFile(data, "w") as archive:
                if directory:
                    archive.writestr("partiflow/", b"")
                archive.writestr("partiflow/__init__.py", b"a")
                archive.writestr("partiflow-0.1.0.dist-info/RECORD", stream.getvalue())
            data.seek(0)
            return zipfile.ZipFile(data)

        record = "partiflow-0.1.0.dist-info/RECORD"
        for rows in ([ [record, "", ""] ],
                     [["partiflow/__init__.py", "tool-generated", "2"], [record, "", ""]],
                     [["partiflow/__init__.py", "", "1"], [record, "", ""]]):
            with self.subTest(rows=rows), make(rows) as archive, self.assertRaises(package.AcceptanceError):
                package.check_record(archive)
        rows = [["partiflow/__init__.py", "tool-generated", "1"], [record, "", ""]]
        with make(rows, directory=True) as archive:
            self.assertEqual(package.check_record(archive), (record, 2))

    def test_notice_copy_requires_sources_and_includes_referenced_license_text(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(package.AcceptanceError):
                package.add_library_notices(root / "output", root / "system")
            for name in ("libgmp10", "libgfortran5"):
                path = root / "system/doc" / name / "copyright"
                path.parent.mkdir(parents=True)
                path.write_text("See /usr/share/common-licenses/GPL-3.\n")
            common = root / "system/common-licenses/GPL-3"
            with self.assertRaises(package.AcceptanceError):
                package.add_library_notices(root / "output", root / "system")
            common.parent.mkdir(parents=True)
            common.write_text("Exact license text\n")
            copied = package.add_library_notices(root / "output", root / "system")
            self.assertEqual(len(copied), 3)
            self.assertEqual((root / "output/common-licenses/GPL-3").read_bytes(), common.read_bytes())

    def test_outputs_reject_checkout_and_parent_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkout = Path(temporary) / "checkout"
            for output in (checkout, checkout / "runs", Path(temporary), Path("/")):
                with self.subTest(output=output), self.assertRaises(package.AcceptanceError):
                    package.external_root(output, checkout)


if __name__ == "__main__":
    unittest.main()
