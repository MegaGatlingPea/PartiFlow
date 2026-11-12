from pathlib import Path
import csv
import io
import subprocess
import sys
import tempfile
import unittest
import zipfile


class WheelRecordTests(unittest.TestCase):
    def test_standard_distribution_records_and_platform(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            command = [
                sys.executable,
                "setup.py",
                "bdist_wheel",
                "--dist-dir",
                temporary,
            ]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr[-3000:])
            (wheel,) = Path(temporary).glob("*.whl")
            with zipfile.ZipFile(wheel) as archive:
                (record,) = (
                    name
                    for name in archive.namelist()
                    if name.endswith(".dist-info/RECORD")
                )
                rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
                self.assertTrue(rows)
                self.assertTrue(all(len(row) == 3 for row in rows))
                self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
                for name, check, size in rows:
                    if name != record:
                        self.assertTrue(check, name)
                        self.assertEqual(int(size), len(archive.read(name)))
                metadata = archive.read(record.replace("RECORD", "WHEEL")).decode()
                self.assertIn("Root-Is-Purelib: false", metadata)
                self.assertNotIn("Tag: py3-none-any", metadata)
                self.assertIn("Tag: py3-none-linux_x86_64", metadata)
                import yaml

                for name in archive.namelist():
                    if "/_resources/configs/tasks/" not in name or not name.endswith(
                        ".yaml"
                    ):
                        continue
                    tasks = yaml.safe_load(archive.read(name))
                    for task in tasks.values():
                        if isinstance(task, dict) and task.get("pdb_path"):
                            relative = Path(task["pdb_path"])
                            self.assertFalse(relative.is_absolute(), name)
                            resource = "partiflow/_resources/" + relative.as_posix()
                            self.assertIn(
                                resource, archive.namelist(), (name, resource)
                            )
                binary = next(
                    info
                    for info in archive.infolist()
                    if info.filename.endswith("/DAlphaBall.gcc")
                )
                self.assertTrue((binary.external_attr >> 16) & 0o111)
            valid = subprocess.run(
                [sys.executable, "-m", "wheel", "unpack", str(wheel), "-d", str(Path(temporary) / "unpacked")],
                capture_output=True, text=True,
            )
            self.assertEqual(valid.returncode, 0, valid.stderr)
            broken_dir = Path(temporary) / "broken"
            broken_dir.mkdir()
            broken = broken_dir / wheel.name
            with zipfile.ZipFile(wheel) as source, zipfile.ZipFile(broken, "w") as target:
                for item in source.infolist():
                    data = source.read(item.filename)
                    if item.filename == "partiflow/__init__.py":
                        data += b"\n# changed after packaging\n"
                    target.writestr(item, data)
            invalid = subprocess.run(
                [sys.executable, "-m", "wheel", "unpack", str(broken), "-d", str(broken_dir / "unpacked")],
                capture_output=True, text=True,
            )
            self.assertNotEqual(invalid.returncode, 0)
