from pathlib import Path
import csv
import io
import subprocess
import sys
import tempfile
import unittest
import zipfile


class WheelRecordTests(unittest.TestCase):
    def test_distribution_records_have_only_paths_and_sizes(self):
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
                self.assertTrue(all(len(row) == 3 and not row[1] for row in rows))
                self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
                for name, _, size in rows:
                    if name != record:
                        self.assertEqual(int(size), len(archive.read(name)))
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
