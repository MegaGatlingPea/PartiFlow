import contextlib
import io
import stat
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from partiflow.download import TARGETS, acquire, extract_archive, resource_status, run


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = self.base / "models"
        self.source = self.base / "source"
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_local_single_checkpoint_import_and_force(self):
        source = self.source / "partiflow.ckpt"
        source.write_bytes(b"original")
        self.assertEqual(
            acquire("partiflow", self.root, from_local=source)["status"], "imported"
        )
        source.write_bytes(b"replacement")
        self.assertEqual(
            acquire("partiflow", self.root, from_local=source)["status"], "skipped"
        )
        self.assertEqual((self.root / "ckpts" / source.name).read_bytes(), b"original")
        acquire("partiflow", self.root, from_local=source, force=True)
        self.assertEqual(
            (self.root / "ckpts" / source.name).read_bytes(), b"replacement"
        )

    def test_checkpoint_zip_container_is_copied_as_checkpoint(self):
        source = self.source / "partiflow.ckpt"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("archive/data.pkl", b"checkpoint")
        acquire("partiflow", self.root, from_local=source)
        self.assertEqual(
            (self.root / "ckpts" / source.name).read_bytes(), source.read_bytes()
        )

    def test_local_af2_archive_imports_all_configured_arrays(self):
        source = self.source / "af2.tar"
        with tarfile.open(source, "w") as archive:
            for asset in TARGETS["af2"]:
                payload = b"array"
                info = tarfile.TarInfo(Path(asset.relative).name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
        acquire("af2", self.root, from_local=source)
        self.assertEqual(resource_status("af2", self.root)["status"], "present")

    def test_partial_af2_source_does_not_publish_partial_import(self):
        (self.source / "params_model_1.npz").write_bytes(b"array")
        with self.assertRaises(FileNotFoundError):
            acquire("af2", self.root, from_local=self.source)
        self.assertFalse((self.root / "af2" / "params_model_1.npz").exists())

    def test_tar_traversal_is_rejected_before_writing(self):
        source = self.source / "unsafe.tar"
        with tarfile.open(source, "w") as archive:
            good = tarfile.TarInfo("good.txt")
            good.size = 1
            archive.addfile(good, io.BytesIO(b"x"))
            bad = tarfile.TarInfo("../outside.txt")
            bad.size = 1
            archive.addfile(bad, io.BytesIO(b"x"))
        destination = self.base / "extract"
        with self.assertRaises(ValueError):
            extract_archive(source, destination)
        self.assertFalse(destination.exists())
        self.assertFalse((self.base / "outside.txt").exists())

    def test_tar_symlink_is_rejected(self):
        source = self.source / "unsafe.tar"
        with tarfile.open(source, "w") as archive:
            info = tarfile.TarInfo("link")
            info.type = tarfile.SYMTYPE
            info.linkname = "/tmp/outside"
            archive.addfile(info)
        with self.assertRaises(ValueError):
            extract_archive(source, self.base / "extract")

    def test_zip_windows_traversal_and_symlink_are_rejected(self):
        for name, mode in [
            ("..\\outside", stat.S_IFREG | 0o644),
            ("link", stat.S_IFLNK | 0o777),
        ]:
            source = self.source / "unsafe.zip"
            with zipfile.ZipFile(source, "w") as archive:
                item = zipfile.ZipInfo(name)
                item.create_system = 3
                item.external_attr = mode << 16
                archive.writestr(item, b"/tmp/outside")
            with self.assertRaises(ValueError):
                extract_archive(source, self.base / "extract")

    def test_custom_file_url_uses_real_streaming_import(self):
        source = self.source / "weights.ckpt"
        source.write_bytes(b"local-url")
        acquire("partiflow", self.root, url=source.as_uri())
        self.assertEqual(
            (self.root / "ckpts" / "partiflow.ckpt").read_bytes(), b"local-url"
        )

    def test_custom_archive_url_preserves_archive_suffix(self):
        source = self.source / "bundle.tar.gz"
        with tarfile.open(source, "w:gz") as archive:
            info = tarfile.TarInfo("partiflow.ckpt")
            info.size = 7
            archive.addfile(info, io.BytesIO(b"weights"))
        acquire("partiflow", self.root, url=source.as_uri())
        self.assertEqual(
            (self.root / "ckpts" / "partiflow.ckpt").read_bytes(), b"weights"
        )

    def test_unconfigured_target_is_explicit_and_read_only(self):
        result = acquire("partiflow", self.root)
        self.assertEqual(result["status"], "unconfigured")
        self.assertFalse(self.root.exists())

    def test_list_is_read_only(self):
        args = SimpleNamespace(targets=[], model_dir=self.root, list=True, all=False)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(run(args), 0)
        self.assertFalse(self.root.exists())

    def test_pepinter_local_data_import(self):
        (self.source / "indexes").mkdir()
        (self.source / "processed").mkdir()
        (self.source / "indexes" / "split.csv").write_text("id,split\na,train\n")
        (self.source / "processed" / "a.pt").write_bytes(b"processed")
        destination = self.base / "data"
        result = acquire(
            "pepinter", self.root, from_local=self.source, data_dir=destination
        )
        self.assertEqual(result["status"], "imported")
        self.assertTrue((destination / "processed" / "a.pt").is_file())

    def test_missing_pepinter_payload_fails(self):
        with self.assertRaises(ValueError):
            acquire("pepinter", self.root, from_local=self.source)


if __name__ == "__main__":
    unittest.main()
