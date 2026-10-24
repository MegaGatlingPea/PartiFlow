import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from partiflow.download import acquire


class DownloadTransferTests(unittest.TestCase):
    def test_truncated_transfer_does_not_publish_or_replace_existing_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "ckpts/autoencoder.ckpt"
            target.parent.mkdir()
            target.write_bytes(b"existing complete checkpoint")
            reader = io.BytesIO(b"partial")
            reader.headers = {"Content-Length": "100"}
            with patch("urllib.request.urlopen", return_value=reader):
                with self.assertRaisesRegex(ValueError, "Incomplete download"):
                    acquire(
                        "autoencoder", root, url="https://example.org/model", force=True
                    )
            self.assertEqual(target.read_bytes(), b"existing complete checkpoint")
            self.assertEqual(list(root.glob(".partiflow-import-*")), [])

    def test_complete_transfer_publishes_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contents = b"complete checkpoint container"
            reader = io.BytesIO(contents)
            reader.headers = {"Content-Length": str(len(contents))}
            with patch("urllib.request.urlopen", return_value=reader):
                result = acquire("autoencoder", root, url="https://example.org/model")
            self.assertEqual(result["status"], "downloaded")
            self.assertEqual((root / "ckpts/autoencoder.ckpt").read_bytes(), contents)
