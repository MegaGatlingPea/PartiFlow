"""Failure-sensitive helpers for actual public resource acceptance runs."""

from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from acceptance import resources


class ResourceAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_file_comparison_reads_content_and_end(self):
        source, destination = self.root / "source", self.root / "destination"
        source.write_bytes(b"0123456789")
        destination.write_bytes(source.read_bytes())
        self.assertEqual(resources.compare_files(source, destination, block_size=3)["bytes"], 10)
        for changed in (b"0123456x89", b"0123456789extra", b""):
            destination.write_bytes(changed)
            with self.subTest(changed=changed), self.assertRaises(AssertionError):
                resources.compare_files(source, destination, block_size=3)

    def test_machine_json_rejects_console_text(self):
        self.assertEqual(resources.parse_output('{"target":"a"}\n{"target":"b"}\n', lines=True),
                         [dict(target="a"), dict(target="b")])
        self.assertEqual(resources.parse_output('{\n "status": "complete"\n}', lines=False), dict(status="complete"))
        for value in ('Progress\n{"status":"complete"}', '\u001b[32m{"status":"complete"}', '[{"status":"complete"}]'):
            with self.subTest(value=value), self.assertRaises((AssertionError, json.JSONDecodeError)):
                resources.parse_output(value, lines=False)

    def write_index(self, rows):
        path = self.root / "indexes" / "split.csv"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("pdbid", "split"))
            writer.writeheader()
            writer.writerows(rows)

    def test_index_rejects_duplicate_ids_and_unmapped_split(self):
        rows = [dict(pdbid=f"sample{index}", split=split)
                for index, split in enumerate(("train", "val", "test"))]
        self.write_index(rows)
        self.assertEqual(resources.indexed_rows(self.root), rows)
        for changed in ([*rows, dict(pdbid="sample0", split="test")], [dict(pdbid="x", split="valid")]):
            self.write_index(changed)
            with self.assertRaises(AssertionError):
                resources.indexed_rows(self.root)

    def test_mpnn_comparison_checks_every_tensor_and_metadata(self):
        import torch
        source, destination = self.root / "source.pt", self.root / "destination.pt"
        checkpoint = dict(num_edges=48, noise_level=0.2,
                          model_state_dict={f"layer{index}": torch.tensor([float(index)]) for index in range(118)})
        torch.save(checkpoint, source)
        torch.save(checkpoint, destination)
        self.assertEqual(resources.compare_mpnn(source, destination)["tensors"], 118)
        changed = deepcopy(checkpoint)
        changed["model_state_dict"]["layer117"][0] += 1
        torch.save(changed, destination)
        with self.assertRaises(AssertionError):
            resources.compare_mpnn(source, destination)
        changed = deepcopy(checkpoint)
        changed["noise_level"] = 0.3
        torch.save(changed, destination)
        with self.assertRaises(AssertionError):
            resources.compare_mpnn(source, destination)


if __name__ == "__main__":
    unittest.main()
