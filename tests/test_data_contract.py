"""PepInter split, archive and actual PyG/PT data contracts."""

from __future__ import annotations
import csv
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from partiflow.training.prepare_data import build_indexes, safe_extract_archive


class ArchiveAndIndexTests(unittest.TestCase):
    def _archive(self, directory, name, *, link=False):
        path = Path(directory) / "data.tar.xz"
        with tarfile.open(path, "w:xz") as archive:
            root = tarfile.TarInfo("pepinter")
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            member = tarfile.TarInfo(name)
            content = b"ATOM\n"
            if link:
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/outside.pdb"
                archive.addfile(member)
            else:
                member.size = len(content)
                archive.addfile(member, io.BytesIO(content))
        return path

    def test_safe_regular_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self._archive(directory, "pepinter/train/sample.pdb")
            root = Path(directory) / "raw"
            self.assertEqual(safe_extract_archive(archive, root), 1)
            self.assertEqual((root / "train/sample.pdb").read_bytes(), b"ATOM\n")

    def test_traversal_and_links_are_rejected(self):
        for name, link in (
            ("pepinter/../outside.pdb", False),
            ("/tmp/escape.pdb", False),
            ("pepinter/link.pdb", True),
        ):
            with tempfile.TemporaryDirectory() as directory:
                archive = self._archive(directory, name, link=link)
                with self.assertRaises(ValueError):
                    safe_extract_archive(archive, Path(directory) / "raw")

    def test_split_mapping_and_sorted_subset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, size in (("train", 70), ("valid", 18), ("test", 3)):
                target = root / "raw" / split
                target.mkdir(parents=True)
                for number in reversed(range(size)):
                    (target / f"{split}_{number:03d}.pdb").write_text("ATOM\n")
            full, subset, counts = build_indexes(root, verify_counts=False)
            self.assertEqual(counts, {"train": 70, "val": 18, "test": 3})
            with subset.open() as handle:
                selected = list(csv.DictReader(handle))
            self.assertEqual(len(selected), 80)
            self.assertEqual([row["split"] for row in selected].count("train"), 64)
            self.assertEqual([row["split"] for row in selected].count("val"), 16)
            self.assertEqual(selected[0]["pdbid"], "train_000")
            self.assertEqual(selected[63]["pdbid"], "train_063")
            self.assertTrue(all(row["split"] != "test" for row in selected))
            self.assertTrue((subset.parent / selected[0]["structure_path"]).is_file())
            with full.open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 91)

    def test_count_mismatch_does_not_publish_index(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "split counts"):
                build_indexes(Path(directory))
            self.assertFalse((Path(directory) / "indexes/split.csv").exists())


HAS_GRAPHS = importlib.util.find_spec("torch_geometric") is not None


@unittest.skipUnless(HAS_GRAPHS, "Torch and PyG are required")
class GraphDataTests(unittest.TestCase):
    def test_relative_pt_paths_and_valid_alias(self):
        import torch
        from torch_geometric.data import Data
        from partiflow.datasets.pdb_data import PepLightningDataModule

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graphs").mkdir()
            sample = Data(
                coords=torch.zeros(2, 37, 3),
                coord_mask=torch.ones(2, 37, dtype=torch.bool),
                id="one",
            )
            torch.save(sample, root / "graphs/one.pt")
            index = root / "split.csv"
            index.write_text(
                "pdbid,split,tag,processed_path\none,valid,pepinter,graphs/one.pt\n"
            )
            module = PepLightningDataModule(
                split_csv=str(index),
                processed_dir=str(root / "processed"),
                num_workers=0,
                validation_split="valid",
            )
            dataset = module.val_dataset()
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0].id, "one")
            self.assertEqual(dataset[0].coords.shape, (2, 37, 3))
            self.assertEqual(dataset[0].dataset_tag, "pepinter")
            with self.assertRaisesRegex(ValueError, "test"):
                module.test_dataset()

    def test_explicit_preparation_records_failure_without_replacement(self):
        from partiflow.datasets.pdb_data import PepLightningDataModule

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "split.csv"
            index.write_text(
                "pdbid,split,structure_path,processed_path\nmissing,train,missing.pdb,graphs/missing.pt\n"
            )
            module = PepLightningDataModule(
                split_csv=str(index), processed_dir=str(root / "graphs"), num_workers=0
            )
            module.prepare_data()
            self.assertFalse((root / "preparation_failures.csv").exists())
            with self.assertRaisesRegex(RuntimeError, "1 selected"):
                module.prepare_data(skip=False)
            with (root / "preparation_failures.csv").open() as handle:
                failures = list(csv.DictReader(handle))
            self.assertEqual(len(failures), 1)
            self.assertEqual(failures[0]["pdbid"], "missing")
            self.assertIn("FileNotFoundError", failures[0]["error"])
            self.assertFalse((root / "graphs/missing.pt").exists())

    def test_real_prepared_pepinter_subset(self):
        import torch

        root = Path(__file__).resolve().parents[1]
        index = root / "train_data/pepinter/indexes/acceptance.csv"
        if not index.exists():
            self.skipTest("The fixed PepInter subset has not been prepared")
        with index.open() as handle:
            rows = list(csv.DictReader(handle))
        paths = [(index.parent / row["processed_path"]).resolve() for row in rows]
        if not all(path.is_file() for path in paths):
            self.skipTest("The fixed PepInter atom37 graphs have not been prepared")
        self.assertEqual(len(rows), 80)
        self.assertEqual(sum(row["split"] == "train" for row in rows), 64)
        self.assertEqual(sum(row["split"] == "val" for row in rows), 16)
        self.assertFalse(any(row["split"] == "test" for row in rows))
        for row, path in zip(rows, paths):
            graph = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(graph.id, row["pdbid"])
            self.assertEqual(tuple(graph.coords.shape[-2:]), (37, 3))
            self.assertEqual(graph.coord_mask.shape, graph.coords.shape[:-1])
            self.assertTrue(torch.isfinite(graph.coords).all())
            self.assertTrue((graph.is_peptide == 1).any())
            self.assertTrue((graph.is_peptide == 0).any())
        import hydra
        from partiflow.training.config import load_config

        for stage in ("vae", "fm"):
            config = load_config(
                root / "configs" / f"training_{stage}_acceptance.yaml",
                ["hardware.accelerator=cpu"],
                stage=stage,
            )
            module = hydra.utils.instantiate(config.dataset.datamodule)
            module.setup("fit")
            self.assertEqual(len(module.train_dataset()), 64)
            self.assertEqual(len(module.val_dataset()), 16)
            train_batch = next(iter(module.train_dataloader()))
            val_batch = next(iter(module.val_dataloader()))
            for batch in (train_batch, val_batch):
                self.assertEqual(tuple(batch["coords"].shape[-2:]), (37, 3))
                self.assertEqual(batch["coords"].shape[0], 1)
                self.assertIn("coords", batch["mask_dict"])
                self.assertTrue(torch.isfinite(batch["coords"]).all())


if __name__ == "__main__":
    unittest.main()
