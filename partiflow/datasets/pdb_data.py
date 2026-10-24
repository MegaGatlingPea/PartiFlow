import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from tqdm import tqdm
from loguru import logger
import pandas as pd
try:
    from graphein.protein.tensor.io import protein_to_pyg
    _PROTEIN_TO_PYG_ERROR = None
except ImportError as error:
    protein_to_pyg = None
    _PROTEIN_TO_PYG_ERROR = error

from partiflow.openfold.np.residue_constants import resname_to_idx
from partiflow.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from partiflow.datasets.base_data import BaseLightningDataModule
from partiflow.datasets.samplers import TaggedRatioDistributedSampler


class PepDataset(Dataset):
    """Lightweight dataset that reads processed protein graphs from disk."""

    def __init__(
        self,
        pdb_codes: List[str],
        processed_dir: str,
        transform: Optional[Callable] = None,
        in_memory: bool = False,
        tags: Optional[Sequence[str]] = None,
        processed_paths: Optional[Sequence[str]] = None,
        sample_weights: Optional[Sequence[float]] = None,
        sample_loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:

        self.pdb_codes = pdb_codes
        self.processed_dir = Path(processed_dir)
        self.transform = transform
        self.in_memory = in_memory
        self.file_names = [f"{pdb_id}.pt" for pdb_id in pdb_codes]
        self.processed_paths = [Path(path) for path in processed_paths] if processed_paths is not None else None
        self.tags = [str(tag) for tag in tags] if tags is not None else None
        self.sample_weights = [float(weight) for weight in sample_weights] if sample_weights is not None else None
        self.sample_loss_weights = {
            str(key): float(value)
            for key, value in (sample_loss_weights or {}).items()
        }
        if self.processed_paths is not None and len(self.processed_paths) != len(self.file_names):
            raise ValueError(
                f"processed_paths length {len(self.processed_paths)} must match dataset length {len(self.file_names)}"
            )
        if self.tags is not None and len(self.tags) != len(self.file_names):
            raise ValueError(
                f"tags length {len(self.tags)} must match dataset length {len(self.file_names)}"
            )
        if self.sample_weights is not None and len(self.sample_weights) != len(self.file_names):
            raise ValueError(
                f"sample_weights length {len(self.sample_weights)} must match dataset length {len(self.file_names)}"
            )

        if self.in_memory:
            logger.info("Reading processed graphs into memory")
            self._memory: List[Dict[str, Data]] = [
                self._load_from_disk(idx) for idx in tqdm(range(len(self.file_names)))
            ]
        else:
            self._memory = []

    def __len__(self) -> int:
        return len(self.file_names)

    def _apply_transform(self, sample: Dict[str, Data] | Data):
        if self.transform is None:
            return sample
        return self.transform(sample)

    def _load_from_disk(self, idx: int):
        path = (
            self.processed_paths[idx]
            if self.processed_paths is not None
            else self.processed_dir / self.file_names[idx]
        )
        try:
            sample = torch.load(path, map_location="cpu", weights_only=False)
            return self._apply_transform(sample)
        except (EOFError, RuntimeError) as e:
            logger.error(f"Failed to load {path}: {e}. File may be corrupted.")
            raise

    def _attach_sample_metadata(self, sample, idx: int):
        tag = self.tags[idx] if self.tags is not None else "default"
        weight = (
            self.sample_weights[idx]
            if self.sample_weights is not None
            else self.sample_loss_weights.get(tag, self.sample_loss_weights.get("*", 1.0))
        )
        weight_tensor = torch.tensor([weight], dtype=torch.float32)

        if isinstance(sample, Data):
            sample.dataset_tag = tag
            sample.sample_loss_weight = weight_tensor
            return sample

        if isinstance(sample, dict):
            sample["dataset_tag"] = tag
            sample["sample_loss_weight"] = weight_tensor
        return sample

    def __getitem__(self, idx: int):
        if self.in_memory:
            return self._attach_sample_metadata(self._memory[idx], idx)
        return self._attach_sample_metadata(self._load_from_disk(idx), idx)


class PepLightningDataModule(BaseLightningDataModule):
    """Minimal LightningDataModule that materialises processed graphs from a structure folder."""

    def __init__(
        self,
        split_csv: str,
        structure_dir: Optional[str] = None,
        processed_dir: Optional[str] = None,
        pep_chain: str = "L",
        fill_value_coords: float = 1e-5,
        overwrite: bool = False,
        in_memory: bool = False,
        batch_padding: bool = True,
        transforms: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = False,
        pad_max_size: Optional[int] = None,
        truncate_to_pad_max_size: bool = False,
        train_sampler: Optional[Dict] = None,
        sample_loss_weights: Optional[Dict[str, float]] = None,
        use_processed_path_column: bool = True,
        validation_split: Optional[str] = None,
        validation_tags: Optional[Sequence[str]] = None,
        validation_max_samples: Optional[int] = None,
    ) -> None:
        super().__init__(
            batch_padding=batch_padding,
            transforms=transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            pad_max_size=pad_max_size,
            truncate_to_pad_max_size=truncate_to_pad_max_size,
        )
        assert structure_dir is not None or processed_dir is not None, (
            "At least one of structure_dir or processed_dir must be provided."
        )
        self.split_csv = Path(split_csv)
        self.structure_dir = (
            Path(structure_dir)
            if structure_dir is not None
            else self.split_csv.parent / "structures"
        )
        self.processed_dir = (
            Path(processed_dir)
            if processed_dir is not None
            else self.structure_dir.parent / "processed"
        )
        self.pep_chain = pep_chain

        self.fill_value_coords = fill_value_coords
        self.overwrite = overwrite
        self.in_memory = in_memory
        self.train_sampler_cfg = train_sampler
        self.sample_loss_weights = sample_loss_weights or {}
        self.use_processed_path_column = bool(use_processed_path_column)
        self.validation_split = "val" if validation_split == "valid" else validation_split
        self.validation_tags = [str(tag) for tag in validation_tags] if validation_tags is not None else None
        self.validation_max_samples = (
            None if validation_max_samples is None else int(validation_max_samples)
        )
        self.structure_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self._split_df: Optional[pd.DataFrame] = None

    def _load_split_df(self) -> pd.DataFrame:
        if self._split_df is None:
            self._split_df = pd.read_csv(self.split_csv)
            missing = {"pdbid", "split"} - set(self._split_df.columns)
            if missing:
                raise ValueError(f"Split CSV is missing columns: {sorted(missing)}")
            self._split_df["split"] = self._split_df["split"].replace({"valid": "val"})
            if self._split_df["pdbid"].duplicated().any():
                raise ValueError(f"Duplicate sample identifiers in {self.split_csv}")
        return self._split_df

    def _index_path(self, value) -> Path:
        path = Path(str(value))
        return path if path.is_absolute() else (self.split_csv.parent / path).resolve()

    def prepare_data(self, skip=True, failure_csv=None) -> None:
        """Prepare raw structures only when explicitly called with skip=False.

        Lightning's automatic prepare_data hook keeps the already processed
        training path unchanged. The preparation command calls this explicitly.
        """
        if skip:
            return
        df = self._load_split_df()
        failures = []
        for row in tqdm(df.to_dict("records"), desc="Processing structures to graphs"):
            pdb_id = str(row["pdbid"])
            target_path = self._index_path(row["processed_path"]) if "processed_path" in row else self.processed_dir / f"{pdb_id}.pt"
            source_path = self._index_path(row["structure_path"]) if "structure_path" in row else self.structure_dir / f"{pdb_id}.pdb"
            if target_path.exists() and not self.overwrite:
                continue
            temporary = target_path.with_suffix(".pt.writing")
            try:
                sample = self._process_single_structure(source_path, pdb_id)
                target_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(sample, temporary)
                temporary.replace(target_path)
            except Exception as error:
                failures.append({"pdbid": pdb_id, "structure_path": str(source_path), "error": f"{type(error).__name__}: {error}"})
                logger.error(f"Preparation failed for {pdb_id}: {error}")
            finally:
                temporary.unlink(missing_ok=True)
        failure_path = Path(failure_csv) if failure_csv else self.split_csv.with_name("preparation_failures.csv")
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["pdbid", "structure_path", "error"])
            writer.writeheader()
            writer.writerows(failures)
        if failures:
            raise RuntimeError(f"Failed to prepare {len(failures)} selected structures; see {failure_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        self._load_split_df()
        assert stage != "predict", "PepLightningDataModule does not support 'predict' stage"
        super().setup(stage)

    def _process_single_structure(self, pdb_path: Path, sample_id: str) -> Data:
        if not pdb_path.exists():
            raise FileNotFoundError(f"Structure file not found: {pdb_path}")
        if protein_to_pyg is None:
            raise RuntimeError(
                f"Graphein protein_to_pyg is required for explicit raw-data preparation: {_PROTEIN_TO_PYG_ERROR}"
            ) from _PROTEIN_TO_PYG_ERROR
        graph = protein_to_pyg(path=str(pdb_path), fill_value_coords=self.fill_value_coords)
        coord_mask = graph.coords != self.fill_value_coords
        graph.coord_mask = coord_mask[..., 0]
        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        residue_indices = []
        fallback_idx = resname_to_idx.get("UNK", resname_to_idx.get("ALA", 0))
        for resname in graph.residues:
            residue_indices.append(resname_to_idx.get(resname, fallback_idx))
        graph.residue_type = torch.tensor(residue_indices, dtype=torch.long)
        graph.residue_pdb_idx = torch.tensor(
            [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
        )
        graph.orig_chain_id = [
            s.split(":")[0] for s in graph.residue_id
        ]
        graph.is_peptide = torch.tensor(
            [1 if s.split(":")[0] == self.pep_chain else 0 for s in graph.residue_id],
            dtype=torch.long,
        )
        graph.seq_pos = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
        graph.id = sample_id
        return graph

    def _get_dataset(self, split: str) -> PepDataset:
        df = self._load_split_df()
        requested_split = split
        if split == "val" and self.validation_split:
            split = str(self.validation_split)
        if split not in df["split"].unique():
            raise ValueError(f"Split '{split}' not found in {self.split_csv}")
        split_df = df[df["split"] == split].reset_index(drop=True)
        if requested_split == "val" and self.validation_tags:
            split_df = split_df[split_df["tag"].astype(str).isin(self.validation_tags)].reset_index(drop=True)
            if split_df.empty:
                raise ValueError(
                    f"No validation samples found for validation_split={split!r} "
                    f"and validation_tags={self.validation_tags} in {self.split_csv}"
                )
        if requested_split == "val" and self.validation_max_samples is not None:
            split_df = split_df.head(max(0, self.validation_max_samples)).reset_index(drop=True)
        pdb_ids = split_df["pdbid"].tolist()
        tags = split_df["tag"].astype(str).tolist() if "tag" in split_df.columns else None
        processed_paths = (
            [str(self._index_path(value)) for value in split_df["processed_path"].tolist()]
            if self.use_processed_path_column and "processed_path" in split_df.columns
            else None
        )
        sample_weights = split_df["sample_loss_weight"].astype(float).tolist() if "sample_loss_weight" in split_df.columns else None
        return PepDataset(
            pdb_codes=pdb_ids,
            processed_dir=self.processed_dir,
            transform=self.transform,
            in_memory=self.in_memory,
            tags=tags,
            processed_paths=processed_paths,
            sample_weights=sample_weights,
            sample_loss_weights=self.sample_loss_weights,
        )

    def _build_train_sampler(self, dataset: PepDataset):
        cfg = self.train_sampler_cfg
        if not cfg:
            return None

        cfg = dict(cfg)
        sampler_type = cfg.pop("type", cfg.pop("_type_", "tagged_ratio"))
        if sampler_type != "tagged_ratio":
            raise ValueError(f"Unsupported train_sampler type: {sampler_type}")
        if dataset.tags is None:
            raise ValueError(
                "train_sampler.type=tagged_ratio requires a 'tag' column in the split CSV."
            )

        return TaggedRatioDistributedSampler(
            dataset=dataset,
            tags=dataset.tags,
            anchor_tag=cfg.pop("anchor_tag"),
            fill_tag=cfg.pop("fill_tag"),
            anchor_ratio=cfg.pop("anchor_ratio"),
            shuffle=cfg.pop("shuffle", True),
            seed=cfg.pop("seed", 0),
            drop_last=cfg.pop("drop_last", False),
            **cfg,
        )

    def train_dataloader(self):
        if self.train_ds is None:
            self.train_ds = self.train_dataset()
        sampler = self._build_train_sampler(self.train_ds)
        if sampler is None:
            return self._get_dataloader(dataset=self.train_ds, shuffle=True)
        return self._get_dataloader(
            dataset=self.train_ds,
            shuffle=False,
            sampler=sampler,
        )
