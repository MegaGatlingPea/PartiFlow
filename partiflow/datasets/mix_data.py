import random
from typing import Optional, Literal
import lightning as L
from torch.utils.data import Dataset
from torch_geometric.data import Data
from loguru import logger
from partiflow.utils.dense_padding_data_loader import DensePaddingDataLoader

class MixDataset(Dataset):
    """
    Mechanism: using idx slicees to ensure every epoch will seen all the peptide samples
    Other samples in the epoch will be randomly drawn from the PDB dataset, ensuring a fixed ratio of peptide samples per epoch.
    This is to prevent overfitting to peptide samples
    """
    def __init__(
        self, 
        pdb_dataset: Dataset, 
        pep_dataset: Dataset, 
        peptide_ratio: float
    ):
        self.pdb_dataset = pdb_dataset
        self.pep_dataset = pep_dataset
        self.len_pep = len(pep_dataset)
        self.len_pdb = len(pdb_dataset)
        self.total_len = int(self.len_pep / peptide_ratio)
        
        logger.info(
            f"MixDataset Init: Pep Samples = {self.len_pep}, "
            f"PDB Samples drawn per epoch = {self.total_len - self.len_pep}, "
            f"Total Epoch Length = {self.total_len}, Ratio = {peptide_ratio:.2f}"
        )

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Data:
        if idx < self.len_pep:
            return self.pep_dataset[idx]
        else:
            pdb_idx = random.randint(0, self.len_pdb - 1)
            return self.pdb_dataset[pdb_idx]


class CombinedLightningDataModule(L.LightningDataModule):
    def __init__(
        self,
        pdb_datamodule: L.LightningDataModule,
        pep_datamodule: L.LightningDataModule,
        peptide_ratio: float = 0.15,
        val_source: Literal["pep", "pdb"] = "pep",
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = False,
        multiprocessing_context: Optional[str] = None,
        pad_max_size: Optional[int] = None,
        truncate_to_pad_max_size: bool = False,
        drop_last: bool = True,
    ):
        super().__init__()
        if not 0 < peptide_ratio <= 1:
            raise ValueError(f"peptide_ratio must be in (0, 1], got {peptide_ratio}")
        self.pdb_datamodule = pdb_datamodule
        self.pep_datamodule = pep_datamodule
        self.peptide_ratio = peptide_ratio
        self.val_source = val_source
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multiprocessing_context = multiprocessing_context
        self.pad_max_size = pad_max_size
        self.truncate_to_pad_max_size = truncate_to_pad_max_size
        self.drop_last = drop_last
        self._propagate_dataloader_settings()

    def _propagate_dataloader_settings(self) -> None:
        self.pdb_datamodule.batch_size = self.batch_size
        self.pdb_datamodule.num_workers = self.num_workers
        self.pdb_datamodule.pin_memory = self.pin_memory
        self.pdb_datamodule.pad_max_size = self.pad_max_size
        self.pdb_datamodule.truncate_to_pad_max_size = self.truncate_to_pad_max_size
        if hasattr(self.pdb_datamodule, "multiprocessing_context"):
            self.pdb_datamodule.multiprocessing_context = self.multiprocessing_context
        self.pep_datamodule.batch_size = self.batch_size
        self.pep_datamodule.num_workers = self.num_workers
        self.pep_datamodule.pin_memory = self.pin_memory
        self.pep_datamodule.pad_max_size = self.pad_max_size
        self.pep_datamodule.truncate_to_pad_max_size = self.truncate_to_pad_max_size
        if hasattr(self.pep_datamodule, "multiprocessing_context"):
            self.pep_datamodule.multiprocessing_context = self.multiprocessing_context
        for datamodule in (self.pdb_datamodule, self.pep_datamodule):
            sync_transform = getattr(datamodule, "_sync_transform_pad_max_size", None)
            if callable(sync_transform):
                sync_transform()

    def _dataloader_kwargs(self, drop_last: bool) -> dict:
        kwargs = dict(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            max_size=self.pad_max_size,
            truncate=self.truncate_to_pad_max_size,
            drop_last=drop_last,
        )
        if self.num_workers > 0 and self.multiprocessing_context:
            kwargs["multiprocessing_context"] = self.multiprocessing_context
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def setup(self, stage: Optional[str] = None):
        self._propagate_dataloader_settings()
        self.pdb_datamodule.setup(stage)
        self.pep_datamodule.setup(stage)

        if stage == "fit" or stage is None:
            pdb_train = self.pdb_datamodule._get_dataset("train")
            pep_train = self.pep_datamodule._get_dataset("train")
            self.train_dataset = MixDataset(pdb_train, pep_train, self.peptide_ratio)

        if stage in ["fit", "validate", "test"] or stage is None:
            if self.val_source == "pep":
                self.val_dataset = self.pep_datamodule._get_dataset("val")
            else:
                self.val_dataset = self.pdb_datamodule._get_dataset("val")

    def train_dataloader(self):
        return DensePaddingDataLoader(
            self.train_dataset,
            shuffle=True,
            **self._dataloader_kwargs(drop_last=self.drop_last),
        )

    def val_dataloader(self):
        return DensePaddingDataLoader(
            self.val_dataset,
            shuffle=False,
            **self._dataloader_kwargs(drop_last=False),
        )
