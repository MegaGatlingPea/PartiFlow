from abc import ABC, abstractmethod
from typing import Callable, Iterable, List, Optional

import lightning as L
from loguru import logger
from torch.utils.data import Sampler
from torch_geometric import transforms as T
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from partiflow.utils.dense_padding_data_loader import DensePaddingDataLoader


class BaseLightningDataModule(L.LightningDataModule, ABC):
    """Base class for all datamodules"""

    def __init__(
        self,
        batch_padding: bool = True,
        transforms: Optional[List[Callable]] = None,
        batch_size: int = 32,
        num_workers: int = 32,
        pin_memory: bool = False,
        multiprocessing_context: Optional[str] = None,
        pad_max_size: Optional[int] = None,
        truncate_to_pad_max_size: bool = False,
    ):
        """Initialising the base data module class.

        Args:
            batch_padding (bool, optional): Whether batches should be padded to a dense representation
                with the length being either a pre-specified max length or the maximum length of the
                sample in the batch (base PyTorch batch) or whether a sparse representation should be
                used (PyG batch). Defaults to True (base PyTorch batch).
            transforms (List[Callable]): List of transforms applied to each example.
            batch_size (int, optional): Batch size used for dataloaders. Defaults to 32.
            num_workers (int, optional): Number of workers used for dataloading. Defaults to 32.
            pin_memory (bool, optional): Whether memory should be pinned. Defaults to False.
        """
        super().__init__()
        self.batch_padding = batch_padding
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.multiprocessing_context = multiprocessing_context
        self.pad_max_size = pad_max_size
        self.truncate_to_pad_max_size = truncate_to_pad_max_size
        self.transform = (
            self._compose_transforms(transforms) if transforms is not None else None
        )
        self._sync_transform_pad_max_size()
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.predict_ds = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_ds = self.train_dataset()
        elif stage == "validation":
            self.val_ds = self.val_dataset()
        elif stage == "test":
            self.test_ds = self.test_dataset()
        elif stage == "predict":
            self.predict_ds = self.predict_dataset()

    def _compose_transforms(self, transforms: Iterable[Callable]) -> T.Compose:
        try:
            return T.Compose(list(transforms.values()))
        except Exception:
            return T.Compose(transforms)

    def _iter_transforms(self):
        if self.transform is None:
            return []
        if hasattr(self.transform, "transforms"):
            return self.transform.transforms
        return [self.transform]

    def _sync_transform_pad_max_size(self) -> None:
        if self.pad_max_size is None:
            return
        for transform in self._iter_transforms():
            set_pad_max_size = getattr(transform, "set_pad_max_size", None)
            if callable(set_pad_max_size):
                set_pad_max_size(self.pad_max_size)
        
    @abstractmethod
    def _get_dataset(self, split: str) -> Dataset:
        """Creates a dataset given a split.

        Args:
            split (str): Split for which to get the dataset, with options "train", "val" or "test"

        Returns:
            Dataset: Dataset created for the respective split
        """
        ...

    def train_dataset(self) -> Dataset:
        return self._get_dataset("train")

    def val_dataset(self) -> Dataset:
        return self._get_dataset("val")

    def test_dataset(self) -> Dataset:
        return self._get_dataset("test")
    
    def predict_dataset(self) -> Dataset:
        return self._get_dataset("predict")

    def _get_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = False,
        drop_last: bool = True,
        pad_batch_to_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
    ) -> DataLoader:
        """Returns the dataloader for the corresponding dataset.

        Args:
            dataset (Dataset): PyG dataset for which the dataloader will be created.
            shuffle (bool, optional): Whether the dataloader should be shuffled. Defaults to False.

        Returns:
            DataLoader: Dataloader to be used by model.
        """
        
        self._sync_transform_pad_max_size()
        shuffle = shuffle and sampler is None
        
        dataloader_class = DensePaddingDataLoader if self.batch_padding else DataLoader
        exclude_keys = [
            "cpcore_cyclic_type",
            "forced_topology_source",
            "forced_topology_anchor_resids",
            "forced_optional_ca_pair_indices",
            "forced_optional_ca_pair_valid_mask",
            "forced_optional_ca_pair_bin_idx",
            "forced_residue_type_condition",
            "forced_residue_type_condition_mask",
        ]

        dataloader_kwargs = dict(
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=drop_last,
        )
        if dataloader_class is DensePaddingDataLoader:
            dataloader_kwargs["max_size"] = self.pad_max_size
            dataloader_kwargs["truncate"] = self.truncate_to_pad_max_size
            dataloader_kwargs["pad_batch_to_size"] = pad_batch_to_size
            dataloader_kwargs["exclude_keys"] = exclude_keys
        if self.num_workers > 0 and self.multiprocessing_context:
            dataloader_kwargs["multiprocessing_context"] = self.multiprocessing_context
        if self.num_workers > 0:
            dataloader_kwargs["persistent_workers"] = True

        return dataloader_class(
            dataset,
            **dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        if self.train_ds is None:
            self.train_ds = self.train_dataset()
        shuffle = True
        train_dl = self._get_dataloader(
            dataset=self.train_ds,
            shuffle=shuffle
        )
        return train_dl

    def val_dataloader(self) -> DataLoader:
        if self.val_ds is None:
            self.val_ds = self.val_dataset()
        shuffle = False
        logger.info(f"Length of validation set: {len(self.val_ds)}")
        val_dl = self._get_dataloader(
            dataset=self.val_ds,
            shuffle=shuffle,
            drop_last=False,
        )
        return val_dl

    def test_dataloader(self) -> DataLoader:
        if self.test_ds is None:
            self.test_ds = self.test_dataset()
        shuffle = False
        test_dl = self._get_dataloader(
            dataset=self.test_ds,
            shuffle=shuffle,
            drop_last=False,
        )
        return test_dl
    
    def predict_dataloader(self):
        if self.predict_ds is None:
            self.predict_ds = self.predict_dataset()
        shuffle = False
        pad_batch_to_size = (
            self.batch_size if getattr(self, "pad_partial_batch", False) else None
        )
        predict_dl = self._get_dataloader(
            dataset=self.predict_ds,
            shuffle=shuffle,
            drop_last=False,
            pad_batch_to_size=pad_batch_to_size,
        )
        return predict_dl
