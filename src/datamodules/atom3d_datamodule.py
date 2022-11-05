# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
from functools import partial
from typing import Any, Dict, Optional, Tuple

import atom3d
import torch
import atom3d.datasets.datasets as da
import torch_geometric

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset

from src.datamodules.components import atom3d_dataset
from src.datamodules.components.sampler import DistributedSamplerWrapper, BatchSampler

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


@typechecked
def get_data_path(
    dataset: str,
    lba_split: int = 30
) -> str:
    data_paths = {
        "PSR": "PSR/splits/split-by-year/data/",
        "LBA": f"LBA/splits/split-by-sequence-identity-{lba_split}/data/",
    }

    if dataset not in data_paths:
        raise NotImplementedError(
            f"Dataset {dataset} is not implemented yet, please choose one of the following datasets: "
            f'{", ".join(list(data_paths.keys()))}')

    return data_paths[dataset]


@typechecked
def get_task_split(
    task: str,
    lba_split: int = 30
) -> str:
    splits = {
        "PSR": "year",
        "LBA": f"sequence-identity-{lba_split}",
    }

    if task not in splits:
        raise NotImplementedError(
            f"Dataset {task} is not implemented yet, please choose one of the following datasets: "
            f'{", ".join(list(splits.keys()))}')
    return splits[task]


class ATOM3DDataModule(LightningDataModule):
    """
    Adapted from https://github.com/sarpaykent/GBPNet

    A data wrapper for the ATOM3D package. It downloads any missing
    data files from Zenodo. Also applies transformations to the
    raw data to gather graph features.

    :param task: name of the task.
    :param data_dir: location where the data is stored for the tasks.
    :param lba_split: data split type for the LBA task (30 or 60).
    :param edge_cutoff: distance threshold value to determine the edges and RBF kernel.
    :param max_neighbors: number of maximum neighbors for a given node.
    :param max_units: maximum number of `unit` allowed in the input graphs.
    :param unit: component of graph topology to size limit with `max_units`.
    :param batch_size: mini-batch size.
    :param num_workers:  number of workers to be used for data loading.
    :param pin_memory: whether to reserve memory for faster data loading.
    """

    def __init__(
            self,
            task: str = "LBA",
            data_dir: str = os.path.join("data", "ATOM3D"),
            lba_split: int = 30,
            edge_cutoff: float = 4.5,
            max_neighbors: int = 32,
            max_units: int = 0,
            unit: str = "edge",
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = {
            "PSR": atom3d_dataset.PSRTransform,
            "LBA": atom3d_dataset.LBATransform,
        }

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"

    def get_datasets(self) -> Tuple[da.LMDBDataset]:
        """Retrieve data from storage.

        Does not assign state (e.g., self.data_train = data_train).
        """
        relative_path = get_data_path(self.hparams.task, self.hparams.lba_split)
        full_path = os.path.join(self.hparams.data_dir, relative_path)

        transform = self.transforms[self.hparams.task](
            edge_cutoff=self.hparams.edge_cutoff,
            max_num_neighbors=self.hparams.max_neighbors
        )
        dataset_class = partial(da.LMDBDataset, transform=transform)

        return (
            dataset_class(full_path + self.train_phase),
            dataset_class(full_path + self.val_phase),
            dataset_class(full_path + self.test_phase)
        )

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (e.g., self.x = y).
        """
        relative_path = get_data_path(self.hparams.task, self.hparams.lba_split)
        full_path = os.path.join(self.hparams.data_dir, relative_path)
        if not os.path.exists(full_path):
            atom3d.datasets.download_dataset(self.hparams.task.split("_")[0],
                                             split=get_task_split(self.hparams.task, self.hparams.lba_split),
                                             out_path=os.path.join(self.hparams.data_dir, os.sep.join(relative_path.split("/")[:2])))

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        Note: This method is called by Lightning with both `trainer.fit()` and `trainer.test()`.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = self.get_datasets()

    @typechecked
    def get_dataloader(
        self,
        dataset: da.LMDBDataset,
        batch_size: int = None,
        pin_memory: bool = False,
        shuffle: bool = False,
        drop_last: bool = False
    ) -> torch_geometric.loader.DataLoader:
        if batch_size is None:
            batch_size = self.hparams.batch_size
        if pin_memory is None:
            pin_memory = self.hparams.pin_memory
        if self.hparams.max_units == 0:
            return torch_geometric.loader.DataLoader(
                dataset,
                num_workers=self.hparams.num_workers,
                batch_size=batch_size,
                prefetch_factor=100,
                worker_init_fn=set_worker_sharing_strategy,
                shuffle=shuffle,
                drop_last=drop_last
            )
        else:
            if torch.distributed.is_initialized():
                return torch_geometric.loader.DataLoader(
                    dataset,
                    num_workers=self.hparams.num_workers,
                    batch_sampler=DistributedSamplerWrapper(
                        BatchSampler(
                            getattr(dataset, self.hparams.unit + "_counts"),
                            max_units=self.hparams.max_units,
                            shuffle=shuffle
                        )
                    ),
                    pin_memory=pin_memory,
                    drop_last=drop_last
                )
            else:
                return torch_geometric.loader.DataLoader(
                    dataset,
                    num_workers=self.hparams.num_workers,
                    batch_sampler=BatchSampler(
                        getattr(dataset, self.hparams.unit + "_counts"),
                        max_units=self.hparams.max_units,
                        shuffle=shuffle
                    ),
                    pin_memory=pin_memory,
                    drop_last=drop_last
                )

    def train_dataloader(self):
        return self.get_dataloader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.get_dataloader(self.data_val, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.get_dataloader(self.data_test, batch_size=self.hparams.batch_size)

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "atom3d_lba.yaml")
    cfg.data_dir = str(root / "data" / "ATOM3D")
    _ = hydra.utils.instantiate(cfg)

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "atom3d_psr.yaml")
    cfg.data_dir = str(root / "data" / "ATOM3D")
    _ = hydra.utils.instantiate(cfg)
