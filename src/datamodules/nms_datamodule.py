# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import torch
import torch_geometric
import pytorch_lightning as pl

from typing import Literal, Optional, Dict, Any

from src.datamodules.components.nms_dataset import NMSDataset
from src.datamodules.components.sampler import BatchSampler, DistributedSamplerWrapper

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


DATA_MODE_TO_DIR = {
    "small": "small",
    "small_20body": "small_20body",
    "static": "static_20body",
    "dynamic": "dynamic_20body"
}


SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class NMSDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = os.path.join("data", "NMS"),
            data_mode: Literal["small", "small_20body", "static", "dynamic"] = "small",
            max_units: int = 0,
            unit: str = "edge",
            rbf_edge_dist_cutoff: float = 4.5,
            num_rbf: int = 16,
            frame_O: int = 30,
            frame_T: int = 40,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):
        self.train_set = NMSDataset(
            partition="train",
            data_root=os.path.join(self.hparams.data_dir, DATA_MODE_TO_DIR[self.hparams.data_mode]),
            data_mode=self.hparams.data_mode,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            frame_0=self.hparams.frame_O,
            frame_T=self.hparams.frame_T
        )
        self.val_set = NMSDataset(
            partition="valid",
            data_root=os.path.join(self.hparams.data_dir, DATA_MODE_TO_DIR[self.hparams.data_mode]),
            data_mode=self.hparams.data_mode,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            frame_0=self.hparams.frame_O,
            frame_T=self.hparams.frame_T
        )
        self.test_set = NMSDataset(
            partition="test",
            data_root=os.path.join(self.hparams.data_dir, DATA_MODE_TO_DIR[self.hparams.data_mode]),
            data_mode=self.hparams.data_mode,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            frame_0=self.hparams.frame_O,
            frame_T=self.hparams.frame_T
        )

    @typechecked
    def get_dataloader(
        self,
        dataset: NMSDataset,
        batch_size: int = None,
        pin_memory: bool = True,
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
        return self.get_dataloader(self.train_set, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_set, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        return self.get_dataloader(self.test_set, batch_size=self.hparams.batch_size)

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "nms.yaml")
    cfg.data_dir = str(root / "data" / "NMS")
    _ = hydra.utils.instantiate(cfg)
