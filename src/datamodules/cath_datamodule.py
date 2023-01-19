# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import torch
import torch_geometric
import pytorch_lightning as pl

from functools import partial
from omegaconf import DictConfig
from typing import Optional, Dict, Any

from src.datamodules.components.cath_dataset import CATHDataset
from src.datamodules.components.protein_graph_dataset import ProteinGraphDataset
from src.datamodules.components.sampler import BatchSampler, DistributedSamplerWrapper

try:
    import rapidjson as json
except:
    import json

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


SHARING_STRATEGY = "file_system"
torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


def set_worker_sharing_strategy(worker_id: int):
    torch.multiprocessing.set_sharing_strategy(SHARING_STRATEGY)


class CATHDataModule(pl.LightningDataModule):
    def __init__(
            self,
            features_cfg: DictConfig,
            data_dir: str = os.path.join("data", "CATH"),
            file_name: str = "chain_set.jsonl",
            splits_file_name: str = "chain_set_splits.json",
            short_file_name: str = "test_split_L100.json",
            single_chain_file_name: str = "test_split_sc.json",
            max_neighbors: int = 30,
            max_units: int = 0,
            unit: str = "edge",
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = True
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train, self.trainset, self.val, self.valset, self.test, self.testset = [], [], [], [], [], []

        data_path = os.path.join(data_dir, file_name)
        if not os.path.exists(data_path):
            os.system(
                f"wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set.jsonl -P {data_dir}/"
            )
            os.system(
                f"wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/cath/chain_set_splits.json -P {data_dir}/"
            )
            os.system(
                f"wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_L100.json -P {data_dir}/"
            )
            os.system(
                f"wget http://people.csail.mit.edu/ingraham/graph-protein-design/data/SPIN2/test_split_sc.json -P {data_dir}/"
            )

        self.cath_dataset = CATHDataset(
            os.path.join(data_dir, file_name),
            os.path.join(data_dir, splits_file_name)
        )

        self.custom_splits_files = {
            "short": short_file_name,
            "single_chain": single_chain_file_name
        }

        self.custom_splits = {
            "short": None,
            "single_chain": None
        }

        if short_file_name:
            self.short_split_path = os.path.join(data_dir, short_file_name)
            with open(self.short_split_path) as f:
                short_data = json.load(f)

                assert "test" in short_data
                self.short = short_data["test"]
                self.custom_splits["short"] = self.short
        else:
            self.short = None

        if single_chain_file_name:
            self.single_chain_split_path = os.path.join(data_dir, single_chain_file_name)
            with open(self.single_chain_split_path) as f:
                single_chain_data = json.load(f)

                assert "test" in single_chain_data
                self.single_chain = single_chain_data["test"]
                self.custom_splits["single_chain"] = self.single_chain
        else:
            self.single_chain = None

    def get_cache_params(self) -> str:
        return f"k{self.hparams.max_neighbors}"

    def setup(self, stage: Optional[str] = None):
        self.train, self.val, self.test = self.cath_dataset.train, self.cath_dataset.val, self.cath_dataset.test

        dataset_class = partial(
            ProteinGraphDataset,
            features_cfg=self.hparams.features_cfg,
            top_k=self.hparams.max_neighbors
        )

        if self.short:
            self.short_data = []

            for entry in self.test:
                if entry["name"] in self.short:
                    self.short_data.append(entry)

            self.shortset = dataset_class(data_list=self.short_data)

        if self.single_chain:
            self.single_chain_data = []

            for entry in self.test:
                if entry["name"] in self.single_chain:
                    self.single_chain_data.append(entry)

            self.single_chain_set = dataset_class(data_list=self.single_chain_data)

        self.trainset, self.valset, self.testset = map(dataset_class, (self.train, self.val, self.test))

    @typechecked
    def get_dataloader(
        self,
        dataset: ProteinGraphDataset,
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
        return self.get_dataloader(self.trainset, batch_size=self.hparams.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return self.get_dataloader(self.valset, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        sets = {"all": self.testset}
        if self.short:
            sets["short"] = self.shortset
        if self.single_chain:
            sets["single_chain"] = self.single_chain_set
            return [self.get_dataloader(sets[key], batch_size=self.hparams.batch_size) for key in sets]

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "cath_cpd.yaml")
    cfg.data_dir = str(root / "data" / "CATH")
    _ = hydra.utils.instantiate(cfg)
