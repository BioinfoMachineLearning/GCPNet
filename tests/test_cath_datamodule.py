# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import hydra
import pytest
import torch

from omegaconf import OmegaConf
from src.datamodules.cath_datamodule import CATHDataModule

test_cases = [
    (32)
]


@pytest.mark.parametrize("batch_size", test_cases)
def test_cath_datamodule(batch_size):
    # note: append ".." to the front of paths when testing in the `tests` dir
    features_cfg_filepath = os.path.join("configs", "datamodule", "features_cfg", "features_cpd.yaml")
    features_cfg = hydra.utils.instantiate(OmegaConf.load(features_cfg_filepath))
    data_dir = os.path.join("data", "CATH")

    dm = CATHDataModule(
        features_cfg=features_cfg,
        data_dir=data_dir,
        file_name="chain_set.jsonl",
        splits_file_name="chain_set_splits.json",
        short_file_name="test_split_L100.json",
        single_chain_file_name="test_split_sc.json",
        max_units=3000,
        unit="edge",
        num_workers=4,
        max_neighbors=32,
        train_size=1.0
    )
    dm.prepare_data()

    assert not dm.trainset and not dm.valset and not dm.testset
    assert Path(data_dir).exists()

    dm.setup()
    assert dm.trainset and dm.valset and dm.testset
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.trainset) + len(dm.valset) + len(dm.testset)
    assert num_datapoints == 19_752

    batch = next(iter(dm.train_dataloader()))
    x, y = (batch, batch.num_graphs)
    assert x.x.dtype == torch.float32
    assert x.seq.dtype == torch.int64
    assert type(x.name) == list
    assert all([f.dtype == torch.float32 for f in [x.h, x.chi, x.e, x.xi]])
    assert x.edge_index.dtype == torch.int64
    assert x.mask.dtype == torch.bool
    assert type(y) == int
