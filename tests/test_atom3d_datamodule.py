# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
from pathlib import Path

import pytest
import torch

from src.datamodules.atom3d_datamodule import ATOM3DDataModule

test_cases = [
    ("LBA", 32),
    ("PSR", 32)
]


@pytest.mark.parametrize("task,batch_size", test_cases)
def test_atom3d_datamodule(task: str, batch_size: int):
    # note: append ".." to the front of paths when testing in the `tests` dir
    data_dir = os.path.join("data", "ATOM3D")

    dm = ATOM3DDataModule(
        task=task,
        data_dir=data_dir,
        lba_split=30,
        edge_cutoff=4.5,
        max_neighbors=32,
        max_units=0,
        unit="edge",
        batch_size=batch_size,
        num_workers=2
    )
    dm.prepare_data()

    assert not dm.data_train and not dm.data_val and not dm.data_test
    assert Path(data_dir, task).exists()

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    num_datapoints = len(dm.data_train) + len(dm.data_val) + len(dm.data_test)
    if task == "LBA":
        assert num_datapoints == 4_463
    if task == "PSR":
        assert num_datapoints == 44_214

    batch = next(iter(dm.train_dataloader()))
    x, y = (batch, batch.num_graphs)
    assert len(x) == batch_size
    assert y == batch_size
    assert x.h.dtype == torch.int64
    assert all([f.dtype == torch.float32 for f in [x.chi, x.e, x.xi, x.x]])
    assert x.edge_index.dtype == torch.int64
    assert type(y) == int
