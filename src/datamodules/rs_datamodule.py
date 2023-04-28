# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import random
import torch
import torch_geometric

import pandas as pd

from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from typing import Any, Dict, Literal, Optional, Tuple

from src.datamodules.components.rs_dataset import MaskedGraphDataset, NegativeBatchSampler, SingleConformerBatchSampler

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


class RSDataModule(LightningDataModule):
    """
    A data wrapper for the molecular R/S classification task.

    Adapted from https://github.com/keiradams/ChIRo

    :param train_data_filepath: location of preprocessed training data.
    :param val_data_filepath: location of preprocessed validation data.
    :param test_data_filepath: location of preprocessed test data.
    :param iteration_mode: means by which to iterate through the dataset.
    :param sample_1_conformer: whether to sample a single conformer for each 3D input molecule.
    :param select_N_enantiomers: How many enantiomers to select, if any. `None` if to select zero.
    :param mask_coordinates: Whether to mask out missing atom coordinates.
    :param stereo_mask: Whether to use apply a stereoisomeric mask during dataset construction.
    :param grouping: How to group dataset examples.
    :param num_pos: How many positives to consider.
    :param num_neg: How many negatives to consider.
    :param D_max: Maximum distance to consider when computing radial basis function (RBF) features.
    :param num_rbf: Number of RBF features to compute.
    :param batch_size: mini-batch size.
    :param num_workers:  number of workers to be used for data loading.
    :param pin_memory: whether to reserve memory for faster data loading.
    """

    def __init__(
            self,
            train_data_filepath: str,
            val_data_filepath: str,
            test_data_filepath: str,
            seed: int,
            iteration_mode: str = "stereoisomers",
            sample_1_conformer: bool = False,
            select_N_enantiomers: Optional[int] = None,
            mask_coordinates: bool = False,
            stereo_mask: bool = True,
            grouping: Literal["none", "stereoisomers", "graphs"] = "none",
            stratified: bool = False,
            without_replacement: bool = True,
            num_pos: int = 0,
            num_neg: int = 1,
            D_max: float = 4.5,
            num_rbf: int = 16,
            batch_size: int = 64,
            num_workers: int = 0,
            pin_memory: bool = False
    ):
        super().__init__()

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.train_dl: Optional[torch_geometric.loader.DataLoader] = None
        self.val_dl: Optional[torch_geometric.loader.DataLoader] = None
        self.test_dl: Optional[torch_geometric.loader.DataLoader] = None
        self.train_phase, self.val_phase, self.test_phase = "train", "val", "test"

    def get_dataloaders(self) -> Tuple[torch_geometric.loader.DataLoader]:
        """Retrieve data from storage.

        Does not assign state (e.g., self.data_train = data_train).
        """
        # load all DataFrames
        train_df = pd.read_pickle(self.hparams.train_data_filepath)
        val_df = pd.read_pickle(self.hparams.val_data_filepath)
        test_df = pd.read_pickle(self.hparams.test_data_filepath)

        # prepare training and validation DataFrames
        if self.hparams.sample_1_conformer == True:
            train_df = train_df.groupby("ID").sample(1, random_state=self.hparams.seed).sort_values("SMILES_nostereo").reset_index(drop=True)
            val_df = val_df.groupby("ID").sample(1, random_state=self.hparams.seed).sort_values("SMILES_nostereo").reset_index(drop=True)

        if self.hparams.select_N_enantiomers:  # note: number of enantiomers to include for training, where the default value is `None` 
            smiles_nostereo = list(set(train_df.SMILES_nostereo))
            random.shuffle(smiles_nostereo)
            select_smiles_nostereo = smiles_nostereo[0:self.select_N_enantiomers]
            train_df = train_df[train_df.SMILES_nostereo.isin(select_smiles_nostereo)].sort_values("SMILES_nostereo").reset_index(drop=True)

        # select iteration style for training and validation datasets
        if self.hparams.iteration_mode == "stereoisomers":
            single_conformer_train_df = train_df.groupby("ID").sample(1)
            single_conformer_val_df = val_df.groupby("ID").sample(1)

            train_batch_sampler = SingleConformerBatchSampler(
                single_conformer_data_source=single_conformer_train_df,
                full_data_source=train_df,
                batch_size=self.hparams.batch_size,
                drop_last=True,
                num_pos=self.hparams.num_pos,
                num_neg=self.hparams.num_neg,
                without_replacement=self.hparams.without_replacement,
                stratified=self.hparams.stratified
            )
            val_batch_sampler = SingleConformerBatchSampler(
                single_conformer_data_source=single_conformer_val_df,
                full_data_source=val_df,
                batch_size=self.hparams.batch_size,
                drop_last=True,
                num_pos=self.hparams.num_pos,
                num_neg=self.hparams.num_neg,
                without_replacement=self.hparams.without_replacement,
                stratified=self.hparams.stratified
            )
        elif self.hparams.iteration_mode == "conformers":
            train_batch_sampler = NegativeBatchSampler(
                data_source=train_df,
                batch_size=self.hparams.batch_size,
                drop_last=True,
                num_neg=self.hparams.num_neg,
                withoutReplacement=self.hparams.without_replacement,
                stratified=self.hparams.stratified
            )
            val_batch_sampler = NegativeBatchSampler(
                data_source=val_df,
                batch_size=self.hparams.batch_size,
                drop_last=True,
                num_neg=self.hparams.num_neg,
                withoutReplacement=self.hparams.without_replacement,
                stratified=self.hparams.stratified
            )

        # construct all datasets
        train_dataset = MaskedGraphDataset(
            train_df, 
            regression="RS_label_binary",  # note: choose from `[top_score, RS_label_binary, sign_rotation]`
            stereo_mask=self.hparams.stereo_mask,
            mask_coordinates=self.hparams.mask_coordinates,
            D_max=self.hparams.D_max,
            num_rbf=self.hparams.num_rbf
        )
        val_dataset = MaskedGraphDataset(
            val_df, 
            regression="RS_label_binary",
            stereo_mask=self.hparams.stereo_mask,
            mask_coordinates=self.hparams.mask_coordinates,
            D_max=self.hparams.D_max,
            num_rbf=self.hparams.num_rbf
        )
        test_dataset = MaskedGraphDataset(
            test_df, 
            regression="RS_label_binary",
            stereo_mask=self.hparams.stereo_mask,
            mask_coordinates=self.hparams.mask_coordinates,
            D_max=self.hparams.D_max,
            num_rbf=self.hparams.num_rbf
        )

        # construct all dataloaders
        train_dl = self.get_dataloader(train_dataset, split="train", batch_sampler=train_batch_sampler)
        val_dl = self.get_dataloader(val_dataset, split="val", batch_sampler=val_batch_sampler)
        test_dl = self.get_dataloader(test_dataset, split="test")

        return train_dl, val_dl, test_dl

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (e.g., self.x = y).
        """
        assert os.path.exists(self.hparams.train_data_filepath), \
            "Training data can be downloaded manually from https://figshare.com/s/e23be65a884ce7fc8543"
        assert os.path.exists(self.hparams.val_data_filepath), \
            "Validation data can be downloaded manually from https://figshare.com/s/e23be65a884ce7fc8543"
        assert os.path.exists(self.hparams.test_data_filepath), \
            "Test data can be downloaded manually from https://figshare.com/s/e23be65a884ce7fc8543"

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.train_dl`, `self.val_dl`, `self.test_dl`.

        Note: This method is called by Lightning with both `trainer.fit()` and `trainer.test()`.
        """
        # load and split dataloaders only if not loaded already
        if not self.train_dl and not self.val_dl and not self.test_dl:
            self.train_dl, self.val_dl, self.test_dl = self.get_dataloaders()

    @typechecked
    def get_dataloader(
        self,
        dataset: Dataset,
        split: Literal["train", "val", "test"],
        batch_sampler: Optional[torch.utils.data.sampler.Sampler] = None
    ) -> torch_geometric.loader.DataLoader:
        if split in ["train", "val"]:
            # prepare training or validation dataloader
            return torch_geometric.loader.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=self.hparams.num_workers
            )
        else:
            # prepare test dataloader
            return torch_geometric.loader.DataLoader(
                dataset,
                shuffle=False,
                batch_size=1000,
                num_workers=self.hparams.num_workers
            )

    def train_dataloader(self):
        return self.train_dl

    def val_dataloader(self):
        return self.val_dl

    def test_dataloader(self):
        return self.test_dl

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

    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "rs.yaml")
    _ = hydra.utils.instantiate(cfg)
