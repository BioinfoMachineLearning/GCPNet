# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

import os
import torch
import torch_geometric
import pytorch_lightning as pl

from typing import List, Optional, Dict, Any

from src.datamodules.components.ar_dataset import ARDataset
from src import utils

from torchtyping import patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked

log = utils.get_pylogger(__name__)


class ARDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = os.path.join("data", "AR"),
            splits_dir: str = os.path.join("data", "AR", "splits"),
            af2_dir: str = os.path.join("data", "AR", "AF2_model"),
            true_dir: str = os.path.join("data", "AR", "true_model"),
            model_data_cache_dir: str = os.path.join("data", "AR", "model_data_cache"),
            split_index: int = 1,
            rbf_edge_dist_cutoff: float = 4.5,
            num_rbf: int = 16,
            k_min: int = 12,
            k_max: int = 18,
            max_tmscore_metric_threshold: float = 1.1,
            python_exec_path: Optional[str] = None,
            pdbtools_dir: Optional[str] = None,
            force_process_data: bool = False,
            load_only_unprocessed_examples: bool = False,
            batch_size: int = 1,
            num_workers: int = 0,
            pin_memory: bool = True
    ):
        super().__init__()

        assert split_index in list(range(1, 11)), "Split index must be between 1 and 10, inclusively."
        assert 0 < k_min <= k_max, "Atoms' minimum number of edges must be greater than zero and not greater than atoms' maximum number of edges."
        assert 0 < max_tmscore_metric_threshold <= 5.0, "Max TM-score metric's threshold must be between 0 (exclusive) and 5.0 (inclusive)."

        # this line allows to access init params with `self.hparams` attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # features - ESM protein sequence embeddings #
        self.esm_model, esm_alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
        self.esm_model = self.esm_model.eval().cpu()
        self.esm_batch_converter = esm_alphabet.get_batch_converter()

        if load_only_unprocessed_examples:
            log.info("Loading only unprocessed examples!")

    @staticmethod
    @typechecked
    def parse_split_pdbs(
        af2_dir: str,
        true_dir: str,
        splits_dir: str,
        split_filename: str,
        max_tm_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        split_entries = []
        with open(os.path.join(splits_dir, split_filename), "r") as f:
            lines = f.readlines()
            for line in lines:
                line = line.rstrip().split()
                target, _, tm = line[0], line[1], float(line[2])
                if tm >= max_tm_threshold:
                    continue
                split_entries.append({
                    "initial_pdb": os.path.join(af2_dir, f"{target}.pdb"),
                    "true_pdb": os.path.join(true_dir, f"{target}.pdb"),
                    "tmscore_metric": tm
                })
        return split_entries

    def setup(self, stage: Optional[str] = None):
        train_pdbs = self.parse_split_pdbs(
            self.hparams.af2_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            f"train{self.hparams.split_index}.lst",
            max_tm_threshold=self.hparams.max_tmscore_metric_threshold
        )
        valid_pdbs = self.parse_split_pdbs(
            self.hparams.af2_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            f"valid{self.hparams.split_index}.lst",
            max_tm_threshold=self.hparams.max_tmscore_metric_threshold
        )
        test_ar_pdbs = self.parse_split_pdbs(
            self.hparams.af2_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            "test_ar.lst",
            max_tm_threshold=self.hparams.max_tmscore_metric_threshold
        )
        test_casp14_pdbs = self.parse_split_pdbs(
            self.hparams.af2_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            "test_casp14.lst",
            max_tm_threshold=self.hparams.max_tmscore_metric_threshold
        )
        test_casp14_refinement_pdbs = self.parse_split_pdbs(
            self.hparams.af2_dir,
            self.hparams.true_dir,
            self.hparams.splits_dir,
            "test_casp14_refinement.lst",
            max_tm_threshold=self.hparams.max_tmscore_metric_threshold
        )

        self.train_set = ARDataset(
            initial_pdbs=train_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            k_min=self.hparams.k_min,
            k_max=self.hparams.k_max,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            force_process_data=self.hparams.force_process_data,
            load_only_unprocessed_examples=self.hparams.load_only_unprocessed_examples,
            is_test_dataset=False
        )
        self.val_set = ARDataset(
            initial_pdbs=valid_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            k_min=self.hparams.k_min,
            k_max=self.hparams.k_max,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            force_process_data=self.hparams.force_process_data,
            load_only_unprocessed_examples=self.hparams.load_only_unprocessed_examples,
            is_test_dataset=False
        )
        self.test_ar_set = ARDataset(
            initial_pdbs=test_ar_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            k_min=self.hparams.k_min,
            k_max=self.hparams.k_max,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            force_process_data=self.hparams.force_process_data,
            load_only_unprocessed_examples=self.hparams.load_only_unprocessed_examples,
            is_test_dataset=True
        )
        self.test_casp14_set = ARDataset(
            initial_pdbs=test_casp14_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            k_min=self.hparams.k_min,
            k_max=self.hparams.k_max,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            force_process_data=self.hparams.force_process_data,
            load_only_unprocessed_examples=self.hparams.load_only_unprocessed_examples,
            is_test_dataset=True
        )
        self.test_casp14_refinement_set = ARDataset(
            initial_pdbs=test_casp14_refinement_pdbs,
            model_data_cache_dir=self.hparams.model_data_cache_dir,
            rbf_edge_dist_cutoff=self.hparams.rbf_edge_dist_cutoff,
            num_rbf=self.hparams.num_rbf,
            k_min=self.hparams.k_min,
            k_max=self.hparams.k_max,
            esm_model=getattr(self, "esm_model", None),
            esm_batch_converter=getattr(self, "esm_batch_converter", None),
            python_exec_path=self.hparams.python_exec_path,
            pdbtools_dir=self.hparams.pdbtools_dir,
            force_process_data=self.hparams.force_process_data,
            load_only_unprocessed_examples=self.hparams.load_only_unprocessed_examples,
            is_test_dataset=True
        )

    @typechecked
    def get_dataloader(
        self,
        dataset: ARDataset,
        batch_size: int,
        pin_memory: bool,
        shuffle: bool,
        drop_last: bool
    ) -> torch_geometric.loader.DataLoader:
        return torch_geometric.loader.DataLoader(
            dataset,
            num_workers=self.hparams.num_workers,
            batch_size=batch_size,
            pin_memory=pin_memory,
            shuffle=shuffle,
            drop_last=drop_last
        )

    def train_dataloader(self):
        return self.get_dataloader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            drop_last=True
        )

    def val_dataloader(self):
        return self.get_dataloader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            drop_last=False
        )

    def test_dataloader(self):
        return [
            self.get_dataloader(
                self.test_ar_set,
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                drop_last=False
            ),
            self.get_dataloader(
                self.test_casp14_set,
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                drop_last=False
            ),
            self.get_dataloader(
                self.test_casp14_refinement_set,
                batch_size=self.hparams.batch_size,
                pin_memory=self.hparams.pin_memory,
                shuffle=False,
                drop_last=False
            )
        ]

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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "ar.yaml")
    cfg.data_dir = str(root / "data" / "AR")
    _ = hydra.utils.instantiate(cfg)
