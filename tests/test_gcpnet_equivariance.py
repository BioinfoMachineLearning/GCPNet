# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for GCPNet (https://github.com/BioinfoMachineLearning/GCPNet):
# -------------------------------------------------------------------------------------------------------------------------------------

from src.models.gcpnet_nms_module import GCPNetNMSLitModule
from src.models.gcpnet_psr_module import GCPNetPSRLitModule
from src.models.gcpnet_lba_module import GCPNetLBALitModule
from src.models.components import localize
from src.datamodules.components.atom3d_dataset import NUM_ATOM_TYPES
from src.models.components.gcpnet import GCP, GCP2, GCPDropout, GCPInteractions, GCPLayerNorm, GCPMessagePassing, ScalarVector
from src.models import randn

import copy
from functools import partial
import os
from typing import Callable
from scipy.spatial.transform import Rotation
import unittest
import pyrootutils
import hydra
from torch_geometric.data import Batch, Data
from omegaconf import OmegaConf

import torch
import torch.nn as nn
import pytorch_lightning as pl

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()  # use before @typechecked


root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)


# reproducibility
RANDOM_SEED = 1
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
pl.seed_everything(RANDOM_SEED)

###                                                                                                                                                        ###
# Note: Permutation equivariance unit tests can be sensitive to random permutations of nodes.                                                                #
# For example, if two random, edge-connected nodes have their indices swapped, the check for permutation equivariance will most likely fail (as expected).   #
# Therefore, we recommend setting `RANDOM_SEED` to a "node shuffling-safe" fixed value (e.g., 42) while unit testing for equivariance.                       #
###                                                                                                                                                        ###

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

node_input_dim = (1, 2)
node_dim = (100, 16)
edge_input_dim = (16, 1)
edge_dim = (32, 4)
n_nodes = 300
n_edges = 10000
batch_size = 8

(h, chi) = randn(n_nodes, node_dim, device=device)
(e, xi) = randn(n_edges, edge_dim, device=device)
edge_index = torch.randint(0, n_nodes, (2, n_edges), device=device)
x = torch.randn(n_nodes, 3, device=device) + torch.randint(low=1, high=100, size=(1,), device=device)
batch_idx = torch.randint(0, batch_size, (n_nodes,), device=device)
seq = torch.randint(0, 20, (n_nodes,), device=device)

# hyperparameters
lba_cfg_filepath = os.path.join("configs", "model", "module_cfg", "gcp_module_lba.yaml")
nms_cfg_filepath = os.path.join("configs", "model", "module_cfg", "gcp_module_nms.yaml")
lba_model_cfg_filepath = os.path.join("configs", "model", "model_cfg", "gcp_model_lba.yaml")
nms_model_cfg_filepath = os.path.join("configs", "model", "model_cfg", "gcp_model_nms.yaml")
lba_interaction_layer_cfg_filepath = os.path.join("configs", "model", "layer_cfg", "gcp_interaction_layer_lba.yaml")
lba_mp_cfg_filepath = os.path.join("configs", "model", "layer_cfg", "mp_cfg", "gcp_mp_lba.yaml")
cfg = hydra.utils.instantiate(OmegaConf.load(lba_cfg_filepath))
nms_cfg = hydra.utils.instantiate(OmegaConf.load(nms_cfg_filepath))
model_cfg = hydra.utils.instantiate(OmegaConf.load(lba_model_cfg_filepath))
nms_model_cfg = hydra.utils.instantiate(OmegaConf.load(nms_model_cfg_filepath))
layer_cfg = hydra.utils.instantiate(OmegaConf.load(lba_interaction_layer_cfg_filepath))
mp_cfg = hydra.utils.instantiate(OmegaConf.load(lba_mp_cfg_filepath))

# test with the maximum number of layers used for any particular task (e.g., 9 encoder layers and 8 message layers)
model_cfg["num_encoder_layers"] = 9
model_cfg["num_decoder_layers"] = 3
mp_cfg["num_message_layers"] = 8

# test with or without normalization of frame values
norm_x_diff = True
cfg["norm_x_diff"] = norm_x_diff
nms_cfg["norm_x_diff"] = norm_x_diff

# work around not using Hydra for nested config instantiations
layer_cfg["mp_cfg"] = mp_cfg

# override default NMS model config arguments
nms_model_cfg["h_input_dim"] = node_input_dim[0]
nms_model_cfg["chi_input_dim"] = node_input_dim[1]
nms_model_cfg["e_input_dim"] = edge_input_dim[0]
nms_model_cfg["xi_input_dim"] = edge_input_dim[1]

###                                                                                                                        ###
# Note: We have grouped certain unit tests into the same class to prevent out-of-memory concerns.                            #
# This means you will need to uncomment each individual class and its unit test functions separately to test each group.     #
###                                                                                                                        ###

##### Unit Tests #####

### GCPEquivarianceTests ###

# class GCPEquivarianceTest1(unittest.TestCase):

#     def test_gcp(self):
#         model = GCP(
#             node_dim,
#             node_dim,
#             vector_gate=False,
#             frame_gate=False,
#             sigma_frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_vector_gate(self):
#         model = GCP(
#             node_dim,
#             node_dim,
#             vector_gate=True,
#             frame_gate=False,
#             sigma_frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_frame_gate(self):
#         model = GCP(
#             node_dim,
#             node_dim,
#             vector_gate=False,
#             frame_gate=True,
#             sigma_frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_sigma_frame_gate(self):
#         model = GCP(
#             node_dim,
#             node_dim,
#             vector_gate=False,
#             frame_gate=False,
#             sigma_frame_gate=True
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_sequence(self):
#         model = nn.ModuleList([
#             GCP(node_dim, node_dim, vector_gate=False, frame_gate=False, sigma_frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_sequence_vector_gate(self):
#         model = nn.ModuleList([
#             GCP(node_dim, node_dim, vector_gate=True, frame_gate=False, sigma_frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_sequence_frame_gate(self):
#         model = nn.ModuleList([
#             GCP(node_dim, node_dim, vector_gate=False, frame_gate=True, sigma_frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_sequence_sigma_frame_gate(self):
#         model = nn.ModuleList([
#             GCP(node_dim, node_dim, vector_gate=False, frame_gate=False, sigma_frame_gate=True),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp_message_passing_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp_message_passing_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp_message_passing_sigma_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = True
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

# class GCPEquivarianceTest2(unittest.TestCase):

#     def test_gcp_message_passing(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp_interactions(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp_interactions_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp_interactions_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp_interactions_sigma_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = True
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

### GCPLigandBindingAffinityEquivarianceTest ###

# class GCPLigandBindingAffinityEquivarianceTest(unittest.TestCase):

#     def test_lba_gcp_model(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp_model_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp_model_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp_model_sigma_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = True
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

### GCPProteinStructureRankingEquivarianceTest ###

# class GCPProteinStructureRankingEquivarianceTest(unittest.TestCase):

#     def test_psr_gcp_model(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp_model_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp_model_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp_model_sigma_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = True
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

### GCPNewtonianManyBodySystemEquivarianceTest ###

# class GCPNewtonianManyBodySystemEquivarianceTest(unittest.TestCase):

#     def test_nms_gcp_model(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp_model_vector_gate(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp_model_frame_gate(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         custom_cfg["sigma_frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp_model_sigma_frame_gate(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         custom_cfg["sigma_frame_gate"] = True
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)


### GCP2EquivarianceTest ###

# class GCP2EquivarianceTest(unittest.TestCase):

#     def test_gcp2_baseline(self):
#         model = GCP2(
#             node_dim,
#             node_dim,
#             ablate_frame_updates=True,
#             vector_gate=False,
#             frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_baseline_vector_gate(self):
#         model = GCP2(
#             node_dim,
#             node_dim,
#             ablate_frame_updates=True,
#             vector_gate=True,
#             frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2(self):
#         model = GCP2(
#             node_dim,
#             node_dim,
#             ablate_frame_updates=False,
#             vector_gate=False,
#             frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_vector_gate(self):
#         model = GCP2(
#             node_dim,
#             node_dim,
#             ablate_frame_updates=False,
#             vector_gate=True,
#             frame_gate=False
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_frame_gate(self):
#         model = GCP2(
#             node_dim,
#             node_dim,
#             ablate_frame_updates=False,
#             vector_gate=False,
#             frame_gate=True
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs): return model(
#             h_V, edge_index_E, frames_E, node_inputs=node_inputs)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_sequence_baseline(self):
#         model = nn.ModuleList([
#             GCP2(node_dim, node_dim, ablate_frame_updates=True, vector_gate=False, frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_sequence_baseline_vector_gate(self):
#         model = nn.ModuleList([
#             GCP2(node_dim, node_dim, ablate_frame_updates=True, vector_gate=True, frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_sequence(self):
#         model = nn.ModuleList([
#             GCP2(node_dim, node_dim, ablate_frame_updates=False, vector_gate=False, frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_sequence_vector_gate(self):
#         model = nn.ModuleList([
#             GCP2(node_dim, node_dim, ablate_frame_updates=False, vector_gate=True, frame_gate=False),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_sequence_frame_gate(self):
#         model = nn.ModuleList([
#             GCP2(node_dim, node_dim, ablate_frame_updates=False, vector_gate=False, frame_gate=True),
#             GCPDropout(0.1),
#             GCPLayerNorm(node_dim)
#         ]).to(device).eval()

#         def test_gcp_sequence_model_fn(h_V, h_E, edge_index_E, frames_E, node_inputs):
#             # work around `nn.Sequential`'s input argument limitations
#             out = h_V
#             for i, module in enumerate(model):
#                 out = module(out, edge_index_E, frames_E, node_inputs) if i == 0 else module(out)
#             return out
#         model_fn = test_gcp_sequence_model_fn
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, node_inputs=True)

#     def test_gcp2_message_passing_baseline(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_message_passing_baseline_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_message_passing(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_message_passing_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_message_passing_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         model = GCPMessagePassing(
#             node_dim,
#             node_dim,
#             edge_dim,
#             custom_cfg,
#             mp_cfg
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_interactions_baseline(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_interactions_baseline_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_interactions(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_interactions_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

#     def test_gcp2_interactions_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         model = GCPInteractions(
#             ScalarVector(node_dim[0], node_dim[1]),
#             ScalarVector(edge_dim[0], edge_dim[1]),
#             custom_cfg,
#             layer_cfg,
#             dropout=0.1
#         ).to(device).eval()
#         def model_fn(h_V, h_E, edge_index_E, frames_E): return model(h_V, h_E, edge_index_E, frames_E)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(e, xi), x, edge_index, test_message_passing=True)

### GCP2LigandBindingAffinityEquivarianceTest ###

# class GCP2LigandBindingAffinityEquivarianceTest(unittest.TestCase):

#     def test_lba_gcp2_model_baseline(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp2_model_baseline_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp2_model(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp2_model_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_lba_gcp2_model_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         model = GCPNetLBALitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

### GCP2ProteinStructureRankingEquivarianceTest ###

# class GCP2ProteinStructureRankingEquivarianceTest(unittest.TestCase):

#     def test_psr_gcp2_model_baseline(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp2_model_baseline_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp2_model(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp2_model_vector_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

#     def test_psr_gcp2_model_frame_gate(self):
#         custom_cfg = copy.copy(cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         model = GCPNetPSRLitModule(
#             layer_class=GCPInteractions,
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg,
#             num_atom_types=NUM_ATOM_TYPES
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_invariance=True)

### GCP2NewtonianManyBodySystemEquivarianceTest ###

# class GCP2NewtonianManyBodySystemEquivarianceTest(unittest.TestCase):

#     def test_nms_gcp2_model_baseline(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp2_model_baseline_vector_gate(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = True
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp2_model(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp2_model_vector_gate(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = True
#         custom_cfg["frame_gate"] = False
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)

#     def test_nms_gcp2_model_frame_gate(self):
#         custom_cfg = copy.copy(nms_cfg)
#         custom_cfg["selected_GCP"] = {"_target_": "src.models.components.gcpnet.GCP2", "_partial_": True}
#         custom_cfg["selected_GCP"] = hydra.utils.instantiate(custom_cfg["selected_GCP"])
#         custom_cfg["ablate_frame_updates"] = False
#         custom_cfg["vector_gate"] = False
#         custom_cfg["frame_gate"] = True
#         model = GCPNetNMSLitModule(
#             layer_class=partial(GCPInteractions, updating_node_positions=True),
#             optimizer=partial(torch.optim.Adam, lr=1e-4, weight_decay=0),
#             scheduler=None,
#             model_cfg=nms_model_cfg,
#             module_cfg=custom_cfg,
#             layer_cfg=layer_cfg
#         ).to(device).eval()
#         def model_fn(b): return model.forward(b)
#         test_equivariance(model_fn, ScalarVector(h, chi), ScalarVector(
#             e, xi), x, edge_index, test_translation_equivariance=True)


@typechecked
def construct_batch_from_random_data_list(batch_size: int, construct_node_types: bool = True, **kwargs):
    data_list = []
    for _ in range(batch_size):
        effective_n_nodes = n_nodes // batch_size
        effective_n_edges = n_edges // batch_size
        a = (
            torch.randint(low=0, high=NUM_ATOM_TYPES, size=(effective_n_nodes,), device=device)
            if construct_node_types
            else torch.randn(effective_n_nodes, node_input_dim[0], device=device)
        )
        _, chi = randn(effective_n_nodes, node_input_dim, device=device)
        b, xi = randn(effective_n_edges, edge_input_dim, device=device)
        x = torch.randn(effective_n_nodes, 3, device=device) + torch.randint(low=1, high=100, size=(1,), device=device)
        edge_index = torch.randint(0, effective_n_nodes, (2, effective_n_edges), device=device)
        lig_flag = torch.randint(low=0, high=2, size=(effective_n_nodes,), device=device)
        data_list.append(Data(h=a, chi=chi, e=b, xi=xi, x=x.squeeze(), edge_index=edge_index, lig_flag=lig_flag))
    batch = Batch.from_data_list(data_list)
    return batch


@typechecked
def test_rotation_and_permutation_equivariance_and_translation_invariance(
    model: Callable,
    random_Q: TensorType[3, 3],
    random_g: TensorType[3, 1]
):
    # package the input features as a batch
    batch_construction_fn = partial(construct_batch_from_random_data_list)
    batch = batch_construction_fn(batch_size)
    batch_trans = copy.deepcopy(batch)
    batch_perm = copy.deepcopy(batch)

    # establish baseline model outputs for later comparison
    a_s, chi_V, b_s, xi_V, x_V, edge_index_E = batch.h, batch.chi, batch.e, batch.xi, batch.x, batch.edge_index
    batch, _ = model(batch)
    out_h, out_chi, out_e, out_xi = batch.h, batch.chi, batch.e, batch.xi

    # test for rotation equivariance w.r.t. vector-valued features and translation invariance w.r.t. input node positions #
    # chi_trans = random_g.transpose(-1, -2) + chi @ random_Q  # note: rotation matrix on the right
    chi_trans = (random_Q @ chi_V.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # out_chi_trans = out_chi @ random_Q  # note: rotation matrix on the right
    out_chi_trans = (random_Q @ out_chi.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # xi_trans = random_g.transpose(-1, -2) + xi @ random_Q  # note: rotation matrix on the right
    xi_trans = (random_Q @ xi_V.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # out_xi_trans = out_xi @ random_Q  # note: rotation matrix on the right
    out_xi_trans = (random_Q @ out_xi.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # x_trans = random_g.transpose(-1, -2) + x_V @ random_Q  # note: rotation matrix on the right
    x_trans = (random_Q @ x_V.transpose(-1, -2) + random_g).transpose(-1, -2)  # note: rotation matrix on the left

    batch_trans.h, batch_trans.chi, batch_trans.e, batch_trans.xi, batch_trans.x = a_s.clone(
    ), chi_trans.clone(), b_s.clone(), xi_trans.clone(), x_trans.clone()
    batch_trans, _ = model(batch_trans)
    out_h_prime, out_chi_prime, out_e_prime, out_xi_prime = batch_trans.h, batch_trans.chi, batch_trans.e, batch_trans.xi

    # test for permutation equivariance #
    a_perm, chi_perm, b_perm, xi_perm, x_perm, edge_index_perm = a_s.clone(
    ), chi_V.clone(), b_s.clone(), xi_V.clone(), x_V.clone(), edge_index_E.clone()
    node_ids_to_swap = torch.randperm(a_perm.shape[0])[:2]
    a_perm[node_ids_to_swap[0]], x_perm[node_ids_to_swap[0]], chi_perm[node_ids_to_swap[0]
                                                                       ] = a_perm[node_ids_to_swap[1]], x_perm[node_ids_to_swap[1]], chi_perm[node_ids_to_swap[1]]
    a_perm[node_ids_to_swap[1]], x_perm[node_ids_to_swap[1]], chi_perm[node_ids_to_swap[1]
                                                                       ] = a_s[node_ids_to_swap[0]], x_V[node_ids_to_swap[0]], chi_V[node_ids_to_swap[0]]

    # note: not all nodes may have the same number of outgoing edges,
    # so we can only swap source and destination node indices and edge features for up to `num_edges_to_swap` edges
    orig_src_node_matches = edge_index_perm[0] == node_ids_to_swap[0]
    new_src_node_matches = edge_index_perm[0] == node_ids_to_swap[1]
    orig_src_node_index = torch.argwhere(orig_src_node_matches)
    new_src_node_index = torch.argwhere(new_src_node_matches)
    num_edges_to_swap = min(len(orig_src_node_index), len(new_src_node_index))
    orig_src_node_index, new_src_node_index = orig_src_node_index[:
                                                                  num_edges_to_swap], new_src_node_index[:num_edges_to_swap]

    edge_index_perm[1, orig_src_node_index] = edge_index_perm[1, new_src_node_index]
    edge_index_perm[0, orig_src_node_index] = node_ids_to_swap[1]
    edge_index_perm[1, new_src_node_index] = edge_index_perm[1, orig_src_node_index]
    edge_index_perm[0, new_src_node_index] = node_ids_to_swap[0]

    b_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap] = b_perm[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap]
    b_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap] = b_s[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap]
    xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap] = xi_perm[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap]
    xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap] = xi_V[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap]
    batch_perm.h = a_perm
    batch_perm.chi = chi_perm
    batch_perm.e = b_perm
    batch_perm.xi = xi_perm
    batch_perm.x = x_perm
    batch_perm.edge_index = edge_index_perm
    batch_perm, _ = model(batch_perm)
    out_h_prime_prime, out_chi_prime_prime, _, _ = batch_perm.h, batch_perm.chi, batch_perm.e, batch_perm.xi

    # check for NaNs
    assert not any([
        out_h.isnan().any(),
        out_h_prime.isnan().any(),
        out_e.isnan().any(),
        out_e_prime.isnan().any(),
        out_chi_trans.isnan().any(),
        out_chi_prime.isnan().any(),
        out_xi_trans.isnan().any(),
        out_xi_prime.isnan().any(),
        out_h_prime_prime.isnan().any(),
        out_chi.isnan().any(),
        out_chi_prime_prime.isnan().any()
    ]), "No NaNs may be present"

    # compare the model's effective types of equivariance with all desired types of equivariance
    condition_1 = torch.allclose(out_h, out_h_prime, atol=1e-4, rtol=1e-4)
    condition_2 = torch.allclose(out_e, out_e_prime, atol=1e-4, rtol=1e-4)
    condition_3 = torch.allclose(out_chi_trans, out_chi_prime, atol=1e-4, rtol=1e-4)
    condition_4 = torch.allclose(out_xi_trans, out_xi_prime, atol=1e-4, rtol=1e-4)
    condition_5 = not torch.allclose(out_h[node_ids_to_swap[0]],
                                     out_h_prime_prime[node_ids_to_swap[0]], atol=1e-1, rtol=1e-4)
    condition_6 = not torch.allclose(out_h[node_ids_to_swap[1]],
                                     out_h_prime_prime[node_ids_to_swap[1]], atol=1e-1, rtol=1e-4)
    condition_7 = not torch.allclose(out_chi[node_ids_to_swap[0]],
                                     out_chi_prime_prime[node_ids_to_swap[0]], atol=1e-1, rtol=1e-4)
    condition_8 = not torch.allclose(out_chi[node_ids_to_swap[1]],
                                     out_chi_prime_prime[node_ids_to_swap[1]], atol=1e-1, rtol=1e-4)

    assert condition_1, "Scalar node features must be updated in an SE(3)-invariant manner"
    assert condition_2, "Scalar edge features must be updated in an SE(3)-invariant manner"
    assert condition_3, "Vector node features must be updated in an SO(3)-equivariant and translation-invariant manner"
    assert condition_4, "Vector edge features must be updated in an SO(3)-equivariant and translation-invariant manner"
    assert condition_5, "Scalar node features must be updated in a permutation-equivariant manner (1)"
    assert condition_6, "Scalar node features must be updated in a permutation-equivariant manner (2)"
    assert condition_7, "Vector node features must be updated in a permutation-equivariant manner (1)"
    assert condition_8, "Vector node features must be updated in a permutation-equivariant manner (2)"


@typechecked
def test_rotation_and_permutation_equivariance_and_translation_equivariance(
    model: Callable,
    random_Q: TensorType[3, 3],
    random_g: TensorType[3, 1]
):
    # package the input features as a batch
    batch_construction_fn = partial(construct_batch_from_random_data_list, construct_node_types=False)
    batch = batch_construction_fn(batch_size)
    batch_trans = copy.deepcopy(batch)
    batch_perm = copy.deepcopy(batch)

    # establish baseline model outputs for later comparison
    a_s, chi_V, b_s, xi_V, x_V, edge_index_E = batch.h, batch.chi, batch.e, batch.xi, batch.x, batch.edge_index
    batch, _ = model(batch)
    out_h, out_chi, out_e, out_xi, out_x = batch.h, batch.chi, batch.e, batch.xi, batch.x

    # test for rotation equivariance w.r.t. vector-valued features and translation equivariance w.r.t. input node positions #
    # chi_trans = random_g.transpose(-1, -2) + chi @ random_Q  # note: rotation matrix on the right
    chi_trans = (random_Q @ chi_V.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # out_chi_trans = out_chi @ random_Q  # note: rotation matrix on the right
    out_chi_trans = (random_Q @ out_chi.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # xi_trans = random_g.transpose(-1, -2) + xi @ random_Q  # note: rotation matrix on the right
    xi_trans = (random_Q @ xi_V.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # out_xi_trans = out_xi @ random_Q  # note: rotation matrix on the right
    out_xi_trans = (random_Q @ out_xi.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # x_trans = random_g.transpose(-1, -2) + x_V @ random_Q  # note: rotation matrix on the right
    x_trans = (random_Q @ x_V.transpose(-1, -2) + random_g).transpose(-1, -2)  # note: rotation matrix on the left
    # out_x_trans = random_g.transpose(-1, -2) + out_x @ random_Q  # note: rotation matrix on the right
    out_x_trans = (random_Q @ out_x.transpose(-1, -2) + random_g).transpose(-1, -2)  # note: rotation matrix on the left

    batch_trans.h, batch_trans.chi, batch_trans.e, batch_trans.xi, batch_trans.x = a_s.clone(
    ), chi_trans.clone(), b_s.clone(), xi_trans.clone(), x_trans.clone()
    batch_trans, _ = model(batch_trans)
    out_h_prime, out_chi_prime, out_e_prime, out_xi_prime, out_x_prime = batch_trans.h, batch_trans.chi, batch_trans.e, batch_trans.xi, batch_trans.x

    # test for permutation equivariance #
    a_perm, chi_perm, b_perm, xi_perm, x_perm, edge_index_perm = a_s.clone(
    ), chi_V.clone(), b_s.clone(), xi_V.clone(), x_V.clone(), edge_index_E.clone()
    node_ids_to_swap = torch.randperm(a_perm.shape[0])[:2]
    a_perm[node_ids_to_swap[0]], x_perm[node_ids_to_swap[0]], chi_perm[node_ids_to_swap[0]
                                                                       ] = a_perm[node_ids_to_swap[1]], x_perm[node_ids_to_swap[1]], chi_perm[node_ids_to_swap[1]]
    a_perm[node_ids_to_swap[1]], x_perm[node_ids_to_swap[1]], chi_perm[node_ids_to_swap[1]
                                                                       ] = a_s[node_ids_to_swap[0]], x_V[node_ids_to_swap[0]], chi_V[node_ids_to_swap[0]]

    # note: not all nodes may have the same number of outgoing edges,
    # so we can only swap source and destination node indices and edge features for up to `num_edges_to_swap` edges
    orig_src_node_matches = edge_index_perm[0] == node_ids_to_swap[0]
    new_src_node_matches = edge_index_perm[0] == node_ids_to_swap[1]
    orig_src_node_index = torch.argwhere(orig_src_node_matches)
    new_src_node_index = torch.argwhere(new_src_node_matches)
    num_edges_to_swap = min(len(orig_src_node_index), len(new_src_node_index))
    orig_src_node_index, new_src_node_index = orig_src_node_index[:
                                                                  num_edges_to_swap], new_src_node_index[:num_edges_to_swap]

    edge_index_perm[1, orig_src_node_index] = edge_index_perm[1, new_src_node_index]
    edge_index_perm[0, orig_src_node_index] = node_ids_to_swap[1]
    edge_index_perm[1, new_src_node_index] = edge_index_perm[1, orig_src_node_index]
    edge_index_perm[0, new_src_node_index] = node_ids_to_swap[0]

    b_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap] = b_perm[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap]
    b_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap] = b_s[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap]
    xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap] = xi_perm[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap]
    xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])][:num_edges_to_swap] = xi_V[torch.argwhere(
        edge_index_perm[0] == node_ids_to_swap[1])][:num_edges_to_swap]
    batch_perm.h = a_perm
    batch_perm.chi = chi_perm
    batch_perm.e = b_perm
    batch_perm.xi = xi_perm
    batch_perm.x = x_perm
    batch_perm.edge_index = edge_index_perm
    batch_perm, _ = model(batch_perm)
    out_h_prime_prime, out_chi_prime_prime, _, _ = batch_perm.h, batch_perm.chi, batch_perm.e, batch_perm.xi

    # check for NaNs
    assert not any([
        out_h.isnan().any(),
        out_h_prime.isnan().any(),
        out_e.isnan().any(),
        out_e_prime.isnan().any(),
        out_chi_trans.isnan().any(),
        out_chi_prime.isnan().any(),
        out_xi_trans.isnan().any(),
        out_xi_prime.isnan().any(),
        out_x_trans.isnan().any(),
        out_x_prime.isnan().any(),
        out_h_prime_prime.isnan().any(),
        out_chi.isnan().any(),
        out_chi_prime_prime.isnan().any()
    ]), "No NaNs may be present"

    # compare the model's effective types of equivariance with all desired types of equivariance
    condition_1 = torch.allclose(out_h, out_h_prime, atol=1e-4, rtol=1e-4)
    condition_2 = torch.allclose(out_e, out_e_prime, atol=1e-4, rtol=1e-4)
    condition_3 = torch.allclose(out_chi_trans, out_chi_prime, atol=1e-4, rtol=1e-4)
    condition_4 = torch.allclose(out_xi_trans, out_xi_prime, atol=1e-4, rtol=1e-4)
    condition_5 = torch.allclose(out_x_trans, out_x_prime, atol=1e-4, rtol=1e-4)
    condition_6 = not torch.allclose(out_h[node_ids_to_swap[0]],
                                     out_h_prime_prime[node_ids_to_swap[0]], atol=1e-2, rtol=1e-4)
    condition_7 = not torch.allclose(out_h[node_ids_to_swap[1]],
                                     out_h_prime_prime[node_ids_to_swap[1]], atol=1e-2, rtol=1e-4)
    condition_8 = not torch.allclose(out_chi[node_ids_to_swap[0]],
                                     out_chi_prime_prime[node_ids_to_swap[0]], atol=1e-1, rtol=1e-4)
    condition_9 = not torch.allclose(out_chi[node_ids_to_swap[1]],
                                     out_chi_prime_prime[node_ids_to_swap[1]], atol=1e-1, rtol=1e-4)

    assert condition_1, "Scalar node features must be updated in an SE(3)-invariant manner"
    assert condition_2, "Scalar edge features must be updated in an SE(3)-invariant manner"
    assert condition_3, "Vector node features must be updated in an SO(3)-equivariant and translation-invariant manner"
    assert condition_4, "Vector edge features must be updated in an SO(3)-equivariant and translation-invariant manner"
    assert condition_5, "Node positions must be updated in an SE(3)-equivariant manner"
    assert condition_6, "Scalar node features must be updated in a permutation-equivariant manner (1)"
    assert condition_7, "Scalar node features must be updated in a permutation-equivariant manner (2)"
    assert condition_8, "Vector node features must be updated in a permutation-equivariant manner (1)"
    assert condition_9, "Vector node features must be updated in a permutation-equivariant manner (2)"


@typechecked
def test_rotation_and_permutation_equivariance(
    model: Callable,
    nodes: ScalarVector,
    edges: ScalarVector,
    edge_index: TensorType[2, "num_edges"],
    frames: TensorType["num_edges", 3, 3],
    random_Q: TensorType[3, 3],
    test_message_passing: bool = False,
    node_inputs: bool = True
):
    # cache inputs
    h_s, chi_V, e_s, xi_V = nodes[0].clone(), nodes[1].clone(), edges[0].clone(), edges[1].clone()

    # establish baseline model outputs for later comparison
    if test_message_passing:
        (out_h, out_chi) = model(nodes, edges, edge_index, frames)
    else:
        (out_h, out_chi) = model(nodes, edges, edge_index, frames, node_inputs)

    # test for rotation equivariance #
    # chi_rot = chi @ random_Q  # note: rotation matrix on the right
    chi_rot = (random_Q @ chi_V.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # out_chi_rot = out_chi @ random_Q  # note: rotation matrix on the right
    out_chi_rot = (random_Q @ out_chi.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # xi_rot = xi @ random_Q  # note: rotation matrix on the right
    xi_rot = (random_Q @ xi_V.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left
    # frames_rot = frames @ random_Q  # note: rotation matrix on the right
    frames_rot = (random_Q @ frames.transpose(-1, -2)).transpose(-1, -2)  # note: rotation matrix on the left

    if test_message_passing:
        out_h_prime, out_chi_prime = model(ScalarVector(
            h_s, chi_rot), ScalarVector(e_s, xi_rot), edge_index, frames_rot)
    else:
        out_h_prime, out_chi_prime = model(ScalarVector(h_s, chi_rot), ScalarVector(
            e_s, xi_rot), edge_index, frames_rot, node_inputs)

    # test for permutation equivariance #
    h_perm, chi_perm, e_perm, xi_perm, edge_index_perm, frames_perm = h_s.clone(
    ), chi_V.clone(), e_s.clone(), xi_V.clone(), edge_index.clone(), frames.clone()
    node_ids_to_swap = torch.randperm(h_perm.shape[0])[:2]
    h_perm[node_ids_to_swap[0]], chi_perm[node_ids_to_swap[0]
                                          ] = h_perm[node_ids_to_swap[1]], chi_perm[node_ids_to_swap[1]]
    h_perm[node_ids_to_swap[1]], chi_perm[node_ids_to_swap[1]] = h_s.clone()[node_ids_to_swap[0]], chi_V.clone()[
        node_ids_to_swap[0]]

    # note: not all nodes may have the same number of outgoing edges,
    # so we can only swap source and destination node indices and edge features for up to `num_edges_to_swap` edges
    orig_src_node_matches = edge_index_perm[0] == node_ids_to_swap[0]
    new_src_node_matches = edge_index_perm[0] == node_ids_to_swap[1]
    orig_src_node_index = torch.argwhere(orig_src_node_matches)
    new_src_node_index = torch.argwhere(new_src_node_matches)
    num_edges_to_swap = min(len(orig_src_node_index), len(new_src_node_index))
    orig_src_node_index, new_src_node_index = orig_src_node_index[:
                                                                  num_edges_to_swap], new_src_node_index[:num_edges_to_swap]

    edge_index_perm[1, orig_src_node_index] = edge_index_perm[1, new_src_node_index]
    edge_index_perm[0, orig_src_node_index] = node_ids_to_swap[1]
    edge_index_perm[1, new_src_node_index] = edge_index_perm[1, orig_src_node_index]
    edge_index_perm[0, new_src_node_index] = node_ids_to_swap[0]

    e_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])[:num_edges_to_swap]
           ] = e_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])[:num_edges_to_swap]]
    e_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])[:num_edges_to_swap]
           ] = e_s[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])[:num_edges_to_swap]]
    xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])[:num_edges_to_swap]
            ] = xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])[:num_edges_to_swap]]
    xi_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])[:num_edges_to_swap]
            ] = xi_V[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])[:num_edges_to_swap]]
    frames_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])[:num_edges_to_swap]
                ] = frames_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])[:num_edges_to_swap]]
    frames_perm[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[0])[:num_edges_to_swap]
                ] = frames[torch.argwhere(edge_index_perm[0] == node_ids_to_swap[1])[:num_edges_to_swap]]

    if test_message_passing:
        out_h_prime_prime, out_chi_prime_prime = model(ScalarVector(
            h_perm, chi_perm), ScalarVector(e_perm, xi_perm), edge_index_perm, frames_perm)
    else:
        out_h_prime_prime, out_chi_prime_prime = model(ScalarVector(h_perm, chi_perm), ScalarVector(
            e_perm, xi_perm), edge_index_perm, frames_perm, node_inputs)

    # check for NaNs
    assert not any([
        out_h.isnan().any(),
        out_h_prime.isnan().any(),
        out_chi_rot.isnan().any(),
        out_chi_prime.isnan().any(),
        out_h_prime_prime.isnan().any(),
        out_chi.isnan().any(),
        out_chi_prime_prime.isnan().any()
    ]), "No NaNs may be present"

    # compare the model's effective types of equivariance with all desired types of equivariance
    condition_1 = torch.allclose(out_h, out_h_prime, atol=1e-5, rtol=1e-4)
    condition_2 = torch.allclose(out_chi_rot, out_chi_prime, atol=1e-5, rtol=1e-4)
    condition_3 = not torch.allclose(out_h[node_ids_to_swap[0]],
                                     out_h_prime_prime[node_ids_to_swap[0]], atol=1e-1, rtol=1e-4)
    condition_4 = not torch.allclose(out_h[node_ids_to_swap[1]],
                                     out_h_prime_prime[node_ids_to_swap[1]], atol=1e-1, rtol=1e-4)
    condition_5 = not torch.allclose(out_chi[node_ids_to_swap[0]],
                                     out_chi_prime_prime[node_ids_to_swap[0]], atol=1e-2, rtol=1e-4)
    condition_6 = not torch.allclose(out_chi[node_ids_to_swap[1]],
                                     out_chi_prime_prime[node_ids_to_swap[1]], atol=1e-2, rtol=1e-4)

    assert condition_1, "Scalar node features must be updated in an SO(3)-invariant manner"
    assert condition_2, "Vector node features must be updated in an SO(3)-equivariant manner"
    assert condition_3, "Scalar node features must be updated in a permutation-equivariant manner (1)"
    assert condition_4, "Scalar node features must be updated in a permutation-equivariant manner (2)"
    assert condition_5, "Vector node features must be updated in a permutation-equivariant manner (1)"
    assert condition_6, "Vector node features must be updated in a permutation-equivariant manner (2)"


@typechecked
def test_equivariance(
    model: Callable,
    nodes: ScalarVector,
    edges: ScalarVector,
    x: TensorType["num_nodes", 3],
    edge_index: TensorType[2, "num_edges"],
    node_inputs: bool = True,
    test_translation_invariance: bool = False,
    test_translation_equivariance: bool = False,
    test_message_passing: bool = False
):
    with torch.no_grad():
        frames = localize(x, edge_index)
        random_Q = torch.as_tensor(Rotation.random().as_matrix(),
                                   dtype=torch.float32, device=device)
        if test_translation_invariance or test_translation_equivariance:
            random_g = torch.randn((3, 1), device=device)
            test_fn = (
                test_rotation_and_permutation_equivariance_and_translation_invariance
                if test_translation_invariance
                else test_rotation_and_permutation_equivariance_and_translation_equivariance
            )
            test_fn(
                model,
                random_Q,
                random_g
            )
        else:
            test_rotation_and_permutation_equivariance(
                model,
                nodes,
                edges,
                edge_index,
                frames,
                random_Q,
                test_message_passing,
                node_inputs=node_inputs
            )


if __name__ == "__main__":
    unittest.main()
