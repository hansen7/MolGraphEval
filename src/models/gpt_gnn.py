from typing import Callable

import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class GPTGNNModel(PreTrainerModel):
    """This is the atom prediction head for GPT-GNN.
    Will add edge in the future."""

    def __init__(self, config: TrainingConfig, gnn: GNN):
        super().__init__(config=config)
        self.gnn: nn.Module = gnn
        self.atom_pred: nn.Module = nn.Linear(config.emb_dim, 119)
        self.molecule_readout_func: Callable = global_mean_pool

    def forward(self, batch: Batch):
        node_repr = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        graph_repr = self.molecule_readout_func(node_repr, batch.batch)
        next_node_pred = self.atom_pred(graph_repr)
        return next_node_pred
