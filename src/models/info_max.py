""" InfoMax + Graph
Ref Paper 1: https://arxiv.org/abs/1908.01000
Ref Paper 2: https://arxiv.org/abs/1908.01000 """
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from models import GNN, Discriminator
from models.pre_trainer_model import PreTrainerModel


class InfoMaxModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super(InfoMaxModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        self.pool: Callable = global_mean_pool
        self.infograph_discriminator_SSL_model = Discriminator(config.emb_dim)

    def forward_embedding(self, batch: Batch) -> Tuple[torch.Tensor]:
        node_emb = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        pooled_emb = self.pool(node_emb, batch.batch)
        return node_emb, pooled_emb
