from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class GraphCLModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super(GraphCLModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        self.emb_dim = config.emb_dim
        self.pool: Callable = global_mean_pool
        self.projection_head: nn.Module = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward_cl(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_assignments: Tensor,
    ) -> Tensor:
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch_assignments)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1: Tensor, x2: Tensor) -> Tensor:
        T = 0.1
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
            "i,j->ij", x1_abs, x2_abs
        )
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch), range(batch)]
        # loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = pos_sim / (sim_matrix.sum(dim=1))
        loss = -torch.log(loss).mean()
        return loss
