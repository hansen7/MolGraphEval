# Ref: https://github.com/lsh0520/RGCL/blob/main/transferLearning/chem/pretrain_rgcl.py

from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_scatter import scatter_max
from torch_geometric.nn import global_mean_pool
from config.training_config import TrainingConfig
from models.pre_trainer_model import PreTrainerModel
from models.building_blocks.gnn import GNN, GNN_IMP_Estimator


class RGCLModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super(RGCLModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        self.emb_dim = config.emb_dim
        self.pool: Callable = global_mean_pool
        self.node_imp_estimator = GNN_IMP_Estimator()
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
        node_imp = self.node_imp_estimator(x, edge_index, edge_attr, batch_assignments)
        x = self.gnn(x, edge_index, edge_attr)

        out, _ = scatter_max(torch.reshape(node_imp, (1, -1)), batch_assignments)
        out = out.reshape(-1, 1)
        out = out[batch_assignments]
        node_imp /= out * 10
        node_imp += 0.9
        node_imp = node_imp.expand(-1, 300)

        x = torch.mul(x, node_imp)
        x = self.pool(x, batch_assignments)
        x = self.projection_head(x)

        return x

    def loss_cl(self, x1: Tensor, x2: Tensor, temp=0.1) -> Tensor:
        T = temp
        batch, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
            "i,j->ij", x1_abs, x2_abs
        )
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch), range(batch)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = -torch.log(loss).mean()
        return loss

    def loss_infonce(self, x1, x2, temp=0.1):
        T = temp
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
            "i,j->ij", x1_abs, x2_abs
        )
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / sim_matrix.sum(dim=1)
        loss = -torch.log(loss).mean()
        return loss

    def loss_ra(self, x1, x2, x3, temp=0.1, lamda=0.1):
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)
        x3_abs = x3.norm(dim=1)

        cp_sim_matrix = torch.einsum("ik,jk->ij", x1, x3) / torch.einsum(
            "i,j->ij", x1_abs, x3_abs
        )
        cp_sim_matrix = torch.exp(cp_sim_matrix / temp)

        sim_matrix = torch.einsum("ik,jk->ij", x1, x2) / torch.einsum(
            "i,j->ij", x1_abs, x2_abs
        )
        sim_matrix = torch.exp(sim_matrix / temp)

        pos_sim = sim_matrix[range(batch_size), range(batch_size)]

        ra_loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        ra_loss = -torch.log(ra_loss).mean()

        cp_loss = pos_sim / (cp_sim_matrix.sum(dim=1) + pos_sim)
        cp_loss = -torch.log(cp_loss).mean()

        loss = ra_loss + lamda * cp_loss

        return ra_loss, cp_loss, loss
