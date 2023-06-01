from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class GraphPred(nn.Module):
    def __init__(self, config: TrainingConfig, gnn: GNN, num_tasks: int):
        super(GraphPred, self).__init__()
        self.gnn: nn.Module = gnn
        self.emb_dim = config.emb_dim
        self.pool: Callable = global_mean_pool
        # self.downstream_head: nn.Module = nn.Sequential(
        #     nn.Linear(self.emb_dim, self.emb_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.emb_dim, self.emb_dim),
        # )
        if config.JK == "concat":
            self.downstream_head = nn.Linear(
                (self.num_layer + 1) * self.emb_dim, num_tasks
            )
        else:
            self.downstream_head = nn.Linear(self.emb_dim, num_tasks)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_assignments: Tensor,
    ) -> Tensor:
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch_assignments)
        x = self.downstream_head(x)
        return x

    # def from_pretrained(self, model_file):
    #     self.gnn.load_state_dict(model_file)
    #     return
