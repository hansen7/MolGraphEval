from typing import Callable

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from datasets import RDKIT_PROPS
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class MotifModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super(MotifModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        self.emb_dim = config.emb_dim
        self.num_tasks = len(RDKIT_PROPS)
        self.pool: Callable = global_mean_pool
        self.criterion = nn.BCEWithLogitsLoss()
        self.prediction_model: nn.Module = nn.Sequential(
            nn.Linear(self.emb_dim, self.num_tasks)
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
        x = self.prediction_model(x)
        return x

    def loss_cl(self, y_pred: Tensor, y_actual: Tensor) -> Tensor:
        loss = self.criterion(y_pred, y_actual)
        return loss
