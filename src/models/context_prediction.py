""" GRAPH SSL Pre-Training via Context Prediction (CP)
i.e., maps nodes in similar structural contexts to closer embeddings
Ref Paper: Sec. 3.1.1 and Appendix G of
            https://arxiv.org/abs/1905.12265 ;
Ref Code: ${GitHub_Repo}/chem/pretrain_contextpred.py """

from typing import Callable

import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_mean_pool

from config.training_config import TrainingConfig
from models import GNN
from models.pre_trainer_model import PreTrainerModel


class ContextPredictionModel(PreTrainerModel):
    def __init__(self, gnn: GNN, config: TrainingConfig):
        super(ContextPredictionModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        self.pool: Callable = global_mean_pool

        l1 = self.config.num_layer - 1
        l2 = l1 + self.config.csize
        num_layer = self.config.num_layer
        self.config.num_layer = l2 - l1
        self.context_model: nn.Module = GNN(self.config)
        self.config.num_layer = num_layer

    def forward_substruct_model(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        return self.gnn(x, edge_index, edge_attr)

    def forward_context_model(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor
    ) -> Tensor:
        return self.context_model(x, edge_index, edge_attr)

    def forward_context_repr(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        overlap_context_substruct_idx: Tensor,
        batch_overlapped_context: Tensor,
    ) -> Tensor:
        # creating context representations
        overlapped_node_repr = self.context_model(x, edge_index, edge_attr)[
            overlap_context_substruct_idx
        ]

        # positive context representation
        # readout -> global_mean_pool by default
        context_repr = self.pool(overlapped_node_repr, batch_overlapped_context)
        return context_repr


if __name__ == "__main__":
    pass
