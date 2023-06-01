import torch
from torch_geometric.data import Batch

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class EdgePredictionModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super().__init__(config=config)
        self.gnn: torch.nn.Module = gnn

    def forward(self, batch: Batch) -> torch.Tensor:
        node_repr = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        return node_repr
