import torch
from torch_geometric.data import Batch

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class AttributeMaskingModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super().__init__(config=config)
        self.gnn: torch.nn.Module = gnn
        self.molecule_atom_masking_model = torch.nn.Linear(config.emb_dim, 119)

    def forward(self, batch: Batch) -> torch.Tensor:
        node_repr = self.gnn(batch.masked_x, batch.edge_index, batch.edge_attr)
        node_pred = self.molecule_atom_masking_model(
            node_repr[batch.masked_atom_indices]
        )
        return node_pred
