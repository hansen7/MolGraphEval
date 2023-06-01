import torch.nn as nn
from torch import Tensor

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.pre_trainer_model import PreTrainerModel


class ContextualModel(PreTrainerModel):
    def __init__(self, gnn: GNN, config: TrainingConfig):
        super(ContextualModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        self.emb_dim = config.emb_dim
        self.criterion = nn.CrossEntropyLoss()
        self.atom_vocab_size = config.atom_vocab_size
        self.atom_vocab_model: nn.Module = nn.Sequential(
            nn.Linear(self.emb_dim, self.atom_vocab_size)
        )

    def forward_cl(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
        batch_assignments: Tensor,
    ) -> Tensor:
        x = self.gnn(x, edge_index, edge_attr)
        x = self.atom_vocab_model(x)
        return x

    def loss_cl(self, y_pred: Tensor, y_actual: Tensor) -> Tensor:
        loss = self.criterion(y_pred, y_actual)
        return loss
