# Ref: https://github.com/THUDM/GraphMAE/blob/main/chem/pretraining.py#L50

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from functools import partial
from config.training_config import TrainingConfig
from models.pre_trainer_model import PreTrainerModel
from models.building_blocks.gnn import GNN, GNNDecoder

NUM_NODE_ATTR = 119


class GraphMAEModel(PreTrainerModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super(GraphMAEModel, self).__init__(config)
        self.gnn: nn.Module = gnn
        # self.emb_dim = config.emb_dim
        # self.pool: Callable = global_mean_pool
        self.atom_pred_decoder = GNNDecoder(
            config.emb_dim, NUM_NODE_ATTR, JK=config.JK, gnn_type="gin"
        )
        # ref: https://github.com/THUDM/GraphMAE/blob/6d2636e942f6597d70f438e66ce876f80f9ca9e0/chem/pretraining.py#L137
        self.bond_pred_decoder = None

    def forward(self, batch) -> Tensor:
        node_rep = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        pred_node = self.atom_pred_decoder(
            node_rep, batch.edge_index, batch.edge_attr, batch.masked_atom_indices
        )
        return pred_node

    def loss(self, pred_node, batch, alpha_l=1.0, loss_fn="sce") -> Tensor:
        def sce_loss(x, y, alpha=1):
            x = F.normalize(x, p=2, dim=-1)
            y = F.normalize(y, p=2, dim=-1)
            loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)
            return loss.mean()

        node_attr_label = batch.node_attr_label
        # node_attr_label = batch.mask_node_label
        masked_node_indices = batch.masked_atom_indices

        if loss_fn == "sce":
            criterion = partial(sce_loss, alpha=alpha_l)
            loss = criterion(node_attr_label, pred_node[masked_node_indices])
        else:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(
                pred_node.double()[masked_node_indices], batch.mask_node_label[:, 0]
            )

        return loss
