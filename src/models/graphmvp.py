from typing import Callable, Union, List

# import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data.batch import Batch
from torch_geometric.nn import global_mean_pool
from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.building_blocks.schnet import SchNet
from models.building_blocks.auto_encoder import AutoEncoder, VariationalAutoEncoder
from models.pre_trainer_model import PreTrainerModel
from util import dual_CL


class GraphMVPModel(PreTrainerModel):
    def __init__(
        self,
        config: TrainingConfig,
        gnn_2d: GNN,
        gnn_3d: SchNet,
        ae2d3d: Union[AutoEncoder, VariationalAutoEncoder],
        ae3d2d: Union[AutoEncoder, VariationalAutoEncoder],
    ):
        super(GraphMVPModel, self).__init__(config)
        self.gnn: nn.Module = gnn_2d
        self.gnn_3d: nn.Module = gnn_3d
        self.ae2d3d: nn.Module = ae2d3d
        self.ae3d2d: nn.Module = ae3d2d
        self.config: TrainingConfig = config
        self.pool: Callable = global_mean_pool

    def forward(self, batch: Batch) -> List[Tensor]:

        # x, edge_index, edge_attr, positions, batch_assignments = batch.x, batch.edg
        repr_node = self.gnn(batch.x, batch.edge_index, batch.edge_attr)
        repr_2d = self.pool(repr_node, batch.batch)
        repr_3d = self.gnn_3d(batch.x[:, 0], batch.positions, batch.batch)

        return repr_2d, repr_3d

    def loss(self, repr_2d: Tensor, repr_3d: Tensor) -> Tensor:

        CL_loss, _ = dual_CL(repr_2d, repr_3d, self.config)
        AE_loss_1 = self.ae2d3d(repr_2d, repr_3d)
        AE_loss_2 = self.ae3d2d(repr_3d, repr_2d)
        AE_loss = (AE_loss_1 + AE_loss_2) / 2

        loss = CL_loss * self.config.GMVP_alpha1 + AE_loss * self.config.GMVP_alpha2
        # loss = -torch.log(loss).mean()
        return loss
