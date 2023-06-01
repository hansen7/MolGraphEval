import torch.nn as nn

from config.training_config import TrainingConfig
from models.building_blocks.gnn import GNN
from models.graph_cl import GraphCLModel


class JOAOv2Model(GraphCLModel):
    def __init__(self, config: TrainingConfig, gnn: GNN):
        super(JOAOv2Model, self).__init__(config=config, gnn=gnn)
        self.gnn: nn.Module = gnn
        self.projection_head: nn.ModuleList = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.emb_dim, self.emb_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.emb_dim, self.emb_dim),
                )
                for _ in range(5)
            ]
        )

    def forward_cl(self, x, edge_index, edge_attr, batch, n_aug=0):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head[n_aug](x)
        return x
