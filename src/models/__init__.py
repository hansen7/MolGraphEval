from models.building_blocks.gnn import GNN, GNN_graphpred
from .attribute_masking import AttributeMaskingModel
from .building_blocks.auto_encoder import (
    AutoEncoder,
    EnergyVariationalAutoEncoder,
    ImportanceWeightedAutoEncoder,
    NormalizingFlowVariationalAutoEncoder,
    VariationalAutoEncoder,
)
from .context_prediction import ContextPredictionModel
from .contextual import ContextualModel
from .discriminator import Discriminator
from .edge_prediction import EdgePredictionModel
from .building_blocks.flow import (
    AffineFlow,
    BatchNormFlow,
    NormalizingFlow,
    PlanarFlow,
    PReLUFlow,
    RadialFlow,
)
from .gpt_gnn import GPTGNNModel
from .graph_cl import GraphCLModel
from .info_max import InfoMaxModel
from .joao_v2 import JOAOv2Model
from .motif import MotifModel
from .graphpred import GraphPred
from .graphmvp import GraphMVPModel
from .rgcl import RGCLModel
from .graphmae import GraphMAEModel
