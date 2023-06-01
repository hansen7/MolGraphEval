from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    MessagePassing,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_self_loops, softmax
from torch_scatter import scatter_add

from config.training_config import TrainingConfig
from config.validation_config import ValidationConfig

num_atom_type = 120
num_chirality_tag = 3

num_bond_type = 6
num_bond_direction = 3


class GINConv(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge info by concatenation.

    See https://arxiv.org/abs/1810.00826"""

    def __init__(self, emb_dim, aggr="add"):
        """
        :param emb_dim: int, dimensionality of embeddings for nodes and edges.
        :param aggr: aggregation method, option: "add" or "mean" or "max"
        """
        super(GINConv, self).__init__()
        self.aggr = aggr
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim)
        )
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index[0], x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        # in message,
        # x_j: (1514, 300)
        # edge_attr: (1514, 300)
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super(GCNConv, self).__init__()

        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        # assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))
        # edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        norm = self.norm(edge_index[0], x.size(0), x.dtype)
        x = self.linear(x)

        # return self.propagate(
        #     self.aggr, edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm
        # )
        return self.propagate(
            edge_index=edge_index[0], x=x, edge_attr=edge_embeddings, norm=norm
        )

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GINConv_Ext(MessagePassing):
    """
    Extension of GIN aggregation to incorporate edge information by concatenation.
    Args:
        emb_dim (int): dimensionality of embeddings for nodes and edges.
        embed_input (bool): whether to embed input or not.

    See https://arxiv.org/abs/1810.00826
    """

    def __init__(self, emb_dim, out_dim, aggr="add", **kwargs):
        kwargs.setdefault("aggr", aggr)
        self.aggr = aggr
        super(GINConv_Ext, self).__init__(**kwargs)
        # multi-layer perceptron
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, 2 * emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * emb_dim, out_dim),
        )
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GCNConv_Ext(MessagePassing):
    def __init__(self, in_dim, out_dim, aggr="add"):
        super(GCNConv_Ext, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, out_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, out_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones(
            (edge_index.size(1),), dtype=dtype, device=edge_index.device
        )
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        norm = self.norm(edge_index, x.size(0), x.dtype)

        # https://github.com/THUDM/GraphMAE/blob/6d2636e942f6597d70f438e66ce876f80f9ca9e0/chem/model.py#L103
        x = self.linear(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings, norm=norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


class GATConv(MessagePassing):
    def __init__(self, emb_dim, heads=2, negative_slope=0.2, aggr="add"):
        super(GATConv, self).__init__()

        self.aggr = aggr
        self.heads = heads
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        self.weight_linear = nn.Linear(emb_dim, heads * emb_dim)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * emb_dim))

        self.bias = nn.Parameter(torch.Tensor(emb_dim))

        self.edge_embedding1 = nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, heads * emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.weight_linear(x).view(-1, self.heads, self.emb_dim)
        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, edge_index, x_i, x_j, edge_attr):
        edge_attr = edge_attr.view(-1, self.heads, self.emb_dim)
        x_j += edge_attr

        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index[0])

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out


class GraphSAGEConv(MessagePassing):
    def __init__(self, emb_dim, aggr="mean"):
        super(GraphSAGEConv, self).__init__()

        self.aggr = aggr
        self.emb_dim = emb_dim
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, emb_dim)

        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

    def forward(self, x, edge_index, edge_attr):
        # add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(
            edge_attr[:, 1]
        )

        x = self.linear(x)

        return self.propagate(self.aggr, edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p=2, dim=-1)


class GNN_IMP_Estimator(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """

    def __init__(self, num_layer=3, emb_dim=300, JK="last", drop_ratio=0):
        super(GNN_IMP_Estimator, self).__init__()
        self.drop_ratio = drop_ratio
        self.num_layer = num_layer
        self.JK = JK

        # self.num_layer = 3
        # self.drop_ratio = 0
        # emb_dim = 300
        # self.JK = "last"

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.gnns = torch.nn.ModuleList()
        self.gnns.append(GCNConv_Ext(emb_dim, 128))
        self.gnns.append(GCNConv_Ext(128, 64))
        self.gnns.append(GCNConv_Ext(64, 32))

        self.batch_norms = torch.nn.ModuleList()
        self.batch_norms.append(torch.nn.BatchNorm1d(128))
        self.batch_norms.append(torch.nn.BatchNorm1d(64))
        self.batch_norms.append(torch.nn.BatchNorm1d(32))

        self.linear = torch.nn.Linear(32, 1)

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]
        for layer in range(len(self.gnns)):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == len(self.gnns) - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        node_representation = h_list[-1]
        node_representation = self.linear(node_representation)
        node_representation = softmax(node_representation, batch)

        return node_representation


class GNNDecoder(nn.Module):
    # https://github.com/THUDM/GraphMAE/blob/main/chem/model.py
    def __init__(self, hidden_dim, out_dim, JK="last", drop_ratio=0, gnn_type="gin"):
        super().__init__()
        self._dec_type = gnn_type
        if gnn_type == "gin":
            self.conv = GINConv_Ext(hidden_dim, out_dim, aggr="add")
        elif gnn_type == "gcn":
            self.conv = GCNConv_Ext(hidden_dim, out_dim, aggr="add")
        elif gnn_type == "linear":
            self.dec = nn.Linear(hidden_dim, out_dim)
        else:
            raise NotImplementedError(f"{gnn_type}")
        self.dec_token = nn.Parameter(torch.zeros([1, hidden_dim]))
        self.enc_to_dec = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.activation = nn.PReLU()
        self.temp = 0.2

    def forward(self, x, edge_index, edge_attr, mask_node_indices):
        if self._dec_type == "linear":
            out = self.dec(x)
        else:
            x = self.activation(x)
            x = self.enc_to_dec(x)
            x[mask_node_indices] = 0
            out = self.conv(x, edge_index, edge_attr)
        return out


class GNN(nn.Module):
    """Wrapper of GNN models"""

    def __init__(self, config: Union[TrainingConfig, ValidationConfig]):
        """
        :param config: include following parameters:
            drop_ratio (float): dropout rate in the {}
            num_layer (int): the number of GNN layers
            JK (str): node reprs, "last", "concat", "max" or "sum"
            gnn_type (str): "gin", "gcn", "graphsage" or "gat" """
        super(GNN, self).__init__()
        self.drop_ratio = config.dropout_ratio
        self.num_layer = config.num_layer
        self.JK = config.JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = nn.Embedding(num_atom_type, config.emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, config.emb_dim)

        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # Graph pooling
        graph_pooling = config.graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # stacking GNN layers
        self.gnns = nn.ModuleList()
        for _ in range(config.num_layer):
            if config.gnn_type == "gin":
                self.gnns.append(GINConv(config.emb_dim, aggr=config.aggr))
            elif config.gnn_type == "gcn":
                self.gnns.append(GCNConv(config.emb_dim))
            elif config.gnn_type == "gat":
                self.gnns.append(GATConv(config.emb_dim))
            elif config.gnn_type == "graphsage":
                self.gnns.append(GraphSAGEConv(config.emb_dim))

        # adding batchnorms
        self.batch_norms = nn.ModuleList()
        for _ in range(config.num_layer):
            self.batch_norms.append(nn.BatchNorm1d(config.emb_dim))

    def forward(self, *argv):
        """Feed forward computation
        :param argv: x, edge_index, edge_attr"""
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

        h_list = [x]  # list(x)
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            # h = F.relu(h)
            # h = F.dropout(h, self.drop_ratio, training=self.training)
            if layer == self.num_layer - 1:
                # remove relu in the last layer
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.relu(h)
                h = F.dropout(h, self.drop_ratio, training=self.training)
            h_list.append(h)

        # Different implementations of Jk-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise NotImplementedError
        return node_representation

    def get_embeddings(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_embeddings = self.forward(x, edge_index, edge_attr)
        graph_embeddings = self.pool(node_embeddings, batch)
        return node_embeddings, graph_embeddings


class GNN_graphpred(nn.Module):
    """Unused, wrapper for Graph-level tasks"""

    # TODO: Combine with model.graphpred
    def __init__(self, args, num_tasks, molecule_model=None):
        """
        :param args: which contains
            num_layer (int): number of GNN layers
            emb_dim (int): dimensions of embeddings
            JK (str): same as in class GNN
        :param num_tasks (int): number of tasks in the MTL scenario
        :param molecule_model: GNN class, for load pre-trained weights"""
        super(GNN_graphpred, self).__init__()
        self.molecule_model = molecule_model
        self.num_layer = args.num_layer
        self.emb_dim = args.emb_dim
        self.num_tasks = num_tasks
        self.JK = args.JK

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        graph_pooling = args.graph_pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        self.mult = 1

        if self.JK == "concat":
            self.graph_pred_linear = nn.Linear(
                self.mult * (self.num_layer + 1) * self.emb_dim, self.num_tasks
            )
        else:
            self.graph_pred_linear = nn.Linear(self.mult * self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file):
        self.molecule_model.load_state_dict(torch.load(model_file))

    def get_embeddings(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_pred_linear(graph_representation)
        return node_representation, graph_representation, pred

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = (
                data.x,
                data.edge_index,
                data.edge_attr,
                data.batch,
            )
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.molecule_model(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        output = self.graph_pred_linear(graph_representation)
        return output
