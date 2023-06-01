from typing import Callable, Optional

import torch.nn as nn
from torch import Tensor


def get_activation(name: str, leaky_relu: Optional[float] = 0.5) -> nn.Module:
    str_to_activation = {
        "leaky_relu": nn.LeakyReLU(leaky_relu),
        "rrelu": nn.RReLU(),
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "gelu": nn.GELU(),
        "prelu": nn.PReLU(),
        "selu": nn.SELU(),
    }

    return str_to_activation[name]


def create_batch_norm_1d_layers(num_layers: int, dim_hidden: int):
    batch_norm_layers = nn.ModuleList()
    for i in range(num_layers - 1):
        batch_norm_layers.append(nn.BatchNorm1d(num_features=dim_hidden))
    return batch_norm_layers


def create_linear_layers(
    num_layers: int, dim_input: int, dim_hidden: int, dim_output: int
):
    linear_layers = nn.ModuleList()
    # Input layer
    linear_layers.append(nn.Linear(in_features=dim_input, out_features=dim_hidden))
    # Hidden layers
    for i in range(1, num_layers - 1):
        linear_layers.append(nn.Linear(in_features=dim_hidden, out_features=dim_hidden))
    # Output layer
    linear_layers.append(nn.Linear(dim_hidden, dim_output))
    return linear_layers


def init_layers(initializer_name: str, layers: nn.ModuleList):
    initializer = get_initializer(initializer_name)
    for layer in layers:
        initializer(layer.weight)


def get_initializer(name: str = "xavier") -> Callable:
    str_to_init = {
        "orthogonal": nn.init.orthogonal_,
        "xavier": nn.init.xavier_uniform_,
        "kaiming": nn.init.kaiming_uniform_,
    }
    return str_to_init[name]


class MLP(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        num_layers: int,
        batch_norm: bool,
        initializer: str,
        dropout: float,
        activation: str,
        leaky_relu: float,
        is_output_activation: bool,
    ):
        super().__init__()
        self.layers = create_linear_layers(
            num_layers=num_layers,
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=dim_output,
        )
        init_layers(initializer_name=initializer, layers=self.layers)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.batch_norm_layers = (
            create_batch_norm_1d_layers(num_layers=num_layers, dim_hidden=dim_hidden)
            if batch_norm
            else None
        )
        self.activation_function = get_activation(
            name=activation, leaky_relu=leaky_relu
        )
        self.is_output_activation = is_output_activation

    def forward(self, x: Tensor):
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation_function(x)
            if self.batch_norm_layers:
                x = self.batch_norm_layers[i](x)
            if self.dropout:
                x = self.dropout(x)
        x = self.layers[-1](x)
        if self.is_output_activation:
            x = self.activation_function(x)
        return x
