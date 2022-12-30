import torch
import torch.nn as nn
from typing import Any, List, Optional

class LinearBlock(nn.Module):
    """A customizable linear layer

    @param activation_before: activation between input and linear; can be None
    @param activation_after: activation after linear; can be None
    @param bias: whether the linear layer should have a bias
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: Optional[Any],
        bias: bool = True
    ):
        super().__init__()

        layers = []
        layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        layers.append(activation)

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        return self.layers(x)


class LinearNet(nn.Module):
    """A multi-layer feed-forward neural network"""

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        widths: List[int],
        activation: Any
    ):
        super().__init__()

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.widths = widths
        self.full_widths = [input_dims] + widths
        self.activation = activation
        self.layers = nn.Sequential()

        for layer_num, (in_dim, out_dim) in enumerate(zip(self.full_widths[0:-1], self.full_widths[1:])):

            self.layers.add_module(
                f"{LinearBlock.__name__}_{layer_num}",
                LinearBlock(in_dim, out_dim, activation=activation),
            )

        self.layers.add_module("linear_out", nn.Linear(self.full_widths[-1], output_dims))

        self.__name__ = str(self)

    def forward(self, x: torch.Tensor):
        z = self.layers(x)
        return z
