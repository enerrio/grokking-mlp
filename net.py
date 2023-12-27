from einops.layers.torch import Rearrange
import torch
import torch.nn as nn


def layer_norm(net: nn.Module) -> torch.Tensor:
    """Get the weight norm for given layer"""
    output_layer = net.net[-1]
    return torch.linalg.norm(output_layer.weight.data.detach(), ord=2)


class MLP(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Embedding(vocab_size, 128),
            Rearrange("batch token d_model -> batch (token d_model)"),
            nn.Linear(4 * 128, 128),
            nn.ReLU(),
            nn.Linear(128, vocab_size),
        )
        self.initialize_params()

    def initialize_params(self) -> None:
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
