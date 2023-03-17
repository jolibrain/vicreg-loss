import torch
import torch.nn as nn


class VICRegModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_layers: int,
        hidden_size: int,
        representation_size: int,
    ):
        super().__init__()

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_size, kernel_size=3, padding="same"),
            nn.GELU(),
            nn.BatchNorm2d(hidden_size),
        )

        self.conv_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=2),
                    nn.GELU(),
                    nn.BatchNorm2d(hidden_size),
                )
                for _ in range(n_layers)
            ]
        )

        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(hidden_size),
        )
        self.expander = nn.Sequential(
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, representation_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for layer in self.conv_layers:
            x = layer(x)
        h = self.encoder(x)
        z = self.expander(h)
        return z
