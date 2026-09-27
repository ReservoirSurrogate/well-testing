"""Differentiable physical-coordinate model, (r [m], t [s]) -> p [Pa]."""
import torch
from torch import nn


class PressurePINN(nn.Module):
    def __init__(self, problem, network):
        super().__init__()
        self.register_buffer("scales", torch.tensor(
            [problem.rw, problem.re, problem.t_scale, problem.t_max, problem.pwf, problem.dp],
            dtype=torch.float64,
        ))
        layers = []
        size = 2
        for _ in range(network.depth):
            layers.extend((nn.Linear(size, network.width, dtype=torch.float64), nn.Tanh()))
            size = network.width
        layers.append(nn.Linear(size, 1, dtype=torch.float64))
        self.core = nn.Sequential(*layers)
        for layer in self.core:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, coordinates):
        """coordinates: (N, 2) tensor with columns r and t. Returns (N, 1) pressure."""
        rw, re, t_scale, t_max, pwf, dp = self.scales.unbind()
        r, t = coordinates[:, :1], coordinates[:, 1:2]
        x = 2 * torch.log(r / rw) / torch.log(re / rw) - 1
        s = 2 * torch.log1p(t / t_scale) / torch.log1p(t_max / t_scale) - 1
        return pwf + dp * self.core(torch.cat((x, s), dim=1))

    @torch.no_grad()
    def predict_pressure(self, r, t):
        """Broadcast physical radius/time inputs and return pressure with that shape."""
        r = torch.as_tensor(r, dtype=self.scales.dtype, device=self.scales.device)
        t = torch.as_tensor(t, dtype=self.scales.dtype, device=self.scales.device)
        r, t = torch.broadcast_tensors(r, t)
        if not (torch.isfinite(r).all() and torch.isfinite(t).all()):
            raise ValueError("Coordinates must be finite.")
        if torch.any((r < self.scales[0]) | (r > self.scales[1]) | (t < 0) | (t > self.scales[3])):
            raise ValueError("Query lies outside the configured radius/time domain.")
        if torch.any((r == self.scales[0]) & (t == 0)):
            raise ValueError("Pressure at the incompatible initial well corner is not a prediction target.")
        return self(torch.stack((r.flatten(), t.flatten()), dim=1)).reshape(r.shape)
