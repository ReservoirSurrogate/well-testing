"""Inverse model (build-up project, Milestone 6): build-up curve -> (8 ring ln k, ln C_D, ln r_eD) with uncertainties.

    x (B, C_IN, G) on the log-spaced shut-in grid (examples.py)
    -> lift (pointwise, width W)
    -> L x [spectral convolution along ln dt (rFFT of the zero-padded signal, lowest `modes` complex modes mixed per
            channel pair) + pointwise conv; GELU]
    -> pooling over the valid part of the test (masked mean and max; channel 6 is the validity mask)
    -> MLP head: mean and log-variance of the N_TARGET normalized targets + the no-storage logit

Loss: beta-NLL (Seitzer et al. 2022), var.detach()^beta * 0.5 (log var + (y - mu)^2 / var), averaged over unmasked
targets (rings beyond r_eD and ln C_D of no-storage samples are masked), plus the binary cross-entropy of the
no-storage flag. beta = 0 is the plain Gaussian NLL, whose mean gradient (y - mu) / var vanishes for targets the model
still assigns a large variance (r_eD stayed at its prior spread in the first run); beta = 0.5 keeps it O(1).
"""
import torch
from torch import nn

from examples import C_IN, G, MEASURED, N_TARGET

LOG_VAR_RANGE = (-10.0, 5.0)


class SpectralConv(nn.Module):
    def __init__(self, width, modes, pad):
        super().__init__()
        self.modes, self.pad = modes, pad
        self.weight = nn.Parameter(torch.randn(modes, width, width, dtype=torch.cfloat) / width)

    def forward(self, x):
        n = x.shape[-1]
        x_hat = torch.fft.rfft(nn.functional.pad(x, (0, self.pad)))[..., :self.modes]
        y_hat = torch.einsum("bck,kcd->bdk", x_hat, self.weight)
        return torch.fft.irfft(y_hat, n=n + self.pad)[..., :n]


class InverseFNO(nn.Module):
    def __init__(self, width=48, modes=24, n_layers=4, pad=32, head=128, c_in=C_IN):
        """c_in: number of input channels the model uses (the first c_in of examples.py; 8 for runs before the
        noise-level channel)."""
        super().__init__()
        self.c_in = c_in
        self.lift = nn.Conv1d(c_in, width, 1)
        self.spectral = nn.ModuleList(SpectralConv(width, modes, pad) for _ in range(n_layers))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(n_layers))
        self.head = nn.Sequential(nn.Linear(2 * width, head), nn.GELU(), nn.Linear(head, head), nn.GELU(),
                                  nn.Linear(head, 2 * N_TARGET + 1))

    def forward(self, x):
        """Returns (mu (B, N_TARGET), log_var (B, N_TARGET), no_storage_logit (B,))."""
        x = x[:, :self.c_in]
        mask = x[:, 6:7]                                           # (B, 1, G)
        # measured curves after the end of the test carry no information: zero them here too (the spectral layers
        # are global, so anything in the tail would reach the valid part)
        keep = torch.ones(self.c_in, dtype=x.dtype)
        keep[list(MEASURED)] = 0.0
        x = x * (keep[None, :, None] + (1 - keep)[None, :, None] * mask)
        h = self.lift(x)
        for spec, pw in zip(self.spectral, self.pointwise):
            h = nn.functional.gelu(spec(h) + pw(h))
        mean = (h * mask).sum(-1) / mask.sum(-1).clamp(min=1.0)
        mx = h.masked_fill(mask == 0, float("-inf")).amax(-1)
        out = self.head(torch.cat([mean, mx], dim=1))
        mu, log_var = out[:, :N_TARGET], out[:, N_TARGET:2 * N_TARGET].clamp(*LOG_VAR_RANGE)
        return mu, log_var, out[:, -1]


def loss_terms(mu, log_var, logit, y, no_storage, beta=0.5):
    """(beta-NLL over unmasked targets, BCE of the no-storage flag); y has NaN where masked."""
    valid = ~torch.isnan(y)
    y0 = torch.where(valid, y, torch.zeros_like(y))
    nll = 0.5 * (log_var + (y0 - mu) ** 2 * torch.exp(-log_var))
    if beta:
        nll = nll * torch.exp(beta * log_var).detach()
    nll = (nll * valid).sum() / valid.sum().clamp(min=1)
    bce = nn.functional.binary_cross_entropy_with_logits(logit, no_storage)
    return nll, bce


def n_params(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


__all__ = ["G", "InverseFNO", "loss_terms", "n_params"]
