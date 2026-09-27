"""Automatic physical derivatives and the four collocation loss terms."""
import torch


def first_derivatives(model, coordinates):
    xy = coordinates.detach().clone().requires_grad_(True)
    pressure = model(xy)
    gradient = torch.autograd.grad(pressure, xy, torch.ones_like(pressure), create_graph=True)[0]
    return xy, pressure, gradient[:, :1], gradient[:, 1:2]


def pressure_derivatives(model, coordinates):
    xy, pressure, dp_dr, dp_dt = first_derivatives(model, coordinates)
    if dp_dr.requires_grad:
        second = torch.autograd.grad(dp_dr, xy, torch.ones_like(dp_dr), create_graph=True,
                                     allow_unused=True)[0]
        d2p_dr2 = torch.zeros_like(dp_dr) if second is None else second[:, :1]
    else:
        d2p_dr2 = torch.zeros_like(dp_dr)
    return pressure, dp_dr, dp_dt, d2p_dr2


def pde_residual(model, coordinates, problem):
    _, dp_dr, dp_dt, d2p_dr2 = pressure_derivatives(model, coordinates)
    return dp_dt - problem.alpha * (d2p_dr2 + dp_dr / coordinates[:, :1])


def physics_losses(model, points, problem, time_weighting=True):
    """Each MSE is dimensionless. No analytical/reference pressures enter training."""
    p = problem
    # Across decades in time, f ~ 1/t can be tiny even for an inaccurate pressure field.
    # Use a logarithmic positive multiplier so long physical horizons do not create
    # enormous gradients (the old t_scale + t multiplier was poorly conditioned for years).
    scale = (p.t_scale * (1 + torch.log1p(points["domain"][:, 1:2] / p.t_scale))
             if time_weighting else p.t_scale)
    residual = pde_residual(model, points["domain"], p) * scale / p.dp
    initial = (model(points["ic"]) - p.pi) / p.dp
    well = (model(points["bc1"]) - p.pwf) / p.dp
    _, _, outer_slope, _ = first_derivatives(model, points["bc2"])
    outer = p.re * outer_slope / p.dp
    return {"pde": residual.square().mean(), "ic": initial.square().mean(),
            "bc1": well.square().mean(), "bc2": outer.square().mean()}


def weighted_loss(losses, weights):
    return sum(weights[name] * value for name, value in losses.items())
