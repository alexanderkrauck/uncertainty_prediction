import torch
from torch import Tensor
import torch.nn.functional as F

def nlll(distribution_class, y, weights, mu, sigma, reduce="mean", numeric_stability=1e-6, **kwargs):
    distribution = distribution_class(
        mu, sigma + numeric_stability
    )  # for numerical stability if predicted sigma is too close to 0
    log_prob = distribution.log_prob(y.reshape(-1, 1))
    loss = (
        torch.exp(log_prob) + numeric_stability
    )  # for numerical stability because outliers can cause this to be 0
    loss = torch.sum(loss * weights, dim=1)
    loss = -torch.log(loss)
    if reduce == "mean":
        loss = torch.mean(loss)
    elif reduce == "sum":
        loss = torch.sum(loss)
    return loss

def miscalibration_area_fn(
    distribution_class,
    y: Tensor,
    weights: Tensor,
    mu: Tensor,
    sigma: Tensor,
    n_samples: int = 100,
    grumble_tau: float = 0.1,
    sigmoid_steepness: float = 50,
    reduce="mean",
    **kwargs,
):
    device = y.device

    drawn_samples = distribution_class(mu, sigma + 1e-6).rsample((n_samples,)).transpose(0, 1)

    weights = weights.unsqueeze(1).expand(-1, n_samples, -1)
    component_indices = F.gumbel_softmax(
        weights, tau=grumble_tau, hard=False
    )

    effective_samples = (component_indices * drawn_samples).sum(dim=-1)
    y = y.squeeze(-1)

    quantiles = torch.arange(5, 96, 10, device=device) / 100

    upper_bounds = torch.quantile(effective_samples, quantiles, dim=-1)
    y_r = (
        torch.sigmoid(sigmoid_steepness * (upper_bounds - y)).sum(dim=-1)
        / y.shape[0]
    )

    miscalibration_area = (
        (y_r - quantiles).abs().mean()
    )  # maybe use trapz shomehow instead of mean

    if reduce == "sum":
        miscalibration_area = miscalibration_area * y.shape[0]

    return miscalibration_area