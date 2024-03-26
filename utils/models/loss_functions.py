"""
Loss functions for the mixture density networks.

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Type

def nlll(distribution_class: Type[torch.distributions.Distribution], y, weights, mu, sigma, reduce="mean", numeric_stability=1e-8, **kwargs):
    """
    Parameters
    ----------
    distribution_class : Type[torch.distributions.Distribution]
        Distribution class from torch.distributions
    y : Tensor
        Target values of shape (batch_size, y_size)
    weights : Tensor
        Weights of shape (batch_size, n_distributions)
    mu : Tensor
        Means of shape (batch_size, y_size, n_distributions)
    sigma : Tensor
        Standard deviations of shape (batch_size, y_size, n_distributions)
    reduce : str, optional
        Reduction method, by default "mean"
    numeric_stability : float, optional
        Small number for numerical stability, by default 1e-8
    """

    distribution = distribution_class(
        mu, sigma + numeric_stability
    )  # for numerical stability if predicted sigma is too close to 0
    log_prob = distribution.log_prob(y.unsqueeze(-1))
    log_prob = log_prob.sum(dim=1) #sum over multidimensional y
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

def miscalibration_area_fn( #NOTE: Actually this is "mean absulute calibration error"
    distribution_class,
    y: Tensor,
    weights: Tensor,
    mu: Tensor,
    sigma: Tensor,
    n_samples: int = 100,
    gumbel_tau: float = 0.1,
    sigmoid_steepness: float = 50,
    reduce="mean",
    **kwargs,
):
    """ Calculates an approximation to the mean absolute calibration error that is differentiable and can be used as a loss function or a regularizer.

    Parameters
    ----------
    distribution_class : Type[torch.distributions.Distribution]
        Distribution class from torch.distributions
    y : Tensor
        Target values of shape (batch_size, y_size)
    weights : Tensor
        Weights of shape (batch_size, n_distributions)
    mu : Tensor
        Means of shape (batch_size, y_size, n_distributions)
    sigma : Tensor
        Standard deviations of shape (batch_size, y_size, n_distributions)
    n_samples : int, optional
        Number of samples to draw from the distribution. Higher numbers give a more accurate approximation, by default 100
    grumble_tau : float, optional
        Tau parameter for the Gumbel softmax, by default 0.1
    sigmoid_steepness : float, optional
        Steepness of the sigmoid, by default 50
    reduce : str, optional
        Reduction method, by default "mean". Can be "mean" or "sum"
    """


    device = y.device

    drawn_samples = distribution_class(mu, sigma + 1e-6).rsample((n_samples,)).transpose(0, 1)

    weights = weights.unsqueeze(1).expand(-1, n_samples, -1)
    component_indices = F.gumbel_softmax(
        weights, tau=gumbel_tau, hard=False
    )

    effective_samples = (component_indices.unsqueeze(-2) * drawn_samples).sum(dim=-1)
    

    quantiles = torch.arange(5, 96, 10, device=device) / 100

    upper_bounds = torch.quantile(effective_samples, quantiles, dim=-2)
    y_r = (
        torch.sigmoid(sigmoid_steepness * (upper_bounds - y)).sum(dim=-2)
        / y.shape[0]
    )

    miscalibration_area = (
        (y_r - quantiles.unsqueeze(-1)).abs().mean() #NOTE: currently we average over multidimensional y
    )  # maybe use trapz shomehow instead of mean

    if reduce == "sum":
        miscalibration_area = miscalibration_area * y.shape[0]

    return miscalibration_area

def pinball_loss_fn(y: Tensor, mu: Tensor, sort_mu: bool = True, reduce: str="mean", **kwargs):
    """Calculates the pinball loss
    
    Parameters
    ----------
    y : Tensor
        Target values of shape (batch_size, 1)
    mu : Tensor
        Means of shape (batch_size, 1, n_quantiles)
    sort_mu : bool, optional
        Whether to sort the mu values, by default True
    reduce : str, optional
        Reduction method, by default "mean". Can be "mean" or "sum"
    """

    assert mu.shape[1] == 1 and y.shape[1] == 1, "we only do 1 output dimension with the pinball loss"

    device = y.device

    mu = mu.flatten(1)

    n_quantiles = mu.shape[-1]
    quantiles = torch.linspace(1.0/(n_quantiles+1), n_quantiles/(n_quantiles+1), n_quantiles, device=device).unsqueeze(0)

    if sort_mu:
        mu, _ = torch.sort(mu, dim=-1)

    errors = y - mu # -> (batch_size, n_quantiles)

    pinball_loss = torch.max((quantiles - 1) * errors, quantiles * errors) # -> (batch_size, n_quantiles)
    pinball_loss = pinball_loss.mean(dim=-1) # -> (batch_size)

    if reduce == "mean":
        pinball_loss = pinball_loss.mean()
    elif reduce == "sum":
        pinball_loss = pinball_loss.sum()
    
    return pinball_loss
