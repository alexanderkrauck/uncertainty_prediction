from .basic_architectures import MLP, ConditionalDensityEstimator
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
import numpy as np

ACTIVATION_FUNCTION_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}

DISTRIBUTION_MAP = {
    "gaussian": torch.distributions.Normal,
    "laplacian": torch.distributions.Laplace,
}


class MDN(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module,
        n_hidden: list,
        n_distributions: int,
        dropout_rate: float = 0.2,
        activation_function: str = "relu",
        distribution_type: str = "gaussian",
        **kwargs,
    ):
        super().__init__()

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        self.mean_x, self.std_x = train_data_module.train_dataset.scaler_x
        self.mean_y, self.std_y = train_data_module.train_dataset.scaler_y

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        n_inputs = train_data_module.train_dataset.x.shape[1]
        self.distribution_class = DISTRIBUTION_MAP[distribution_type.lower()]
        self.n_distributions = n_distributions
        self.mlp = MLP(
            n_inputs,
            n_hidden,
            n_distributions * 3,
            dropout_rate,
            activation_function,
            **kwargs,
        )

    def forward(self, x, y = None):
        x = (x - self.mean_x) / self.std_x

        mlp_out = self.mlp(x)
        logits, mu, log_sigma = torch.split(mlp_out, self.n_distributions, dim=1)

        sigma = torch.exp(log_sigma)
        weights = F.softmax(logits, dim=1)

        mu = self.std_y * mu + self.mean_y
        sigma = self.std_y * sigma

        return_dict = {"weights": weights, "mu": mu, "sigma": sigma}
        return return_dict

    def eval_output(self, y, output, reduce = "mean", reliability_loss_weight: float = 0.0, **kwargs):
        mdn_loss = self.mdn_loss_fn(y, **output, reduce=reduce)
        if reliability_loss_weight > 0:
            reliability_loss = self.reliabiltiy_loss_fn(y, **output, reduce=reduce)
            loss = mdn_loss + reliability_loss_weight * reliability_loss
            return loss, {
                "loss": loss.item(),
                "mdn_loss": mdn_loss.item(),
                "reliability_loss": reliability_loss.item(),
            }
        return mdn_loss, {"loss": mdn_loss.item(), "mdn_loss": mdn_loss.item()}

    def get_density(
        self,
        x: Tensor,
        y: Tensor,
        numeric_stability: float = 1e-6,
        weights: Optional[Tensor] = None,
        mu: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
    ) -> Tensor:
        if weights is None or mu is None or sigma is None:
            weights, mu, sigma = self(x)
        distribution = self.distribution_class(
            mu, sigma + numeric_stability
        )  # for numerical stability if predicted sigma is too close to 0
        densities = (
            torch.exp(distribution.log_prob(y)) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        density = torch.sum(densities * weights, dim=1)
        return density

    def mdn_loss_fn(self, y, weights, mu, sigma, reduce="mean", numeric_stability=1e-6):
        distribution = self.distribution_class(
            mu, sigma + numeric_stability
        )  # for numerical stability if predicted sigma is too close to 0
        loss = (
            torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        loss = torch.sum(loss * weights, dim=1)
        loss = -torch.log(loss)
        if reduce == "mean":
            loss = torch.mean(loss)
        elif reduce == "sum":
            loss = torch.sum(loss)
        if loss.item() == np.inf or loss.item() == -np.inf:
            print("inf loss")
        return loss

    def reliabiltiy_loss_fn(
        self,
        y: Tensor,
        weights: Tensor,
        mu: Tensor,
        sigma: Tensor,
        numeric_stability: int = 1e-6,
        n_samples: int = 100,
        reduce="mean",
    ):
        device = y.device

        distribution = self.distribution_class(mu, sigma + numeric_stability)
        drawn_samples = distribution.sample((n_samples,)).transpose(0, 1)
        component_indices = torch.multinomial(weights, n_samples, replacement=True)
        effective_samples = torch.gather(
            drawn_samples, -1, component_indices.unsqueeze(-1)
        ).squeeze(-1)
        y = y.squeeze(-1)

        quantiles = torch.arange(5, 96, 10, device=device) / 100

        upper_bounds = torch.quantile(effective_samples, quantiles, dim=-1)
        y_r = (y < upper_bounds).sum(dim=-1) / y.shape[0]

        reliability_loss = (
            (y_r - quantiles).abs().mean()
        )  # maybe use trapz shomehow instead of mean

        if reduce == "sum":
            reliability_loss = reliability_loss * y.shape[0]

        return reliability_loss
