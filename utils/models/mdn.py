from .basic_architectures import MLP, ConditionalDensityEstimator
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional
from torch import Tensor
import numpy as np

from ..data_module import TrainingDataModule #TODO: not ideal class dependencies here. Idealy would have some sort of models module class that contains the data dependencies. But this is not a priority right now

ACTIVATION_FUNCTION_MAP = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
    "elu": nn.ELU(),
    "selu": nn.SELU(),
}

DISTRIBUTION_MAP = {
    "gaussian": torch.distributions.Normal,
    "laplacian": torch.distributions.Laplace,
}


class MDN(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden: list,
        n_distributions: int,
        dropout_rate: float = 0.2,
        activation_function: str = "relu",
        distribution_type: str = "gaussian",
        std_stability_mode: str = "tanh",
        tanh_std_stability: float = 3.0,
        **kwargs,
    ):
        """
        
        Parameters
        ----------
        train_data_module : 
            The train data module used to train the model. This is used to get the mean and std of the data to normalize the input and output.
        n_hidden : list
            List of hidden layer sizes.
        n_distributions : int
            Number of distributions to use.
        dropout_rate : float, optional
            Dropout rate, by default 0.2
        activation_function : str, optional
            Activation function to use, by default "relu"
        distribution_type : str, optional
            Distribution type to use, by default "gaussian"
        std_stability_mode : str, optional
            Mode to use for std stability. Can be "none", "tanh" or "softplus", by default "none"
        tanh_std_stability : Optional[float]
            Tanh std stability value, by default 3.0. Only used if std_stability_mode is "tanh"
        """
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

        self.tanh_std_stability = tanh_std_stability
        self.std_stability_mode = std_stability_mode.lower()
        if self.std_stability_mode not in ["none", "tanh", "softplus"]:
            raise ValueError(
                f"std_stability_mode must be one of 'none', 'tanh' or 'softplus', but is {self.std_stability_mode}"
            )
        if self.std_stability_mode == "tanh" and self.tanh_std_stability is None:
            raise ValueError(
                "tanh_std_stability must be set if std_stability_mode is tanh"
            )

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        n_inputs = train_data_module.train_dataset.x.shape[1]
        distribution_type = distribution_type.lower()
        if distribution_type in DISTRIBUTION_MAP:
            self.distribution_class = DISTRIBUTION_MAP[distribution_type.lower()]
        else:
            raise ValueError(
                f"Distribution type {distribution_type} not supported."
            )
        self.n_distributions = n_distributions
        self.mlp = MLP(
            n_inputs,
            n_hidden,
            n_distributions * 3,
            dropout_rate,
            activation_function,
            **kwargs,
        )

    def forward(self, x, y=None):
        x = (x - self.mean_x) / self.std_x

        mlp_out = self.mlp(x)
        logits, mu, log_sigma = torch.split(mlp_out, self.n_distributions, dim=1) 

        if self.std_stability_mode == "tanh":
            log_sigma = F.tanh(log_sigma) * self.tanh_std_stability #TODO !!!!!!!!!!!!!!! justify that. its intended to be a hack to prevent sigmas from getting too large
            sigma = torch.exp(log_sigma)
        elif self.std_stability_mode == "softplus":
            sigma = F.softplus(log_sigma)
        else:
            sigma = torch.exp(log_sigma) #this can lead to instability easily as sigmas can get extremely large

        weights = F.softmax(logits, dim=1)

        mu = self.std_y * mu + self.mean_y
        sigma = self.std_y * sigma

        return_dict = {"weights": weights, "mu": mu, "sigma": sigma}
        return return_dict

    def eval_output(
        self, y, output, reduce="mean", reliability_loss_weight: float = 0.0, **kwargs
    ):
        mdn_loss = self.mdn_loss_fn(y, **output, reduce=reduce)

        metric_dict = {}
        for key, value in output.items():
            val = value.detach().cpu().numpy()
            metric_dict["mean_" + key] = (
                np.mean(val) if reduce == "mean" else np.mean(val) * y.shape[0]
            )
            metric_dict["std_" + key] = (
                np.std(val) if reduce == "mean" else np.std(val) * y.shape[0]
            )

        if reliability_loss_weight > 0:
            reliability_loss = self.reliabiltiy_loss_fn(y, **output, reduce=reduce)
            loss = mdn_loss + reliability_loss_weight * reliability_loss
            return loss, {**metric_dict, **{
                "loss": loss.item(),
                "mdn_loss": mdn_loss.item(),
                "reliability_loss": reliability_loss.item(),
            }}
        return mdn_loss, {**metric_dict, **{
            "loss": mdn_loss.item(),
            "mdn_loss": mdn_loss.item(),
        }}

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

        distribution = self.distribution_class(mu, sigma + numeric_stability) # this should be differentiable (ChatGPT) but maybe confirm it
        drawn_samples = distribution.sample((n_samples,)).transpose(0, 1)
        component_indices = torch.multinomial(weights, n_samples, replacement=True) #this is non differentiable
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
