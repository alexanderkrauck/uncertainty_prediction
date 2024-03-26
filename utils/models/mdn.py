"""
This file contains the implementation of the Mixture Density Network (MDN) architecture.

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

# Third-party libraries
import torch
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F

# Typing
from typing import Tuple, Optional

# Local/Application Specific
from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
    DISTRIBUTION_MAP,
)
from .loss_functions import nlll, miscalibration_area_fn, pinball_loss_fn
from ..data_module import (
    TrainingDataModule,
)  # TODO: not ideal class dependencies here. Ideally would have some sort of models module class that contains the data dependencies. But this is not a priority right now


class MDN(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden: list = [32, 32],
        n_distributions: int = 10,
        dropout_rate: float = 0.2,
        activation_function: str = "relu",
        distribution_type: str = "gaussian",
        std_stability_mode: str = "softplus",
        tanh_std_stability: float = 3.0,
        force_equal_weights: bool = False,
        force_equal_std: bool = False,
        train_equal_std: bool = False,
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
        force_equal_weights : bool, optional
            Whether to force the weights to be equal, by default False
            This in particular used for having essentially multiple quantile regression.
        force_equal_std : bool, optional
            Whether to force the std to be equal, by default False.
            This is in particular useful when we use the multiple quantile regression method for predicting the CDE
            We reach that by fitting a small gaussian at every quantile and forcing the std to be equal.
        train_equal_std : bool, optional
            Only used if 'force_equal_std' is True. Whether to train the std to be equal or not, by default False. It is initialized with
            one divided by the square root of components.
        """
        super().__init__(train_data_module)

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

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

        distribution_type = distribution_type.lower()
        if distribution_type in DISTRIBUTION_MAP:
            self.distribution_class = DISTRIBUTION_MAP[distribution_type.lower()]
        else:
            raise ValueError(f"Distribution type {distribution_type} not supported.")
        self.n_distributions = n_distributions
        self.split_sizes = [
            self.n_distributions,
            self.n_distributions * self.y_size,
            self.n_distributions * self.y_size,
        ]

        input_size = self.x_size
        if (
            self.time_series
        ):  # If time series data, we use an RNN as the first layer and remove the first layer size from the list as that is the RNN size
            self.rnn = nn.GRU(
                input_size, n_hidden[0], batch_first=True, dropout=dropout_rate
            )
            input_size = n_hidden[0]
            n_hidden = n_hidden[1:]

        self.mlp = MLP(
            input_size,
            n_hidden,
            (n_distributions * self.y_size) * 2 + n_distributions,
            dropout_rate,
            activation_function,
            **kwargs,
        )

        self.force_equal_weights = force_equal_weights
        self.force_equal_std = force_equal_std
        self.train_equal_std = train_equal_std
        if self.force_equal_std:
            self.equal_component_std = nn.Parameter(
                torch.ones(1, self.y_size, 1) / self.n_distributions**0.5,
                requires_grad=train_equal_std,
            )

    def forward(self, x, y=None, normalised_output_domain: bool = False):
        x = (x - self.mean_x) / self.std_x

        if self.time_series:
            x, _ = self.rnn(x)
            x = x[:, -1]  # We only take the last output of the RNN

        mlp_out = self.mlp(x)
        logits_weights, mu, log_sigma = torch.split(mlp_out, self.split_sizes, dim=1)

        if self.force_equal_std:
            # This is useful for the case where we want to force the model to predict the same std for all distributions
            # but for each sample in the batch separately.
            # NOTE: we need to do that before applying std stability because otherwise while training the std can
            # become negative which breaks the model.
            log_sigma = torch.ones_like(log_sigma, device=log_sigma.device) * self.equal_component_std
        if self.std_stability_mode == "tanh":
            log_sigma = F.tanh(log_sigma) * self.tanh_std_stability
            sigma = torch.exp(log_sigma)
        elif (
            self.std_stability_mode == "softplus"
        ):  # softplus is a more stable version of the exponential function
            sigma = F.softplus(log_sigma)
        else:
            sigma = torch.exp(
                log_sigma
            )  # this can lead to instability easily as sigmas can get extremely large

        weights = F.softmax(logits_weights, dim=1)

        mu = mu.reshape(-1, self.y_size, self.n_distributions)
        sigma = sigma.reshape(-1, self.y_size, self.n_distributions)

        if self.force_equal_weights:
            weights = torch.ones_like(weights) / self.n_distributions

        if not normalised_output_domain:
            mu = self.std_y.unsqueeze(-1) * mu + self.mean_y.unsqueeze(-1)
            sigma = self.std_y.unsqueeze(-1) * sigma

        return_dict = {"weights": weights, "mu": mu, "sigma": sigma}
        return return_dict

    def eval_output(
        self,
        y,
        output,
        normalised_output_domain: bool = False,
        reduce="mean",
        miscalibration_area_loss_weight: float = 0.0,
        weights_entropy_loss_weight: float = 0.0,
        pinball_loss_weight: float = 0.0,
        nll_loss_weight: float = 1.0,
        force_alternative_loss_calculations: bool = True,
        **kwargs,
    ):
        """Evaluate the output of the model.

        Parameters
        ----------
        y : Tensor
            The target values.
        output : dict
            The output of the model.
        normalised_output_domain : bool, optional
            Whether the output should be in the normalised domain, by default False
        reduce : str, optional
            The reduction type, by default "mean"
        miscalibration_area_loss_weight : float, optional
            The weight of the miscalibration area loss, by default 0.0
        weights_entropy_loss_weight : float, optional
            The weight of the weights entropy loss, by default 0.0
        pinball_loss_weight : float, optional
            The weight of the pinball loss, by default 0.0
            The quantiles of the pinball loss are assumed uniformly distributed between 0 and 1 with the number of
            quantiles equal to the number of distributions. So the predicted 'mu' are the quantiles.
            If only the pinball loss is used, this is equivalent to multiple quantile regression.
        nll_loss_weight : float, optional
            The weight of the negative log likelihood loss, by default 1.0 (this is the default because it is the main loss)
        force_alternative_loss_calculations : bool, optional
            Whether to force the alternative loss calculations even if their weights are 0 for metrics, by default True
        """

        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        
        if self.force_equal_weights and self.force_equal_std:
            # We detach here the means and weights only, because if we force equal stds we don't want to detach them
            # because either we do not have grad anyways or we want to train them, but only the stds here.
            loss = nlll(
                self.distribution_class,
                y,
                output["weights"].detach(),
                output["mu"].detach(),
                output["sigma"],
                reduce=reduce,
                **kwargs,
            )
        else:
            loss = nlll(self.distribution_class, y, **output, reduce=reduce, **kwargs)

        metric_dict = {}
        for key, value in output.items():
            val = value.detach().cpu().numpy()
            metric_dict["mean_" + key] = (
                np.mean(val) if reduce == "mean" else np.mean(val) * y.shape[0]
            )
            metric_dict["std_" + key] = (
                np.std(val) if reduce == "mean" else np.std(val) * y.shape[0]
            )

        if normalised_output_domain:
            metric_dict["nll_loss_normalized"] = loss.item()
            if reduce == "mean":
                metric_dict["nll_loss"] = (loss + torch.log(self.std_y).sum()).item()
            else:
                metric_dict["nll_loss"] = (
                    (loss + torch.log(self.std_y).sum() * y.shape[0])
                ).item()
        else:
            metric_dict["nll_loss"] = loss.item()
            if reduce == "mean":
                metric_dict["nll_loss_normalized"] = (
                    loss - torch.log(self.std_y).sum()
                ).item()
            else:
                metric_dict["nll_loss_normalized"] = (
                    (loss - torch.log(self.std_y).sum() * y.shape[0])
                ).item()

        if weights_entropy_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            weights_entropy = torch.distributions.categorical.Categorical(
                probs=output["weights"]
            ).entropy()
            if reduce == "mean":
                weights_entropy = weights_entropy.mean()
            else:
                weights_entropy = weights_entropy.sum()
            metric_dict["weights_entropy"] = weights_entropy.item()

        if miscalibration_area_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            miscalibration_area = miscalibration_area_fn(
                self.distribution_class, y, **output, reduce=reduce, **kwargs
            )
            metric_dict["misclibration_area"] = miscalibration_area.item()

        if pinball_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):

            pinball_loss = pinball_loss_fn(y, **output, reduce=reduce, **kwargs)
            metric_dict["pinball_loss"] = pinball_loss.item()

        # We add the losses to the main loss
        loss *= nll_loss_weight

        if weights_entropy_loss_weight > 0:
            loss = loss - weights_entropy_loss_weight * weights_entropy

        if miscalibration_area_loss_weight > 0:
            loss = loss + miscalibration_area_loss_weight * miscalibration_area

        if pinball_loss_weight > 0:
            loss = loss + pinball_loss_weight * pinball_loss

        if (
            normalised_output_domain
        ):  # Because we don't want to add the loss if we are not in the normalised domain as it is not meaningful
            metric_dict["loss"] = loss.item()

        return loss, metric_dict

    def get_density(
        self,
        x: Tensor,
        y: Tensor,
        normalised_output_domain: bool = False,
        numeric_stability: float = 1e-8,
        weights: Optional[Tensor] = None,
        mu: Optional[Tensor] = None,
        sigma: Optional[Tensor] = None,
    ) -> Tensor:
        if weights is None or mu is None or sigma is None:
            output = self(x, y, normalised_output_domain=normalised_output_domain)
            weights, mu, sigma = output["weights"], output["mu"], output["sigma"]
        distribution = self.distribution_class(
            mu, sigma + numeric_stability
        )  # for numerical stability if predicted sigma is too close to 0

        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        densities = (
            torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        densities = densities.sum(1)
        density = torch.sum(densities * weights, dim=1)

        # if not normalised_output_domain:
        #    density = density * (1 / self.std_y).prod() #(we multiply by the jacobian determinant of the transformation from the normalised to the unnormalised domain)

        return density
