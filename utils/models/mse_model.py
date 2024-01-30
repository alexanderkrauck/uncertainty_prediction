"""
This file contains the implementation of the MSE model for conditional density estimation.

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
from torch.utils.data import DataLoader, TensorDataset

# Typing
from typing import Tuple, Optional

# Local/Application Specific
from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
    DISTRIBUTION_MAP,
)
from .loss_functions import nlll, miscalibration_area_fn
from ..data_module import (
    TrainingDataModule,
)  # TODO: not ideal class dependencies here. Ideally would have some sort of models module class that contains the data dependencies. But this is not a priority right now


class MSEModel(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden: list = [32, 32],
        dropout_rate: float = 0.2,
        activation_function: str = "relu",
        **kwargs,
    ):
        """

        Parameters
        ----------
        train_data_module :
            The train data module used to train the model. This is used to get the mean and std of the data to normalize the input and output.
        n_hidden : list
            List of hidden layer sizes.
        dropout_rate : float, optional
            Dropout rate, by default 0.2
        activation_function : str, optional
            Activation function to use, by default "relu"
        """
        super().__init__()

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        self.mean_x, self.std_x = (
            train_data_module.train_dataset.mean_x,
            train_data_module.train_dataset.std_x,
        )
        self.mean_y, self.std_y = (
            train_data_module.train_dataset.mean_y,
            train_data_module.train_dataset.std_y,
        )

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        self.x_size = train_data_module.train_dataset.x.shape[1]
        self.y_size = train_data_module.train_dataset.y.shape[1]

        self.mlp = MLP(
            self.x_size,
            n_hidden,
            self.y_size,
            dropout_rate,
            activation_function,
            **kwargs,
        )

        uses_validation_data = train_data_module.val_dataset is not None
        if uses_validation_data:
            self.mse_calibration_samples_x = torch.tensor(
                train_data_module.val_dataset.x
            )
            self.mse_calibration_samples_y = torch.tensor(
                train_data_module.val_dataset.y
            )
        else:
            self.mse_calibration_samples_x = torch.tensor(
                train_data_module.train_dataset.x
            )
            self.mse_calibration_samples_y = torch.tensor(
                train_data_module.train_dataset.y
            )

        self.calibrate_mse_std()

    def forward(self, x, y=None, normalised_output_domain: bool = False):
        x = (x - self.mean_x) / self.std_x

        mu = self.mlp(x)

        if not normalised_output_domain:
            mu = self.std_y.unsqueeze(-1) * mu + self.mean_y.unsqueeze(-1)

        return_dict = {"mu": mu}
        return return_dict

    def calibrate_mse_std(self):
        with torch.no_grad():
            device = self.parameters().__next__().device
            dataset = TensorDataset(self.mse_calibration_samples_x, self.mse_calibration_samples_y)
        
            # Create a DataLoader to handle batching of the dataset
            data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
            

            running_loss = 0.0
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                output = self(batch_x, batch_y, normalised_output_domain=False)["mu"]
                loss = F.mse_loss(output, batch_y, reduction="sum")
                running_loss = running_loss + loss

            self.std = torch.sqrt(running_loss / len(dataset))

    def eval_output(
        self,
        y,
        output,
        normalised_output_domain: bool = False,
        reduce="mean",
        miscalibration_area_loss_weight: float = 0.0,
        force_all_loss_calculation: bool = True,
        **kwargs,
    ):
        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        loss = F.mse_loss(output["mu"], y, reduction=reduce)

        metric_dict = {}

        if normalised_output_domain:
            metric_dict["mse_loss_normalized"] = loss.item()
        else:
            metric_dict["mse_loss"] = loss.item()

        if miscalibration_area_loss_weight > 0 or (
            not normalised_output_domain and force_all_loss_calculation
        ):
            
            mu = output["mu"]
            weights = torch.ones_like(mu)
            mu = mu.unsqueeze(-1)
            sigma = torch.ones_like(mu) * self.std_y

            metric_dict["nll_loss"] = nlll(
                torch.distributions.Normal, y, weights, mu, sigma, reduce, **kwargs
            ).item()
            metric_dict["misclibration_area"] = miscalibration_area_fn(
                torch.distributions.Normal,
                y,
                weights,
                mu,
                sigma,
                reduce=reduce,
                **kwargs,
            ).item()

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
        mu: Optional[Tensor] = None,
    ) -> Tensor:
        if mu is None:
            output = self(x, y, normalised_output_domain=normalised_output_domain)
            mu, = output["mu"]

        distribution = torch.distributions.Normal(
            mu, torch.ones_like(mu) * self.std_y
        )  # for numerical stability if predicted sigma is too close to 0

        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        densities = (
            torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability
        )  # for numerical stability because outliers can cause this to be 0
        densities = densities.sum(1)
        density = torch.sum(densities, dim=1)

        # if not normalised_output_domain:
        #    density = density * (1 / self.std_y).prod() #(we multiply by the jacobian determinant of the transformation from the normalised to the unnormalised domain)

        return density

    def eval(self):
        super().eval()
        
        self.calibrate_mse_std()