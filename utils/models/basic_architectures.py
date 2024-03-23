"""
This file contains the basic architectures used in the conditional density estimators.

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

# Standard libraries
from abc import ABC, abstractmethod
from typing import List, Dict

# Third-party libraries
import torch
from torch import nn, Tensor

from utils.data_module import (
    TrainingDataModule,
)
from utils.utils import make_to_pass_precomputed_variables

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


class ConditionalDensityEstimator(ABC, torch.nn.Module):
    """
    Abstract base class for conditional density estimators.
    """

    def __init__(self, train_data_module: TrainingDataModule, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

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

        self.x_size = train_data_module.train_dataset.x.shape[-1]
        self.y_size = train_data_module.train_dataset.y.shape[1]
        self.y_space = train_data_module.train_dataset.y_space
        self.y_space  = nn.Parameter(self.y_space, requires_grad=False)
        self.time_series = train_data_module.train_dataset.is_timeseries

    @abstractmethod
    def forward(
        self, x: Tensor, y: Tensor, normalised_output_domain: bool
    ) -> Dict[str, Tensor]:
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            The input features.
        y : torch.Tensor
            The target features.
        normalised_output_domain : bool
            Whether the output domain should be normalised or not.

        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary containing the outputs of the network.
        """
        pass

    @abstractmethod
    def eval_output(
        self,
        y: Tensor,
        output: Tensor,
        normalised_output_domain: bool = True,
        reduce: str = "mean",
        **loss_hyperparameters
    ) -> Tensor:
        """
        Returns the loss of the output distribution at the given points."""
        pass

    def training_pass(
        self,
        x: Tensor,
        y: Tensor,
        loss_calculation_in_normalised_domain: bool = True,
        **loss_hyperparameters
    ) -> Tensor:
        output = self(
            x,
            y,
            normalised_output_domain=loss_calculation_in_normalised_domain,
        )

        return self.eval_output(
            y,
            output,
            normalised_output_domain=loss_calculation_in_normalised_domain,
            **loss_hyperparameters
        )

    @abstractmethod
    def get_density(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        normalised_output_domain: bool = True,
        numeric_stability: float = 1e-6,
        **precomputed_variables
    ) -> torch.Tensor:
        """
        Returns the density of the output distribution at the given points."""
        pass

    def get_nll(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        normalised_output_domain: bool = True,
        numeric_stability: float = 1e-6,
        **kwargs
    ) -> torch.Tensor:
        """
        Returns the negative log-likelihood of the output distribution at the given points.
        """

        probabilities = self.get_density(
            x, y, normalised_output_domain, numeric_stability
        )
        nll = -torch.log(probabilities)

        return nll

    
    def get_statistics(
        self,
        x: torch.Tensor,
        normalised_output_domain: bool = True,
        numeric_stability: float = 1e-6,
        **precomputed_variables
    ):
        
        medians = []
        modes = []
        means = []
        with torch.no_grad():
            for x_val in x:
                x_val = x_val.unsqueeze(0).expand(self.y_space.shape[0], *[-1 for _ in range(len(x_val.shape))])

                densities = self.get_density(
                    x_val,
                    self.y_space.unsqueeze(-1),
                    normalised_output_domain,
                    numeric_stability,
                    **precomputed_variables
                )
                #we just take the nearest to 0.5
                medians.append(self.y_space[torch.argmin(torch.abs(torch.cumsum(densities/densities.sum(),0) - 0.5))])
                modes.append(
                    self.y_space[torch.argmax(densities)]
                )
                means.append(
                    (self.y_space * densities / densities.sum()).sum()
                )

        return {
            "means": torch.tensor(means),
            "modes": torch.tensor(modes),
            "medians": torch.tensor(medians),
        }
    
    @staticmethod
    def avg_hellinger_dist(distributions):
        """
        Calculates the average Hellinger distance between all pairs of distributions.
        
        Args:
            distributions (torch.Tensor): A tensor of shape [n_distributions, n_grid_elements]
                                        containing the estimated densities.
        
        Returns:
            float: The average Hellinger distance between all pairs of distributions.
        """
        n_distributions = distributions.size(0)
        sqrt_dists = torch.sqrt(distributions.unsqueeze(1))
        
        dists = sqrt_dists - sqrt_dists.transpose(0, 1)
        hellinger_dists = torch.sqrt(0.5 * (dists ** 2).sum(dim=-1))
        
        # Remove diagonal elements (distances to self)
        hellinger_dists = hellinger_dists[~torch.eye(n_distributions, dtype=bool)].view(-1)
        
        return hellinger_dists.mean()

    def monte_carlo_dropout_distributional_uncertainty(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        n_samples: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Returns the Monte Carlo estimate of the Hellinger Distance between multiple samples from the model distribution with a
        prior distribution. See  https://arxiv.org/abs/1506.02142 for more details.

        This should be a kind of measure of uncertainty in the model. If the model is very uncertain, the Hellinger distance
        will be large, since the model will predict very different distributions for the same input.
        """

        num_steps = len(self.y_space)
        hellinger_distances = []

        self.train(True) #set the model to training mode so we get different dropout masks
        with torch.no_grad():
            
            for x_val in x:
                sample_x = x_val.unsqueeze(0).expand(n_samples, *[-1 for _ in range(len(x_val.shape))])
                sample_y = y.unsqueeze(0).expand(n_samples, -1, -1)
                precomputed_variables = self.forward(sample_x, sample_y, False)

                #for each of those precomputed variables, we need to predict the grid of y-value densities
                densities = []
                for idx in range(n_samples):
                    to_pass_precomputed_variables = make_to_pass_precomputed_variables(
                            precomputed_variables, num_steps, idx
                    )
                    estimated_densities = self.get_density(
                        None, self.y_space.unsqueeze(-1), False, **to_pass_precomputed_variables
                    )
                    densities.append(estimated_densities)

                densities = torch.stack(densities, dim=0)
                densities = densities / densities.sum(dim=1, keepdim=True) #normalise the densities
               

                #now we need to calculate some average difference between the distributions
                #Hellinger distance is a good choice since it scales with the probability mass
                #KL divergence is a bad choice since it can explode. Moreover it isn't symmetric
                avg_hellinger_dist = self.avg_hellinger_dist(densities)
                hellinger_distances.append(avg_hellinger_dist)

        return torch.tensor(hellinger_distances)




class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: List[int],
        n_output: int,
        dropout_rate: float = 0.1,
        activation_function: nn.Module = nn.ReLU(),
        use_layer_norm: bool = False,
        **kwargs
    ):
        """
        Initializes the MLP.

        Parameters:
        -----------
        n_input : int
            Number of input features.
        n_hidden : list of int
            List containing the number of neurons in each hidden layer.
        n_output : int
            Number of output features.
        dropout_rate : float
            Dropout rate to be used in dropout layers.
        activation_function : nn.Module
            Activation function to be used in hidden layers.
        use_layer_norm: bool
            Whether to use layer normalization or not.
        """
        super(MLP, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.use_layer_norm = use_layer_norm
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, n_hidden[0])])
        self.hidden_layers.extend(
            [nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)]
        )
        self.output_layer = nn.Linear(n_hidden[-1], n_output)
        self.activation_function = activation_function

        self.initialize_weights()

    def initialize_weights(self):
        """
        Initializes the weights of the network.
        """

        # By default, PyTorch uses Kaiming initialization for ReLU and its variants
        # For sigmoid and tanh, we use Xavier initialization
        if isinstance(self.activation_function, nn.Tanh) or isinstance(
            self.activation_function, nn.Sigmoid
        ):
            for layer in self.hidden_layers:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
        -----------
        x : torch.Tensor
            Input tensor to the network.

        Returns:
        --------
        torch.Tensor
            Output tensor of the network after passing through all layers.
        """
        x = self.dropout(x)
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            if self.use_layer_norm:
                x = torch.layer_norm(x, x.shape[1:])
            x = self.activation_function(x)
            x = self.dropout(x)
        x = self.output_layer(x)
        return x
