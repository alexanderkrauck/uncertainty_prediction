"""
Utitlity functions for evaluation. (Without gradients)

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"


# Standard libraries
from abc import ABC, abstractmethod

# Third-party libraries
import torch
from torch import Tensor
import numpy as np
from typing import Dict

# Local/Application Specific
from .models.basic_architectures import ConditionalDensityEstimator


def make_to_pass_precomputed_variables(
    precomputed_variables: dict, num_steps: int, idx: int
):
    to_pass_precomputed_variables = {}
    for key, value in precomputed_variables.items():
        if isinstance(value, tuple):
            new_tuple = tuple(
                tup_element[idx].unsqueeze(0).expand(num_steps, *tup_element[idx].shape)
                for tup_element in value
            )
            to_pass_precomputed_variables[key] = new_tuple
        elif isinstance(value, Tensor):
            to_pass_precomputed_variables[key] = (
                value[idx].unsqueeze(0).expand(num_steps, *value[idx].shape)
            )
        else:
            raise ValueError(
                f"precomputed_variables must be a dict of tensors or tuples. {key} is not."
            )

class BaseEvaluationFunction(ABC):

    @abstractmethod
    def is_density_evaluation_function(self) -> bool:
        pass

    @abstractmethod
    def __call__(
        self,
        y_space: Tensor,
        densities: Tensor,
        x_batch: Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, Tensor] = None,
        reduce="mean",
        **kwargs,
    ) -> float:
        pass

class HellingerDistance(BaseEvaluationFunction):

    def is_density_evaluation_function(self):
        return True

    def __call__(  # TODO make y multidimensial-able
        y_space: Tensor,
        densities: Tensor,
        x_batch: Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, Tensor] = None,
        reduce="mean",
        **kwargs,
    ) -> float:
        """
        Calculates the Hellinger distance between the true density and the estimated density for each x in x_batch.

        Parameters
        ----------
        y_space : Tensor
            y space to evaluate the density at of shape (num_steps, 1)
        densities : Tensor
            True densities of shape (batch_size, num_steps)
        x_batch : Tensor
            x values of shape (batch_size, x_size)
        model : ConditionalDensityEstimator
            Model to use to estimate the density.
        precomputed_variables : Dict[str, Tensor], optional
            Precomputed variables to pass to the model, by default None. Can be used to speed up the calculation of the density significantly.
        reduce : str, optional
            How to reduce the Hellinger distance across the batch, by default "mean".

        Returns
        -------
        float
            Hellinger distance
        """

        num_steps = y_space.shape[0]
        if model.y_size != 1:  # NOTE: Hellinger distance is only implemented for 1D y
            return -1

        hellinger_distances = []

        for idx in range(x_batch.shape[0]):
            # Calculate true density
            x_space = x_batch[idx].unsqueeze(0).expand(num_steps, -1)
            true_densities = densities[idx]  # True density

            # Calculate estimated density
            if precomputed_variables:
                to_pass_precomputed_variables = make_to_pass_precomputed_variables(
                    precomputed_variables, num_steps, idx
                )

                estimated_densities = model.get_density(
                    x_space, y_space, False, **to_pass_precomputed_variables
                )
            else:
                estimated_densities = model.get_density(x_space, y_space, False)
            # Calculate Hellinger distance component wise
            # sqrt(p(x)) - sqrt(q(x)) and then square
            diff_sq = (
                torch.sqrt(
                    torch.tensor(true_densities, dtype=torch.float32, device=x_batch.device)
                )
                - torch.sqrt(estimated_densities)
            ) ** 2
            h_distance = torch.sqrt(
                torch.sum(diff_sq) / 2
            )  # Integrate and multiply by 1/sqrt(2)

            hellinger_distances.append(h_distance.item())

        if reduce == "mean":
            hellinger_distances = np.mean(hellinger_distances)
        elif reduce == "sum":
            hellinger_distances = np.sum(hellinger_distances)

        return hellinger_distances


class KLDivergence(BaseEvaluationFunction):

    def is_density_evaluation_function(self):
        return True

    def __call__(
        y_space: torch.Tensor,
        densities: torch.Tensor,
        x_batch: torch.Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, torch.Tensor] = None,
        reduce="mean",
        **kwargs,
    ) -> float:
        """
        Calculates the KL divergence between the true density and the estimated density for each x in x_batch.

        Parameters
        ----------
        y_space : torch.Tensor
            y space to evaluate the density at of shape (num_steps, 1)
        densities : torch.Tensor
            True densities of shape (batch_size, num_steps)
        x_batch : torch.Tensor
            x values of shape (batch_size, x_size)
        model : ConditionalDensityEstimator
            Model to use to estimate the density.
        precomputed_variables : Dict[str, torch.Tensor], optional
            Precomputed variables to pass to the model, by default None.
        reduce : str, optional
            How to reduce the KL divergence across the batch, by default "mean".

        Returns
        -------
        float
            KL divergence
        """

        if model.y_size != 1:  # NOTE: Hellinger distance is only implemented for 1D y
            return -1

        num_steps = y_space.shape[0]
        kl_divergences = []

        for idx in range(x_batch.shape[0]):
            x_space = x_batch[idx].unsqueeze(0).expand(num_steps, -1)
            true_densities = densities[idx]  # True density P

            if precomputed_variables:
                to_pass_precomputed_variables = make_to_pass_precomputed_variables(
                    precomputed_variables, num_steps, idx
                )
                estimated_densities = model.get_density(
                    x_space, y_space, False, **to_pass_precomputed_variables
                )
            else:
                estimated_densities = model.get_density(x_space, y_space, False)

            # Avoid division by zero and log of zero by adding a small constant
            epsilon = 1e-12
            true_densities = true_densities + epsilon
            estimated_densities = estimated_densities + epsilon

            # KL Divergence calculation
            kl_div = torch.sum(
                true_densities * torch.log(true_densities / estimated_densities)
            )

            kl_divergences.append(kl_div.item())

        if reduce == "mean":
            return np.mean(kl_divergences)
        elif reduce == "sum":
            return np.sum(kl_divergences)

        return kl_divergences

class WassersteinDistance(BaseEvaluationFunction):

    def is_density_evaluation_function(self):
        return True

    def __call__(
        y_space: torch.Tensor,
        densities: torch.Tensor,
        x_batch: torch.Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, torch.Tensor] = None,
        reduce="mean",
        wasserstein_p: float = 1.0,
        **kwargs,
    ) -> float:
        """
        Calculates the Wasserstein distance between the true density and the estimated density for each x in x_batch.

        Parameters
        ----------
        y_space : torch.Tensor
            y space to evaluate the density at of shape (num_steps, 1)
        densities : torch.Tensor
            True densities of shape (batch_size, num_steps)
        x_batch : torch.Tensor
            x values of shape (batch_size, x_size)
        model : ConditionalDensityEstimator
            Model to use to estimate the density.
        precomputed_variables : Dict[str, torch.Tensor], optional
            Precomputed variables to pass to the model, by default None.
        reduce : str, optional
            How to reduce the Wasserstein distance across the batch, by default "mean".
        wasserstein_p : float
            p for the Wasserstein distance. 1 for the 1-Wasserstein distance, 2 for the 2-Wasserstein distance, etc.

        Returns
        -------
        float
            Wasserstein distance
        """

        num_steps = y_space.shape[0]
        wasserstein_distances = []

        for idx in range(x_batch.shape[0]):
            x_space = x_batch[idx].unsqueeze(0).expand(num_steps, -1)
            true_densities = densities[idx]  # True density

            if precomputed_variables:
                to_pass_precomputed_variables = make_to_pass_precomputed_variables(
                    precomputed_variables, num_steps, idx
                )
                estimated_densities = model.get_density(
                    x_space, y_space, False, **to_pass_precomputed_variables
                )
            else:
                estimated_densities = model.get_density(x_space, y_space, False)

            # Calculate cumulative sums
            true_cumulative = torch.cumsum(true_densities, dim=0)
            estimated_cumulative = torch.cumsum(estimated_densities, dim=0)

            # Wasserstein distance calculation
            w_dist = torch.sum(
                torch.abs((true_cumulative - estimated_cumulative) ** wasserstein_p)
            ) ** (1 / wasserstein_p)

            wasserstein_distances.append(w_dist.item())

        if reduce == "mean":
            return np.mean(wasserstein_distances)
        elif reduce == "sum":
            return np.sum(wasserstein_distances)

        return wasserstein_distances

class MiscalibrationArea(BaseEvaluationFunction):

    def is_density_evaluation_function(self):
        return False
    
    def __call__(
            x_batch: torch.Tensor,
            y_batch: torch.Tensor,
            model: ConditionalDensityEstimator,
            **kwargs):
        raise NotImplementedError("MiscalibrationArea is not implemented yet.")
        
