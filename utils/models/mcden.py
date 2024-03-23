"""
This file contains the implementation of a mixture of conditional density estimators model.

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
from typing import Tuple, Optional, List

# Local/Application Specific
from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
)
from ..data_module import (
    TrainingDataModule,
)  # TODO: not ideal class dependencies here. Ideally would have some sort of models module class that contains the data dependencies. But this is not a priority right now


class MCDEN(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        sub_architectures: List[ConditionalDensityEstimator],
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
        sub_architectures :
            A list of conditional density estimators that are used as the sub-architectures of the mixture model.
        n_hidden :
            The number of hidden units in the MLP that is used to calculate the weights of the mixture model.
        dropout_rate :
            The dropout rate used in the MLP.
        activation_function :
            The activation function used in the MLP.
        """
        super().__init__(train_data_module)

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        self.mlp = MLP(
            self.x_size,
            n_hidden,
            len(sub_architectures),
            dropout_rate,
            activation_function,
            **kwargs,
        )

        self.sub_archs = nn.ModuleList(sub_architectures)


    def forward(self, x, y=None, normalised_output_domain: bool = False):
        
        sub_arch_outputs = []
        for sub_arch in self.sub_archs:
            output = sub_arch(x, y, normalised_output_domain=normalised_output_domain)
            sub_arch_outputs.append(output)

        x = (x - self.mean_x) / self.std_x

        mlp_out = self.mlp(x)

        weights = F.softmax(mlp_out, dim=1)

        return_dict = {"weights": weights}

        for i, sub_arch_output in enumerate(sub_arch_outputs):
            for key, value in sub_arch_output.items():
                return_dict[key + f"__{i}"] = value

        return return_dict

    def eval_output(
        self,
        y,
        output,
        normalised_output_domain: bool = False,
        reduce="mean",
        miscalibration_area_loss_weight: float = 0.0,
        weights_entropy_loss_weight: float = 0.0,
        force_alternative_loss_calculations: bool = True,
        **kwargs,
    ):
        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        density = self.get_density(y, normalised_output_domain=normalised_output_domain, **output)
        loss = -torch.log(density + 1e-8)

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
                metric_dict["nll_loss_normalized"] = (loss - torch.log(self.std_y).sum()).item()
            else:
                metric_dict["nll_loss_normalized"] = (
                    (loss - torch.log(self.std_y).sum() * y.shape[0]) 
                ).item()

        if weights_entropy_loss_weight > 0 or (
            not normalised_output_domain and force_alternative_loss_calculations
        ):
            weights_entropy = (
                torch.distributions.categorical.Categorical(probs=output["weights"])
                .entropy()
            )
            if reduce == "mean":
                weights_entropy = weights_entropy.mean()
            else:
                weights_entropy = weights_entropy.sum()
            metric_dict["weights_entropy"] = weights_entropy.item()



        if weights_entropy_loss_weight > 0:
            loss = (
                loss - weights_entropy_loss_weight * weights_entropy
            )  # because we want to regularize for high entropy



        if (
            normalised_output_domain
        ):  # Because we don't want to add the loss if we are not in the normalised domain as it is not meaningful
            metric_dict["loss"] = loss.item()

        return loss, metric_dict

    def get_density(
        self,
        x: Optional[Tensor], #if we have weights we do not need x
        y: Tensor,
        normalised_output_domain: bool = False,
        numeric_stability: float = 1e-8,
        weights: Optional[Tensor] = None,
        **other_outputs,
    ) -> Tensor:
        if weights is None:#Assume that if anything is None then also weights is None
            output = self(x, y, normalised_output_domain=normalised_output_domain)
            weights = output["weights"]
            other_outputs = output

        sub_output_dicts = [{} for _ in range(len(self.sub_archs))]
        for key in other_outputs.keys():
            splitkey = key.split("__")
            sub_output_dicts[int(splitkey[1])][splitkey[0]] = other_outputs[key]
        
        densities = torch.empty((y.shape[0], len(self.sub_archs)))
        for idx in range(len(self.sub_archs)):
            densities[:, idx] = self.sub_archs[idx].get_density(y, normalised_output_domain=normalised_output_domain, **sub_output_dicts[idx])
        
        density = (densities * weights).sum(1)


        # if not normalised_output_domain:
        #    density = density * (1 / self.std_y).prod() #(we multiply by the jacobian determinant of the transformation from the normalised to the unnormalised domain)

        return density
