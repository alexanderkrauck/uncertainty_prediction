"""
This file contains the implementation of a transformer that does a CDE. Only for 1 output dimension for now.

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
    DISTRIBUTION_MAP,
)

from ..data_module import (
    TrainingDataModule,
)  # TODO: not ideal class dependencies here. Ideally would have some sort of models module class that contains the data dependencies. But this is not a priority right now

from torch.nn import TransformerEncoder, TransformerEncoderLayer


class LearnedPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model, n_bins):
        super(LearnedPositionalEmbedding, self).__init__()

        self.pos_embedding = torch.nn.Embedding(n_bins, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(0)).unsqueeze(-1).expand(-1, x.size(1)).to(x.device)
        return x + self.pos_embedding(positions)


class CDETransformer(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        n_hidden_in: list = [32],
        n_hidden_out: list = [32],
        n_hidden_intermediate: int = 64,
        embedding_size: int = 32,
        n_bins: int = 128,
        num_heads: int = 4,
        num_transformer_layers: int = 1,
        dropout_rate: float = 0.1,
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
        super().__init__(train_data_module)

        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        self.min_y, self.max_y = (
            train_data_module.train_dataset.y.min(),
            train_data_module.train_dataset.y.max(),
        )

        self.n_bins = n_bins

        self.min_y = nn.Parameter(self.min_y, requires_grad=False)
        self.max_y = nn.Parameter(self.max_y, requires_grad=False)
        self.bin_range = nn.Parameter(
            self.max_y - self.min_y + self.std_y, requires_grad=False
        )
        self.bin_width = nn.Parameter(self.bin_range / n_bins, requires_grad=False)

        self.mlp_in = MLP(
            self.x_size,
            n_hidden_in,
            embedding_size,
            dropout_rate,
            activation_function,
        )

        self.mlp_out = MLP(
            embedding_size,
            n_hidden_out,
            1,
            dropout_rate,
            activation_function,
        )

        encoder_layers = TransformerEncoderLayer(
            embedding_size,
            num_heads,
            dim_feedforward=n_hidden_intermediate,
            dropout=dropout_rate,
            activation=activation_function,
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, num_layers=num_transformer_layers
        )

        self.positional_embedding = LearnedPositionalEmbedding(embedding_size, n_bins)
        # y_space is the center of the output bins
        self.y_space = nn.Parameter(
            torch.linspace(
                (self.min_y - self.std_y + self.bin_width / 2).item(),
                (self.max_y + self.std_y - self.bin_width / 2).item(),
                n_bins,
            ),
            requires_grad=False,
        )

    def forward(self, x, y=None, normalised_output_domain: bool = False):

        x = (x - self.mean_x) / self.std_x

        embedding = self.mlp_in(x)
        embedding = embedding.unsqueeze(0).expand(self.n_bins, -1, -1)
        embedding = self.positional_embedding(embedding)

        embedding = self.transformer_encoder(embedding)

        embedding = self.mlp_out(embedding).squeeze(-1)
        #embedding = F.softmax(embedding, dim=0)
        embedding = F.softplus(embedding)
        embedding = embedding / torch.sum(embedding, dim=0, keepdim=True) #here bin size 1 is assumed after normalisation
        embedding = embedding.T #transpose to have the same shape as the output [batch_size, n_bins]

        #here we need to rescale the embeddings/densities using the jacobian determinant of the transformation.
        embedding = embedding * self.n_bins / self.bin_range #have to scale to the jacobi determinant of the transformation
        if normalised_output_domain:
            embedding = embedding * self.std_y #scale to the normalised domain using the jacobi determinant of the transformation

        return_dict = {
            "bin_weights": embedding
        }  

        return return_dict

    def eval_output(
        self,
        y,
        output,
        normalised_output_domain: bool = False,
        reduce="mean",
        **kwargs,
    ):
        if normalised_output_domain:
            y = (y - self.mean_y) / self.std_y

        density = self.get_density(
            None, y, normalised_output_domain=normalised_output_domain, **output
        )
        if reduce == "mean":
            loss = -torch.log(density + 1e-8).mean()
        elif reduce == "sum":
            loss = -torch.log(density + 1e-8).sum()

        metric_dict = {}

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

        if (
            normalised_output_domain
        ):  # Because we don't want to add the loss if we are not in the normalised domain as it is not meaningful
            metric_dict["loss"] = loss.item()

        return loss, metric_dict

    def get_density(
        self,
        x: Optional[Tensor],
        y: Tensor,
        normalised_output_domain: bool = False,
        numeric_stability: float = 1e-8,
        bin_weights: Optional[Tensor] = None,
        **other_outputs,
    ) -> Tensor:
        if (
            bin_weights is None
        ):  # Assume that if anything is None then also weights is None
            output = self(x, y, normalised_output_domain=normalised_output_domain)
            bin_weights = output["bin_weights"]

        # Don't need to consider the normalised_output_domain here, because the bin_weights are already normalised or not depending on the normalised_output_domain
        out_bounds = (y < self.min_y - self.std_y) | (
            y > self.max_y + self.std_y
        )  # If y is outside the range of the bins, then the density is 0 for this method
        #    return torch.zeros(y.shape[0], device = y.device)

        y_bin = (y - self.y_space).abs().argmin(dim=-1)

        densities = bin_weights.gather(1, y_bin.unsqueeze(1)).squeeze()
        densities[out_bounds.squeeze()] = 0

        # TODO consider adding smoothing to the bin_weights
        return densities
