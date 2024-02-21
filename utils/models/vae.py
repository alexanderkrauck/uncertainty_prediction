from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
)
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from typing import List

from ..data_module import (
    TrainingDataModule,
)


class VAEConditionalDensityEstimator(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module: TrainingDataModule,
        input_dim,
        latent_dim,
        n_hidden_enc: List[int],
        n_hidden_dec: List[int],
        dropout_rate: float = 0.0,
        activation_function: str = "relu",
        **kwargs,
    ):
        super(VAEConditionalDensityEstimator, self).__init__()

        self.latent_dim = latent_dim

        activation_function = activation_function.lower()
        if activation_function not in ACTIVATION_FUNCTION_MAP:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )
        activation_function = ACTIVATION_FUNCTION_MAP[activation_function]

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

        self.latent_dim = latent_dim
        self.latent_distribution = torch.distributions.Normal(
            torch.zeros(latent_dim), torch.ones(latent_dim)
        )

        # Initialize encoder and decoder as MLPs
        self.encoder = MLP(
            n_input=input_dim,
            n_hidden=n_hidden_enc,
            n_output=latent_dim * 2,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            **kwargs,
        )  # *2 for mean and logvar
        self.decoder = MLP(
            n_input=latent_dim,
            n_hidden=n_hidden_dec,
            n_output=input_dim,
            dropout_rate=dropout_rate,
            activation_function=activation_function,
            **kwargs,
        )

    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = x.chunk(
            2, dim=-1
        )  # Split the encoder output into mean and logvar
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_hat = self.decode(z)

        output_dict = {
            "x_hat": x_hat,
            "mean": mean,
            "logvar": logvar,
        }
        return output_dict

    def get_density(
        self,
        x: Tensor,
        y: Tensor,
        numeric_stability: float = 1e-6,
        **precomputed_variables,
    ) -> Tensor:
        net_input = torch.cat([x, y], dim=1)
        mean, logvar = self.encode(net_input)
        densities = self.latent_distribution.log_prob(mean).sum(-1).exp()

        lower_bound = self.mean_y - self.std_y * 4
        upper_bound = self.mean_y + self.std_y * 4
        step_size = (upper_bound - lower_bound) / 100
        steps = torch.arange(lower_bound, upper_bound, step_size).view(-1, 1).float()

        for x_inner, y_inner in zip(x, y):
            x_inner = x_inner.repeat(steps.shape[0], 1)
            net_input_inner = torch.cat([x_inner, steps], dim=1)

            mean_inner, logvar_inner = self.encode(net_input_inner)

            densities_inner = (
                self.latent_distribution.log_prob(mean_inner).sum(-1).exp()
            )

            diffs_inner = mean_inner[1:] - mean_inner[:-1]
            norms_inner = diffs_inner.norm(dim=-1)
            averages_inner = torch.zeros(mean_inner.shape[0])
            averages_inner[0] = norms_inner[0]
            averages_inner[-1] = norms_inner[-1]
            averages_inner[1:-1] = (norms_inner[1:] + norms_inner[:-1]) / 2
            averages_inner = averages_inner / torch.sum(averages_inner)

            prior_x = (averages_inner * densities_inner).sum()

    def eval_output(
        self, y: Tensor, output: Tensor, reduce: str = "mean", **loss_hyperparameters
    ) -> Tensor:
        pass
