from .basic_architectures import MLP
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

activation_functions = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "sigmoid": nn.Sigmoid(),
    "leaky_relu": nn.LeakyReLU(),
}


class MDN(nn.Module):
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
        if activation_function in activation_functions:
            activation_function = activation_functions[activation_function]
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
        self.distribution_type = distribution_type
        self.n_distributions = n_distributions
        self.mlp = MLP(
            n_inputs,
            n_hidden,
            n_distributions * 3,
            dropout_rate,
            activation_function,
            **kwargs,
        )

    def forward(self, x):
        x = (x - self.mean_x) / self.std_y

        mlp_out = self.mlp(x)
        logits, mu, log_sigma = torch.split(mlp_out, self.n_distributions, dim=1)

        sigma = torch.exp(log_sigma)
        weights = F.softmax(logits, dim=1)

        mu = self.std_y * mu + self.mean_y
        sigma = self.std_y * sigma

        return weights, mu, sigma
