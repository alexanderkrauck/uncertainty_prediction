from .basic_architectures import MLP
import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor
from typing import List


class VAE(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dims_enc: List[int],
        hidden_dims_dec: List[int],
    ):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim

        # Initialize encoder and decoder as MLPs
        self.encoder = MLP(
            n_input=input_dim, n_hidden=hidden_dims_enc, n_output=latent_dim * 2
        )  # *2 for mean and logvar
        self.decoder = MLP(
            n_input=latent_dim, n_hidden=hidden_dims_dec, n_output=input_dim
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
        return x_hat, mean, logvar
