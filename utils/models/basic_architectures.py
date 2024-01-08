from torch import nn
import torch
from typing import List


class MLP(nn.Module):
    def __init__(
        self,
        n_input: int,
        n_hidden: List[int],
        n_output: int,
        dropout_rate: float = 0.1,
        activation_function: nn.Module = nn.ReLU(),
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
        """
        super(MLP, self).__init__()

        self.dropout = nn.Dropout(p=dropout_rate)
        self.hidden_layers = nn.ModuleList([nn.Linear(n_input, n_hidden[0])])
        self.hidden_layers.extend(
            [nn.Linear(n_hidden[i], n_hidden[i + 1]) for i in range(len(n_hidden) - 1)]
        )
        self.output_layer = nn.Linear(n_hidden[-1], n_output)
        self.activation_function = activation_function

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
            x = self.activation_function(hidden_layer(x))
            x = self.dropout(x)
        x = self.output_layer(x)
        return x