import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .basic_architectures import (
    MLP,
    ConditionalDensityEstimator,
    ACTIVATION_FUNCTION_MAP,
    DISTRIBUTION_MAP,
)

from typing import Tuple, Optional, List

from abc import ABC, abstractmethod


class Flow(ABC, nn.Module):
    """
    Abstract base class for normalizing flows. In particular normalizing flows that have estimated parameters.
    """

    def __init__(self, n_inputs: int):
        super().__init__()
        self.n_inputs = n_inputs

    @abstractmethod
    def forward(self, x: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param x: input tensor
        :return: output tensor and log determinant of the Jacobian
        """
        pass

    @abstractmethod
    def inverse(self, y: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        """
        :param y: output tensor
        :return: input tensor and log determinant of the Jacobian
        """
        pass


class InvertedPlanarFlow(Flow):
    def __init__(self, n_inputs: int):
        super().__init__(n_inputs)
        self.n_parameters = 2 * n_inputs + 1
        self.n_inputs = n_inputs

    def forward(self, x: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Do not require Forward pass for this application")

    def inverse(self, y: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        w = parameters[:, : self.n_inputs]
        u = parameters[:, self.n_inputs : 2 * self.n_inputs]
        b = parameters[:, 2 * self.n_inputs]

        # need to consider that we have a batch dimension
        x = y + u * torch.tanh(torch.sum(y * w, dim=1) + b)

        psi = (1 - torch.tanh(torch.sum(y * w, dim=1) + b) ** 2) * w
        log_det_jacobian = torch.log(torch.abs(1 + torch.sum(psi * u, dim=1)))

        return x, log_det_jacobian


class InverseRadialFlow(Flow):
    def __init__(self, n_inputs: int):
        super().__init__(n_inputs)
        self.n_parameters = n_inputs + 2
        self.n_inputs = n_inputs

    def forward(self, x: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("Do not require Forward pass for this application")

        

    def inverse(self, y: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:

        difference = y - parameters[:, : self.n_inputs]
        r = torch.norm(difference, p=1, dim=1)

        alpha = F.softplus(parameters[:, self.n_inputs])
        beta = torch.exp(parameters[:, self.n_inputs + 1]) - 1 #this will be enough to make the log parts positive

        divisor = 1 / (alpha + r)
        x = y + (beta * alpha * divisor).unsqueeze(-1) * difference#Trippe & Turner alternative

        log_det_jac = (self.n_inputs - 1) * torch.log(
            1 + alpha * beta * divisor
        ) + torch.log(1 + alpha * beta * divisor - alpha * beta * r * (divisor**2))
        return x, log_det_jac


class AffineFlow(Flow):
    def __init__(self, n_inputs: int):
        super().__init__(n_inputs)
        self.n_parameters = 2 * n_inputs

    def forward(self, x: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor. Shape (batch_size, n_inputs)
        parameters : Tensor
            Parameters of the flow. Shape (batch_size, 2 * n_inputs)
        """

        a = parameters[:, : self.n_inputs]
        b = parameters[:, self.n_inputs :]

        y = x * torch.exp(a) + b
        log_det_jacobian = torch.sum(a, dim=1)

        return y, log_det_jacobian

    def inverse(self, y: Tensor, parameters: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        y : Tensor
            Input tensor. Shape (batch_size, n_inputs)
        parameters : Tensor
            Parameters of the flow. Shape (batch_size, 2 * n_inputs)
        """

        a = parameters[:, : self.n_inputs]
        b = parameters[:, self.n_inputs :]

        x = (y - b) * torch.exp(-a)
        log_det_jacobian = -torch.sum(a, dim=1)

        return x, log_det_jacobian


FLOW_MAP = {"planar": InvertedPlanarFlow, "radial": InverseRadialFlow, "affine": AffineFlow}


class NFDensityEstimator(ConditionalDensityEstimator):
    def __init__(
        self,
        train_data_module,
        n_hidden: int,
        n_flows: int = 5,
        flows: Optional[List[str]] = None,
        dropout_rate=0.2,
        activation_function="relu",
        **kwargs,
    ):
        super().__init__()
        activation_function = activation_function.lower()
        if activation_function in ACTIVATION_FUNCTION_MAP:
            activation_function = ACTIVATION_FUNCTION_MAP[activation_function]
        else:
            raise ValueError(
                f"Activation function {activation_function} not supported."
            )

        if flows is None:  # As implemented in the paper (Rothfuss et al.)
            flows = ["affine"] + ["radial" for _ in range(n_flows)]
        if len(flows) == 0:
            raise ValueError("Must have at least one flow")
        flows = [flow.lower() for flow in flows]
        for flow in flows:
            if flow not in FLOW_MAP:
                raise ValueError(f"Flow {flow} not supported")

        self.mean_x, self.std_x = train_data_module.train_dataset.scaler_x
        self.mean_y, self.std_y = train_data_module.train_dataset.scaler_y

        self.mean_x = nn.Parameter(self.mean_x, requires_grad=False)
        self.std_x = nn.Parameter(self.std_x, requires_grad=False)
        self.mean_y = nn.Parameter(self.mean_y, requires_grad=False)
        self.std_y = nn.Parameter(self.std_y, requires_grad=False)

        n_inputs = train_data_module.train_dataset.x.shape[1]
        self.y_dim = train_data_module.train_dataset.y.shape[1]

        flows = [FLOW_MAP[flow](self.y_dim) for flow in flows]
        self.n_flows_params = [flow.n_parameters for flow in flows]
        self.flows = nn.ModuleList(flows)
        self.n_flows = len(self.flows)

        n_output = sum(self.n_flows_params)

        self.mlp = MLP(
            n_inputs, n_hidden, n_output, dropout_rate, activation_function, **kwargs
        )

    def forward(self, x, y, normalised_output_domain: bool = False):
        x = (x - self.mean_x) / self.std_x
        params = self.mlp(x)

        # Split the parameters into the different flows
        params = torch.split(params, self.n_flows_params, dim=1)

        return_dict = {"flow_params": params}
        return return_dict

    def eval_output(
        self,
        y: Tensor,
        output: Tensor,
        normalised_output_domain: bool = True,
        reduce: str = "mean",
        **loss_hyperparameters,
    ) -> Tensor:
        y = (y - self.mean_y) / self.std_y

        loss = self.nlll(
            y,
            **output,
            normalised_output_domain=normalised_output_domain,
            reduce=reduce,
            **loss_hyperparameters,
        )

        metric_dict = {}
        if normalised_output_domain:
            metric_dict["nll_loss_normalized"] = loss.item()
        else:
            metric_dict["nll_loss"] = loss.item()

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
        numeric_stability: float = 1e-6,
        flow_params: Optional[Tuple[Tensor]] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor
            Input tensor. Shape (batch_size, x_dim)
        y : Tensor
            Input tensor. Shape (batch_size, y_dim)
        """

        device = y.device

        if flow_params is None:
            flow_params = self(x, y)["flow_params"]

        y = (y - self.mean_y) / self.std_y

        log_det_jacobian = 0
        for flow, param in zip(self.flows, flow_params):
            y, log_det_jac = flow.inverse(y, param)
            log_det_jacobian = log_det_jac + log_det_jacobian

        log_prob = torch.distributions.MultivariateNormal(
            torch.zeros(self.y_dim, device=device), torch.eye(self.y_dim, device=device)
        ).log_prob(y)

        log_prob_original = log_prob + log_det_jacobian
        if not normalised_output_domain:
            log_prob_original = log_prob_original - torch.log(self.std_y)

        return torch.exp(log_prob_original)

    def nlll(
        self,
        y: Tensor,
        flow_params: Tuple[Tensor],
        normalised_output_domain: bool = True,
        reduce: str = "mean",
        **loss_hyperparameters,
    ) -> Tensor:
        """
        Parameters
        ----------
        y : Tensor
            Input tensor. Shape (batch_size, y_dim)
        flow_params : Tuple
            Tuple of parameters for each flow. Shape (batch_size, n_flows)
        """

        device = y.device

        # Apply the flows
        log_det_jacobian = 0
        for flow, param in zip(self.flows, flow_params):
            y, log_det_jac = flow.inverse(y, param)
            log_det_jacobian = log_det_jac + log_det_jacobian

        # Calculate the log likelihood
        log_prob = torch.distributions.MultivariateNormal(
            torch.zeros(self.y_dim, device=device), torch.eye(self.y_dim, device=device)
        ).log_prob(y)
        log_prob_original = log_prob + log_det_jacobian

        if not normalised_output_domain:
            log_prob_original = log_prob_original - torch.log(self.std_y)

        if reduce == "mean":
            nll_loss = -torch.mean(log_prob_original, dim=0)
        elif reduce == "sum":
            nll_loss = -torch.sum(log_prob_original, dim=0)

        return nll_loss
