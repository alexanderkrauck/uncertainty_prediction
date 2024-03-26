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
from typing import Dict, Tuple, Union, List
Numeric = Union[int, float, np.ndarray]
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib

from scipy.interpolate import interp1d

matplotlib.use("Agg")

# Local/Application Specific
from utils.models.basic_architectures import ConditionalDensityEstimator

from utils.utils import make_to_pass_precomputed_variables



class BaseEvaluationFunction(ABC):
    @abstractmethod
    def is_density_evaluation_function(self) -> bool:
        pass

    @abstractmethod
    def is_slow(self) -> bool:
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

    @property
    def output_size(self):
        return 1


class HellingerDistance(BaseEvaluationFunction):
    def is_density_evaluation_function(self):
        return True
    
    def is_slow(self):
        return True

    def __call__(  # TODO make y multidimensial-able
        self,
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
        step_size = (y_space[1] - y_space[0]).abs()

        densities = densities * step_size

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
            # step_size = y_space[1] - y_space[0]
            estimated_densities *= step_size

            diff_sq = (
                torch.sqrt(true_densities) - torch.sqrt(estimated_densities)
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
    
    def is_slow(self):
        return True

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
        step_size = (y_space[1] - y_space[0]).abs()
        kl_divergences = []

        densities = densities * step_size + 1e-12

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

            estimated_densities = estimated_densities + 1e-12
            estimated_densities *= step_size

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
    
    def is_slow(self):
        return True

    def __call__(
        self,
        y_space: torch.Tensor,
        densities: torch.Tensor,
        x_batch: torch.Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, torch.Tensor] = None,
        reduce="mean",
        wasserstein_p: float = 2.0,
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
        step_size = (y_space[1] - y_space[0]).abs()
        wasserstein_distances = []

        densities = densities * step_size

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

            estimated_densities *= step_size

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


def infer_quantalized_conformal_p(
        model,
        val_data_loader,
        device,
        y_space,
        has_distribution,
        conformal_p: float = 0.9,
        **training_hyperparameters,
):
    
    if isinstance(y_space, torch.Tensor):
        y_space = y_space.clone().to(device).view(-1, 1)
    else:
        y_space = torch.tensor(y_space, device=device).view(-1, 1)

    all_required_ps = []
    cp = ConformalPrediction()

    for idx, batch in enumerate(val_data_loader):
        if has_distribution:
            x_batch, y_batch, densities = batch
        else:
            x_batch, y_batch = batch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        model_output = model(x_batch, y_batch)

        all_required_ps.extend(cp(y_space, x_batch, y_batch, model, model_output, return_required_conformal_p_instead=True, **training_hyperparameters))

    return np.quantile(all_required_ps, conformal_p)

class ConformalPrediction(BaseEvaluationFunction):
    """Conformal Prediction evaluation function.

    The idea is to predict the conditional density function and then to use Highest Density Regions (HDRs) with p-values to make predictions.
    The p-values are then used to calculate the size of the HDRs and the proportion of the true value that is in the HDRs.
    """


    def is_density_evaluation_function(self):
        return False
    
    def is_slow(self):
        return True
    
    @property
    def output_size(self):
        return 4
    
    def get_y_space_fine(self, y_space: torch.Tensor, conformal_finer: int):
        return torch.linspace(
            y_space.min().item(), y_space.max().item(),  y_space.shape[0] * conformal_finer, device=y_space.device
        )

    def __call__(
        self,
        y_space: torch.Tensor,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, torch.Tensor] = None,
        reduce="mean",
        conformal_p: float = 0.90, 
        conformal_finer: float = 10,
        return_required_conformal_p_instead: bool = False,
        return_grid_instead: bool = False,
        **kwargs,
    ):
        #TODO: Implement Conformal Prediction
        num_steps = y_space.shape[0]
        step_size = (y_space[1] - y_space[0]).abs().item()
        fine_step_size = step_size / conformal_finer

        y_space_fine = self.get_y_space_fine(y_space, conformal_finer)

        conformal_sizes = []
        conformal_in_set = []
        conformal_sizes_contiguous = []
        conformal_in_set_contiguous = []

        required_ps = []
        for idx in range(x_batch.shape[0]):
            x_space = x_batch[idx].unsqueeze(0).expand(num_steps, -1)

            if precomputed_variables:
                to_pass_precomputed_variables = make_to_pass_precomputed_variables(
                    precomputed_variables, num_steps, idx
                )
                estimated_densities = model.get_density(
                    x_space, y_space, False, **to_pass_precomputed_variables
                )
            else:
                estimated_densities = model.get_density(x_space, y_space, False)

            interp = interp1d(y_space.flatten().detach().cpu().numpy(), estimated_densities.flatten().detach().cpu().numpy(), kind="linear")
            estimated_densities_fine = torch.tensor(interp(y_space_fine.detach().cpu().numpy()), device=y_space.device)
            estimated_densities_fine_normalized = (
                estimated_densities_fine / estimated_densities_fine.sum()
            )

            #normalized_estimated_densities = estimated_densities / estimated_densities.sum(dim=0)
            sorted_indices = torch.argsort(estimated_densities_fine_normalized, descending=True)
            cumulative_sum = torch.cumsum(estimated_densities_fine_normalized[sorted_indices], dim=0)

            # Find the smallest set of indices such that the sum of the estimated densities is greater than the conformal p
            if return_required_conformal_p_instead:
                wanted_index = torch.argmin(torch.abs(y_batch[idx] - y_space_fine[sorted_indices])).item() #this is the index of the closest bin to the true value in the sorted indices
                required_p = cumulative_sum[wanted_index] #this is the smallests conformal p that we would have required to include the true value in the set
                required_ps.append(required_p.item())
            else:
                smaller_count = (cumulative_sum < conformal_p).sum()
                conformal_set = sorted_indices[:smaller_count + 1]

                if return_grid_instead:
                    grid = torch.zeros_like(y_space_fine, dtype=torch.bool)
                    grid[conformal_set] = True
                    required_ps.append(grid)
                    continue

                if torch.min(torch.abs(y_batch[idx] - y_space_fine[conformal_set])).item() < fine_step_size/2:
                    conformal_in_set.append(1)
                else:
                    conformal_in_set.append(0)

                conformal_size = fine_step_size * len(conformal_set)
                conformal_sizes.append(conformal_size)

                # Now calculate if we want a contiguous set of indices. NOTE: If sizes are equal then the set is contiguous already
                conformal_size_contiguous = (conformal_set.max() - conformal_set.min() + 1) * fine_step_size #because conformal set are indices (+ one because we need to contain the whole histogram bins in the set)
                if y_space_fine[conformal_set.min()].item() - fine_step_size/2 < y_batch[idx] <= y_space_fine[conformal_set.max()].item() + fine_step_size/2:
                    conformal_in_set_contiguous.append(1)
                else:
                    conformal_in_set_contiguous.append(0)
                conformal_sizes_contiguous.append(conformal_size_contiguous.item())

        if return_required_conformal_p_instead or return_grid_instead:
            return required_ps
        
        if reduce == "mean":
            conformal_sizes = np.mean(conformal_sizes)
            conformal_in_set = np.mean(conformal_in_set)
            conformal_sizes_contiguous = np.mean(conformal_sizes_contiguous)
            conformal_in_set_contiguous = np.mean(conformal_in_set_contiguous)
        elif reduce == "sum":
            conformal_sizes = np.sum(conformal_sizes)
            conformal_in_set = np.sum(conformal_in_set)
            conformal_sizes_contiguous = np.sum(conformal_sizes_contiguous)
            conformal_in_set_contiguous = np.sum(conformal_in_set_contiguous)

        return np.array([conformal_sizes, conformal_in_set, conformal_sizes_contiguous, conformal_in_set_contiguous], dtype=np.float32)


class Miscalibration(BaseEvaluationFunction):
    def is_density_evaluation_function(self):
        return False
    
    def is_slow(self) -> bool:
        return True
    
    @property
    def output_size(self):
        return 3

    def __call__(
        self,
        y_space: torch.Tensor,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor,
        model: ConditionalDensityEstimator,
        precomputed_variables: Dict[str, torch.Tensor] = None,
        reduce="mean",
        **kwargs,
    ):
        
        num_steps = y_space.shape[0]
        quantiles = torch.arange(5, 96, 10, device=y_space.device) / 100

        samples = []
        for idx in range(x_batch.shape[0]):
            x_space = x_batch[idx].unsqueeze(0).expand(num_steps, -1)

            if precomputed_variables:
                to_pass_precomputed_variables = make_to_pass_precomputed_variables(
                    precomputed_variables, num_steps, idx
                )
                estimated_densities = model.get_density(
                    x_space, y_space, False, **to_pass_precomputed_variables
                )
            else:
                estimated_densities = model.get_density(x_space, y_space, False)

            normalized_estimated_densities = estimated_densities / estimated_densities.sum(dim=0)
            
            cumsum = normalized_estimated_densities.cumsum(dim=0)
            quantile_indices = (cumsum.unsqueeze(-1) - quantiles).abs().argmin(0)

            sums = y_space[quantile_indices] - y_batch[idx] > 0
            samples.append(sums)
            
        samples = torch.cat(samples, dim=1).T
        empirical_quantiles = samples.float().mean(dim=0)
        empirical_quantile_diffs = (empirical_quantiles - quantiles).abs()
        mean_absolute_calibration_error = empirical_quantile_diffs.mean().item()
        root_mean_squared_calibration_error = torch.sqrt(empirical_quantile_diffs.square().mean()).item()
        miscalibration_area = Miscalibration.miscalibration_area_from_proportions(quantiles.cpu().numpy(), empirical_quantiles.cpu().numpy())

        if reduce == "sum":
            mean_absolute_calibration_error = mean_absolute_calibration_error * x_batch.shape[0]
            root_mean_squared_calibration_error = root_mean_squared_calibration_error * x_batch.shape[0]
            miscalibration_area = miscalibration_area * x_batch.shape[0]

        return np.array([mean_absolute_calibration_error, root_mean_squared_calibration_error, miscalibration_area], dtype=np.float32)
    
    def miscalibration_area_from_proportions(
        exp_proportions: np.ndarray, obs_proportions: np.ndarray
    ) -> float:
        """Miscalibration area from expected and observed proportions lists.

        This function returns the same output as `miscalibration_area` directly from a list
        of expected proportions (the proportion of data that you expect to observe within
        prediction intervals) and a list of observed proportions (the proportion data that
        you observe within prediction intervals).

        Args:
            exp_proportions: expected proportion of data within prediction intervals.
            obs_proportions: observed proportion of data within prediction intervals.

        Returns:
            A single scalar that contains the miscalibration area.
        """
        areas = Miscalibration.trapezoid_area(
            exp_proportions[:-1],
            exp_proportions[:-1],
            obs_proportions[:-1],
            exp_proportions[1:],
            exp_proportions[1:],
            obs_proportions[1:],
            absolute=True,
        )
        return areas.sum()

    @staticmethod
    def trapezoid_area(
        xl: np.ndarray,
        al: np.ndarray,
        bl: np.ndarray,
        xr: np.ndarray,
        ar: np.ndarray,
        br: np.ndarray,
        absolute: bool = True,
    ) -> Numeric:
        """
        Calculate the area of a vertical-sided trapezoid, formed connecting the following points:
            (xl, al) - (xl, bl) - (xr, br) - (xr, ar) - (xl, al)

        This function considers the case that the edges of the trapezoid might cross,
        and explicitly accounts for this.

        Args:
            xl: The x coordinate of the left-hand points of the trapezoid
            al: The y coordinate of the first left-hand point of the trapezoid
            bl: The y coordinate of the second left-hand point of the trapezoid
            xr: The x coordinate of the right-hand points of the trapezoid
            ar: The y coordinate of the first right-hand point of the trapezoid
            br: The y coordinate of the second right-hand point of the trapezoid
            absolute: Whether to calculate the absolute area, or allow a negative area (e.g. if a and b are swapped)

        Returns: The area of the given trapezoid.

        """

        # Differences
        dl = bl - al
        dr = br - ar

        # The ordering is the same for both iff they do not cross.
        cross = dl * dr < 0

        # Treat the degenerate case as a trapezoid
        cross = cross * (1 - ((dl == 0) * (dr == 0)))

        # trapezoid for non-crossing lines
        area_trapezoid = (xr - xl) * 0.5 * ((bl - al) + (br - ar))
        if absolute:
            area_trapezoid = np.abs(area_trapezoid)

        # Hourglass for crossing lines.
        # NaNs should only appear in the degenerate and parallel cases.
        # Those NaNs won't get through the final multiplication so it's ok.
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = Miscalibration.intersection((xl, bl), (xr, br), (xl, al), (xr, ar))[0]
        tl_area = 0.5 * (bl - al) * (x_intersect - xl)
        tr_area = 0.5 * (br - ar) * (xr - x_intersect)
        if absolute:
            area_hourglass = np.abs(tl_area) + np.abs(tr_area)
        else:
            area_hourglass = tl_area + tr_area

        # The nan_to_num function allows us to do 0 * nan = 0
        return (1 - cross) * area_trapezoid + cross * np.nan_to_num(area_hourglass)

    @staticmethod
    def intersection(
        p1: Tuple[Numeric, Numeric],
        p2: Tuple[Numeric, Numeric],
        p3: Tuple[Numeric, Numeric],
        p4: Tuple[Numeric, Numeric],
    ) -> Tuple[Numeric, Numeric]:
        """
        Calculate the intersection of two lines between four points, as defined in
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection.

        This is an array option and works can be used to calculate the intersections of
        entire arrays of points at the same time.

        Args:
            p1: The point (x1, y1), first point of Line 1
            p2: The point (x2, y2), second point of Line 1
            p3: The point (x3, y3), first point of Line 2
            p4: The point (x4, y4), second point of Line 2

        Returns: The point of intersection of the two lines, or (np.nan, np.nan) if the lines are parallel

        """

        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        D = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

        x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / D
        y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / D

        return x, y


def log_plot(
    summary_writer: SummaryWriter,
    model: ConditionalDensityEstimator,
    data_loader: DataLoader,
    y_space: Tensor,
    num_samples: int = 2,
    device: str = "cpu",
    show_other_metrics: bool = True,
):
    num_steps = y_space.shape[0]

    model.eval()

    y_space = torch.tensor(y_space, device=device).view(-1, 1)
    data_loader = DataLoader(data_loader.dataset, batch_size=num_samples, shuffle=False)

    with torch.no_grad():
        minibatch = next(iter(data_loader))
        x_batch, y_batch, densities = minibatch
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        densities = densities.to(device)

        if show_other_metrics:
            hellinger_dist = HellingerDistance()(
                y_space, densities, x_batch, model, reduce=None
            )
            wasserstein_dist = WassersteinDistance()(
                y_space, densities, x_batch, model, reduce=None
            )
            kl_divergence = KLDivergence()(
                y_space, densities, x_batch, model, reduce=None
            )

        for idx in range(x_batch.shape[0]):
            x = x_batch[idx].unsqueeze(0).expand(num_steps, -1)

            model.eval()

            estimated_density = model.get_density(x, y_space, False)

            plt.figure(figsize=(10, 10))
            plt.plot(
                y_space.cpu().numpy(),
                densities[idx].cpu().numpy(),
                label="True Densities",
            )
            plt.plot(
                y_space.cpu().numpy(),
                estimated_density.cpu().numpy(),
                label="Estimated Densities",
            )
            plt.axvline(
                y_batch[idx].cpu().numpy(),
                color="black",
                linestyle="--",
                label="True y",
            )
            plt.legend()
            plt.title(f"Sample {idx}")
            if show_other_metrics:
                plt.figtext(
                    0.75,
                    0.5,
                    "Hellinger Distance: {:.4f}\nWasserstein Distance: {:.4f}\nKL Divergence: {:.4f}".format(
                        hellinger_dist[idx], wasserstein_dist[idx], kl_divergence[idx]
                    ),
                    fontsize=12,
                    ha="center",
                    va="center",
                    wrap=True,
                )
            # plt.show()
            summary_writer.add_figure(f"Sample {idx}", plt.gcf())

def log_permutation_feature_importance(
    summary_writer: SummaryWriter,
    model: ConditionalDensityEstimator,
    data_loader: DataLoader,
    device: str = "cpu",
    show_plot_instead: bool = False,
    feature_names: List[str] = None
):
    model.eval()
    model.to(device)

    num_samples = len(data_loader.dataset)
    num_features = data_loader.dataset.x.shape[1]

    feature_importances = torch.empty((num_samples, num_features))

    current_sample = 0

    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            if len(batch) == 3:
                x_batch, y_batch, densities = batch
            else:
                x_batch, y_batch = batch
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            original_density = model.get_density(x_batch, y_batch, False)
            
            for feature_idx in range(x_batch.shape[1]):
                permutation = torch.randperm(x_batch.shape[0])

                x_permuted = x_batch.clone()
                x_permuted[:, feature_idx] = x_permuted[permutation, feature_idx]

                estimated_density = model.get_density(x_permuted, y_batch, False)



                feature_importances[current_sample:current_sample + x_batch.shape[0], feature_idx] = (original_density.log() - estimated_density.log()).abs()
            
            current_sample += x_batch.shape[0]
    
    feature_importances = feature_importances.mean(dim=0)

    if summary_writer:
        for idx, importance in enumerate(feature_importances):
            summary_writer.add_scalar(f"Feature Importance/Feature {idx}", importance.item())

    #Create a bar plot of the feature importances
    plt.figure(figsize=(10, 10))
    plt.bar(np.arange(num_features), feature_importances.cpu().numpy())
    plt.title("Feature Importances")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    if feature_names:
        plt.xticks(np.arange(num_features), feature_names, rotation='vertical')
    else:
        plt.xticks(np.arange(num_features), np.arange(num_features), rotation='vertical')
    if show_plot_instead:
        plt.show()

        sorted_idx = np.argsort(feature_importances.cpu().numpy())
        for idx in sorted_idx:
            feature_name = feature_names[idx] if feature_names else f"Feature {idx}"
            print(f"{feature_name}: {feature_importances[idx].item()}")
    else:
        summary_writer.add_figure("Feature Importances", plt.gcf())




