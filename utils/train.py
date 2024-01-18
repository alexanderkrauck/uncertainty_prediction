import torch
import os
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.data_module import TrainingDataModule, DataModule

import wandb

from typing import List, Type, Dict

from .models.basic_architectures import ConditionalDensityEstimator
from .setup import load_model


# from chatgpt
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def mdn_loss_fn(y, weights, mu, sigma, reduce="mean", numeric_stability=1e-6):
    distribution = torch.distributions.Normal(
        mu, sigma + numeric_stability
    )  # for numerical stability if predicted sigma is too close to 0
    loss = (
        torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability
    )  # for numerical stability because outliers can cause this to be 0
    loss = torch.sum(loss * weights, dim=1)
    loss = -torch.log(loss)
    if reduce == "mean":
        loss = torch.mean(loss)
    elif reduce == "sum":
        loss = torch.sum(loss)
    if loss.item() == np.inf or loss.item() == -np.inf:
        print("inf loss")
    return loss


def hellinger_distance(
    distribution,
    x: Tensor,
    model: ConditionalDensityEstimator,
    precomputed_variables: Dict[str, Tensor] = None,
    reduce="mean",
    numeric_stability=1e-6,
    num_steps=100,
):
    y_space = torch.linspace(-10, 10, num_steps, device=x.device).view(-1, 1)
    hellinger_distances = []

    for idx in range(x.shape[0]):
        # Calculate true density
        x_space = x[idx].unsqueeze(0).expand(num_steps, -1)
        true_densities = distribution.pdf(
            x_space.detach().cpu().numpy(), y_space.detach().cpu().numpy()
        )  # True density

        # Calculate estimated density
        if precomputed_variables:
            to_pass_precomputed_variables = {
                key: value[idx].unsqueeze(0).expand(num_steps, -1)
                for key, value in precomputed_variables.items()
            }
            estimated_densities = model.get_density(
                x_space, y_space, numeric_stability, **to_pass_precomputed_variables
            )
        else:
            estimated_densities = model.get_density(x_space, y_space, numeric_stability)
        # Calculate Hellinger distance component wise
        # sqrt(p(x)) - sqrt(q(x)) and then square
        diff_sq = (
            torch.sqrt(
                torch.tensor(true_densities, dtype=torch.float32, device=x.device)
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


def evaluate_model(
    model: ConditionalDensityEstimator,
    val_loader: DataLoader,
    device: str,
    loss_hyperparameters: dict,
    distribution=None,
    hellinger_dist_first_n_batches: int = 3,
    is_test: bool = False,
):
    model.eval()
    hellinger_dist = 0
    first_n_batch_sizes = 0
    eval_metrics = {}
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            model_output = model(x, y)
            _, current_eval_metrics = model.eval_output(
                y, model_output, "sum", **loss_hyperparameters
            )
            if idx == 0:
                eval_metrics = current_eval_metrics
            else:
                for key, value in current_eval_metrics.items():
                    eval_metrics[key] += value

            if distribution and (
                idx < hellinger_dist_first_n_batches
                or hellinger_dist_first_n_batches == -1
            ):
                hellinger_dist += hellinger_distance(
                    distribution, x, model, model_output, reduce="sum"
                )
                first_n_batch_sizes += x.shape[0]

    for key, value in eval_metrics.items():
        eval_metrics[key] /= len(val_loader.dataset)

    prefix = "test_" if is_test else "val_"
    return_dict = {prefix + key: value for key, value in eval_metrics.items()}

    if distribution:
        hellinger_dist /= first_n_batch_sizes
        eval_metrics[prefix + "hellinger_dist"] = hellinger_dist
    return return_dict


optimizer_map = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def train_model(
    model: ConditionalDensityEstimator,
    train_data_module: TrainingDataModule,
    optimizer: str,
    optimizer_hyperparameters: dict,
    epochs: int,
    batch_size: int,
    device: str,
    loss_hyperparameters: dict,
    eval_metric_for_best_model: str = "val_loss",
    input_noise_x: float = 0.0,
    input_noise_y: float = 0.0,
    clip_gradient_norm: float = 5.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    eval_every_n: int = 1,
    eval_mode: str = "epoch",
    num_steps: int = None,
):
    """Train a model using the given hyperparameters and return the best model and its validation metrics

    Parameters
    ----------
    model : ConditionalDensityEstimator
        Model to train
    train_data_module : TrainingDataModule
        DataModule containing the data (training and validation)
    optimizer : str
        Name of the optimizer to use
    optimizer_hyperparameters : dict
        Hyperparameters for the optimizer
    epochs : int
        Number of epochs to train for
    batch_size : int
        Batch size to use
    device : str
        Device to use
    loss_hyperparameters : dict
        Hyperparameters for the loss function
    eval_metric_for_best_model : str, optional
        Metric to use for early stopping, by default "val_loss"
    input_noise_x : float, optional
        Standard deviation of Gaussian noise to add to the input x, by default 0.0
    input_noise_y : float, optional
        Standard deviation of Gaussian noise to add to the input y, by default 0.0
    clip_gradient_norm : float, optional
        Maximum norm of the gradients, by default 5.0
    early_stopping_patience : int, optional
        Number of epochs/evaluations to wait before early stopping, by default 15
    early_stopping_min_delta : float, optional
        Minimum change in the metric to be considered an improvement, by default 1e-4
    eval_every_n : int, optional
        Evaluate the model every n epochs/steps, by default 1
    eval_mode : str, optional
        Whether to evaluate the model every n epochs or every n steps, by default "epoch". If "step", then eval_every_n is interpreted as the number of steps. If "step" then num_steps must be provided.
    num_steps : int, optional
        Number of steps to do. Required if eval_mode is "step". If provided and eval_mode is "epoch", then it stops after num_steps if it is hit before the number of epochs.
    """

    if eval_mode == "step":
        if num_steps is None:
            raise ValueError("num_steps must be provided if eval_mode is step")

    optimizer = optimizer_map[optimizer.lower()](
        model.parameters(), **optimizer_hyperparameters
    )

    train_loader = train_data_module.get_train_dataloader(batch_size)
    val_loader = train_data_module.get_val_dataloader(batch_size)
    if train_data_module.has_distribution():
        distribution = train_data_module.distribution
    else:
        distribution = None

    best_val_metrics = None
    best_val_loss = np.inf
    best_params = None

    bar = tqdm(range(epochs))
    step = 0
    early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_delta)

    val_metrics = evaluate_model(
        model, val_loader, device, loss_hyperparameters, distribution
    )
    wandb.log({"step": step, **val_metrics, "epoch": 0})

    def log_evaluation(step, epoch=None):
        nonlocal val_metrics, best_val_loss, best_val_metrics, best_params

        val_metrics = evaluate_model(
            model, val_loader, device, loss_hyperparameters, distribution
        )
        log_data = {"step": step, **val_metrics}
        if epoch is not None:
            log_data["epoch"] = epoch

        wandb.log(log_data)

        if val_metrics[eval_metric_for_best_model] < best_val_loss:
            best_val_loss = val_metrics[eval_metric_for_best_model]
            best_val_metrics = val_metrics
            best_val_metrics["val_epoch"] = epoch if epoch is not None else step
            best_params = model.state_dict()

        early_stopping(val_metrics[eval_metric_for_best_model])
        if early_stopping.early_stop:
            print("Early stopping")
            return True  # Indicate that early stopping condition is met

        return False  # Continue training

    outer_break = False
    for epoch in bar:
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            step += 1
            x, y = x.to(device), y.to(device)

            if input_noise_x > 0.0:
                x = x + torch.randn_like(x) * input_noise_x
            if input_noise_y > 0.0:
                y = y + torch.randn_like(y) * input_noise_y

            optimizer.zero_grad()
            loss, train_metrics = model.training_pass(x, y, **loss_hyperparameters)
            wandb.log({**train_metrics, "step": step})
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=clip_gradient_norm
            )
            optimizer.step()
            if eval_mode == "step" and step % eval_every_n == 0:
                if log_evaluation(step, epoch):
                    outer_break = True
                    break

            if num_steps is not None and step >= num_steps:
                log_evaluation(step, epoch)
                outer_break = True
                break

        if outer_break:
            break

        if eval_mode == "epoch" and epoch % eval_every_n == 0:
            if log_evaluation(step, epoch):
                break
        elif eval_mode == "epoch":
            wandb.log({"step": step, "epoch": epoch})
        bar.set_description(str(val_metrics["val_loss"]))

    return best_params, best_val_metrics


def outer_train(
    train_data_module: TrainingDataModule,
    test_dataloader: DataLoader,
    config_id,
    seed: int,
    config: dict,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    wandb_mode: str,
    project_name: str,
):
    group_name = f"config_{config_id}"
    run_name = f"config_{config_id}_seed_{seed}"

    if wandb_mode != "disabled":
        os.makedirs(os.path.join("runs", project_name, "wandb"), exist_ok=True)

    wandb.init(
        project=project_name,
        config=config,
        name=run_name,
        group=group_name,
        dir=os.path.join("runs", project_name),
        mode=wandb_mode,
    )

    model = load_model(train_data_module, **model_hyperparameters).to(device)

    wandb.watch(model, log="all", log_freq=100)

    best_params, best_val_metrics = train_model(
        model, train_data_module, **training_hyperparameters, device=device
    )

    best_params_path = wandb.run.dir + "/best_params.pt"
    torch.save(best_params, best_params_path)
    artifact = wandb.Artifact(name="best_model", type="model")
    artifact.add_file(best_params_path)
    wandb.log_artifact(artifact)

    model.load_state_dict(best_params)
    test_metrics = evaluate_model(
        model,
        test_dataloader,
        device,
        training_hyperparameters["loss_hyperparameters"],
        train_data_module.distribution
        if train_data_module.has_distribution()
        else None,
        hellinger_dist_first_n_batches=-1,
        is_test=True,
    )

    best_val_metrics = {"best_" + key: value for key, value in best_val_metrics.items()}
    wandb.log(test_metrics)
    wandb.log(best_val_metrics)

    wandb.finish()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cv_experiment(
    data_module: DataModule,
    config_id,
    data_seed: int,
    seeds: List[int],
    config: dict,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    wandb_mode: str,
    project_name: str,
):
    seed_idx = 0

    for train_data_module in data_module.iterable_cv_splits(len(seeds), data_seed):
        seed_all(seeds[seed_idx])
        print(f"Running with seed {seeds[seed_idx]}")
        outer_train(
            train_data_module,
            data_module.get_test_dataloader(128),
            config_id,
            seeds[seed_idx],
            config,
            model_hyperparameters,
            training_hyperparameters,
            device,
            wandb_mode,
            project_name,
        )

        seed_idx += 1


def seeded_experiment(
    data_module: DataModule,
    config_id,
    seeds: List[int],
    config: dict,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    wandb_mode: str,
    project_name: str,
):
    for seed in seeds:
        seed_all(seed)
        print(f"Running with seed {seed}")
        outer_train(
            data_module,
            data_module.get_test_dataloader(128),
            config_id,
            seed,
            config,
            model_hyperparameters,
            training_hyperparameters,
            device,
            wandb_mode,
            project_name,
        )
