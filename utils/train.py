import torch
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
):
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

        val_metrics = evaluate_model(
            model, val_loader, device, loss_hyperparameters, distribution
        )
        wandb.log({"epoch": epoch, **val_metrics})
        bar.set_description(str(val_metrics))

        if val_metrics[eval_metric_for_best_model] < best_val_loss:
            best_val_loss = val_metrics[eval_metric_for_best_model]
            best_val_metrics = val_metrics
            best_val_metrics["val_epoch"] = epoch
            best_params = model.state_dict()

    return best_params, best_val_metrics


def outer_train(
    model_class: Type[nn.Module],
    train_data_module: TrainingDataModule,
    test_dataloader: DataLoader,
    config_id,
    seed: int,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    disable_wandb: bool,
):
    group_name = f"config_{config_id}"
    run_name = f"config_{config_id}_seed_{seed}"

    config = {
        "training_hyperparameters": training_hyperparameters,
        "model_hyperparameters": model_hyperparameters,
    }
    print(config)
    if disable_wandb:
        wandb.init(
            project="mdn_synthetic1",
            config=config,
            name=run_name,
            group=group_name,
            mode="disabled",
        )
    else:
        wandb.init(
            project="mdn_synthetic1", config=config, name=run_name, group=group_name
        )

    model = model_class(train_data_module, **model_hyperparameters).to(device)

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
    print("Test metrics:", test_metrics, "\n Best val metrics:", best_val_metrics)

    wandb.finish()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cv_experiment(
    model_class: Type[nn.Module],
    data_module: DataModule,
    config_id,
    data_seed: int,
    seeds: List[int],
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    disable_wandb: bool,
):
    seed_idx = 0

    for train_data_module in data_module.iterable_cv_splits(len(seeds), data_seed):
        seed_all(seeds[seed_idx])
        outer_train(
            model_class,
            train_data_module,
            data_module.get_test_dataloader(128),
            config_id,
            seeds[seed_idx],
            model_hyperparameters,
            training_hyperparameters,
            device,
            disable_wandb,
        )

        seed_idx += 1


def seeded_experiment(
    model_class: Type[nn.Module],
    data_module: DataModule,
    config_id,
    seeds: List[int],
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    disable_wandb: bool,
):
    for seed in seeds:
        seed_all(seed)
        outer_train(
            model_class,
            data_module,
            data_module.get_test_dataloader(128),
            config_id,
            seed,
            model_hyperparameters,
            training_hyperparameters,
            device,
            disable_wandb,
        )
