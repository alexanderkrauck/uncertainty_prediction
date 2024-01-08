import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.data_module import TrainingDataModule, DataModule

import wandb

from typing import List, Type


def mdn_loss_fn(y, weights, mu, sigma, reduce="mean", numeric_stability=1e-6):
    distribution = torch.distributions.Normal(mu, sigma + numeric_stability) # for numerical stability if predicted sigma is too close to 0
    loss = torch.exp(distribution.log_prob(y.unsqueeze(-1))) + numeric_stability #for numerical stability because outliers can cause this to be 0
    loss = torch.sum(loss * weights, dim=1)
    loss = -torch.log(loss)
    if reduce == "mean":
        loss = torch.mean(loss)
    elif reduce == "sum":
        loss = torch.sum(loss)
    if loss.item() == np.inf or loss.item() == -np.inf:
        print("inf loss")
    return loss


def hellinger_distance(distribution, x, weights, mu, sigma, reduce="mean", numeric_stability=1e-6):
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()
    if not isinstance(weights, np.ndarray):
        weights = weights.detach().cpu().numpy()
    if not isinstance(mu, np.ndarray):
        mu = mu.detach().cpu().numpy()
    if not isinstance(sigma, np.ndarray):
        sigma = sigma.detach().cpu().numpy()

    linspace = np.linspace(-10, 10, 100)  # Adjust range as necessary
    y_space = np.reshape(linspace, (-1, 1))

    hellinger_distances = []

    for idx in range(weights.shape[0]):
        # Calculate true density
        x_space = np.repeat(np.reshape(x[idx], (1, -1)), y_space.shape[0], axis=0)
        true_densities = distribution.pdf(x_space, y_space)  # True density

        # Calculate estimated density
        dist = torch.distributions.Normal(
            torch.from_numpy(mu[idx]), torch.from_numpy(sigma[idx]) + numeric_stability
        )
        estimated_densities = torch.exp(dist.log_prob(torch.from_numpy(y_space)))
        estimated_densities = torch.sum(
            estimated_densities * torch.from_numpy(weights[idx]), dim=1
        )

        # Calculate Hellinger distance component wise
        # sqrt(p(x)) - sqrt(q(x)) and then square
        diff_sq = (
            torch.sqrt(torch.tensor(true_densities, dtype=torch.float32))
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


def reliabiltiy_loss_fn(
    y: Tensor, weights: Tensor, mu: Tensor, sigma: Tensor, numeric_stability:int = 1e-6 ,n_samples: int = 100
):
    device = y.device

    distribution = torch.distributions.Normal(mu, sigma + numeric_stability)
    drawn_samples = distribution.sample((n_samples,)).transpose(0, 1)
    component_indices = torch.multinomial(weights, n_samples, replacement=True)
    effective_samples = torch.gather(
        drawn_samples, -1, component_indices.unsqueeze(-1)
    ).squeeze(-1)
    y = y.squeeze(-1)

    quantiles = torch.arange(5, 96, 10, device=device) / 100

    upper_bounds = torch.quantile(effective_samples, quantiles, dim=-1)
    y_r = (y < upper_bounds).sum(dim=-1) / y.shape[0]

    reliability_loss = (
        (y_r - quantiles).abs().mean()
    )  # maybe use trapz shomehow instead of mean

    return reliability_loss


def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    distribution=None,
    hellinger_dist_first_n_batches: int = 3,
    is_test: bool = False,
):
    model.eval()
    val_loss = 0
    hellinger_dist = 0
    reliability_loss = 0
    first_n_batch_sizes = 0
    with torch.no_grad():
        for idx, (x, y) in enumerate(val_loader):
            x, y = x.to(device), y.to(device)
            weights, mu, sigma = model(x)
            if distribution and (
                idx < hellinger_dist_first_n_batches
                or hellinger_dist_first_n_batches == -1
            ):
                hellinger_dist += hellinger_distance(
                    distribution, x, weights, mu, sigma, reduce="sum"
                )
                first_n_batch_sizes += x.shape[0]
            val_loss += mdn_loss_fn(y, weights, mu, sigma, reduce="sum").item()
            reliability_loss += reliabiltiy_loss_fn(y, weights, mu, sigma) * y.shape[0]
    val_loss /= len(val_loader.dataset)
    reliability_loss /= len(val_loader.dataset)

    prefix = "test_" if is_test else "val_"
    return_dict = {
        prefix + "loss": val_loss,
        prefix + "reliability_loss": reliability_loss.item(),
    }
    if distribution:
        hellinger_dist /= first_n_batch_sizes
        return_dict[prefix + "hellinger_dist"] = hellinger_dist
    return return_dict


optimizer_map = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def train_model(
    model: nn.Module,
    train_data_module: TrainingDataModule,
    optimizer: str,
    optimizer_hyperparameters: dict,
    epochs: int,
    batch_size: int,
    device: str,
    reliability_loss_weight: float = 50.0,
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
        distribution = train_data_module.distribution  # TODO, might not exist or so
    else:
        distribution = None

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
            weights, mu, sigma = model(x)
            loss = mdn_loss_fn(y, weights, mu, sigma)
            train_metrics = {"train_loss": loss.item()}
            if reliability_loss_weight > 0.0:
                reliabiltiy_loss = reliabiltiy_loss_fn(y, weights, mu, sigma)
                loss = loss + reliabiltiy_loss * reliability_loss_weight
                train_metrics["train_reliability_loss"] = reliabiltiy_loss.item()
            wandb.log({**train_metrics, "step": step})
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_gradient_norm)
            optimizer.step()

        val_metrics = evaluate_model(model, val_loader, device, distribution)
        wandb.log({"epoch": epoch, **val_metrics})
        bar.set_description(str(val_metrics))

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_params = model.state_dict()

    return best_params, best_val_loss


def outer_train(
    model_class: Type[nn.Module],
    train_data_module: TrainingDataModule,
    test_dataloader: DataLoader,
    config_id,
    seed: int,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
):
    group_name = f"config_{config_id}"
    run_name = f"config_{config_id}_seed_{seed}"

    config = {
        "training_hyperparameters": training_hyperparameters,
        "model_hyperparameters": model_hyperparameters,
    }
    wandb.init(project="mdn_synthetic1", config=config, name=run_name, group=group_name)

    model = model_class(train_data_module, **model_hyperparameters).to(device)

    wandb.watch(model, log="all", log_freq=100)

    best_params, best_val_loss = train_model(
        model, train_data_module, **training_hyperparameters, device=device
    )

    best_params_path =  wandb.run.dir + "/best_params.pt"
    torch.save(best_params, best_params_path)
    artifact = wandb.Artifact(name="best_model", type="model")
    artifact.add_file(best_params_path)
    wandb.log_artifact(artifact)

    model.load_state_dict(best_params)
    test_metrics = evaluate_model(
        model,
        test_dataloader,
        device,
        train_data_module.distribution if train_data_module.has_distribution() else None,
        hellinger_dist_first_n_batches=-1,
        is_test=True,
    )
    wandb.log(test_metrics)

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
        )