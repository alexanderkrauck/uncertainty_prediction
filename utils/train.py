import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from utils.data_module import TrainingDataModule

import wandb


def mdn_loss_fn(y, weights, mu, sigma, reduce="mean"):
    distribution = torch.distributions.Normal(mu, sigma)
    loss = torch.exp(distribution.log_prob(y.unsqueeze(-1)))
    loss = torch.sum(loss * weights, dim=1)
    loss = -torch.log(loss)
    if reduce == "mean":
        loss = torch.mean(loss)
    elif reduce == "sum":
        loss = torch.sum(loss)
    return loss


def hellinger_distance(distribution, x, weights, mu, sigma, reduce="mean"):
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
            torch.from_numpy(mu[idx]), torch.from_numpy(sigma[idx])
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
    y: Tensor, weights: Tensor, mu: Tensor, sigma: Tensor, n_samples: int = 100
):
    device = y.device

    distribution = torch.distributions.Normal(mu, sigma)
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
    model, val_loader, device, distribution=None, hellinger_dist_first_n_batches=2
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
            if distribution and idx < hellinger_dist_first_n_batches:
                hellinger_dist += hellinger_distance(
                    distribution, x, weights, mu, sigma, reduce="sum"
                )
                first_n_batch_sizes += x.shape[0]
            val_loss += mdn_loss_fn(y, weights, mu, sigma, reduce="sum").item()
            reliability_loss += reliabiltiy_loss_fn(y, weights, mu, sigma) * y.shape[0]
    val_loss /= len(val_loader.dataset)
    reliability_loss /= len(val_loader.dataset)

    return_dict = {"val_loss": val_loss, "val_reliability_loss": reliability_loss.item()}
    if distribution:
        hellinger_dist /= first_n_batch_sizes
        return_dict["val_hellinger_dist"] = hellinger_dist
    return return_dict


optimizer_map = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
}


def train_model(
    model,
    train_data_module: TrainingDataModule,
    optimizer: str,
    optimizer_hyperparameters: dict,
    epochs,
    batch_size,
    device,
    reliability_loss_weight: float = 50.0,
    input_noise_x=0.0,
    input_noise_y=0.0,
):
    optimizer = optimizer_map[optimizer.lower()](
        model.parameters(), **optimizer_hyperparameters
    )

    train_loader = train_data_module.get_train_dataloader(batch_size)
    val_loader = train_data_module.get_val_dataloader(batch_size)
    distribution = train_data_module.distribution  # TODO, might not exist or so

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
            optimizer.step()

        val_metrics = evaluate_model(model, val_loader, device, distribution)
        wandb.log({"epoch": epoch, **val_metrics})
        bar.set_description(str(val_metrics))

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            best_params = model.state_dict()

    return best_params, best_val_loss



def outer_train(
    model_class,
    train_data_module: TrainingDataModule,
    config_id,
    seed,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device,
):
    group_name = f"config_{config_id}"
    run_name = f"config_{config_id}_seed_{seed}"

    config = {
        "training_hyperparameters": training_hyperparameters,
        "model_hyperparameters": model_hyperparameters,
    }
    wandb.init(project="mdn_synthetic1", config=config, name=run_name, group=group_name)

    model = model_class(train_data_module, **model_hyperparameters).to(device)

    best_params, best_val_loss = train_model(model, train_data_module, **training_hyperparameters, device=device)

    wandb.finish()


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cv_experiment(
    model_class,
    data_module,
    config_id,
    data_seed,
    seeds,
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
            config_id,
            seeds[seed_idx],
            model_hyperparameters,
            training_hyperparameters,
            device,
        )

        seed_idx += 1


def experiment(
    model_class,
    data_module,
    config_id,
    seeds,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
):
    for seed in seeds:
        seed_all(seed)
        outer_train(
            model_class,
            data_module,
            config_id,
            seed,
            model_hyperparameters,
            training_hyperparameters,
            device,
        )
