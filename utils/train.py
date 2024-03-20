"""
Utitlity functions for training the models.

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

# Standard libraries
import os

# Third-party libraries
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import wandb
from typing import List, Optional

# Local/Application Specific
from utils.data_module import TrainingDataModule, DataModule
from .models.basic_architectures import ConditionalDensityEstimator
from .setup import load_model
from .evaluation_functions import (
    HellingerDistance,
    KLDivergence,
    WassersteinDistance,
    BaseEvaluationFunction,
    ConformalPrediction,
    log_plot,
    log_permutation_feature_importance,
    infer_quantalized_conformal_p,
    Miscalibration,
)
from .utils import flatten_dict, make_lists_strings_in_dict
from copy import deepcopy

EVALUATION_FUNCTION_MAP = {
    "hellinger_distance": HellingerDistance(),
    "wasserstein_distance": KLDivergence(),
    "kl_divergence": WassersteinDistance(),
    "conformal_prediction": ConformalPrediction(),
    "miscalibration": Miscalibration(),
}


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


def evaluate_model(
    model: ConditionalDensityEstimator,
    val_loader: DataLoader,
    device: str,
    loss_hyperparameters: dict,
    y_space: Optional[np.ndarray] = None,
    evaluation_function_names: List[str] = [],
    slow_first_n_batches: int = 3,
    is_test: bool = False,
    has_distribution: bool = False,
    **kwargs,
):
    """Evaluate a model on the validation set and return the metrics

    Parameters
    ----------
    model : ConditionalDensityEstimator
        Model to evaluate
    val_loader : DataLoader
        Dataloader containing the validation data
    device : str
        Device to use
    loss_hyperparameters : dict
        Hyperparameters for the loss function
    y_space : Optional[np.ndarray], optional
        Tensor containing the y space, by default None. If provided then the val loader is expected to return (x, y, densities) tuples.
    slow_first_n_batches : int, optional
        Number of batches to use to calculate the Density Metrics, by default 3. Since it is a slow operation, it is only calculated for the first n batches as an approximate.
    is_test : bool, optional
        Whether this is a test evaluation or not, by default False

    Returns
    -------
    dict
        Dictionary of the metrics
    """

    model.eval()
    first_n_batch_sizes = 0
    eval_metrics = {}

    if isinstance(y_space, torch.Tensor):
        y_space = y_space.clone().to(device).view(-1, 1)
    else:
        y_space = torch.tensor(y_space, device=device).view(-1, 1)

    evaluation_function_names = [func.lower() for func in evaluation_function_names]
    for func in evaluation_function_names:
        if func not in EVALUATION_FUNCTION_MAP:
            raise ValueError(f"Evaluation function {func} not supported.")
        if (
            not has_distribution
            and EVALUATION_FUNCTION_MAP[func].is_density_evaluation_function()
        ):
            raise ValueError(
                f"Evaluation function {func} requires densities, but the dataset does not contain densities."
            )

    additional_eval_metrics = {
        func: np.zeros((EVALUATION_FUNCTION_MAP[func].output_size))
        for func in evaluation_function_names
    }

    with torch.no_grad():
        for idx, minibatch in enumerate(val_loader):
            eval_input_dict = {}
            if has_distribution:
                x_batch, y_batch, densities = minibatch
                x_batch, y_batch, densities = (
                    x_batch.to(device),
                    y_batch.to(device),
                    densities.to(device),
                )
                eval_input_dict["densities"] = densities
            else:
                x_batch, y_batch = minibatch
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            model_output = model(x_batch, y_batch)

            eval_input_dict["x_batch"] = x_batch
            eval_input_dict["y_batch"] = y_batch
            eval_input_dict["model"] = model
            eval_input_dict["precomputed_variables"] = model_output
            eval_input_dict["reduce"] = "sum"
            eval_input_dict["y_space"] = y_space

            _, current_eval_metrics = model.eval_output(
                y_batch, model_output, False, "sum", **loss_hyperparameters
            )
            if idx == 0:
                eval_metrics = current_eval_metrics
            else:
                for key, value in current_eval_metrics.items():
                    eval_metrics[key] += value

            if idx < slow_first_n_batches or is_test or slow_first_n_batches == -1:
                first_n_batch_sizes += x_batch.shape[0]

            for evaluation_function_name in evaluation_function_names:
                evaluation_function = EVALUATION_FUNCTION_MAP[evaluation_function_name]
                if evaluation_function.is_density_evaluation_function():
                    if not has_distribution:
                        continue
                if (
                    idx >= slow_first_n_batches
                    and not is_test
                    and slow_first_n_batches != -1
                    and evaluation_function.is_slow()
                ):
                    continue
                additional_eval_metrics[
                    evaluation_function_name
                ] += evaluation_function(**eval_input_dict, **kwargs)

    for key, value in eval_metrics.items():
        eval_metrics[key] /= len(val_loader.dataset)

    for evaluation_function_name in evaluation_function_names:
        evaluation_function = EVALUATION_FUNCTION_MAP[evaluation_function_name]
        if evaluation_function.is_slow():
            additional_eval_metrics[evaluation_function_name] /= first_n_batch_sizes
        else:
            additional_eval_metrics[evaluation_function_name] /= len(val_loader.dataset)
    for metric, val in additional_eval_metrics.items():
        if len(val) == 1:
            eval_metrics[metric] = val[0]
        else:
            for i, v in enumerate(val):
                eval_metrics[f"{metric}{i}part"] = v

    prefix = "test_" if is_test else "val_"
    return_dict = {prefix + key: value for key, value in eval_metrics.items()}

    return return_dict


OPTIMIZER_MAP = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop,
    "adamw": optim.AdamW,
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
    summary_writer: SummaryWriter,
    eval_metric_for_best_model: str = "val_nll_loss",
    input_noise_x: float = 0.0,
    input_noise_y: float = 0.0,
    clip_gradient_norm: float = 5.0,
    early_stopping_patience: int = 15,
    early_stopping_min_delta: float = 1e-4,
    eval_every_n: int = 1,
    eval_mode: str = "epoch",
    num_steps: int = None,
    use_validation_set: bool = True,
    verbose: bool = True,
    noisy_start: bool = False,
    noise_decay: float = 0.99,
    noise_level: float = 0.9,
    noise_stop: int = 100,
    log_train_every_n: int = 1,
    use_lr_scheduler: bool = False,
    **kwargs,
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
    summary_writer : SummaryWriter
        SummaryWriter to use for tensorboard logging
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
    use_validation_set : bool
        Whether to use a validation set or not, by default True
    verbose : bool
        Whether to print progress or not, by default True

    Returns
    -------
    Tuple[dict, dict]
        Tuple of the best model parameters and the validation metrics of the best model, if use_validation_set is True.
    or
    Tuple[dict, None]
        Tuple of the best model parameters and None if use_validation_set is False.
    """

    if eval_mode == "step":
        if num_steps is None:
            raise ValueError("num_steps must be provided if eval_mode is step")

    optimizer = OPTIMIZER_MAP[optimizer.lower()](
        model.parameters(), **optimizer_hyperparameters
    )
    if use_lr_scheduler:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5, min_lr=5e-5, cooldown=3, verbose=verbose
        )

    train_loader = train_data_module.get_train_dataloader(batch_size)
    val_loader = train_data_module.get_val_dataloader(batch_size)

    if noisy_start:
        min_y = train_data_module.train_dataset.y.min()
        max_y = train_data_module.train_dataset.y.max()

    has_distribution = train_data_module.has_distribution()

    best_val_metrics = None
    best_val_loss = np.inf
    best_params = None
    if verbose:
        bar = tqdm(range(epochs))
    else:
        bar = range(epochs)
    step = 0
    early_stopping = EarlyStopping(early_stopping_patience, early_stopping_min_delta)

    if use_validation_set:
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            loss_hyperparameters,
            train_data_module.y_space,
            has_distribution=train_data_module.has_distribution(),
            **kwargs,
        )
        wandb.log({"step": step, **val_metrics, "epoch": 0})
        for key, value in val_metrics.items():
            summary_writer.add_scalar(key, value, 0)

    def log_evaluation(step, epoch=None):
        if not use_validation_set:
            return False  # We train until num_steps/epochs is reached

        nonlocal val_metrics, best_val_loss, best_val_metrics, best_params

        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            loss_hyperparameters,
            train_data_module.y_space,
            has_distribution=train_data_module.has_distribution(),
            **kwargs,
        )
        log_data = {"step": step, **val_metrics}
        if epoch is not None:
            log_data["epoch"] = epoch

        wandb.log(log_data)
        for key, value in log_data.items():
            summary_writer.add_scalar(key, value, step)

        if val_metrics[eval_metric_for_best_model] < best_val_loss:
            best_val_loss = val_metrics[eval_metric_for_best_model]
            best_val_metrics = val_metrics
            best_val_metrics["val_epoch"] = epoch if epoch is not None else step
            best_params = model.state_dict()

        early_stopping(val_metrics[eval_metric_for_best_model])
        if use_lr_scheduler:
            lr_scheduler.step(val_metrics[eval_metric_for_best_model])
        if early_stopping.early_stop:
            if verbose:
                print("Early stopping")
            return True  # Indicate that early stopping condition is met

        return False  # Continue training

    outer_break = False
    for epoch in bar:
        model.train()

        for batch_idx, minibatch in enumerate(train_loader):
            step += 1
            if has_distribution:
                x, y, _ = minibatch
            else:
                x, y = minibatch
            x, y = x.to(device), y.to(device)

            # TODO: consider adding rule of thumb noise:
            if input_noise_x > 0.0:
                x = x + torch.randn_like(
                    x
                ) * input_noise_x * train_loader.dataset.std_x.to(device)
            if input_noise_y > 0.0:
                y = y + torch.randn_like(
                    y
                ) * input_noise_y * train_loader.dataset.std_y.to(device)

            if noisy_start and epoch < noise_stop:
                mask = torch.rand(y.shape[0]) < noise_level
                random_values = torch.empty(
                    mask.sum().item(), y.shape[-1], device=device
                ).uniform_(min_y, max_y)
                y[mask] = random_values
                noise_level = noise_level * noise_decay
            optimizer.zero_grad()
            loss, train_metrics = model.training_pass(x, y, **loss_hyperparameters)
            if step % log_train_every_n == 0:
                wandb.log({**train_metrics, "step": step})
                for key, value in train_metrics.items():
                    summary_writer.add_scalar(key, value, step)

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
            summary_writer.add_scalar("epoch", epoch, step)
        if use_validation_set and verbose:
            bar.set_description(
                f"{eval_metric_for_best_model}: {str(val_metrics[eval_metric_for_best_model])}"
            )
    if use_validation_set:
        return best_params, best_val_metrics
    else:
        return model.state_dict(), None


def outer_train(
    train_data_module: TrainingDataModule,
    test_dataloader: Optional[DataLoader],
    config_id,
    seed: int,
    config: dict,
    model_hyperparameters: dict,
    training_hyperparameters: dict,
    device: str,
    wandb_mode: str,
    project_name: str,
    use_validation_set: bool = True,
    verbose: bool = True,
):
    """
    Run a single training run with the given hyperparameters and seed. Also initializes wandb logging and does test evaluation if test_dataloader is provided.

    Parameters
    ----------
    train_data_module : TrainingDataModule
        DataModule containing the data (training and validation)
    test_dataloader : Optional[DataLoader]
        Dataloader for the test set.
    config_id : str
        Id of the config
    seed : int
        Seed to use for minibatch shuffling and weight initialization
    config : dict
        Config to use (for logging)
    model_hyperparameters : dict
        Hyperparameters for the model
    training_hyperparameters : dict
        Hyperparameters for the training
    device : str
        Device to use
    wandb_mode : str
        Mode to use for wandb
    project_name : str
        Name of the project
    use_validation_set : bool, optional
        Whether to use the validation set or not, by default True
    verbose : bool, optional
        Whether to print progress to console or not, by default True

    Returns
    -------
    dict
        Test metrics if test_dataloader is provided, else the best validation metrics.
    """

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
    summary_writer = SummaryWriter(os.path.join("runs", project_name, run_name))

    model = load_model(train_data_module, **model_hyperparameters).to(device)

    wandb.watch(model, log="all", log_freq=100)

    best_params, best_val_metrics = train_model(
        model,
        train_data_module,
        **training_hyperparameters,
        device=device,
        use_validation_set=use_validation_set,
        verbose=verbose,
        summary_writer=summary_writer,
    )

    if wandb_mode != "disabled":
        best_params_path = wandb.run.dir + "/best_params.pt"
        torch.save(best_params, best_params_path)
        artifact = wandb.Artifact(name="best_model", type="model")
        artifact.add_file(best_params_path)
        wandb.log_artifact(artifact)
    metrics_dict = {}
    model.load_state_dict(best_params)
    if test_dataloader is not None:
        test_metrics = evaluate_model(
            model,
            test_dataloader,
            device,
            y_space=train_data_module.y_space,
            is_test=True,
            has_distribution=train_data_module.has_distribution(),
            **training_hyperparameters,
        )
        metrics_dict.update(test_metrics)
        if train_data_module.has_distribution():
            log_plot(
                summary_writer,
                model,
                test_dataloader,
                train_data_module.y_space,
                5,
                device,
            )
        log_permutation_feature_importance(
            summary_writer, model, test_dataloader, device
        )
    elif (
        use_validation_set
    ):  # If we don't have a test set, we log the best validation metrics with the slow metrics calculated for the whole validation set
        copied_training_hyperparameters = deepcopy(training_hyperparameters)
        copied_training_hyperparameters["slow_first_n_batches"] = -1
        # Here we also do the calcluation of the conformity quantile that we want of the score.
        if (
            "evaluation_function_names" in training_hyperparameters
            and "conformal_prediction"
            in training_hyperparameters["evaluation_function_names"]
        ):
            required_conformal_p = infer_quantalized_conformal_p(
                model,
                train_data_module.get_val_dataloader(128),
                device,
                train_data_module.y_space,
                train_data_module.has_distribution(),
                **training_hyperparameters,
           )
            best_val_metrics["val_required_conformal_p"] = required_conformal_p
            training_hyperparameters = training_hyperparameters.copy()
            training_hyperparameters["conformal_p"] = required_conformal_p #we set the conformal p to the required one so below in the evaluation we use the correct one.
        best_val_metrics.update(
            evaluate_model(
                model,
                train_data_module.get_val_dataloader(128),
                device,
                y_space=train_data_module.y_space,
                is_test=False,
                has_distribution=train_data_module.has_distribution(),
                **training_hyperparameters,
            )
        )
        

    if use_validation_set:
        best_val_metrics = {
            "best_" + key: value for key, value in best_val_metrics.items()
        }
        metrics_dict.update(best_val_metrics)

    wandb.log(metrics_dict)
    summary_writer.add_hparams(
        make_lists_strings_in_dict(flatten_dict(config)), metrics_dict
    )
    summary_writer.close()

    wandb.finish()
    if test_dataloader is not None:
        return test_metrics
    return best_val_metrics


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
    use_test_set: bool = True,
    verbose: bool = True,
):
    """
    Run a cross validation experiment with the given hyperparameters and seeds.

    Parameters
    ----------
    data_module : DataModule
        DataModule containing the data
    config_id : int
        Id of the config
    data_seed : int
        Seed to use to split the data into folds
    seeds : List[int]
        List of seeds to use for each fold (for minibatch shuffling and weight initialization)
    config : dict
        Config to use
    model_hyperparameters : dict
        Hyperparameters for the model
    training_hyperparameters : dict
        Hyperparameters for the training
    device : str
        Device to use
    wandb_mode : str
        Mode to use for wandb
    project_name : str
        Name of the project
    use_test_set : bool, optional
        Whether to use the test set or not, by default True
    verbose : bool, optional
        Whether to print progress to console or not, by default True

    Returns
    -------
    List[dict]
        List of the test metrics or best validation metrics (if use_test_set is false) for each fold
    """
    seed_idx = 0

    metrics_list = []
    for train_data_module in data_module.iterable_cv_splits(len(seeds), data_seed):
        seed_all(seeds[seed_idx])
        if verbose:
            print(f"Running with seed {seeds[seed_idx]}")
        metrics = outer_train(
            train_data_module,
            data_module.get_test_dataloader(128) if use_test_set else None,
            config_id,
            seeds[seed_idx],
            config,
            model_hyperparameters,
            training_hyperparameters,
            device,
            wandb_mode,
            project_name,
            verbose=verbose,
        )
        metrics_list.append(metrics)

        seed_idx += 1
    return metrics_list


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
    use_validation_set: bool = True,
    verbose: bool = True,
):
    """
    Run an experiment multiple times with different seeds but same data splits and hyperparameters.

    Parameters
    ----------
    data_module : DataModule
        DataModule containing the data
    config_id : int
        Id of the config
    seeds : List[int]
        List of seeds to use for each run (for minibatch shuffling and weight initialization)
    config : dict
        Config to use
    model_hyperparameters : dict
        Hyperparameters for the model
    training_hyperparameters : dict
        Hyperparameters for the training
    device : str
        Device to use
    wandb_mode : str
        Mode to use for wandb
    project_name : str
        Name of the project
    use_validation_set : bool, optional
        Whether to use the validation set or not, by default True
    verbose : bool, optional
        Whether to print progress to console or not, by default True

    Returns
    -------
    List[dict]
        List of the test metrics for each run.
    """

    test_metrics_list = []
    for seed in seeds:
        seed_all(seed)
        if verbose:
            print(f"Running with seed {seed}")
        test_metrics = outer_train(
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
            use_validation_set=use_validation_set,
            verbose=verbose,
        )
        test_metrics_list.append(test_metrics)
    return test_metrics_list
