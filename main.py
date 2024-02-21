#!/usr/bin/python

import argparse
import yaml
import random
import os
import traceback

from utils.setup import sync_wandb, load_data_module, generate_configs, find_keys
from utils.train import cv_experiment, seeded_experiment
from utils.data_module import DataModule

import torch
import numpy as np
from copy import deepcopy

from tqdm import tqdm
import optuna
from optuna import Trial

def objective(trial: Trial, config: dict, data_module: DataModule, device: str, wandb_mode:str, project_name:str):
    """
    Optuna objective function for tuning hyperparameters.

    Args:
        trial: An Optuna `Trial` object.
        config: A dictionary containing the configuration.
        data_module: A `DataModule` object that contains the data.
        device: The device to use for training.
        wandb_mode: The wandb mode.
        project_name: The name of the wandb project.

    Returns:
        score
    """

    conf = deepcopy(config)
    paths = find_keys(config, None, target_keys=["choose"])
    for path in paths:
        inner_conf = conf
        for key in path[:-2]:
            inner_conf = inner_conf[key]
        options = {key["value"]: idx for idx, key in enumerate(inner_conf[path[-2]][path[-1]])}
        choice = trial.suggest_categorical(str(path), list(options.keys()))
        choice = options[choice]
        for key, val in inner_conf[path[-2]][path[-1]][choice].items():
            if key != "value":
                inner_conf[key] = val
        inner_conf[path[-2]] = inner_conf[path[-2]][path[-1]][choice]["value"]
        
    paths = find_keys(conf, None, target_keys=["tune", "rangetunefloat", "logtunefloat", "rangetuneint", "logtuneint"])
    for path in paths:
        inner_conf = conf
        for key in path[:-2]:
            inner_conf = inner_conf[key]
        options = inner_conf[path[-2]][path[-1]]
        if path[-1] == "tune":
            opts = {str(option): option for option in options}
            choice = trial.suggest_categorical(str(path), [str(option) for option in options])
            choice = opts[choice]
        elif path[-1] == "rangetunefloat":
            choice = trial.suggest_float(str(path), options[0], options[1])
        elif path[-1] == "logtunefloat":
            choice = trial.suggest_float(str(path), options[0], options[1], log=True)
        elif path[-1] == "rangetuneint":
            choice = trial.suggest_int(str(path), options[0], options[1])
        elif path[-1] == "logtuneint":
            choice = trial.suggest_int(str(path), options[0], options[1], log=True)
        inner_conf[path[-2]] = choice

    metric_list = cv_experiment(
        data_module,
        conf["config_id"],
        conf["data_seed"],
        conf["seeds"],
        conf,
        conf["model_hyperparameters"],
        conf["training_hyperparameters"],
        device,
        wandb_mode,
        project_name,
        use_test_set=False,
        verbose=False,
    )
    mean_metrics = {}
    for key in metric_list[0].keys():
        mean_metrics[key] = np.mean([metric[key] for metric in metric_list])
    trial.set_user_attr("metrics", mean_metrics)
    trial.set_user_attr("config", conf)
    score = mean_metrics["best_"+conf["training_hyperparameters"]["eval_metric_for_best_model"]]

    return score

    
    



def main(
    config_file=None,
    log_directory=None,
    eval_mode="default",
    choose_n_configs: int = None,
    wandb_mode: str = "offline",
    project_name="debug_project",
    nested_cv_results_file="nested_cv_results.log",
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    """Main function for running deep learning experiments.

    Parameters
    ----------
    config_file : str
        Path to the config file.
    log_directory : str
        Path to the log directory.
    eval_mode : str
        Evaluation mode. Either 'default', 'cv' or 'nested_cv'. If using nested_cv, the config file should have a very particular structure.
    choose_n_configs : int
        Choose n configs randomly from the config file. If None, all configs are run.
    wandb_mode : str
        Wandb mode. Either 'online', 'offline' or 'disabled'.
    project_name : str
        Name of the wandb project.
    nested_cv_results_file : str
        Name of the file where the nested cv results are saved.
    """

    wandb_mode = wandb_mode.lower()
    project_name = project_name.lower()
    if not project_name.startswith("cde_"):
        project_name = "cde_" + project_name
    # Your code for running deep learning experiments goes here
    print("Running deep learning experiments...")
    print("Parameters:")
    print("hyperparameters:", config_file)
    print("log_directory:", log_directory)
    print("eval_mode:", eval_mode)
    print("wandb_mode:", wandb_mode)
    print("choose_n_configs:", choose_n_configs)
    print("project_name:", project_name)

    config_files = config_file.split(",")

    if len(config_files) == 1 and os.path.isdir(config_file):
        config_files = [
            os.path.join(config_file, f)
            for f in os.listdir(config_file)
            if f.endswith(".yml")
        ]
        print("Found the following config files:")
        print(config_files)

    for cf in config_files:
        main_config_run(
            config_file=cf,
            log_directory=log_directory,
            eval_mode=eval_mode,
            choose_n_configs=choose_n_configs,
            wandb_mode=wandb_mode,
            project_name=project_name,
            nested_cv_results_file=nested_cv_results_file,
            device=device
        )



def main_config_run(    
    config_file=None,
    log_directory=None,
    eval_mode="default",
    choose_n_configs: int = None,
    wandb_mode: str = "offline",
    project_name="debug_project",
    nested_cv_results_file="nested_cv_results.log",
    device="cuda" if torch.cuda.is_available() else "cpu",
    ):

        # Load hyperparameters
    with open(config_file, "r") as f:
        true_config = yaml.safe_load(f)

    eval_mode = eval_mode.lower()
    if eval_mode == "nested_cv":
        print("Saving nested cv results to", nested_cv_results_file)
    if eval_mode not in ["default", "cv", "nested_cv", "optuna"]:
        raise ValueError(f"Evaluation mode {eval_mode} not supported yet.")

    if eval_mode == "nested_cv" or eval_mode == "optuna":
        data_configs = generate_configs(config=true_config["data_hyperparameters"])
        with open(nested_cv_results_file, "a") as f:
                f.write(f"\n Model Type: {true_config['model_hyperparameters']['model_class']} - Dataset: {true_config['data_hyperparameters']}\n")

        test_metrics_list = []
        for outer_idx, data_config in enumerate(data_configs):
            config_grid = deepcopy(true_config)
            config_grid["data_hyperparameters"] = data_config

            data_module = load_data_module(**data_config)

            if eval_mode == "optuna":
                os.makedirs(f"runs/{project_name}/", exist_ok=True)
                study = optuna.create_study(
                    direction="minimize",
                    study_name=f"{project_name}_{true_config['config_id']}_{data_config['data_type']}_split{outer_idx}",
                    storage=f"sqlite:///runs/{project_name}/{true_config['config_id']}_optuna.db",#f"sqlite:///{log_directory}/{true_config['model_hyperparameters']['model_class']}_{data_config['data_type']}.db",
                    load_if_exists=True,
                )
                objective_fn = lambda trial: objective(trial, config_grid, data_module, device, wandb_mode, project_name)
                study.optimize(objective_fn, n_trials=choose_n_configs if choose_n_configs is not None else 1000, show_progress_bar=True)

                best_trial = study.best_trial
                best_config = best_trial.user_attrs["config"]
                best_epoch = int(best_trial.user_attrs["metrics"]["best_val_epoch"])
                best_eval_score = best_trial.value

            if eval_mode == "nested_cv":
                configs = [conf for conf in generate_configs(config=config_grid)]
                if choose_n_configs and len(configs) > choose_n_configs:
                    random.shuffle(configs)
                    print(
                        f"Randomly choosing {choose_n_configs} configs from {len(configs)} configs."
                    )
                    configs = configs[:choose_n_configs]
                best_eval_score = np.inf
                best_config = None
                best_epoch = None
                for idx, config in enumerate(tqdm(configs)):
                    metric_list = cv_experiment(
                        data_module,
                        config["config_id"],
                        config["data_seed"],
                        config["seeds"],
                        config,
                        config["model_hyperparameters"],
                        config["training_hyperparameters"],
                        device,
                        wandb_mode,
                        project_name,
                        use_test_set=False,
                        verbose=False,
                    )
                    mean_metrics = {}
                    for key in metric_list[0].keys():
                        mean_metrics[key] = np.mean([metric[key] for metric in metric_list])
                    score = mean_metrics["best_"+config["training_hyperparameters"]["eval_metric_for_best_model"]]

                    if score < best_eval_score:
                        best_eval_score = score
                        best_config = config
                        best_epoch = int(mean_metrics["best_val_epoch"])

            with open(nested_cv_results_file, "a") as f:
                f.write(f"Best Config cv run {outer_idx}:\n")
                f.write(str(best_config))
                f.write(f"Best mean cv score: {best_eval_score}\n")

            best_config["training_hyperparameters"]["epochs"] = best_epoch # Because we don't know whats best if we don't use the validation set
            print(best_config)
            test_metrics = seeded_experiment(
                data_module,
                best_config["config_id"],
                best_config["seeds"],
                best_config,
                best_config["model_hyperparameters"],
                best_config["training_hyperparameters"],
                device,
                "online", #For the test set we always want to log to wandb because we don't care about speed
                project_name,
                False,
                verbose=False
            )
            test_metrics_list.extend(test_metrics)

        final_test_metrics_mean = {}
        final_test_metrics_std = {}
        final_test_metrics_all = {}
        for key in test_metrics_list[0].keys():
            final_test_metrics_all[key] = [test_metrics[key] for test_metrics in test_metrics_list]
            final_test_metrics_mean[key] = np.mean(final_test_metrics_all[key])
            final_test_metrics_std[key] = np.std(final_test_metrics_all[key])
        print("The final test metrics are:")
        with open(nested_cv_results_file, "a") as f:
            f.write(f"Results for {true_config['model_hyperparameters']['model_class']}\n")
            for key in final_test_metrics_mean.keys():
                print(f"{key}: {final_test_metrics_mean[key]} +- {final_test_metrics_std[key]} --- {final_test_metrics_all[key]}")
                f.write(f"{key}: {final_test_metrics_mean[key]} +- {final_test_metrics_std[key]} --- {final_test_metrics_all[key]}\n")



    # Run experiments in default or cv mode
    else:

        configs = [conf for conf in generate_configs(config=true_config)]

        if choose_n_configs and len(configs) > choose_n_configs:
            random.shuffle(configs)
            print(
                f"Randomly choosing {choose_n_configs} configs from {len(configs)} configs."
            )
            configs = configs[:choose_n_configs]
        for idx, config in enumerate(configs):
            if len(configs) > 1:
                config["config_id"] = config["config_id"] + "_cnf" + str(idx)
            print(f"Running config {idx}/{len(configs)}:\n", config)

            data_module = load_data_module(**config["data_hyperparameters"])

            if eval_mode not in ["default", "cv"]:
                raise ValueError(f"Evaluation mode {eval_mode} not supported yet.")

            def run_experiments():
                if eval_mode == "default":
                    # Run experiment
                    seeded_experiment(
                        data_module,
                        config["config_id"],
                        config["seeds"],
                        config,
                        config["model_hyperparameters"],
                        config["training_hyperparameters"],
                        device,
                        wandb_mode,
                        project_name,
                    )
                elif eval_mode == "cv":
                    # Run cross-validation experiment
                    cv_experiment(
                        data_module,
                        config["config_id"],
                        config["data_seed"],
                        config["seeds"],
                        config,
                        config["model_hyperparameters"],
                        config["training_hyperparameters"],
                        device,
                        wandb_mode,
                        project_name,
                    )

            try:
                torch.autograd.set_detect_anomaly(False)
                run_experiments()
            except ValueError as e:
                print(e)
                traceback.print_exc()
                print("Redoing this config with anomaly detection on:")
                print(config)
                torch.autograd.set_detect_anomaly(
                    True
                )  # makes it much slower but can better see where the error is

                try:
                    run_experiments()
                except Exception as e:
                    print(e)
                    traceback.print_exc()
                    print("Now skipping this config.")
                    continue

                continue
    if wandb_mode == "offline":
        sync_wandb(project_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Experiment")

    # Add your command-line arguments here
    parser.add_argument(
        "--config_file",
        type=str,
        help="Name of the file where hyperparameters are stored",
    )
    parser.add_argument(
        "--log_directory", type=str, help="Directory where wandb logs are saved"
    )
    parser.add_argument("--eval_mode", type=str, help="Evaluation mode")
    parser.add_argument("--wandb_mode", type=str, help="Wandb mode")
    parser.add_argument(
        "--choose_n_configs", type=int, help="Choose n configs randomly"
    )
    parser.add_argument("--project_name", type=str, help="Wandb project name")
    parser.add_argument(
        "--nested_cv_results_file", type=str, help="Nested cv results file"
    )
    parser.add_argument("--device", type=str, help="Device to use")

    args = parser.parse_args()
    pass_args = {k: v for k, v in dict(args._get_kwargs()).items() if v is not None}
    main(**pass_args)
