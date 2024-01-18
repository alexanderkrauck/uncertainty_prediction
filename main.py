#!/usr/bin/python

import argparse
import yaml
import random

from utils.setup import sync_wandb, load_data_module, load_model_class, generate_configs
from utils.train import cv_experiment, seeded_experiment

import torch
import copy





def main(
    config_file=None,
    log_directory=None,
    eval_mode="default",
    choose_n_configs: int = None,
    wandb_mode:str="offline",
    project_name="debug_project",
):
    
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

    # Load hyperparameters
    with open(config_file, "r") as f:
        true_config = yaml.safe_load(f)


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
        print(f"Running config {idx}:\n", config)

        data_module = load_data_module(**config["data_hyperparameters"])

        device = "cuda" if torch.cuda.is_available() else "cpu"

        eval_mode = eval_mode.lower()
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
            print("Redoing this config with anomaly detection on:")
            print(config)
            torch.autograd.set_detect_anomaly(
                True
            )  # makes it much slower but can better see where the error is

            try:
                run_experiments()
            except Exception as e:
                print(e)
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

    args = parser.parse_args()
    pass_args = {k: v for k, v in dict(args._get_kwargs()).items() if v is not None}
    main(**pass_args)
