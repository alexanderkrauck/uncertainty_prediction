#!/usr/bin/python

import argparse
import yaml

from utils.models.mdn import MDN
from utils.data_module import SyntheticDataModule
from utils.train import outer_train, cv_experiment, seeded_experiment

import torch

def main(config_file=None, log_directory=None, model_class="gmm", eval_mode="default"):
    # Your code for running deep learning experiments goes here
    print("Running deep learning experiments...")
    print("Parameters:")
    print("hyperparameters:", config_file)
    print("log_directory:", log_directory)
    print("model_class:", model_class)
    print("eval_mode:", eval_mode)

    # Load hyperparameters
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load model class
    if model_class.lower() == "gmm":
        model_class = MDN
    else:
        raise ValueError(f"Model class {model_class} not supported.")

    # Load data
    data_hyperparams = config["data_hyperparameters"]
    data_type = data_hyperparams["data_type"]
    if data_type == "synthetic":
        data_module = SyntheticDataModule(**data_hyperparams)
    else:
        raise ValueError(f"Data type {data_type} not supported yet.")
        #TODO: Add support for real data
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if eval_mode == "default":
        # Run experiment
        seeded_experiment(
            model_class,
            data_module,
            config["config_id"],
            config["seeds"],
            config["model_hyperparameters"],
            config["training_hyperparameters"],
            device
        )
    elif eval_mode == "cv":
        # Run cross-validation experiment
        cv_experiment(
            model_class,
            data_module,
            config["config_id"],
            config["data_seed"],
            config["seeds"],
            config["model_hyperparameters"],
            config["training_hyperparameters"],
            device
        )



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep Learning Experiment")
    
    # Add your command-line arguments here
    parser.add_argument("--config_file", type=str, help="Name of the file where hyperparameters are stored")
    parser.add_argument("--log_directory", type=str, help="Directory where wandb logs are saved")
    parser.add_argument("--model_class", type=str, help="Model class")
    parser.add_argument("--eval_mode", type=str, help="Evaluation mode")
    
    args = parser.parse_args()
    pass_args = {k: v for k, v in dict(args._get_kwargs()).items() if v is not None}
    main(**pass_args)
