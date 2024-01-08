#!/usr/bin/python

import argparse
import yaml

from utils.models.mdn import MDN
from utils.models.vae import VAE
from utils.data_module import SyntheticDataModule, VoestDataModule, UCIDataModule
from utils.train import outer_train, cv_experiment, seeded_experiment

import torch
import copy

def generate_configs(config, tune_key="tune"):
    """
    Recursively generate all combinations of configurations.
    Each time a 'tune' element is found, create new configs for each option in the 'tune' list.
    This function handles nested 'tune' elements as well.
    """

    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, dict):
                if tune_key in value:
                    for option in value[tune_key]:
                        new_config = copy.deepcopy(config)
                        new_config[key] = option  # Replace 'tune' with the actual value
                        yield from generate_configs(new_config)  # Recurse with the new config
                    return  # Once all 'tune' options for a key are processed, return
                else:
                    # Recurse into nested dictionaries
                    results = [result for result in generate_configs(value)]
                    if len(results) > 1:
                        for result in results:
                            new_config = copy.deepcopy(config)
                            new_config[key] = result
                            yield from generate_configs(new_config)
                        return
                    


    # If the current config or value is not a dict or no 'tune' elements are found, yield the config/value as is
    yield config


def main(config_file=None, log_directory=None, model_class="gmm", eval_mode="default"):
    # Your code for running deep learning experiments goes here
    print("Running deep learning experiments...")
    print("Parameters:")
    print("hyperparameters:", config_file)
    print("log_directory:", log_directory)
    print("model_class:", model_class)
    print("eval_mode:", eval_mode)

    torch.autograd.set_detect_anomaly(True)

    # Load hyperparameters
    with open(config_file, 'r') as f:
        true_config = yaml.safe_load(f)
    
    # Load model class
    if model_class.lower() == "gmm":
        model_class = MDN
    elif model_class.lower() == "vae":
        model_class = VAE
    else:
        raise ValueError(f"Model class {model_class} not supported.")

    configs = [conf for conf in generate_configs(config=true_config)]
    for idx, config in enumerate(configs):
        if len(configs) > 1:
            config["config_id"] = config["config_id"] + "_cnf" + str(idx)

        # Load data
        data_hyperparams = config["data_hyperparameters"]
        data_type = data_hyperparams["data_type"]
        if data_type == "synthetic":
            data_module = SyntheticDataModule(**data_hyperparams)
        elif data_type == "voest":
            data_module = VoestDataModule(**data_hyperparams)
        elif data_type == "uci":
            data_module = UCIDataModule(**data_hyperparams)
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
