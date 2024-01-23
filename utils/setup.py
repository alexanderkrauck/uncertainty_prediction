import os, subprocess
from utils.data_module import SyntheticDataModule, VoestDataModule, UCIDataModule, DataModule, TrainingDataModule, NYCTaxiDataModule
from utils.models import MDN, VAE, ConditionalDensityEstimator, GaussianKMN, NFDensityEstimator
from copy import deepcopy

def sync_wandb(project_name: str):
    for item in os.listdir(os.path.join("runs", project_name, "wandb")):
        if item.startswith("offline-run"):
            run_path = os.path.join("runs", project_name, "wandb", item)
            sync_wand_run(run_path)
        
def sync_wand_run(run_path: str):
    subprocess.run(
        [
            "wandb",
            "sync",
            run_path,
        ]
    )
        
def load_data_module(data_type: str, **data_hyperparameters:dict) -> DataModule:

    data_type = data_type.lower()
    if data_type == "synthetic":
        return SyntheticDataModule(**data_hyperparameters)
    elif data_type == "voest":
        return VoestDataModule(**data_hyperparameters)
    elif data_type == "uci":
        return UCIDataModule(**data_hyperparameters)
    elif data_type == "nyctaxi":
        return NYCTaxiDataModule(**data_hyperparameters)
    else:
        raise ValueError(f"Data type {data_type} not supported yet.")
    
def load_model_class(model_class: str) -> ConditionalDensityEstimator:
    model_class = model_class.lower()
    if model_class == "mdn":
        return MDN
    elif model_class == "kmn":
        return GaussianKMN
    elif model_class == "nf":
        return NFDensityEstimator
    elif model_class == "vae":
        return VAE
    else:
        raise ValueError(f"Model class {model_class} not supported yet.")
    
def load_model(train_data_module: TrainingDataModule, model_class: str, **model_hyperparameters: dict):
    model_class = load_model_class(model_class)
    return model_class(train_data_module, **model_hyperparameters)
    
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
                        new_config = deepcopy(config)
                        new_config[key] = option  # Replace 'tune' with the actual value
                        yield from generate_configs(
                            new_config
                        )  # Recurse with the new config
                    return  # Once all 'tune' options for a key are processed, return
                else:
                    # Recurse into nested dictionaries
                    results = [result for result in generate_configs(value)]
                    if len(results) > 1:
                        for result in results:
                            new_config = deepcopy(config)
                            new_config[key] = result
                            yield from generate_configs(new_config)
                        return

    # If the current config or value is not a dict or no 'tune' elements are found, yield the config/value as is
    yield config