import os, subprocess
from utils.data_module import SyntheticDataModule, VoestDataModule, UCIDataModule, DataModule
from utils.models import MDN, VAE

def sync_wandb(project_name: str):
    for item in os.listdir(os.path.join("runs", project_name, "wandb")):
        if item.startswith("offline-run"):
            subprocess.run(
                [
                    "wandb",
                    "sync",
                    os.path.join("runs", project_name, "wandb", item),
                ])
            
        
def load_data_module(data_type: str, **data_hyperparameters:dict) -> DataModule:

    data_type = data_type.lower()
    if data_type == "synthetic":
        return SyntheticDataModule(**data_hyperparameters)
    elif data_type == "voest":
        return VoestDataModule(**data_hyperparameters)
    elif data_type == "uci":
        return UCIDataModule(**data_hyperparameters)
    else:
        raise ValueError(f"Data type {data_type} not supported yet.")
    
def load_model_class(model_class: str):
    model_class = model_class.lower()
    if model_class == "mdn":
        return MDN
    elif model_class == "vae":
        return VAE
    else:
        raise ValueError(f"Model class {model_class} not supported yet.")