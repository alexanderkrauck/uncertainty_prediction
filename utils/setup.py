import os, subprocess


def sync_wandb(project_name: str):
    for item in os.listdir(os.path.join("runs", project_name, "wandb")):
        if item.startswith("offline-run"):
            subprocess.run(
                [
                    "wandb",
                    "sync",
                    os.path.join("runs", project_name, "wandb", item),
                ])