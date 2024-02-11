# Neural Conditional Density Estimation for Regression Tasks

This repository contains conditional density estimation approaches for regression tasks. The Folders of the repository contain:

- `configs`: Hyperparameter Configurations for the experiments done.
- `graphics`: Graphs/Visualizations made that illustrate methods and contents of this repository.
- `important_logs`: Log files for important experiments.
- `notebooks`: Jupyter Notebooks that are used by me to do experiments and explorative analysis of methods and approaches. They are **not** constructed in a way to be executable from top to bottom necissarely but rather served as a tool for me to do "dirty work".
- `notes`: Results from other repositories that I use to conduct experiments.
- `utils`: Main logic of my pipeline.
- `main.py`: Command Line Entry Point for running experiments. Run experiments by calling `python main.py --config_file=...`.

The code in this repository facilitates a robust pipeline for running experiments in the field of conditional density estimation. To work with this repository, you need to install the required conda environment in `environment.yml` by running.

```conda env create -f environment.yml```
