# A New Perspective on Uncertainty Techniques in Regression

This repository contains conditional density estimation approaches for regression tasks. The associated master's thesis can be found in `Master Thesis - A New Perspective on Uncertainty Techniques in Regression.pdf`.

This code is not meant to be a comprehensive framework for general use but rather an experimental code-stack that I used for my research. Nevertheless, I am happy if others can learn from it or find it useful in some way.

The folders in this repository include:

- `configs`: Hyperparameter configurations for the experiments.
- `graphics`: Graphs and visualizations illustrating methods and contents of this repository.
- `important_logs`: Log files for significant experiments.
- `notebooks`: Jupyter Notebooks used for experiments and exploratory analysis. They are not necessarily constructed to be executable from top to bottom but served as a tool for exploratory work.
- `notes`: Results from other repositories used for conducting experiments.
- `utils`: Main logic of the pipeline.
- `main.py`: Command line entry point for running experiments. Run experiments by calling `python main.py --config_file=...`.

The code in this repository facilitates a robust pipeline for running experiments in the field of conditional density estimation. To work with this repository, install the required conda environment specified in `environment.yml` by running:

```bash
conda env create -f environment.yml
```

## Citation

If you use this work or find it helpful, please cite as follows

```bibtex
@mastersthesis{YourName2023,
  title={A New Perspective on Uncertainty Techniques in Regression},
  author={Alexander Krauck},
  year={2024},
  school={Johannes Kepler University Linz},
  url={https://github.com/alexanderkrauck/uncertainty_prediction}
}
```
