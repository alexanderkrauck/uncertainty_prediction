# Notes for the master thesis project "Uncertainty prediction with Machine Learning"

## Used for Installation

conda create -n uncertainty_prediction python pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge numpy pandas matplotlib scikit-learn jupyterlab
pip install alpaca-ml xlrd openpyxl seaborn

alpaca-ml: <https://alpaca-ml.readthedocs.io/en/latest/genindex.html> <https://github.com/stat-ml/alpaca>

## Other Random Notes

The original way of predicting the standard deviation of the mixture density network components is to predict the log std instead and then exponentiate it. However, for certain datasets it can lead to instability. I think that is because some datasets have strong outliers and then the predicted standard deviation can be very large. If it now is too large, then exponentiating it will make it go to infinity sometimes or something that can not be dealt with numerically. I fixed it now with using tanh on the predicted log std together with a scaling of the tanh to still allow bigger values. That means the exponent will not be too extreme and will always be able to be handeld numerically.
