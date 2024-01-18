# Notes for the master thesis project "Uncertainty prediction with Machine Learning"

## Used for Installation

conda create -n uncertainty_prediction python pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge numpy pandas matplotlib scikit-learn jupyterlab
pip install alpaca-ml xlrd openpyxl seaborn

alpaca-ml: <https://alpaca-ml.readthedocs.io/en/latest/genindex.html> <https://github.com/stat-ml/alpaca>

## TanH stability

The original way of predicting the standard deviation of the mixture density network components is to predict the log std instead and then exponentiate it. However, for certain datasets it can lead to instability. I think that is because some datasets have strong outliers and then the predicted standard deviation can be very large. If it now is too large, then exponentiating it will make it go to infinity sometimes or something that can not be dealt with numerically. I fixed it now with using tanh on the predicted log std together with a scaling of the tanh to still allow bigger values. That means the exponent will not be too extreme and will always be able to be handeld numerically.

The TanH seems to work fine. I evaluated it on a dataset where it is not required (synthetic dataset) and the performance is comparable. Thus I decide on always using it for further experiments.

## Evaluation in Rothfuss paper about noise regularisation

Reported in Table 1, are averages over 3 different train/test splits and 5 seeds each for initializing the neural networks. The heldout test set amounts to 20% of the respective data set. Consistent with the results of the simulation study, noise regularization outperforms the other methods across the great majority of data sets and CDE models.

## Correctness of the (negative) log likelihood as an objective function

The NLL Loss actually is the ideal loss function for CDE as implicitly we are having this $\arg \max _{\theta \in \Theta} \sum_{i=1}^n \log \hat{f}_\theta\left(x_i\right)=\arg \min _{\theta \in \Theta} \mathcal{D}_{K L}\left(p_{\mathcal{D}} \| \hat{f}_\theta\right)$ with $p_{\mathcal{D}}(x)=\frac{1}{n} \sum_{i=1}^n \delta\left(\left\|x-x_i\right\|\right)$. $p_{\mathcal{D}}(x)$ is the mixture function of point masses that we empiricly have. Of course this is only of very theoretic nature but if we assume that the training datapoints correctly represent the sourcing distribution, then we want in fact exactly $p_{\mathcal{D}}(x)$. However, clearly as long as we have a limited amount of training samples this will not actually resemble the sourcing distribution as this would be dense. Consequently, we need regularisation metrics that induce smoothness assumtions into the model.

## Scaling of Distributions for Loss Calculation

I was doing the loss calculation on the scale of the domain of the input data (the scale of $y$ to be precise). I realised that it is not the same. Now for training the predicted distributions are kept in the normalised domain and the loss is calculated w.r.t. them which means that also the likelihood is different (as the distribution is scaled). For evaluation however I figured that it would be better to stay in the original domain of the data to keep it more comparable to other results.
