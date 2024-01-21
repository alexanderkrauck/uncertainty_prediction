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

## Issues with the Log-Likelihood as a metric for CDE

The mean log-likelihood is a commonly used evaluation metric in probabilistic models like conditional density estimation, valued for its direct assessment of how well a model's probability distribution fits the data. However, it has several limitations: being sensitive to the data's scale, it can yield misleading comparisons across datasets with different variances. As a relative measure, its absolute value lacks a standardized interpretation, making it less intuitive than metrics like accuracy or RMSE, particularly for those not familiar with probabilistic models. Its sensitivity to outliers means that a few anomalous data points can disproportionately affect the metric, potentially leading to skewed evaluations. Additionally, for models involving complex or high-dimensional distributions, computing the log-likelihood can be computationally challenging, limiting its practicality in certain applications. These factors necessitate careful consideration when using mean log-likelihood for model evaluation, especially when comparing across diverse datasets or communicating results to a non-specialist audience.

## Idea: New metric sigma-y-overlap-score

The evaluation metric for conditional density estimation (CDE) models treats each test data point $y_i$ as a probability distribution (e.g., a Gaussian with fixed variance centered around $y_i$) rather than as a fixed value. The model's predicted distribution for a given $x_i$ is then compared with this intrinsic distribution of $y_i$. The metric quantifies the degree of overlap between these two distributions by integrating the area of their intersection. This intersection area, indicative of the alignment between the model's predictions and the test data's probabilistic nature, is averaged over all test samples to provide a performance metric. While this method offers a nuanced view of the model's predictive accuracy, it necessitates careful consideration of the scaling of $y$ and the choice of hyperparameters for the intrinsic distribution to ensure consistency and comparability across different datasets.

## KMN

The kernel mixture network in its most general form works by the following steps:

1. Determine $n$ kernel centers from the train data. That can be done by multiple different options like randomly, kmeans or something like that.
2. Decide for $k$ different kernel types. The kernel types can be the same with different parameters too, like two Gaussian kernels with different scales. Those kernel hyperparameters can also be trainable if so desired.
3. Construct a MLP with arbitrary activations and structure that takes $\mathbf{x}$ as input and outputs one weight for each of the kernels. In total we have $k \cdot n$ kernel functions that essentially work in a flat manner; It does not really matter which kernel function a kernel belongs to. So the MLP output needs to have shape $l \cdot n$.
4. Now we just scale those kernels and treat them like distributions for the most part. So we basically have a density mixture now with estimated weights on the basis of $\mathbf{x}$.
5. We can calculate the likelihood of the $y$ part of samples now.

It is a convenient choice to just take Gaussian kernels as this greatly simplifies the required code.

## Normalizing Flows vs. VAE: My thoughts

Normalising flows are despite the intial thought similarity quite different from VAEs. The reason is that while variational autoencoders still try to maintain a sort of similar semantic structure in the latent space as in the original space the normalising flow does not actually. The normalising flow simply does everything in its power to transform the domain space to the simple distribution.

In particular, in a VAE we do not actually predict the laten variable but instead a distribution over it and sample from it. Thereby we implicity enfore a structural similarity. In a Normalising flow wo don't. Actually, in a normalising flow a point that is on the very peak of the simple distribution with the highest likelihood could just as well map to a point in the domain space that is extremely unlikely; all that because the density of the distribution is transformed in almost arbitrary ways and thus the densities can be completely different. In a VAE that would not happen because in a VAE we actually inherntly assume that the domain data follows the latent distribution in one way or another. All that because of the sampling in the latent space.

However, i need to do more research on this topic!
