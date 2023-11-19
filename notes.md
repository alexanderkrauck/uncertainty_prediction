# Used for Installation

conda create -n uncertainty_prediction python pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge numpy pandas matplotlib scikit-learn jupyterlab
pip install alpaca-ml xlrd openpyxl

alpaca-ml: <https://alpaca-ml.readthedocs.io/en/latest/genindex.html> <https://github.com/stat-ml/alpaca>
