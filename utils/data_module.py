from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch

import pickle
import sys
import numpy as np
import os

from abc import ABC, abstractmethod
from typing import Iterable

path_to_repo = os.path.abspath(".") + '/other_repos/Conditional_Density_Estimation/'

if path_to_repo not in sys.path:
    sys.path.insert(0, path_to_repo)


from cde.density_simulation import GaussianMixture

class CustomDataset(Dataset):
    def __init__(self, x, y):


        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)
    
        self.scaler_x = (torch.mean(self.x, dim=0), torch.std(self.x, dim=0))
        self.scaler_y = (torch.mean(self.y, dim=0), torch.std(self.y, dim=0))

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class TrainingDataModule:
    def __init__(self, train_dataset, val_dataset, distribution = None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.distribution = distribution

    def get_train_dataloader(self, batch_size:int):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_val_dataloader(self, batch_size:int):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

class DataModule(ABC):
    
        @abstractmethod
        def get_train_dataloader(self, batch_size:int) -> DataLoader:
            pass
        
        @abstractmethod
        def get_val_dataloader(self, batch_size:int) -> DataLoader:
            pass
    
        @abstractmethod
        def get_test_dataloader(self, batch_size:int) -> DataLoader:
            pass
        
        @abstractmethod
        def iterable_cv_splits(self, n_splits: int, seed: int) -> Iterable[TrainingDataModule]:
            pass

        @abstractmethod
        def has_distribution(self) -> bool:
            pass

class SyntheticDataModule(DataModule):

    def __init__(self, data_path:str, **kwargs):

        self.data_path = data_path
        with open(data_path+".pkl", 'rb') as file:
            self.distribution = pickle.load(file)

        self.samples_x_train = np.loadtxt(data_path + "_x_train.csv", delimiter=",")
        self.samples_y_train = np.loadtxt(data_path + "_y_train.csv", delimiter=",")
        self.samples_x_test = np.loadtxt(data_path + "_x_test.csv", delimiter=",")
        self.samples_y_test = np.loadtxt(data_path + "_y_test.csv", delimiter=",")
        self.samples_x_val = np.loadtxt(data_path + "_x_val.csv", delimiter=",")
        self.samples_y_val = np.loadtxt(data_path + "_y_val.csv", delimiter=",")

        self.train_dataset = CustomDataset(self.samples_x_train, self.samples_y_train)
        self.val_dataset = CustomDataset(self.samples_x_val, self.samples_y_val)
        self.test_dataset = CustomDataset(self.samples_x_test, self.samples_y_test)


    def get_test_dataloader(self, batch_size:int):
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    
    def get_train_dataloader(self, batch_size:int):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
    
    def get_val_dataloader(self, batch_size:int):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)
    
    def iterable_cv_splits(self, n_splits: int, seed: int):
        # Ensure numpy array type for compatibility with KFold
        x = np.array(self.samples_x_train)
        y = np.array(self.samples_y_train)

        # Create a KFold object
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        # Generator function to yield train and validation DataLoaders
        def cv_generator():
            for train_indices, val_indices in kfold.split(x):
                # Create subsets for this fold
                x_train_fold, y_train_fold = x[train_indices], y[train_indices]
                x_val_fold, y_val_fold = x[val_indices], y[val_indices]

                # Wrap them in TensorDataset and DataLoader
                train_dataset = CustomDataset(x_train_fold, y_train_fold)
                val_dataset = CustomDataset(x_val_fold, y_val_fold)

                yield TrainingDataModule(train_dataset, val_dataset, self.distribution)

        # Return the generator object
        return cv_generator()
    
    def has_distribution(self):
        return True
    
