"""
Utitlity functions for handling data.

Copyright (c) 2024 Alexander Krauck

This code is distributed under the MIT license. See LICENSE.txt file in the 
project root for full license information.
"""

__author__ = "Alexander Krauck"
__email__ = "alexander.krauck@gmail.com"
__date__ = "2024-02-01"

# Standard libraries
import os
import sys
import pickle

# Third-party libraries
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from typing import Iterable, Optional

# Local/Application Specific
path_to_repo = os.path.abspath(".") + "/other_repos/Conditional_Density_Estimation/"
if path_to_repo not in sys.path:
    sys.path.insert(0, path_to_repo)
from cde.density_simulation import GaussianMixture  # type: ignore

import alpaca.utils.datasets.config as alpaca_config
alpaca_config.DATA_DIR = "./datasets/alpaca_datasets"
from alpaca.utils.datasets.builder import build_dataset


class SampleDensities:
    def __init__(self, y_space:np.ndarray, densities: np.ndarray):
        self.y_space = y_space
        self.densities = densities

    def __len__(self):
        return len(self.densities)

    def __getitem__(self, idx):
        return self.densities[idx]


class CustomDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_densities: Optional[SampleDensities] = None,
    ):
        self.x = torch.tensor(x, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.float)

        self.mean_x, self.std_x = torch.mean(self.x, dim=0), torch.std(self.x, dim=0)
        self.mean_y, self.std_y = torch.mean(self.y, dim=0), torch.std(self.y, dim=0)

        self.sample_densities = sample_densities

    @property
    def y_space(self):
        return self.sample_densities.y_space if self.sample_densities is not None else None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.sample_densities is not None:
            return self.x[idx], self.y[idx], self.sample_densities[idx]
        return self.x[idx], self.y[idx]


class TrainingDataModule:
    def __init__(self, train_dataset: CustomDataset, val_dataset: CustomDataset):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    @property
    def y_space(self):
        return self.train_dataset.y_space

    def get_train_dataloader(self, batch_size: int):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

    def get_val_dataloader(self, batch_size: int):
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False)

    def has_distribution(self):
        return self.train_dataset.sample_densities is not None


class DataModule(ABC, TrainingDataModule):
    def __init__(self, pre_normalize_datasets: bool = False, **kwargs):
        self.pre_normalize_datasets = pre_normalize_datasets

        self.initialize_data(**kwargs)
        self.fix_dimensions()
        self.create_datasets()

    def get_train_dataloader(self, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_val_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=batch_size, shuffle=shuffle)

    def get_test_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=batch_size, shuffle=shuffle)

    @abstractmethod
    def iterable_cv_splits(
        self, n_splits: int, seed: int
    ) -> Iterable[TrainingDataModule]:
        pass

    @abstractmethod
    def initialize_data(self, **kwargs) -> None:
        pass

    @abstractmethod
    def has_distribution(self) -> bool:
        pass

    def create_datasets(self) -> None:
        if self.pre_normalize_datasets:
            self.normalize_datasets()

        if self.has_distribution():
            self.train_dataset = CustomDataset(
                self.x_train,
                self.y_train,
                self.densities_train,
            )
            self.val_dataset = CustomDataset(self.x_val, self.y_val, self.densities_val)
            self.test_dataset = CustomDataset(
                self.x_test, self.y_test, self.densities_test
            )
        else:
            self.train_dataset = CustomDataset(self.x_train, self.y_train)
            self.val_dataset = CustomDataset(self.x_val, self.y_val)
            self.test_dataset = CustomDataset(self.x_test, self.y_test)

    def normalize_datasets(self) -> None:
        x_scaler = StandardScaler().fit(self.x_train)
        y_scaler = StandardScaler().fit(self.y_train)

        self.x_train = x_scaler.transform(self.x_train)
        self.x_val = x_scaler.transform(self.x_val)
        self.x_test = x_scaler.transform(self.x_test)

        self.y_train = y_scaler.transform(self.y_train)
        self.y_val = y_scaler.transform(self.y_val)
        self.y_test = y_scaler.transform(self.y_test)

    def fix_dimensions(self) -> None:
        self.x_train = self.x_train.reshape(self.x_train.shape[0], -1)
        self.x_val = self.x_val.reshape(self.x_val.shape[0], -1)
        self.x_test = self.x_test.reshape(self.x_test.shape[0], -1)
        self.y_train = self.y_train.reshape(self.y_train.shape[0], -1)
        self.y_val = self.y_val.reshape(self.y_val.shape[0], -1)
        self.y_test = self.y_test.reshape(self.y_test.shape[0], -1)


class SyntheticDataModule(DataModule):
    def initialize_data(
        self, data_path: str, n_density_steps=100, min_max_density_stds=0.5, **kwargs
    ):
        self.data_path = data_path
        with open(data_path + ".pkl", "rb") as file:
            self.distribution = pickle.load(file)

        self.x_train = np.loadtxt(data_path + "_x_train.csv", delimiter=",")
        self.y_train = np.loadtxt(data_path + "_y_train.csv", delimiter=",")
        self.x_test = np.loadtxt(data_path + "_x_test.csv", delimiter=",")
        self.y_test = np.loadtxt(data_path + "_y_test.csv", delimiter=",")
        self.x_val = np.loadtxt(data_path + "_x_val.csv", delimiter=",")
        self.y_val = np.loadtxt(data_path + "_y_val.csv", delimiter=",")

        self.density_y_min = (
            self.y_train.min() - min_max_density_stds * self.y_train.std()
        )
        self.density_y_max = (
            self.y_train.max() + min_max_density_stds * self.y_train.std()
        )
        self.n_density_steps = n_density_steps

        self.y_space = np.linspace(self.density_y_min, self.density_y_max, n_density_steps)
        self.densities_train = self.get_sample_densities(self.x_train)
        self.densities_val = self.get_sample_densities(self.x_val)
        self.densities_test = self.get_sample_densities(self.x_test)
    
    def get_sample_densities(self, x: np.ndarray) -> SampleDensities:
        y_space = self.y_space.reshape(-1, 1)

        return SampleDensities(
            self.y_space,
            np.array([self.distribution.pdf(np.tile(inner_x, (self.n_density_steps, 1)), y_space) for inner_x in x]),
        )

    def iterable_cv_splits(self, n_splits: int, seed: int):
        # Ensure numpy array type for compatibility with KFold
        x = np.array(self.x_train)
        y = np.array(self.y_train)

        # Create a KFold object
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Generator function to yield train and validation DataLoaders
        def cv_generator():
            for train_indices, val_indices in kfold.split(x):
                # Create subsets for this fold
                x_train_fold, y_train_fold = x[train_indices], y[train_indices]
                x_val_fold, y_val_fold = x[val_indices], y[val_indices]

                # Wrap them in TensorDataset and DataLoader
                train_dataset = CustomDataset(x_train_fold, y_train_fold, self.get_sample_densities(x_train_fold))
                val_dataset = CustomDataset(x_val_fold, y_val_fold, self.get_sample_densities(x_val_fold))

                yield TrainingDataModule(train_dataset, val_dataset)

        # Return the generator object
        return cv_generator()

    def has_distribution(self):
        return True


class UCIDataModule(DataModule):
    def initialize_data(
        self,
        dataset_name: str,
        val_split: float = 0.0,
        test_split: float = 0.0,
        random_state: int = 42,
        **kwargs,
    ):
        self.x_total, self.y_total = UCIDataModule.load_full_ds(dataset_name)

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_total, self.y_total, test_size=val_split, random_state=random_state
        )
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_train,
            self.y_train,
            test_size=test_split
            * (
                (1 / (1 - val_split)) - 1
            ),  # because test_split is relative to total size
            random_state=random_state,
        )

    @staticmethod
    def load_full_ds(dataset_name: str, val_split: float = 0.0, random_state: int = 42):
        """
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset to be loaded from DATASETS.keys()
        val_split : float
            Fraction of the dataset to be used for validation. If 0, the entire dataset is returned.
        """
        dataset = build_dataset(dataset_name, val_split=1)
        x_train_tmp, y_train_tmp = dataset.dataset("train")
        x_val_tmp, y_val_tmp = dataset.dataset("val")
        x_total = np.concatenate([x_train_tmp, x_val_tmp])
        y_total = np.concatenate([y_train_tmp, y_val_tmp])

        if val_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                x_total, y_total, test_size=val_split, random_state=random_state
            )
            return X_train, y_train, X_val, y_val
        else:
            return x_total, y_total

    def has_distribution(self) -> bool:
        return False

    def iterable_cv_splits(
        self, n_splits: int, seed: int
    ) -> Iterable[TrainingDataModule]:
        # Ensure numpy array type for compatibility with KFold
        x = np.concatenate((self.x_train, self.x_val), axis=0)
        y = np.concatenate((self.y_train, self.y_val), axis=0)

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

                yield TrainingDataModule(train_dataset, val_dataset)

        # Return the generator object
        return cv_generator()


class VoestDataModule(DataModule):
    def initialize_data(
        self,
        data_path: str = "datasets/voest_datasets",
        original: bool = False,
        val_split: float = 0.15,
        test_split: float = 0.15,
        split_random: bool = False,
        random_state: int = 42,
        **kwargs,
    ):
        self.data_path = data_path
        self.original = original
        if original:
            meta_cols = 4
            filename = "Export_Nov22_Oktl23_Rohdaten_Kontr.csv"
            remove_first_n = 0
        else:
            meta_cols = 1
            filename = "Export_Nov22_Oktl23_Rohdaten_Train.csv"
            remove_first_n = 56

        voest_ds = pd.read_csv(
            filepath_or_buffer=f"{data_path}/{filename}", sep=";", dtype=str
        )
        voest_ds = voest_ds.iloc[remove_first_n:]
        voest_ds = VoestDataModule.sort_columns_excluding_first_n(voest_ds, meta_cols)

        for col in voest_ds.columns:
            if str(col).isdigit():
                if voest_ds[col].dtype == "float64":
                    voest_ds[col] = voest_ds[col].astype(np.float32)
                else:
                    voest_ds[col] = (
                        voest_ds[col].str.replace(",", ".").astype(np.float32)
                    )

        voest_ds = VoestDataModule.move_column_to_first(voest_ds, "85")
        meta_cols += 1

        # new_columns = [id_to_name.get(int(col), col) if str(col).isdigit() else col for col in voest_ds_1.columns]
        # voest_ds_1.columns = new_columns

        self.x_total = voest_ds.iloc[:, meta_cols:].to_numpy()
        self.y_total = voest_ds["85"].to_numpy()

        mean_vals = np.nanmean(self.x_total, axis=0)
        std_exclude = np.nan_to_num(np.nanstd(self.x_total, axis=0)) < 1e-7
        nan_mask = np.isnan(self.x_total)
        self.x_total[nan_mask] = np.take(mean_vals, np.where(nan_mask)[1])
        self.x_total = np.nan_to_num(self.x_total)

        self.x_total = self.x_total[
            :, ~std_exclude
        ]  # remove columns that are constant as they do not contribute to the model at all

        if split_random:
            self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
                self.x_total,
                self.y_total,
                test_size=test_split,
                random_state=random_state,
            )
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
                self.x_train,
                self.y_train,
                test_size=0.2 * ((1 / (1 - test_split)) - 1),
                random_state=random_state,
            )
        else:
            n_rows = self.x_total.shape[0]

            self.x_train = self.x_total[: int(n_rows * (1 - val_split - test_split))]
            self.x_val = self.x_total[
                int(n_rows * (1 - val_split - test_split)) : int(
                    n_rows * (1 - test_split)
                )
            ]
            self.x_test = self.x_total[int(n_rows * (1 - test_split)) :]

            self.y_train = self.y_total[: int(n_rows * (1 - val_split - test_split))]
            self.y_val = self.y_total[
                int(n_rows * (1 - val_split - test_split)) : int(
                    n_rows * (1 - test_split)
                )
            ]
            self.y_test = self.y_total[int(n_rows * (1 - test_split)) :]

    def has_distribution(self) -> bool:
        return False

    def iterable_cv_splits(
        self, n_splits: int, seed: int
    ) -> Iterable[TrainingDataModule]:
        # Ensure numpy array type for compatibility with KFold
        x = np.array(self.x_train + self.x_val)
        y = np.array(self.y_train + self.y_val)

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

                yield TrainingDataModule(train_dataset, val_dataset)

        # Return the generator object
        return cv_generator()

    @staticmethod
    def sort_columns_excluding_first_n(df, n):
        """
        Sort all but the first n columns of a DataFrame by their column names.

        Parameters:
        df (pd.DataFrame): The original DataFrame.
        n (int): The number of first columns to exclude from sorting.

        Returns:
        pd.DataFrame: A new DataFrame with all but the first n columns sorted by column name.
        """

        # Step 1: Split the DataFrame
        excluded_part = df.iloc[:, :n]  # First n columns to exclude from sorting
        sortable_part = df.iloc[:, n:]  # The rest of the columns

        # Step 2: Sort the rest of the columns by their names
        sorted_rest = sortable_part[sorted(sortable_part.columns, key=lambda x: int(x))]

        # Step 3: Reassemble the DataFrame
        df_sorted = pd.concat([excluded_part, sorted_rest], axis=1)

        return df_sorted

    @staticmethod
    def move_column_to_first(df, column_name="85"):
        """
        Move a specified column by name to the first position in the DataFrame.

        Parameters:
        df (pd.DataFrame): The original DataFrame.
        column_name (str): The name of the column to move.

        Returns:
        pd.DataFrame: A new DataFrame with the specified column moved to the first position.
        """

        # Ensure the column exists in the DataFrame
        if column_name in df.columns:
            # Create a list of columns with the specified column at the first position
            cols = [column_name] + [col for col in df.columns if col != column_name]

            # Reindex the DataFrame with the new column order
            return df[cols]
        else:
            # Return the original DataFrame if the column name doesn't exist
            print(f"Column '{column_name}' does not exist in the DataFrame.")
            return df


class RothfussDataModule(DataModule):
    def initialize_data(
        self,
        dataset_name: str,
        data_path: str = "datasets/nyc_taxi_dataset/",
        val_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42,
        **kwargs,
    ):
        from .data import rothfuss_dataset

        rothfuss_dataset.DATA_DIR = data_path
        os.makedirs(data_path, exist_ok=True)

        dataset_name = dataset_name.lower()
        if dataset_name == "nyc_taxi":
            (
                self.x_total,
                self.y_total,
            ) = rothfuss_dataset.NCYTaxiDropoffPredict().get_target_feature_split()
        elif dataset_name == "energy":
            (
                self.x_total,
                self.y_total,
            ) = rothfuss_dataset.Energy().get_target_feature_split()
        elif dataset_name == "concrete":
            (
                self.x_total,
                self.y_total,
            ) = rothfuss_dataset.Concrete().get_target_feature_split()
        elif dataset_name == "boston_housing":
            (
                self.x_total,
                self.y_total,
            ) = rothfuss_dataset.BostonHousing().get_target_feature_split()
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet.")

        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_total, self.y_total, test_size=val_split, random_state=random_state
        )
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x_train,
            self.y_train,
            test_size=test_split
            * (
                (1 / (1 - val_split)) - 1
            ),  # because test_split is relative to total size
            random_state=random_state,
        )

    def has_distribution(self) -> bool:
        return False

    def iterable_cv_splits(
        self, n_splits: int, seed: int
    ) -> Iterable[TrainingDataModule]:
        # Ensure numpy array type for compatibility with KFold
        x = np.concatenate((self.x_train, self.x_val), axis=0)
        y = np.concatenate((self.y_train, self.y_val), axis=0)

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

                yield TrainingDataModule(train_dataset, val_dataset)

        # Return the generator object
        return cv_generator()