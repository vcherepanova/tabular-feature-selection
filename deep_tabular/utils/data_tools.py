""" data_tools.py
    Tools for building tabular datasets
    Developed for deep-tabular project
    April 2022
"""

import logging
import os
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import openml
import pandas as pd
import sklearn.preprocessing
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from synthetic_data.synthetic_data import make_tabular_data

# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115



def get_categories_full_cat_data(full_cat_data_for_encoder):
    return (
        None
        if full_cat_data_for_encoder is None
        else [
            len(set(full_cat_data_for_encoder.values[:, i]))
            for i in range(full_cat_data_for_encoder.shape[1])
        ]
    )

def get_data_locally(name, add_noise, noise_percent):
    path_type_1_c = f'/data/{name}/C_train.npy'
    path_type_1_n = f'/data/{name}/N_train.npy'
    path_type_2_c = f'/data/{name}/X_cat_train.npy'
    path_type_2_n = f'/data/{name}/X_num_train.npy'

    targets = np.concatenate(
        [np.load('/data/{}/y_{}.npy'.format(name, part)) for part in ['train', 'val', 'test']])
    targets = pd.Series(targets, name='target')

    if os.path.exists(path_type_1_c) or os.path.exists(path_type_1_n):
        path_template_c = '/data/{}/C_{}.npy'
        path_template_n = '/data/{}/N_{}.npy'
    else:
        path_template_c = '/data/{}/X_cat_{}.npy'
        path_template_n = '/data/{}/X_num_{}.npy'

    cat_path = path_template_c.format(name, 'train')
    if os.path.exists(cat_path):
        warnings.warn('The dataset contains categorical features and this code was not tested on datasets with categorical features properly.')
        categorical_array = np.vstack(
            [np.load(path_template_c.format(name, part)) for part in ['train', 'val', 'test']])
        categorical_columns = ['cat_{}'.format(i) for i in range(categorical_array.shape[1])]
        cat_df = pd.DataFrame(categorical_array, columns=categorical_columns)
    else:
        cat_df = pd.DataFrame()
        categorical_columns = []

    num_path = path_template_n.format(name, 'train')
    if os.path.exists(num_path):
        numerical_array = np.vstack(
            [np.load(path_template_n.format(name, part)) for part in ['train', 'val', 'test']])
        if name=='microsoft':
            indices = np.random.choice(numerical_array.shape[0], size=500000, replace=False)
            numerical_array = numerical_array[indices]
            targets = targets[indices]
        # adding uninformative features
        if add_noise == 'random_feats':
            np.random.seed(0)
            n_feats = int(numerical_array.shape[1]/(1-noise_percent)*noise_percent)
            uninformative_features = np.random.randn(numerical_array.shape[0], n_feats)
            numerical_array = np.concatenate([numerical_array, uninformative_features], axis=1)
        elif add_noise == 'corrupted_feats':
            np.random.seed(0)
            n_max = int(numerical_array.shape[1] / 0.1 * 0.9)
            n_feats = int(numerical_array.shape[1]/(1-noise_percent)*noise_percent)
            features_idx = np.random.choice(numerical_array.shape[1], n_max, replace=True)[:n_feats]
            features_copy = numerical_array[:,features_idx]
            features_std = np.nanstd(features_copy, axis=0)
            alpha_noise = 0.5
            corrupted_features = (1-alpha_noise)*features_copy + alpha_noise*np.random.randn(numerical_array.shape[0], n_feats)*features_std
            numerical_array = np.concatenate([numerical_array, corrupted_features], axis=1)
        elif add_noise == 'secondorder_feats':
            np.random.seed(0)
            n_max = int(numerical_array.shape[1]/0.1*0.9)
            n_feats = int(numerical_array.shape[1]/(1-noise_percent)*noise_percent)
            features_1 = np.random.choice(numerical_array.shape[1], n_max, replace=True)[:n_feats]
            features_2 = np.random.choice(numerical_array.shape[1], n_max, replace=True)[:n_feats]
            second_order_features = numerical_array[:,features_1]*numerical_array[:,features_2]
            numerical_array = np.concatenate([numerical_array, second_order_features], axis=1)

        numerical_columns = ['num_{}'.format(i) for i in range(numerical_array.shape[1])]
        num_df = pd.DataFrame(numerical_array, columns=numerical_columns)
    else:
        num_df = pd.DataFrame()
        numerical_columns = []

    data = pd.concat([num_df, cat_df], axis=1)

    return data, targets, categorical_columns, numerical_columns

def get_data(dataset_id, task, datasplit=[.65, .15, .2], seed=0, add_noise=False, noise_percent=0.0):

    data, targets, categorical_columns, numerical_columns = get_data_locally(dataset_id, add_noise, noise_percent)
    np.random.seed(seed)
    # reindex and find NaNs/Missing values in categorical columns
    data, targets = data.reset_index(drop=True), targets.reset_index(drop=True)

    data[categorical_columns] = data[categorical_columns].fillna("___null___")

    if task != 'regression':
        l_enc = LabelEncoder()
        targets = l_enc.fit_transform(targets)
    else:
        targets = targets.to_numpy()

    # split data into train/val/test
    train_size, test_size, valid_size = datasplit[0], datasplit[2], datasplit[1]/(1-datasplit[2])
    if task != 'regression':
        data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=test_size, random_state=seed, stratify = targets)
        data_train, data_val, targets_train, targets_val = train_test_split(data_train, targets_train, test_size=valid_size, random_state=seed, stratify = targets_train)
    else:
        data_train, data_test, targets_train, targets_test = train_test_split(data, targets, test_size=test_size, random_state=seed)
        data_train, data_val, targets_train, targets_val = train_test_split(data_train, targets_train, test_size=valid_size, random_state=seed)

    data_cat_train = data_train[categorical_columns].values
    data_num_train = data_train[numerical_columns].values

    data_cat_val = data_val[categorical_columns].values
    data_num_val = data_val[numerical_columns].values

    data_cat_test = data_test[categorical_columns].values
    data_num_test = data_test[numerical_columns].values

    info = {"name": dataset_id,
            "task_type": task,
            "n_num_features": len(numerical_columns),
            "n_cat_features": len(categorical_columns),
            "train_size": data_train.shape[0],
            "val_size": data_val.shape[0],
            "test_size": data_test.shape[0]}

    if task == "multiclass":
        info["n_classes"] = len(set(targets))
    if task == "binclass":
        info["n_classes"] = 1
    if task == "regression":
        info["n_classes"] = 1

    if len(numerical_columns) > 0:
        numerical_data = {"train": data_num_train, "val": data_num_val, "test": data_num_test}
    else:
        numerical_data = None

    if len(categorical_columns) > 0:
        categorical_data = {"train": data_cat_train, "val": data_cat_val, "test": data_cat_test}
    else:
        categorical_data = None

    targets = {"train": targets_train, "val": targets_val, "test": targets_test}

    if len(categorical_columns) > 0:
        full_cat_data_for_encoder = data[categorical_columns]
    else:
        full_cat_data_for_encoder = None

    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder

@dataclass
class TabularDataset:
    x_num: Optional[Dict[str, np.ndarray]]
    x_cat: Optional[Dict[str, np.ndarray]]
    y: Dict[str, np.ndarray]
    info: Dict[str, Any]
    normalization: Optional[str]
    cat_policy: str
    seed: int
    full_cat_data_for_encoder: Optional[pd.DataFrame]
    y_policy: Optional[str] = None

    @property
    def is_binclass(self):
        return self.info['task_type'] == "binclass"

    @property
    def is_multiclass(self):
        return self.info['task_type'] == "multiclass"

    @property
    def is_regression(self):
        return self.info['task_type'] == "regression"

    @property
    def n_num_features(self):
        return self.info["n_num_features"]

    @property
    def n_cat_features(self):
        return self.info["n_cat_features"]

    @property
    def n_features(self):
        return self.n_num_features + self.n_cat_features

    @property
    def n_classes(self):
        return self.info["n_classes"]

    @property
    def parts(self):
        return self.x_num.keys() if self.x_num is not None else self.x_cat.keys()

    def size(self, part: str):
        x = self.x_num if self.x_num is not None else self.x_cat
        assert x is not None
        return len(x[part])

    def normalize(self, x_num, noise=1e-3):
        x_num_train = x_num['train'].copy()
        if self.normalization == 'standard':
            normalizer = sklearn.preprocessing.StandardScaler()
        elif self.normalization == 'quantile':
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(x_num['train'].shape[0] // 30, 1000), 10),
                subsample=int(1e9),
                random_state=self.seed,
            )
            if noise:
                stds = np.std(x_num_train, axis=0, keepdims=True)
                noise_std = noise / np.maximum(stds, noise)
                x_num_train += noise_std * np.random.default_rng(self.seed).standard_normal(x_num_train.shape)

        else:
            raise ValueError('Unknown Normalization')
        normalizer.fit(x_num_train)
        return {k: normalizer.transform(v) for k, v in x_num.items()}

    def handle_missing_values_numerical_features(self, x_num):
        num_nan_masks = {k: np.isnan(v) for k, v in x_num.items()}
        if any(x.any() for x in num_nan_masks.values()):
            num_new_values = np.nanmean(self.x_num['train'], axis=0)
            for k, v in x_num.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
        return x_num

    def encode_categorical_features(self, x_cat):
        encoder = sklearn.preprocessing.OrdinalEncoder(handle_unknown='error', dtype='int64')
        encoder.fit(self.full_cat_data_for_encoder.values)
        x_cat = {k: encoder.transform(v) for k, v in x_cat.items()}
        return x_cat

    def concatenate_data(self, x_cat, x_num):
        if self.cat_policy == 'indices':
            result = [x_num, x_cat]
        elif self.cat_policy == 'ohe':
            raise ValueError('Not implemented')
        return result

    def preprocess_data(self):
        logging.info('Building Dataset')
        if self.x_num:
            x_num = deepcopy(self.x_num)
            x_num = self.handle_missing_values_numerical_features(x_num)
            if self.normalization:
                x_num = self.normalize(x_num)
        else:
            # if x_num is None replace with empty tensor for dataloader
            x_num = {part: torch.empty(self.size(part), 0) for part in self.parts}

        # if there are no categorical features, return only numerical features
        if self.cat_policy == 'drop' or not self.x_cat:
            assert x_num is not None
            x_num = to_tensors(x_num)
            # if x_cat is None replace with empty tensor for dataloader
            x_cat = {part: torch.empty(self.size(part), 0) for part in self.parts}
            return [x_num, x_cat]

        x_cat = deepcopy(self.x_cat)
        x_cat = self.encode_categorical_features(x_cat)
        x_cat, x_num = to_tensors(x_cat), to_tensors(x_num)
        result = self.concatenate_data(x_cat, x_num)
        return result

    def build_y(self):
        if self.is_regression:
            assert self.y_policy == 'mean_std'
        y = deepcopy(self.y)
        if self.y_policy:
            if not self.is_regression:
                warnings.warn('y_policy is not None, but the task is NOT regression')
                info = None
            elif self.y_policy == 'mean_std':
                mean, std = self.y['train'].mean(), self.y['train'].std()
                y = {k: (v - mean) / std for k, v in y.items()}
                info = {'policy': self.y_policy, 'mean': mean, 'std': std}
            else:
                raise ValueError('Unknown y policy')
        else:
            info = None

        y = to_tensors(y)
        if self.is_regression or self.is_binclass:
            y = {part: y[part].float() for part in self.parts}
        return y, info


def to_tensors(data):
    return {k: torch.as_tensor(v) for k, v in data.items()}
