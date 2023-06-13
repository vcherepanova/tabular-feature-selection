import logging
import os
import random

import torch
from icecream import ic
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.utils.data import TensorDataset, DataLoader
import deep_tabular.models as models
from .data_tools import get_data, get_categories_full_cat_data, TabularDataset
from .warmup import ExponentialWarmup, LinearWarmup
import numpy as np

def get_dataloaders(cfg, return_raw=False):

    cfg_dataset = cfg.dataset
    x_numerical, x_categorical, y, info, full_cat_data_for_encoder = get_data(dataset_id=cfg_dataset.name,
                                                                              task=cfg_dataset.task,
                                                                              datasplit=[.65, .15, .2],
                                                                              seed=0,
                                                                              add_noise=cfg.dataset.add_noise,
                                                                              noise_percent = cfg.dataset.noise_percent,
                                                                              )

    if cfg.mode=='downstream' and cfg.importance_path is not None:
        importances = torch.load(cfg.importance_path)
        # convert % of features to number of features
        if cfg.topk<=1:
            cfg.topk=max(1, round(len(importances)*cfg.topk))
        impo_vals, impo_locs = torch.topk(importances, cfg.topk)
        impo_locs = np.array(impo_locs.tolist())
        x_keys = x_numerical.keys()
        assert cfg.topk <= x_numerical['train'].shape[1], 'topk should be less than number of total feature count'
        for k in x_keys:
            x_numerical[k] = x_numerical[k][:,impo_locs]

        info['n_num_features'] = cfg.topk

    data_seed = 0
    dataset = TabularDataset(x_numerical,
                             x_categorical,
                             y,
                             info,
                             normalization=cfg_dataset.normalization,
                             cat_policy="indices",
                             seed=data_seed,
                             full_cat_data_for_encoder=full_cat_data_for_encoder,
                             y_policy=cfg_dataset.y_policy,
                             )

    X = dataset.preprocess_data()
    Y, y_info = dataset.build_y()
    unique_categories = get_categories_full_cat_data(full_cat_data_for_encoder)
    n_numerical = dataset.n_num_features
    n_categorical = dataset.n_cat_features
    n_classes = dataset.n_classes
    logging.info(f"Task: {cfg_dataset.task}, Dataset: {cfg_dataset.name}, n_numerical: {n_numerical}, "
                 f"n_categorical: {n_categorical}, n_classes: {n_classes}, n_train_samples: {dataset.size('train')}, "
                 f"n_val_samples: {dataset.size('val')}, n_test_samples: {dataset.size('test')}")

    if return_raw:
        raw_data = {"train": (X[0]["train"], X[1]["train"], Y["train"]), "val": (X[0]["val"], X[1]["val"], Y["val"]), "test": (X[0]["test"], X[1]["test"], Y["test"])}
        return raw_data, unique_categories, n_numerical, n_classes

    trainset = TensorDataset(X[0]["train"], X[1]["train"], Y["train"])
    valset = TensorDataset(X[0]["val"], X[1]["val"], Y["val"])
    testset = TensorDataset(X[0]["test"], X[1]["test"], Y["test"])

    trainloader = DataLoader(trainset, batch_size=cfg.hyp.train_batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=cfg.hyp.test_batch_size, shuffle=False, drop_last=False)
    testloader = DataLoader(testset, batch_size=cfg.hyp.test_batch_size, shuffle=False, drop_last=False)

    loaders = {"train": trainloader, "val": valloader, "test": testloader}
    return loaders, unique_categories, n_numerical, n_classes


def get_model(model, num_numerical, unique_categories, num_outputs, d_embedding, model_params):
    model = model.lower()
    net = getattr(models, model)(num_numerical, unique_categories, num_outputs, d_embedding, model_params)
    return net


def get_optimizer_for_single_net(optim_args, net, state_dict):
    warmup = ExponentialWarmup if optim_args.warmup_type == "exponential" else LinearWarmup

    all_params = [{"params": [p for n, p in net.named_parameters()]}]
    if optim_args.optimizer.lower() == "sgd":
        optimizer = SGD(all_params, lr=optim_args.lr, weight_decay=optim_args.weight_decay,
                        momentum=optim_args.momentum)
    elif optim_args.optimizer.lower() == "adam":
        optimizer = Adam(all_params, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    elif optim_args.optimizer.lower() == "adamw":
        optimizer = AdamW(all_params, lr=optim_args.lr, weight_decay=optim_args.weight_decay)
    else:
        raise ValueError(f"{ic.format()}: Optimizer choice of {optim_args.optimizer.lower()} not yet implmented. "
                         f"Should be one of ['sgd', 'adam', 'adamw'].")

    if state_dict is not None:
        optimizer.load_state_dict(state_dict)
        warmup_scheduler = warmup(optimizer, warmup_period=0)
    else:
        warmup_scheduler = warmup(optimizer, warmup_period=optim_args.warmup_period)

    if optim_args.lr_decay.lower() == "step":
        lr_scheduler = MultiStepLR(optimizer, milestones=optim_args.lr_schedule,
                                   gamma=optim_args.lr_factor, last_epoch=-1)
    elif optim_args.lr_decay.lower() == "cosine":
        lr_scheduler = CosineAnnealingLR(optimizer, optim_args.epochs, eta_min=0, last_epoch=-1, verbose=False)
    else:
        raise ValueError(f"{ic.format()}: Learning rate decay style {optim_args.lr_decay} not yet implemented.")

    return optimizer, warmup_scheduler, lr_scheduler

def get_criterion(task):
    if task == "multiclass":
        criterion = torch.nn.CrossEntropyLoss()
    elif task == "binclass":
        criterion = torch.nn.BCEWithLogitsLoss()
    elif task == "regression":
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError(f"No loss function implemented for task {task}.")
    return criterion


def load_model_from_checkpoint(model_args, num_numerical, unique_categories, num_outputs, device):
    model = model_args.name
    model_path = model_args.model_path
    d_embedding = model_args.d_embedding
    epoch = 0
    optimizer = None

    net = get_model(model, num_numerical, unique_categories, num_outputs, d_embedding, model_args)
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
    if model_path is not None:
        logging.info(f"Loading model from checkpoint {model_path}...")
        state_dict = torch.load(model_path, map_location=device)
        net.load_state_dict(state_dict["net"])
        epoch = state_dict["epoch"] + 1
        optimizer = state_dict["optimizer"]

    return net, epoch, optimizer
