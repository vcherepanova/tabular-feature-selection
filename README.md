# tabular-feature-selection
Repository for feature selection with tabular models. This repository contains code for the paper *"A Performance-Driven Benchmark for Feature
Selection in Tabular Deep Learning"*.
## Data 
Our benchmark for feature selection builds upon the datasets from two benchmark papers: 
1. *Revisiting Deep Learning Models for Tabular Data [1]* 
2. *On Embeddings for Numerical Features in Tabular Deep Learning [2]*

Please, follow the instructions [here](https://github.com/Yura52/tabular-dl-revisiting-models#33-data) and [here](https://github.com/Yura52/tabular-dl-num-embeddings#data) to download the datasets and put them in folder /data. 

[1] Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A., 2021. Revisiting deep learning models for tabular data. Advances in Neural Information Processing Systems, 34, pp.18932-18943.

[2] Gorishniy, Y., Rubachev, I. and Babenko, A., 2022. On embeddings for numerical features in tabular deep learning. Advances in Neural Information Processing Systems, 35, pp.24991-25004.

## Requirements 
We include the environment requirements in requirements.txt. 

## Overview of the Repository 

- ```config``` contains Hydra configs for main scripts, for model parameters, training parameters and datasets. 
- ```deep tabular``` 

    * ```utils``` contains util files for loading and preprocessing data, training and testing loops, feature importance calculation
    * ```models``` contains implementations of deep tabular models

- ```launch``` contains examples of job runs 
- ```train_deep_model.py``` uses ```config/train_model.yaml``` config for training a specified deep tabular model (MLP or FT-Transformer) on a specified dataset (see arguments in ```config/train_model.yaml```)
- ```train_classical.py``` uses ```config/train_model.yaml``` config for training a specified classical model (Random Forest, XGBoost, Linear/Logistic Regression or Univariate Statistical test)
- ```tune_baseline.py``` uses ```config/tune_config.yaml``` config for tuning the hyperparameters of a model on a specified dataset (see arguments in ```config/train_model.yaml```)
- ```tune_baseline.py``` uses ```config/tune_full_pipeline_config.yaml``` config for tuning the hyperparameters of upstream feature selection model and downstream model simultaneously with respect to the downstream performance (see arguments in ```config/tune_full_pipeline_config.yaml```)

## How to use the code
### No Hyperparameter tuning
1. Training MLP on California Housing dataset with 50% second-order features with default hyperparameters

```python3 train_deep_model.py mode=downstream dataset=california_housing name=no_fs model=mlp hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5```

Results of this job will be saved in stats.json file in the folder specified in ```config/train_model.yaml``` config file.

2. Compute and save feature importance with XGBoost on California Housing dataset with 50% second-order features with default hyperparameters

```python3 train_classical.py mode=feature_selection dataset=california_housing name=fs_xgboost model=xgboost hyp=hyp_for_xgboost dataset.add_noise=secondorder_feats dataset.noise_percent=0.5```

Results of this job will be saved in stats.json file in the folder specified in ```config/train_model.yaml``` config file. Computed feature importances are saved in feature_importances.pt

3. Use pre-computed feature importances to select 50% of all features and run downstream FT-Transformer model on selected features: 

```python3 train_deep_model.py mode=downstream dataset=california_housing name=ft_transformer_fs_xgboost model=ft_transformer hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5 importance_path=feature_importances.pt topk=0.5```

### Hyperparameter tuning

1. Tune the hyperparameters of FT-Transformer model on California Housing dataset with 50% corrupted features:

```python3 tune_baseline.py mode=downstream model=ft_transformer dataset=california_housing name=tune_ft_ch hyp=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5```

This job will save the best performing hyperparameters in best_config.json, results for the best hyperparameters in best_stats.json and stats from all trials in all_stats.json and trials.csv.

2. Tune the hyperparameters for both upstream feature selection model and downstream model simultaneously. The upstream model is MLP with Deep Lasso feature selection, and the downstream model is also MLP. 

```python3 tune_full_pipeline.py model=mlp model_downstream=mlp dataset=california_housing name=tune_ft_ch_full hyp=hyp_for_neural_network hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 hyp.regularization='deep_lasso' topk=0.5```

This job will save the best performing hyperparameters of both upstream feature selection and downstream models as well as performance stats of their combination. 
More examples can be found in the ```launch``` folder. 
