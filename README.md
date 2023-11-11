# tabular-feature-selection
Repository for feature selection with deep tabular models. This repository contains code for the paper [*"A Performance-Driven Benchmark for Feature
Selection in Tabular Deep Learning"*](https://openreview.net/forum?id=v4PMCdSaAT).
## Data 
Our benchmark for feature selection builds upon the datasets from two papers: 
1. *Revisiting Deep Learning Models for Tabular Data [1]* 
2. *On Embeddings for Numerical Features in Tabular Deep Learning [2]*

Please, follow the instructions [here](https://github.com/Yura52/tabular-dl-revisiting-models#33-data) and [here](https://github.com/Yura52/tabular-dl-num-embeddings#data) to download the datasets and put them in folder ```/data```. 

[1] Gorishniy, Y., Rubachev, I., Khrulkov, V. and Babenko, A., 2021. Revisiting deep learning models for tabular data. Advances in Neural Information Processing Systems, 34, pp.18932-18943.

[2] Gorishniy, Y., Rubachev, I. and Babenko, A., 2022. On embeddings for numerical features in tabular deep learning. Advances in Neural Information Processing Systems, 35, pp.24991-25004.


## Requirements 
We include the environment requirements in ```requirements.txt```. 

## Overview of the Repository 

- ```config``` contains Hydra configs for main scripts, for model parameters, training parameters and datasets. 
- ```deep tabular``` 

    * ```utils``` contains util files for loading and preprocessing data, training and testing loops, feature importance calculation
    * ```models``` contains implementations of deep tabular models

- ```launch``` contains examples of training jobs
- ```train_deep_model.py``` uses ```config/train_model.yaml``` config for training a specified deep tabular model (MLP or FT-Transformer) on a specified dataset (see arguments in ```config/train_model.yaml```)
- ```train_classical.py``` uses ```config/train_model.yaml``` config for training a specified classical feature selection model (Random Forest, XGBoost, Linear/Logistic Regression or Univariate Statistical test)
- ```tune_baseline.py``` uses ```config/tune_config.yaml``` config for tuning the hyperparameters of a model on a specified dataset (see arguments in ```config/train_model.yaml```)
- ```tune_full_pipeline.py``` uses ```config/tune_full_pipeline_config.yaml``` config for tuning the hyperparameters of upstream feature selection model and downstream model simultaneously with respect to the downstream performance (see arguments in ```config/tune_full_pipeline_config.yaml```)

## Benchmark 
Utils for constructing the benchmark are contained in ```deep tabular/utils/data_tools```. In particular, ```get_data_locally``` function reads the data from ```data``` folder using the ```name``` of the dataset, and adds extraneous features to the dataset.  
- ```add_noise``` argument controls the type of extraneous features and can be ```random_feats```, ```corrupted_feats```, or ```secondorder_feats```
- ```noise_percent``` argument controls the proportion of added extraneous features. 

## Limitations 
Currently, the code in this repo works for datasets with numerical features only. Using our code for datasets with categorical features will result in an error. 

## Deep Lasso 
We include the code for Deep Lasso regularizer in ```deep_lasso.py``` file. The file also contains the code for selecting important features using Deep Lasso. 
## How to use the code
### Training downstrem deep tabular models
To train a deep tabular model (either MLP or FT-Transformer) on a dataset with extraneous features, please use the ```train_deep_model.py``` script. For example, to train MLP on California dataset with extraneous second-order features, constituting 50% of all features, run 

```python3 train_deep_model.py mode=downstream dataset=california_housing name=no_fs model=mlp hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5```

Results of this job will be saved in ```stats.json``` file in the folder specified in ```config/train_model.yaml``` config file.

### Running Feature Selection 
To compute feature selection with classical algorithms (Lasso, XGBoost, Random Forest, etc), use ```python3 train_classical.py``` script and specify ```mode=feature_selection```. For example, to compute feature importance with xgboost model on california housing dataset with 50% extraneous second-order features, run

```python3 train_classical.py mode=feature_selection dataset=california_housing name=fs_xgboost model=xgboost hyp=hyp_for_xgboost dataset.add_noise=secondorder_feats dataset.noise_percent=0.5```

To compute feature selection with deep learning based algorithms (Deep Lasso, First-Layer Lasso, Attention Map Importance), use ```python3 train_deep_model.py``` and specify ```mode=feature_selection```. For example, to compute feature importance with Deep Lasso based on FT-Transformer model on california housing dataset with 50% extraneous second-order features, run

```python3 train_deep_model.py mode=feature_selection dataset=california_housing name=fs_deep_lasso model=ft_transformer hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5 hyp.regularization=deep_lasso hyp.reg_weight=0.1```
- ```hyp.regularization=deep_lasso``` for Deep Lasso feature importance 
- ```hyp.regularization=first_lasso``` for First-Layer Lasso feature importance 
-  ```model=ft_transformer_attention``` for Attention Map feature importance

 Computed feature importances will be saved in ```feature_importances.pt```

### Training downstream deep tabular models on selected features 

To Use pre-computed feature importances, please specify the ```importance_path=feature_importances.pt``` and indicate the proportions of the most important features to incude in the dataset ```topk=0.5``` 

```python3 train_deep_model.py mode=downstream dataset=california_housing name=ft_transformer_fs_xgboost model=ft_transformer hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5 importance_path=feature_importances.pt topk=0.5```

### Hyperparameter tuning for the downstream model 

To tune the hyperparameters of the downstream deep tabular models, use the ```python3 tune_baseline.py``` script. For example, to tune hyperparameters of FT-Transformer on California Housing dataset with 50% corrupted features and no feature selection:

```python3 tune_baseline.py mode=downstream model=ft_transformer dataset=california_housing name=tune_ft_ch hyp=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5```

This job will save the best performing hyperparameters in ```best_config.json```, results for the best hyperparameters in ```best_stats.json``` and stats from all trials in ```all_stats.json``` and ```trials.csv```.

### Hyperparameter tuning for both feature selector and downstream model
To tune the hyperparameters for feature selection algorithm and downstream model simultaneously, use the ```python3 tune_full_pipeline.py``` script. For example, to tune MLP-based Deep Lasso feature selector, and the downstream MLP model: 

```python3 tune_full_pipeline.py model=mlp model_downstream=mlp dataset=california_housing name=tune_ft_ch_full hyp=hyp_for_neural_network hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 hyp.regularization='deep_lasso' topk=0.5```

This job will save the best performing hyperparameters of both upstream feature selection and downstream models as well as performance stats of their combination. 
More examples can be found in the ```launch``` folder. 

## How to reproduce results in the main tables 

1. First, tune the hyperparameters of both feature selection and downstream models for each fs_method-model-dataset configuration:

```python3 tune_full_pipeline.py model={FS MODEL} model_downstream={DOWNSTREAM MODEL} dataset={DATASET NAME} name={NAME OF EXPERIMENT} hyp={CONFIG FOR FS MODEL} hyp_downstream={CONFIG FOR DOWNSTREAM MODEL} dataset.add_noise={NOISE SETUP} dataset.noise_percent={% OF NOISE IN DATASET} hyp.regularization={FS REGULARIZATION} topk={% OF FEATURES TO SELECT}```

For example for XGBoost feature selection and downstream MLP model run: 

```python3 tune_full_pipeline.py model=xgboost model_downstream=mlp dataset=california_housing name=xgboost_mlp hyp=hyp_for_xgboost hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5```

2. Then, run training job for the best hyperparameters for 10 different seeds:

```python3 run_full_pipeline.py --multirun model=xgboost model_downstream=mlp dataset=california_housing name=xgboost_mlp hyp=hyp_for_xgboost hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5 hyp.seed=0,1,2,3,4,5,6,7,8,9```

This script loads the ```best_config.json``` file and runs feature selection and downstream models with the specified hyperparameters for 10 seeds. Results are saved in ```final_stats.json``` files in folders corresponding to the seed number in the same directory. 

Please, find more examples in ```launch/feature_selection_California_Housing.sh```