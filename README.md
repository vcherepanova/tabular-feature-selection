# ✨ Feature Selection for Deep Tabular Models ✨
Welcome to our repository for feature selection with deep tabular models!
This repository contains code for our paper [*"A Performance-Driven Benchmark for Feature
Selection in Tabular Deep Learning"*](https://openreview.net/forum?id=v4PMCdSaAT).

## Overview 
Academic tabular benchmarks often contain small sets of curated features. In contrast, data scientists typically collect as many features as possible into their datasets, and even engineer new features from existing ones. To prevent over-fitting in subsequent downstream modeling, practitioners commonly use automated feature selection methods that identify a reduced subset of informative features. Existing benchmarks for tabular feature selection consider classical downstream models, toy synthetic datasets, or do not evaluate feature selectors on the basis of downstream performance. We construct a challenging feature selection benchmark evaluated on downstream neural networks including transformers, using real datasets and multiple methods for generating extraneous features. We also propose an input-gradient-based analogue of LASSO for neural networks, called Deep Lasso, that outperforms classical feature selection methods on challenging problems such as selecting from corrupted or second-order features.

## Citation 

Please, consider citing our work if you find our benchmark and Deep Lasso methods helpful: 
```
@inproceedings{cherepanova2023performance,
  title={A Performance-Driven Benchmark for Feature Selection in Tabular Deep Learning},
  author={Cherepanova, Valeriia and Levin, Roman and Somepalli, Gowthami and Geiping, Jonas and Bruss, C Bayan and Wilson, Andrew Gordon and Goldstein, Tom and Goldblum, Micah},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```


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

    * ```utils``` contains utilities for loading and preprocessing data, training and testing loops, feature importance calculation
    * ```models``` contains implementations of deep tabular models

- ```launch``` contains examples of training jobs
- ```train_deep_model.py``` uses ```config/train_model.yaml``` config for training a specified deep tabular model (MLP or FT-Transformer) on a specified dataset (see arguments in ```config/train_model.yaml```)
- ```train_classical.py``` uses ```config/train_model.yaml``` config for training a specified classical feature selection model (Random Forest, XGBoost, Linear/Logistic Regression or Univariate Statistical test)
- ```tune_baseline.py``` uses ```config/tune_config.yaml``` config for tuning the hyperparameters of a model on a specified dataset (see arguments in ```config/train_model.yaml```)
- ```tune_full_pipeline.py``` uses ```config/tune_full_pipeline_config.yaml``` config for tuning the hyperparameters of upstream feature selection model and downstream model simultaneously with respect to the downstream performance (see arguments in ```config/tune_full_pipeline_config.yaml```)

## Benchmark 
The benchmark construction utilities are in ```deep tabular/utils/data_tools```,  including the ```get_data_locally``` function for reading and augmenting datasets with extraneous features.  
- ```add_noise``` argument controls the type of extraneous features and can be ```random_feats```, ```corrupted_feats```, or ```secondorder_feats```
- ```noise_percent``` argument controls the proportion of added extraneous features. 

## Limitations 
Note: Our current implementation is tailored for numerical features only; applying it to categorical features may result in errors.

## Deep Lasso 
Find the implementation of our novel Deep Lasso regularizer in ```deep_lasso.py```, along with the feature selection functionality.

## How to use the code
### Training downstrem deep tabular models
To train a deep tabular model, such as an MLP or FT-Transformer, on a dataset containing extraneous features, use the ```train_deep_model.py``` script. For instance, to train an MLP on the California Housing dataset with 50% of the features being extraneous second-order features, execute the following command:

```python3 train_deep_model.py mode=downstream dataset=california_housing name=no_fs model=mlp hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5```

Results of this job will be saved in ```stats.json``` file, located in the directory specified in the ```config/train_model.yaml``` file.

### Running Feature Selection 
For feature selection with classical algorithms (Lasso, XGBoost, Random Forest, etc), use ```python3 train_classical.py``` script and specify ```mode=feature_selection```. For example, to calculate feature importance using the XGBoost model on the California Housing dataset with 50% extraneous second-order features, run:

```python3 train_classical.py mode=feature_selection dataset=california_housing name=fs_xgboost model=xgboost hyp=hyp_for_xgboost dataset.add_noise=secondorder_feats dataset.noise_percent=0.5```

To perform feature selection with deep learning-based algorithms (like Deep Lasso, First-Layer Lasso, Attention Map Importance), use the ```python3 train_deep_model.py``` script and specify ```mode=feature_selection```. For instance, to determine feature importance using Deep Lasso with the FT-Transformer model on the California Housing dataset with 50% extraneous second-order features, execute:

```python3 train_deep_model.py mode=feature_selection dataset=california_housing name=fs_deep_lasso model=ft_transformer hyp=hyp_for_neural_network dataset.add_noise=secondorder_feats dataset.noise_percent=0.5 hyp.regularization=deep_lasso hyp.reg_weight=0.1```
- ```hyp.regularization=deep_lasso``` for Deep Lasso feature importance 
- ```hyp.regularization=first_lasso``` for First-Layer Lasso feature importance 
-  ```model=ft_transformer_attention``` for Attention Map feature importance

 Computed feature importances will be saved in ```feature_importances.pt```

### Training downstream deep tabular models on selected features 

To leverage pre-computed feature importances, specify the path using ```importance_path=feature_importances.pt``` and indicate the proportion of the most significant features to include in the dataset using ```topk``` argument:

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

First, tune the hyperparameters of both feature selection and downstream models for each fs_method-model-dataset configuration:

```python3 tune_full_pipeline.py model={FS MODEL} model_downstream={DOWNSTREAM MODEL} dataset={DATASET NAME} name={NAME OF EXPERIMENT} hyp={CONFIG FOR FS MODEL} hyp_downstream={CONFIG FOR DOWNSTREAM MODEL} dataset.add_noise={NOISE SETUP} dataset.noise_percent={% OF NOISE IN DATASET} hyp.regularization={FS REGULARIZATION} topk={% OF FEATURES TO SELECT}```

For example for XGBoost feature selection and downstream MLP model run: 

```python3 tune_full_pipeline.py model=xgboost model_downstream=mlp dataset=california_housing name=xgboost_mlp hyp=hyp_for_xgboost hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5```

Then, run training job for the best hyperparameters for 10 different seeds:

```python3 run_full_pipeline.py --multirun model=xgboost model_downstream=mlp dataset=california_housing name=xgboost_mlp hyp=hyp_for_xgboost hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5 hyp.seed=0,1,2,3,4,5,6,7,8,9```

This script loads the ```best_config.json``` file and runs feature selection and downstream models with the specified hyperparameters for 10 seeds. Results are saved in ```final_stats.json``` files in folders corresponding to the seed number in the same directory. 

Please, find more examples in ```launch/feature_selection_California_Housing.sh```