import train_deep_model, train_classical
import hydra
import sys
import deep_tabular as dt
import os
import copy
from omegaconf import DictConfig, OmegaConf
import json
import numpy as np

def load_parameters(model, hyp, hypers_dict, mode=''):
    if mode == '_upstream':
        pass
    elif mode == '_downstream':
        pass
    elif mode == '':
        pass
    else:
        raise ValueError('invalid value for mode')

    if model=='ft_transformer' or model=='ft_transformer_attention_map':
        model_params = {
            'd_embedding': hypers_dict[f"d_embedding{mode}"],
            'n_layers': hypers_dict[f"n_layers{mode}"],
            'd_ffn_factor': hypers_dict[f"d_ffn_factor{mode}"],
            'attention_dropout': hypers_dict[f"attention_dropout{mode}"],
            'ffn_dropout': hypers_dict[f"ffn_dropout{mode}"],
            'residual_dropout': hypers_dict[f"residual_dropout{mode}"] if hypers_dict[
                                                                     f"optional_residual_dropout{mode}"] is True else 0.0,
        }
        training_params = {
            'lr':  hypers_dict[f"lr{mode}"],
            'weight_decay':  hypers_dict[f"weight_decay{mode}"],
            }

    if model == 'resnet':
        model_params = {
            'd_embedding': hypers_dict[f"d_embedding{mode}"],
            'd_hidden_factor': hypers_dict[f"hidden_factor{mode}"],
            'n_layers': hypers_dict[f"n_layers{mode}"],
            'hidden_dropout': hypers_dict[f"hidden_dropout{mode}"],
            'residual_dropout': hypers_dict[f"residual_dropout{mode}"] if hypers_dict[
                                                                       f"optional_residual_dropout{mode}"] is True else 0.0,
        }
        training_params = {
            'lr':  hypers_dict[f"lr{mode}"],
            'weight_decay':  hypers_dict[f"weight_decay{mode}"] if hypers_dict[f"optional_weight_decay{mode}"] is True else 0.0,
            }

    if model == 'mlp':
        if not hyp.regularization == 'lasso':
            n_layers = hypers_dict[f"n_layers{mode}"]
            d_first = [hypers_dict[f"d_first{mode}"]] if n_layers else []
            d_middle = ([hypers_dict[f"d_middle{mode}"]] * (n_layers - 2) if n_layers > 2 else [])
            d_last = [hypers_dict[f"d_last{mode}"]] if n_layers > 1 else []
            layers = d_first + d_middle + d_last

            model_params = {
                'd_embedding':  hypers_dict[f"d_embedding{mode}"],
                'd_layers': layers,
                'dropout': hypers_dict[f"dropout{mode}"] if hypers_dict[f"optional_dropout{mode}"] is True else 0.0,
                }
        training_params = {
            'lr':  hypers_dict[f"lr{mode}"],
            'weight_decay':  hypers_dict[f"weight_decay{mode}"] if hypers_dict[f"optional_weight_decay{mode}"] is True else 0.0,
            }

    if model=='xgboost':

        model_params={
            'max_depth': hypers_dict[f'max_depth{mode}'],
            'min_child_weight': hypers_dict[f'min_child_weight{mode}'],
            'subsample': hypers_dict[f'subsample{mode}'],
            'learning_rate': hypers_dict[f'learning_rate{mode}'],
            'colsample_bytree': hypers_dict[f'colsample_bytree{mode}'],
            'gamma': hypers_dict[f'gamma{mode}'],
            'lambda': hypers_dict[f'lambda{mode}'],
        }
        training_params = {}

    if model == 'univariate':
        model_params = {}
        training_params = {}
    if model == 'lasso':
        model_params = {}
        model_params['alpha'] = hypers_dict[f'alpha{mode}']
        training_params = {}
    if model=='forest':
        model_params = {'n_estimators': hypers_dict[f'n_estimators{mode}'],
                        'max_depth': hypers_dict[f'max_depth{mode}']}
        training_params = {}

    if model in ['mlp', 'ft_transformer']:
        if hyp.regularization == 'deep_lasso' or hyp.regularization=='first_lasso':
            training_params['reg_weight'] = hypers_dict[f'reg_weight{mode}']
        if hyp.regularization=='lasso':
            training_params['reg_weight'] = hypers_dict[f'reg_weight{mode}']

    return model_params, training_params


def run_experiment(cfg: DictConfig, hypers_dict):

    config = copy.deepcopy(cfg) # create config for train_model with suggested parameters
    config.mode = 'feature_selection'

    model_params, training_params = load_parameters(cfg.model.name, cfg.hyp, hypers_dict,  mode='_upstream')
    for par, value in model_params.items():
        config.model[par] = value
    for par, value in training_params.items():
        config.hyp[par] = value

    # feature selection part
    if config.model.name in ['xgboost', 'univariate', 'lasso', 'forest']:
        _, importances = train_classical.main(config)
    else:
        _, importances = train_deep_model.main(config)

    # create config for training downstream model with suggested parameters
    config_dwnst = copy.deepcopy(cfg)
    config_dwnst.model = config_dwnst.model_downstream
    config_dwnst.hyp = config_dwnst.hyp_downstream
    config_dwnst.hyp.seed = config.hyp.seed

    model_params_dwnst, training_params_dwnst = load_parameters(cfg.model_downstream.name, cfg.hyp_downstream, hypers_dict, mode='_downstream')


    for par, value in model_params_dwnst.items():
        config_dwnst.model[par] = value
    for par, value in training_params_dwnst.items():
        config_dwnst.hyp[par] = value

    config_dwnst.mode = 'downstream'
    if config_dwnst.model.name not in ['xgboost', 'univariate', 'lasso', 'forest']:
        config_dwnst.hyp.regularization = None
        config_dwnst.hyp.reg_weight = 0
        stats = train_deep_model.main(config_dwnst)
    else:
        stats = train_classical.main(config_dwnst)

    return stats, config, config_dwnst, importances


@hydra.main(config_path="config", config_name="run_full_pipeline_config")
def main(cfg):
    hypers_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), os.pardir)), 'best_config.json')

    with open(hypers_path) as json_file:
        hypers_dict = json.load(json_file)

    stats, config, config_dwnst, importances = run_experiment(cfg, hypers_dict)

    with open(os.path.join("final_stats.json"), "w") as fp:
        json.dump(stats, fp, indent=4)
    with open("config_downstream.yaml", "w") as f:
        OmegaConf.save(config_dwnst, f)
    with open("config_upstream.yaml", "w") as f:
        OmegaConf.save(config, f)
    np.savetxt('importances.out', importances.numpy(), delimiter=',')

if __name__ == "__main__":
    main()



