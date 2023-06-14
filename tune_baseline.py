import train_deep_model, train_classical
import hydra
import optuna
import sys
import deep_tabular as dt
import os
import copy
from omegaconf import DictConfig, OmegaConf
import plotly.io as pio
import json


def sample_value_with_default(trial, name, distr, min, max, default):
    # chooses suggested or default value with 50/50 chance
    if distr == 'uniform':
        value_suggested = trial.suggest_uniform(name, min, max)
    elif distr == 'loguniform':
        value_suggested = trial.suggest_loguniform(name, min, max)
    value = value_suggested if trial.suggest_categorical(f'optional_{name}', [False, True]) else default
    return value


def get_parameters(model, trial, hyp, mode = ''):
    """mode is either upstream or downstream to sample the corresponding model params"""
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
            'd_embedding':  trial.suggest_int(f'd_embedding{mode}', 64, 512, step=8), #using n_heads = 8 by default
            'n_layers': trial.suggest_int(f'n_layers{mode}', 1, 4),
            'd_ffn_factor': trial.suggest_uniform(f'd_ffn_factor{mode}', 2/3, 8/3),
            'attention_dropout': trial.suggest_uniform(f'attention_dropout{mode}', 0.0, 0.5),
            'ffn_dropout' : trial.suggest_uniform(f'ffn_dropout{mode}', 0.0, 0.5),
            'residual_dropout': sample_value_with_default(trial, f'residual_dropout{mode}', 'uniform', 0.0, 0.2, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform(f'lr{mode}', 1e-5, 1e-3),
            'weight_decay':  trial.suggest_loguniform(f'weight_decay{mode}', 1e-6, 1e-3),
            }

    if model=='resnet':
        model_params = {
            'd_embedding':  trial.suggest_int(f'd_embedding{mode}', 32, 512, step=8),
            'd_hidden_factor': trial.suggest_uniform(f'd_hidden_factor{mode}', 1.0, 4.0),
            'n_layers': trial.suggest_int(f'n_layers{mode}', 1, 8,),
            'hidden_dropout': trial.suggest_uniform(f'residual_dropout{mode}', 0.0, 0.5),
            'residual_dropout': sample_value_with_default(trial, f'residual_dropout{mode}', 'uniform', 0.0, 0.5, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform(f'lr{mode}', 1e-5, 1e-3),
            'weight_decay':  sample_value_with_default(trial, f'weight_decay{mode}', 'loguniform', 1e-6, 1e-3, 0.0),
            }

    if model=='mlp':
        n_layers = trial.suggest_int(f'n_layers{mode}', 1, 8)
        suggest_dim = lambda name: trial.suggest_int(name, 1, 512)
        d_first = [suggest_dim(f'd_first{mode}')] if n_layers else []
        d_middle = ([suggest_dim(f'd_middle{mode}')] * (n_layers - 2) if n_layers > 2 else [])
        d_last = [suggest_dim(f'd_last{mode}')] if n_layers > 1 else []
        layers = d_first + d_middle + d_last

        model_params = {
            'd_embedding':  trial.suggest_int(f'd_embedding{mode}', 32, 512, step=8),
            'd_layers': layers,
            'dropout': sample_value_with_default(trial, f'dropout{mode}', 'uniform', 0.0, 0.5, 0.0),
            }
        training_params = {
            'lr':  trial.suggest_loguniform(f'lr{mode}', 1e-5, 1e-2),
            'weight_decay':  sample_value_with_default(trial, f'weight_decay{mode}', 'loguniform', 1e-6, 1e-3, 0.0),
            }

    if model=='xgboost':
        model_params={
            'max_depth': trial.suggest_int(f'max_depth{mode}', 3, 10),
            'min_child_weight': trial.suggest_loguniform(f'min_child_weight{mode}', 1e-08, 100000.0),
            'subsample': trial.suggest_uniform(f'subsample{mode}', 0.5, 1.0),
            'learning_rate': trial.suggest_loguniform(f'learning_rate{mode}', 1e-05, 1),
            'colsample_bytree': trial.suggest_uniform(f'colsample_bytree{mode}', 0.5, 1),
            'gamma': sample_value_with_default(trial, f'gamma{mode}', 'loguniform', 0.001, 100.0, 0.0),
            'lambda': sample_value_with_default(trial, f'lambda{mode}', 'loguniform', 0.1, 10.0, 0.0),
        }
        training_params = {}

    if hyp.regularization=='deep_lasso' or hyp.regularization=='first_lasso':
        training_params['reg_weight'] = trial.suggest_loguniform(f'reg_weight{mode}', 1e-2, 5e-1)
    if hyp.regularization=='lasso':
        training_params['reg_weight'] = trial.suggest_loguniform(f'reg_weight{mode}', 1e-3, 5e-1)
    if model == 'univariate':
        model_params = {}
        training_params = {}
    if model == 'lasso':
        model_params = {'alpha': trial.suggest_uniform(f'alpha{mode}', 1e-4, 1.0)}
        training_params = {}
    if model=='forest':
        model_params = {'n_estimators': trial.suggest_int(f'n_estimators{mode}', 10, 2000),
                        'max_depth': trial.suggest_int(f'max_depth{mode}', 3, 10)}
        training_params = {}

    return model_params, training_params


def objective(trial, cfg: DictConfig, trial_configs, trial_stats):
    config_dwnst = copy.deepcopy(cfg)

    model_params_dwnst, training_params_dwnst = get_parameters(cfg.model.name, trial, cfg.hyp,
                                                               mode='_downstream')

    for par, value in model_params_dwnst.items():
        config_dwnst.model[par] = value
    for par, value in training_params_dwnst.items():
        config_dwnst.hyp[par] = value

    config_dwnst.mode = 'downstream'
    if config_dwnst.model.name not in ['xgboost', 'univariate', 'lasso', 'forest']:
        stats = train_deep_model.main(config_dwnst)
    else:
        stats = train_classical.main(config_dwnst)

    trial_configs.append(config_dwnst)
    trial_stats.append(stats)

    return stats['val_stats']['score']


@hydra.main(config_path="config", config_name="tune_config")
def main(cfg):
    optuna_seed = 10
    n_optuna_trials = 100

    trial_stats = []
    trial_configs = []
    sampler = optuna.samplers.TPESampler(seed=optuna_seed)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=optuna.pruners.MedianPruner())
    func = lambda trial: objective(trial, cfg, trial_configs, trial_stats)
    study.optimize(func, n_trials=n_optuna_trials)

    best_trial = study.best_trial

    for key, value in best_trial.params.items():
        print("{}: {}".format(key, value))

    best_stats = trial_stats[best_trial.number]
    df_trials = study.trials_dataframe(attrs=('number', 'value', 'params', 'state', 'datetime_start', 'datetime_complete'))
    df_trials.to_csv('trials.csv')
    with open(os.path.join("best_stats.json"), "w") as fp:
        json.dump(best_stats, fp, indent=4)
    with open(os.path.join("best_config.json"), "w") as fp:
        json.dump(best_trial.params, fp, indent=4)
    with open(os.path.join("all_stats.json"), "w") as fp:
        json.dump(trial_stats, fp, indent=4)
    fig = optuna.visualization.plot_optimization_history(study)
    pio.write_image(fig, "optimization_history.png")

if __name__ == "__main__":
    main()
