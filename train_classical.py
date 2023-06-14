import json
import logging
import os
import hydra
import torch
import deep_tabular as dt
import xgboost as xgb
import numpy as np
from collections import OrderedDict
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, mean_squared_error, balanced_accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_classical_model(cfg):
	if cfg.dataset.task == 'regression':
		if cfg.model.name=='xgboost':
			model = xgb.XGBRegressor(**cfg.model, seed = cfg.hyp.seed)
		elif cfg.model.name == "univariate":
			model = SelectKBest(score_func=f_regression, k="all")
		elif cfg.model.name == "lasso":
			model = Lasso(alpha=cfg.model.alpha, random_state=cfg.hyp.seed)
		elif cfg.model.name == 'forest':
			model = RandomForestRegressor(n_estimators=cfg.model.n_estimators,
										  max_depth=cfg.model.max_depth,
										  random_state=cfg.hyp.seed,
										  n_jobs=-1)
		else:
			raise NotImplementedError('Model is not implemented')
	else:
		if cfg.model.name == 'xgboost':
			model = xgb.XGBClassifier(**cfg.model, seed = cfg.hyp.seed)
		elif cfg.model.name == "univariate":
			model = SelectKBest(score_func=f_classif, k="all")
		elif cfg.model.name == "lasso":
			model = LogisticRegression(penalty='l1', solver="saga",
									   C=cfg.model.alpha, random_state=cfg.hyp.seed)
		elif cfg.model.name == 'forest':
			model = RandomForestClassifier(n_estimators=cfg.model.n_estimators,
										   max_depth=cfg.model.max_depth,
										   random_state=cfg.hyp.seed,
										   n_jobs=-1)
		else:
			raise NotImplementedError('Model is not implemented')
	return model

def evaluate_classical_model(model, data, task):
	scores = []
	for key in ["test", "val", "train"]:
		X, y = data[key]
		predicted = model.predict(X)
		if task == "multiclass":
			accuracy = float(accuracy_score(y, predicted))
			balanced_accuracy = float(balanced_accuracy_score(y, predicted, adjusted=False))
			balanced_accuracy_adjusted = float(balanced_accuracy_score(y, predicted, adjusted=True))
			score = {"score": accuracy,
					  "accuracy": accuracy,
					  "balanced_accuracy": balanced_accuracy,
					  "balanced_accuracy_adjusted": balanced_accuracy_adjusted}
		elif task == "regression":
			rmse = float(mean_squared_error(y, predicted, squared=False))
			score = {"score": -rmse,
					  "rmse": -rmse}
		elif task == "binclass":
			accuracy = float(accuracy_score(y, predicted))
			score = {"score": accuracy,
					  "accuracy": accuracy}
		scores.append(score)
	return scores

def prepare_input(x_num, x_cat, model_name, selected_features=None):
	x = []
	if x_num is not None:
		if selected_features is not None:
			x_num = x_num[:, selected_features]
		x.append(x_num)
	if x_cat is not None and x_cat.shape[1]>0:
		if model_name == 'xgboost':
			for i in range(x_cat.shape[1]):
				x.append(x_cat[:, i])
				x.append(torch.nn.functional.one_hot(x_cat[:,i]))
		else:
			raise NotImplementedError('datasets with categorical features are not supported yet')
	x = torch.cat(x, dim=-1)
	return x.numpy()


@hydra.main(config_path="config", config_name="train_model")
def main(cfg: DictConfig):
	log = logging.getLogger()
	log.info("\n_________________________________________________\n")
	log.info("train_classical.py main() running.")
	log.info(OmegaConf.to_yaml(cfg))
	if cfg.hyp.save_period < 0:
		cfg.hyp.save_period = 1e8
	torch.manual_seed(cfg.hyp.seed)
	####################################################
	#  Dataset and Model
	data, unique_categories, n_numerical, n_classes = dt.utils.get_dataloaders(cfg, return_raw=True)

	if cfg.mode == "downstream" and cfg.importances_path:
		importances = np.load(cfg.importances_path)
		selected_features = np.argsort(importances)[:cfg.topk]
	else:
		selected_features = None

	X_train, y_train = prepare_input(data["train"][0], data["train"][1], cfg.model.name, selected_features), data["train"][2].numpy()
	X_val, y_val = prepare_input(data["val"][0], data["val"][1], cfg.model.name, selected_features), data["val"][2].numpy()
	X_test, y_test = prepare_input(data["test"][0], data["test"][1], cfg.model.name, selected_features), data["test"][2].numpy()
	data = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}

	####################################################
	#  Train and Evaluate
	log.info(f"Loading {cfg.model.name} model...")
	model = get_classical_model(cfg)
	if cfg.model.name=='xgboost':
		model.fit(X_train, y_train, cfg.hyp.fit, eval_set = [(X_val, y_val)])
	else:
		model.fit(X_train, y_train)

	log.info("Running Final Evaluation...")
	if cfg.model.name=='univariate':
		stats = OrderedDict([])
	else:
		test_stats, val_stats, train_stats = evaluate_classical_model(model,
															   data,
															   cfg.dataset.task)

		log.info(f"Training accuracy: {json.dumps(train_stats, indent=4)}")
		log.info(f"Val accuracy: {json.dumps(val_stats, indent=4)}")
		log.info(f"Test accuracy: {json.dumps(test_stats, indent=4)}")

		stats = OrderedDict([("dataset", cfg.dataset.name),
							 ("model_name", cfg.model.name),
							 ("test_stats", test_stats),
							 ("train_stats", train_stats),
							 ("val_stats", val_stats)])

	if cfg.mode=='downstream':
		with open(os.path.join("stats.json"), "w") as fp:
			json.dump(stats, fp, indent=4)
		log.info(json.dumps(stats, indent=4))

	####################################################
	#  Feature Selection
	if cfg.mode == 'feature_selection':
		if cfg.model.name == 'xgboost':
			importances = model.feature_importances_
		elif cfg.model.name == 'univariate':
			importances = np.abs(model.scores_)
		elif cfg.model.name == 'lasso':
			importances = np.abs(model.coef_)
		elif cfg.model.name == 'forest':
			importances = model.feature_importances_
		else:
			raise NotImplementedError('Model is not implemented')

		torch.save(torch.tensor(importances), f'./feature_importances.pt')
		return stats, torch.tensor(importances)
	else:
		return stats

if __name__ == "__main__":
	main()
