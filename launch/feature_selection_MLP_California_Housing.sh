# Deep Lasso feature selection + downstream MLP on California Housing dataset with 50% corrupted features. 50% features are selected for downstream model.
python3 tune_full_pipeline.py model=mlp model_downstream=mlp dataset=california_housing name=deep_lasso_mlp hyp=hyp_for_neural_network hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 hyp.regularization='deep_lasso' topk=0.5

# First Layer Lasso feature selection
python3 tune_full_pipeline.py model=mlp model_downstream=mlp dataset=california_housing name=1l_lasso_mlp hyp=hyp_for_neural_network hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 hyp.regularization='first_lasso' topk=0.5

# XGBoost feature selection
python3 tune_full_pipeline.py model=xgboost model_downstream=mlp dataset=california_housing name=xgboost_mlp hyp=hyp_for_xgboost hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5

# Random Forest feature selection
python3 tune_full_pipeline.py model=forest model_downstream=mlp dataset=california_housing name=forest_mlp hyp=hyp_for_forest hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5

# Univariate Test feature selection
python3 tune_full_pipeline.py model=univariate model_downstream=mlp dataset=california_housing name=univariate_mlp hyp=hyp_for_univariate hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5

# Attention Map feature selection (slightly different implementation of ft-transformer model) + FT-Transformer downstream model
python3 tune_full_pipeline.py model=ft_transformer_attention_map model_downstream=ft_transformer dataset=california_housing name=am_fttransformer hyp=hyp_for_neural_network hyp_downstream=hyp_for_neural_network dataset.add_noise=corrupted_feats dataset.noise_percent=0.5 topk=0.5
