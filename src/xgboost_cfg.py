from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()
    config.constructor = "xgboost.XGBRegressor"
    config.hparams = ConfigDict({
        'tree_method': "gpu_hist",
        'booster': "gbtree",
        'n_estimators': 250,
        'learning_rate': 1e-2,
        'max_depth': 5,
        'reg_alpha': 1.0,
        'reg_lambda': 1.0,
        'min_child_weight': 0.0,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'num_parallel_tree': 1,
    })
    return config
