from ml_collections import ConfigDict


def get_config(config):
    model_configs = {
        'knn': ConfigDict({
            'constructor': 'sklearn.neighbors.KNeighborsRegressor',
            'hparams': ConfigDict({
                'n_neighbors': 5,
            }),
        }),
        'ridge': ConfigDict({
            'constructor': 'sklearn.linear_model.Ridge',
            'hparams': ConfigDict({
                'alpha': 1.0,
                'normalize': False,
            }),
        }),
    }
    return model_configs[config]

