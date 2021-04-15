import ml_collections
from absl import flags
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import placeholder

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor


# Configure model
FLAGS = flags.FLAGS
config_model = ml_collections.ConfigDict()
config_model.learner = "ridge"
config_flags.DEFINE_config_dict("model", config_model)


def load_model():
    models = {
        'knn': KNeighborsRegressor,
        'rf': RandomForestRegressor,
        'ridge': Ridge,
        'tree': DecisionTreeRegressor,
    }
    model = models[FLAGS.model.learner]()
    return model
