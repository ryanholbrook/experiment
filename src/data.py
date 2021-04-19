from pathlib import Path

import pandas as pd

import ml_collections
from absl import flags
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import placeholder

from sklearn.model_selection import train_test_split
from definitions import DATA_DIR


# Configure data
FLAGS = flags.FLAGS
config_data = ml_collections.ConfigDict()
config_data.path = DATA_DIR / "0_raw" / "concrete.csv"
config_data.frac = 1.0
config_data.test_size = 0.2
config_flags.DEFINE_config_dict("data", config_data)


def load_data():
    X = pd.read_csv(FLAGS.data.path)
    X = X.sample(frac=FLAGS.data.frac).reset_index(drop=True)
    y = X.pop("CompressiveStrength")
    return X, y


def load_train_test_splits():
    X, y = load_data()
    return train_test_split(X, y, test_size=FLAGS.data.test_size)
