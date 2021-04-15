import pandas as pd
from pathlib import Path
from joblib import dump, load

import wandb

import ml_collections
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import placeholder

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


# Configure experiment runner
FLAGS = flags.FLAGS
flags.DEFINE_bool('debug', False, "Show debugging information.")
flags.DEFINE_bool('log', False, "Log this experiment to wandb.")

# Configure experiment tracking
config_wandb = ml_collections.ConfigDict()
config_wandb.project = "my-project"
config_wandb.job_type = placeholder(str)
config_wandb.notes = placeholder(str)
config_flags.DEFINE_config_dict("wandb", config_wandb)

# Configure data
config_data = ml_collections.ConfigDict()
config_data.path = "../data/0_raw/concrete.csv"
config_data.frac = 1.0
config_data.test_size = 0.2
config_flags.DEFINE_config_dict("data", config_data)

# Configure model
config_model = ml_collections.ConfigDict()
config_model.learner = "ridge"
config_model.alpha = 1.0
config_flags.DEFINE_config_dict("model", config_model)


def load_data(path, frac=1.0):
    X = pd.read_csv(path)
    X = X.sample(frac=frac).reset_index(drop=True)
    y = X.pop("CompressiveStrength")
    return X, y


def main(_):
    if FLAGS.log:
        wandb.init(config=FLAGS, **FLAGS.wandb)

    # Pipeline
    ## Seed everything

    ## Prepare data
    X, y = load_data(
        path=FLAGS.data.path,
        frac=FLAGS.data.frac,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FLAGS.data.test_size, 
    )

    ## Train model
    model = Ridge(alpha=FLAGS.model.alpha)
    model.fit(X_train, y_train)

    ## Evaluate
    y_fit = model.predict(X_train)
    residual_rmse = mean_squared_error(y_train, y_fit, squared=False)
    y_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Log
    if FLAGS.log:
        # Log model
        model_artifact = wandb.Artifact('ridge-model', type='model')
        model_path = Path('../artifacts/models/ridge.joblib')
        dump(model, model_path, compress=3)
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

        # Log eval
        wandb.log({'residual_rmse': residual_rmse, 'test_rmse': test_rmse})
    else:
        print(f"Residual RMSE: {residual_rmse}\t Test RMSE: {test_rmse}")


if __name__ == "__main__":
    app.run(main)
