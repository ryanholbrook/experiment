from pathlib import Path
from joblib import dump, load

import wandb

import ml_collections
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import placeholder

from sklearn.metrics import mean_squared_error

from data import load_train_test_splits
from definitions import ARTIFACT_DIR
from model_dispatcher import load_model
from utils import set_seed

# Configure experiment runner
FLAGS = flags.FLAGS
flags.DEFINE_bool('debug', False, "Show debugging information.")
flags.DEFINE_bool('log', False, "Log this experiment to wandb.")

# Configure experiment tracking
config_wandb = ml_collections.ConfigDict()
config_wandb.project = "hparam-src"
config_wandb.job_type = placeholder(str)
config_wandb.notes = placeholder(str)
config_flags.DEFINE_config_dict(
    'wandb', config_wandb, "Configuration for W&B experiment tracking.",
)


def main(_):
    if FLAGS.log:
        wandb.init(config=FLAGS, **FLAGS.wandb)

    # Pipeline
    ## Setup
    set_seed()

    ## Data
    X_train, X_test, y_train, y_test = load_train_test_splits()

    ## Train model
    model = load_model()
    model.fit(X_train, y_train)

    ## Evaluate
    y_fit = model.predict(X_train)
    residual_rmse = mean_squared_error(y_train, y_fit, squared=False)
    y_pred = model.predict(X_test)
    test_rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Log
    if FLAGS.log:
        # Log model
        model_artifact = wandb.Artifact('model', type='model')
        model_path = ARTIFACT_DIR / "models" / "model.joblib"
        dump(model, model_path, compress=3)
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)

        # Log eval
        wandb.log({'residual_rmse': residual_rmse, 'test_rmse': test_rmse})
    else:
        print(f"Residual RMSE: {residual_rmse}\t Test RMSE: {test_rmse}")


if __name__ == "__main__":
    app.run(main)
