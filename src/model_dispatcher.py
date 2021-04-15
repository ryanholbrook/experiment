from importlib import import_module

from absl import flags
from ml_collections.config_flags import config_flags


# Configure model
FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('model')


def load_model():
    module_path, class_name = FLAGS.model.constructor.rsplit('.', 1)
    module = import_module(module_path)
    learner = getattr(module, class_name)
    return learner(**FLAGS.model.hparams)
