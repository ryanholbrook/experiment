import numpy as np
import os
import random


def set_seed(seed=31415):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # If using TensorFlow
    # tf.random.set_seed(seed)
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
