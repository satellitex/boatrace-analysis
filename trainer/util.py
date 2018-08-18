# -*- coding: utf-8 -*-

import random
import string
import logging
import numpy as np
import os

DTYPE = np.float32
logger = logging.getLogger(__name__)


def save_np(dir, base_name, data):
    path = os.path.join(dir, "{}.npy".format(base_name))
    np.save(path, data)
    logger.info("Save {}".format(path))


def load_np(dir, base_name):
    path = os.path.join(dir, "{}.npy".format(base_name))
    return np.load(path)


def random_str(n=10):
    return ''.join([
        random.choice(string.ascii_letters + string.digits) for _ in range(n)
    ])
