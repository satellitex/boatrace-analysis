# -*- coding: utf-8 -*-

from trainer import util
from trainer.resource import Resource
import numpy as np
import random


def test_save_load_np():
    resource = Resource()
    data = np.array([random.randint(1, 100) for _ in range(100)] * 20)

    util.save_np(resource.get_test_dir, 'test', data)
    res = util.load_np(resource.get_test_dir, 'test')
    assert (np.all(data == res))
