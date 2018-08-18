# -*- coding: utf-8 -*-

from trainer.resource import Resource
from trainer import config
from trainer.data_processor import JsonDataProcessor
import chainer as ch
import numpy as np
import os


class JsonTestDataProcessor(JsonDataProcessor):
    def _load_json(self):
        return [
            {
                'a': 1,
                'b': 2,
                'c': 3,
                'ans': 0,
            },
            {
                'a': 3,
                'b': 2,
                'c': 1,
                'ans': 1,
            },
            {
                'a': 2,
                'b': 3,
                'c': 1,
                'ans': 0,
            },
        ]

    def _convert_json_to_input_ndarray(self, json_data_list):
        return np.array([
            np.array([data['a'], data['b'], data['c']])
            for data in json_data_list
        ]).astype(np.float32)

    def _convert_json_to_output_ndarray(self, json_data_list):
        return np.array([np.array([data['ans']])
                         for data in json_data_list]).astype(np.float32)

    def _normalize(self, data):
        return ch.functions.normalize(data)


class TestConfig(config.Config):
    PREPARED_DIR = "test_prepared_dir"


def test_mssp_graph_data_processor():
    resource = Resource(config=TestConfig)

    data_processor = JsonTestDataProcessor(name='test_json', resource=resource)
    data_processor.prepare(force_prepare=True)

    assert (os.path.isfile(
        os.path.join(resource.get_prepared_dir, 'test_json_in.npy')))
    assert (os.path.isfile(
        os.path.join(resource.get_prepared_dir, 'test_json_out.npy')))

    x_data, y_data = data_processor.get_train_data(n=1)

    assert (x_data.shape == (1, 3))
    assert (y_data.shape == (1, 1))
    assert (np.all(x_data == np.array([[1, 2, 3]])))
    assert (np.all(y_data == np.array([[0]])))

    x_data, y_data = data_processor.get_test_data(n=2)
    assert (x_data.shape == (2, 3))
    assert (y_data.shape == (2, 1))
    assert (np.all(x_data == np.array([[3, 2, 1], [2, 3, 1]])))
    assert (np.all(y_data == np.array([[1], [0]])))
