# -*- coding: utf-8 -*-

from trainer.resource import Resource
from trainer import config
from trainer.data_processor import JsonDataProcessor,\
    GreedyJsonDataProcessor, HalfJsonDataProcessor
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

    def _convert_json_to_label_ndarray(self, json_data_list):
        return np.array([np.array([data['ans']])
                         for data in json_data_list]).astype(np.float32)

    def _normalize(self, data):
        return ch.functions.normalize(data)


class TestConfig(config.Config):
    PREPARED_DIR = "test_prepared_dir"


def test_json_data_processor():
    resource = Resource(config=TestConfig)

    data_processor = JsonTestDataProcessor(name='test_json', resource=resource)
    data_processor.prepare(force_prepare=True, normalize_adj_flag=False)

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


def test_load_json():
    data_processor = GreedyJsonDataProcessor(name='test_greedy')
    json_list = data_processor._load_json()

    print(json_list[0])
    print(json_list[10])
    assert (len(json_list) == 11309)


def test_convert_to_label_ndarray():
    test_expected_label = np.array([0, 0, 0, 0, 1])
    data_processor = GreedyJsonDataProcessor()
    json_list = data_processor._load_json()
    test_actual_label = \
        data_processor._convert_json_to_label_ndarray(json_list)[:5]
    print("Actual_label : {}".format(test_actual_label))
    print("Expection_label : {}".format(test_expected_label))
    assert (np.all(test_actual_label == test_expected_label))
    assert (len(json_list) == 11309)


def test_convert_one_hot():
    data_processor = GreedyJsonDataProcessor()

    labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    for a in ['a', 'b', 'f', 'g']:
        one_hots = data_processor._convert_one_hot(a, labels)
        assert (len(one_hots) == 7)
        assert (one_hots[labels.index(a)] == 1)
        assert (np.sum(one_hots) == 1)


def test_convert_before_to_input():
    data_processor = GreedyJsonDataProcessor()

    before_array = data_processor._convert_before_to_input({
        "temp":
        "17.0",
        "weather":
        "晴",
        "windspeed":
        "7",
        "wind":
        "5",
        "watertemp":
        "22.0",
        "wave":
        "7"
    })
    assert (before_array.shape == (26, ))
    assert (np.all(before_array == [
        17., 1., 0., 0., 0., 0., 7., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 22., 7.
    ]))


def test_convert_member_to_input():
    data_processor = GreedyJsonDataProcessor()

    member_array = data_processor._convert_member_to_input({
        "waku":
        "1",
        "number":
        "3947",
        "level":
        "B1",
        "name":
        "寺本昇平",
        "branch":
        "群馬",
        "hometown":
        "神奈川",
        "age":
        "42",
        "weight":
        "49.5",
        "F":
        "0",
        "L":
        "0",
        "ST":
        "0.17",
        "nation_first":
        "4.46",
        "nation_second":
        "16.67",
        "nation_third":
        "37.50",
        "here_first":
        "5.90",
        "here_second":
        "37.50",
        "here_third":
        "63.75",
        "motor_no":
        "39",
        "motor_second":
        "33.96",
        "motor_third":
        "56.60",
        "boat_no":
        "26",
        "boat_second":
        "25.00",
        "boat_third":
        "38.64"
    })

    assert (member_array.shape == (25, ))
    assert (np.all(member_array == np.array([
        1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 42., 49.5, 0., 0., 0.17, 4.46,
        16.67, 37.5, 5.9, 37.5, 63.75, 33.96, 56.6, 25., 38.64
    ]).astype(np.float32)))


def test_convert_json_to_input():
    data_processor = GreedyJsonDataProcessor()
    json_list = data_processor._load_json()
    inp = data_processor._convert_json_to_input_ndarray(json_list)
    # TODO 実際は members が 6人なのと人のデータを入れて変わる
    assert (inp.shape == (11309, 151))


def test_half_json_load():
    data_processor = HalfJsonDataProcessor()
    json_list = data_processor._load_json()

    inp = data_processor._convert_json_to_input_ndarray(json_list)
    out = data_processor._convert_json_to_label_ndarray(json_list)

    assert(inp.dtype == np.float32)
    assert(out.dtype == np.int32)
    assert (np.sum(out == 0) / 3500 > 0.4)
