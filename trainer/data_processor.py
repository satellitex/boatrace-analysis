# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from trainer.resource import Resource
import numpy as np
from trainer import util
import logging
import chainer as ch
import os
import json

DTYPE = np.float32
logger = logging.getLogger(__name__)


class DataProcessor(object, metaclass=ABCMeta):
    def __init__(self, name='prepared', resource=Resource()):
        self.resource = resource
        self.name = name

    @abstractmethod
    def prepare(self, force_prepare=False):
        raise NotImplemented

    @abstractmethod
    def get_train_data(self, n=1):
        raise NotImplemented

    @abstractmethod
    def get_test_data(self, n=1):
        raise NotImplemented

    @property
    def in_name(self):
        return '{}_in'.format(self.name)

    @property
    def out_name(self):
        return '{}_out'.format(self.name)


class JsonDataProcessor(DataProcessor):
    def prepare(self, force_prepare=False, n=100, normalize_adj_flag=False):
        """
        prepare calculation.

        Args:
            force_prepare: it is true, overwrite perepared data.
            n: number of test data
            normalize_adj_flag: bool
                if True, normalized data.
        """

        self.in_array = None
        self.out_array = None
        if force_prepare is False:
            try:
                self.in_array = util.load_np(self.resource.get_prepared_dir,
                                             self.in_name)
                self.out_array = util.load_np(self.resource.get_prepared_dir,
                                              self.out_name)
                logger.info("Success Loading from dir {}".format(
                    self.resource.get_prepared_dir))
                logger.debug(self.in_array[0:2])
                logger.debug(self.out_array[0:2])
                return
            except FileNotFoundError as e:
                logger.info(
                    "File not exist so, new create MSSPGraphData to ndarray")

        json_data_list = self._load_json()
        logger.debug(json_data_list)

        self.in_array = self._convert_json_to_input_ndarray(json_data_list)
        logger.debug(self.in_array)

        self.out_array = self._convert_json_to_output_ndarray(json_data_list)
        logger.debug(self.out_array)

        if normalize_adj_flag:
            self.in_array = self._normalize(self.in_array)
            logger.debug(self.in_array)

        util.save_np(self.resource.get_prepared_dir, self.in_name,
                     self.in_array)
        util.save_np(self.resource.get_prepared_dir, self.out_name,
                     self.out_array)
        logger.info("Success Loading array from GraphData")

    @abstractmethod
    def _load_json(self):
        """
        幾つかの json データをファイルからロードして dict の list で返す。
        Returns: list of dict
            e.g) [{'a': 1, 'b':2}, {'a':3, 'b':5}]
        """
        raise NotImplemented

    @abstractmethod
    def _convert_json_to_input_ndarray(self, json_data_list):
        """
        dict の list を受け取り deepLearning の入力となる ndarray を返す
        Args:
            json_data_list: list of dict
        Returns: ndarray
        """
        raise NotImplemented

    @abstractmethod
    def _convert_json_to_output_ndarray(self, json_data_list):
        """
        dict の list を受け取り deepLearning の出力となる ndarray を返す
        Args:
            json_data_list: list of dict
        Returns: ndarray
        """
        raise NotImplemented

    @abstractmethod
    def _normalize(self, data):
        """
        ndarray を normalize して返す
        Args:
            data: ndarray

        Returns: ndarray
        """

    def get_train_data(self, n=None):
        if n is None:
            n = int(len(self.in_array) / 3)
        self._train_ed = n

        return self.in_array[:n], self.out_array[:n]

    def get_test_data(self, n=None):
        if hasattr(self, '_train_ed'):
            st = self._train_ed
        else:
            st = int(len(self.in_array) / 3)

        if n is None:
            n = int(len(self.in_array) / 3)
        self._test_ed = st + n
        return self.in_array[st:st + n], self.out_array[st:st + n]

    def get_unk_data(self, n=None):
        if hasattr(self, '_test_ed'):
            st = self._test_ed
        else:
            st = int(len(self.in_array) / 3) * 2

        if n is None:
            n = int(len(self.in_array) / 3)
        return self.in_array[st:st + n], self.out_array[st:st + n]


class MockJsonDataProcessor(JsonDataProcessor):
    """
    Mock Json DataProcessor load xor data.
    """

    def _load_json(self):
        return [
            {
                'x': 0,
                'y': 0,
                'xor': 0,
            },
            {
                'x': 1,
                'y': 0,
                'xor': 1,
            },
            {
                'x': 0,
                'y': 1,
                'xor': 1,
            },
            {
                'x': 1,
                'y': 1,
                'xor': 0,
            },
            {
                'x': 0,
                'y': 0,
                'xor': 0,
            },
            {
                'x': 1,
                'y': 0,
                'xor': 1,
            },
            {
                'x': 0,
                'y': 1,
                'xor': 1,
            },
            {
                'x': 1,
                'y': 1,
                'xor': 0,
            },
        ]

    def _convert_json_to_input_ndarray(self, json_data_list):
        return np.array([
            np.array([data['x'], data['y']]) for data in json_data_list
        ]).astype(np.float32)

    def _convert_json_to_output_ndarray(self, json_data_list):
        return np.array([data['xor'] for data in json_data_list]).astype(
            np.int32)

    def _normalize(self, data):
        return ch.functions.normalize(data)


class GreedyJsonDataProcessor(JsonDataProcessor):
    WEATHER_LABELS = ['晴', '曇り', '雨', '雪', '霧']
    WIND_LABELS = [
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
        '14', '15', '16', '17'
    ]

    def _load_json(self):
        path_list = [
            name for name in os.listdir(self.resource.get_original_dir)
        ]
        return [
            self._load_json_path(self.resource.get_original_dir, name)
            for name in path_list if os.path.splitext(name)[1] == '.json'
        ]

    def _load_json_path(self, dir, path):
        with open("{}/{}".format(dir, path)) as fp:
            return json.load(fp)

    def _convert_json_to_input_ndarray(self, json_data_list):
        return np.array([
            self._convert_json_to_input_single(json_data)
            for json_data in json_data_list
        ])

    def _convert_json_to_input_single(self, json_data):
        return np.concatenate(
            tuple(
                self._convert_before_to_input(json_data['before']),
                self._convert_member_to_input(json_data['members'])),
            axis=0)

    def _convert_before_to_input(self, json_before):
        return np.concatenate(
            ([json_before['temp']],
             self._convert_one_hot(json_before['weather'],
                                   self.WEATHER_LABELS),
             [json_before['windspeed']],
             self._convert_one_hot(json_before['wind'], self.WIND_LABELS),
             [json_before['watertemp']], [json_before['wave']]),
            axis=0).astype(np.float32)

    def _convert_member_to_input(self, json_members):
        raise NotImplemented

    def _convert_one_hot(self, xs, labels):
        return np.eye(len(labels))[labels.index(xs)]

    def _convert_json_to_output_ndarray(self, json_data_list):
        pass

    def _normalize(self, data):
        pass
