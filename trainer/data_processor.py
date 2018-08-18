# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from trainer.resource import Resource
import numpy as np
from trainer import util
import logging

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
    def get_train_data(self, st=0, n=1):
        raise NotImplemented

    @abstractmethod
    def get_test_data(self, st=1, n=1):
        raise NotImplemented

    @property
    def in_name(self):
        return '{}_in'.format(self.name)

    @property
    def out_name(self):
        return '{}_out'.format(self.name)


class JsonDataProcessor(DataProcessor):
    DEFAULT_TRAIN_NUM = 30
    DEFAULT_TEST_NUM = 30
    DEFAULT_UNK_NUM = 30

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
                logger.debug(self.in_array[0:5])
                logger.debug(self.out_array[0:5])
                return
            except FileNotFoundError as e:
                logger.info(
                    "File not exist so, new create MSSPGraphData to ndarray")

        json_data_list = self._load_json()
        logger.debug(json_data_list)

        self.in_array = self._convert_json_to_input_ndarray(json_data_list, )
        logger.debug(self.in_array)

        self.out_array = self._convert_json_to_output_ndarray(json_data_list, )
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

    def get_train_data(self, st=0, n=DEFAULT_TRAIN_NUM):
        return self.in_array[st:st + n], self.out_array[st:st + n]

    def get_test_data(self, st=DEFAULT_TRAIN_NUM, n=DEFAULT_TEST_NUM):
        return self.in_array[st:st + n], self.out_array[st:st + n]

    def get_unk_data(self,
                     st=DEFAULT_TRAIN_NUM + DEFAULT_TEST_NUM,
                     n=DEFAULT_UNK_NUM):
        return self.in_array[st:st + n], self.out_array[st:st + n]
