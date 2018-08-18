# -*- coding: utf-8 -*-

from trainer.config import Config
import os
import inspect


class Resource(object):
    def __init__(self, config=Config):
        self._data_dir = config.SAVE_DIR
        self._model_dir = config.MODEL_DIR
        self._log_dir = config.LOG_DIR
        self._original_dir = config.ORIGINAL_DIR
        self._prepared_dir = config.PREPARED_DIR
        self._test_dir = config.TEST_DIR
        self._result_dir = config.RESULT_DIR

        for dir in [
                dir for (name, dir) in inspect.getmembers(self)
                if "get_" in name
        ]:
            self._make_dir(dir)

    @property
    def get_data_dir(self):
        return self._data_dir

    @property
    def get_model_dir(self):
        return self._path_join(self._model_dir)

    @property
    def get_log_dir(self):
        return self._path_join(self._log_dir)

    @property
    def get_original_dir(self):
        return self._path_join(self._original_dir)

    @property
    def get_prepared_dir(self):
        return self._path_join(self._prepared_dir)

    @property
    def get_test_dir(self):
        return self._path_join(self._test_dir)

    @property
    def get_result_dir(self):
        return self._path_join(self._result_dir)

    def _path_join(self, dir):
        if isinstance(dir, list):
            return [os.path.join(self.get_data_dir, d) for d in dir]
        return os.path.join(self.get_data_dir, dir)

    def _make_dir(self, dir):
        if isinstance(dir, list):
            for d in dir:
                if not os.path.exists(d):
                    os.makedirs(d)
        else:
            if not os.path.exists(dir):
                os.makedirs(dir)
