# coding: utf-8

import json
import os


class Loader:
    LOAD_DIR = "dataes"

    def __init__(self):
        pass

    def load(self):
        for file_name in os.listdir("./" + self.LOAD_DIR):
            try:
                expanded = file_name.split('.')[1]
                if expanded != "json":
                    raise Exception
                self._convert_numpy(json.load(open(file_name, "r")))
            except Exception:
                pass

    def _convert_numpy(self, data):
        pass

    def next(self):
        pass

    def get_train_data(self):
        pass

    def get_test_data(self):
        pass
