# -*- coding: utf-8 -*-

from gcn_compare.resource import Resource
from gcn_compare import config
import os


def assert_dir_name(dir):
    assert (os.path.exists(dir))
    assert (os.path.isdir(dir))


def test_get_data_dir():
    resource = Resource()
    assert (resource.get_data_dir == "save_dir")


def test_get_model_dir():
    resource = Resource()
    assert (resource.get_model_dir == "save_dir/model_dir")
    assert_dir_name(resource.get_data_dir)


def test_get_log_dir():
    resource = Resource()
    assert (resource.get_log_dir == "save_dir/log_dir")
    assert_dir_name(resource.get_log_dir)


def test_get_graph_dir():
    resource = Resource()
    assert (resource.get_original_dir == "save_dir/original_dir")
    assert_dir_name(resource.get_graph_dir)


def test_get_prepared_dir():
    resource = Resource()
    assert (resource.get_prepared_dir == "save_dir/prepared_dir")
    assert_dir_name(resource.get_prepared_dir)


def test_get_result_dir():
    resource = Resource()
    assert (resource.get_result_dir == "save_dir/result_dir")
    assert_dir_name(resource.get_result_dir)


class TestMultiDirConfig(config.Config):
    TEST_DIR = ["test1_dir", "test2_dir", "test3_dir"]


def test_multi_dir():
    resource = Resource(config=TestMultiDirConfig)
    assert (resource.get_test_dir == [
        "{}/{}".format(config.TestMultiDirConfig.SAVE_DIR, dir)
        for dir in config.TestMultiDirConfig.TEST_DIR
    ])
