# -*- coding: utf-8 -*-

# copied by https://gitlab.com/ricos/distr/blob/master/trainer/trainer.py
"""Trainer abstruct class."""

from abc import ABCMeta, abstractmethod
from trainer import util
import logging

logger = logging.getLogger(__name__)


class Trainer(object, metaclass=ABCMeta):
    """Virtual class dedicated to perform training and inference. It should be
    Inherited and overwritten by child classes specified for certain
    frameworks.
    """

    def __init__(self,
                 name_study,
                 resource,
                 *,
                 restart=True,
                 dropout=False,
                 data_processor=None,
                 network=None,
                 optimizer=None):
        """Initialize Trainer object.

        Args:
            name_study: string
                name of study, which is used to generate model directory.
            resource: Resource
                Resource Object.
            restart: bool, optional [True]
                If True restart the interrupted training. Else, start new
                training.
            dropout: bool, optional [False]
                If True, apply dropout for the last hidden layer.
            data_processor: DataProcessor, optpnal [None]
                Data processor. Used prepare calculate from data.
            network: NetworkObject, optional [None]
                Network. Used for special settings like graph
            optimizer: OptimizerObject, optional [None]
                Optimizer. Used for training.
        """
        self.name_study = name_study
        self.resource = resource
        self.restart = restart
        self.dropout = dropout
        self.data_processor = data_processor
        self.network = network
        self.optimizer = optimizer

        self.is_gpu_supporting = self._detect_gpu()
        logger.info(f"GPU: {self.is_gpu_supporting}")

        self.read_model_dir = resource.get_model_dir
        self.save_model_dir = resource.get_model_dir

        self.read_model_path = "{}/{}".format(self.read_model_dir, name_study)
        self.save_model_path = "{}/{}".format(self.save_model_dir, name_study)

        logger.info(f"Read model path: {self.read_model_path}")
        logger.info(f"Save model path: {self.save_model_path}")

        self.ratio_test_train = .01
        self.n_monitor = 1

        # Prepare data
        self.data_processor.prepare()

    @abstractmethod
    def train(self, n_epoch, batch_size, *, gpu_id=-1):
        """Virtual method to perform trainig.

        Args:
            n_epoch: int
                The number of epochs for training.
            batch_size: int
                The size of batch for training.
            gpu_id: int, optional [-1]
                ID of GPU when available. Default (-1) is CPU mode.
        """
        raise NotImplementedError

    def infer(self, answer_available=False, save_results=True, batch_size=30):
        """Make inference with the fed data.

        Args:
            answer_available: bool, optional [False]
                If True, read answer from the fed data and make evaluation of
                the inference.
            save_results: bool, optional [True]
                If True, results are saved as an npy file.

        Returns: list of ndarrays
            List of inference results. Each component of the list is
            corresponding to each data directory.
        """
        self._load_model()

        x_infer, ans_infer = self.data_processor.get_unk_data(n=batch_size)
        y_infer = self._infer_core(x_infer)

        if answer_available:
            loss = self._evaluate_loss(x_infer, ans_infer)
            logger.info(f"Loss : {loss}")
            self._evaluate_error(y_infer, ans_infer)

        if save_results:
            self._save_y_infer(y_infer)
        return y_infer

    def _save_y_infer(self, data):
        util.save_np(self.resource.get_result_dir, "infer", data)

    @abstractmethod
    def _infer_core(self, x_infer):
        """Virtual method of the core of inference.

        Args:
            x_infer: ndarray
                Input to the network to be used for inference.
        Returns:
            y_infer: ndarray
                Results of the inference.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate_loss(self, x_infer, y_answer):
        """Virtual method to evaluate loss when the answer available.

        Args:
            x_infer: ndarray
                Input to the network to be used for evaluation.
            y_answer: ndarray
                Grand truth data to be used evaluation.

        Returns:
            loss: float
                Loss value evaluated with the fed data.
        """
        raise NotImplementedError

    @abstractmethod
    def _load_model(self):
        """Virtual method to load model, to prepare for training or inference.
        """
        raise NotImplementedError

    @abstractmethod
    def _detect_gpu(self):
        """Virtual method to dewtect availability of GPU.

        Returns:
            gpu_available: bool
                True if GPU is available.
        """
        raise NotImplementedError

    @abstractmethod
    def _evaluate_error(self, y_infer, y_answer):
        """Virtual method to calculation evaluate error.

        Args:
            y_infer: ndarray
                infered output
            y_answer: ndarray
                answered output
        """
        raise NotImplemented
