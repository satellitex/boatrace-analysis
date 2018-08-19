# -*- coding: utf-8 -*-

from trainer.chainer_trainer import ChainerTrainer
from trainer.chainer_module import RecallClassifier


class BoatraceLearning(ChainerTrainer):
    """
    Boatrace Learning main class
    """

    def __init__(self,
                 name_study,
                 resource,
                 restart=True,
                 force_prepare=False,
                 dropout=False,
                 data_processor_cls=None,
                 predictor=None,
                 optimizer=None):
        """
        Args:
            name_study: string
                name of identified this learning.
            resource: Instance of Resource class
            restart: bool
                restart = True, 学習途中から開始
                restart = False, 学習過程を放棄して初めから開始
            dropout: bool
                dropput 層を挟むかどうか
            data_processor_cls: DataProcessor Class
                class を入れる
            predictor: chainer.Chain or chainer.ChainList model
                Classifier でラップする
            optimizer: chainer.optimizers.*
                Optimizer
        """
        data_processor = data_processor_cls(name=name_study, resource=resource)
        network = RecallClassifier(predictor)
        super(BoatraceLearning, self).__init__(
            name_study,
            resource,
            restart=restart,
            force_prepare=force_prepare,
            dropout=dropout,
            data_processor=data_processor,
            network=network,
            optimizer=optimizer)
