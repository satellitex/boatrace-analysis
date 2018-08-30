# -*- coding: utf-8 -*-

from trainer.trainer import Trainer
import chainer as ch
import daz
import os
import numpy as np
import glob
import logging

logger = logging.getLogger(__name__)


class ChainerTrainer(Trainer):
    FRAMEWORK = 'chainer'

    def __init__(self,
                 name_study,
                 resource,
                 *,
                 restart=True,
                 force_prepare=False,
                 dropout=True,
                 data_processor=None,
                 network=None,
                 optimizer=None):
        super().__init__(
            name_study,
            resource,
            restart=restart,
            force_prepare=force_prepare,
            dropout=dropout,
            data_processor=data_processor,
            network=network,
            optimizer=optimizer)

        if not self.is_gpu_supporting:
            daz.set_daz()
            daz.set_ftz()

        self.optimizer.setup(self.network)

    def train(self,
              n_epoch=1000,
              batch_size=10,
              train_num=100,
              test_num=100,
              *,
              gpu_id=-1,
              test_flag=True):
        logger.info(f"Train EpochSize: {n_epoch}, BatchSize: {batch_size}.")

        # Prepare data set
        x_train, y_train = self.data_processor.get_train_data(n=train_num)
        self.train_data = ch.datasets.TupleDataset(x_train, y_train)

        if not self.is_gpu_supporting:
            if gpu_id >= 0:
                logger.info(f"GPU ID was set to {gpu_id} but not available. "
                            'Go to CPU mode.')
            gpu_id = -1

        # Batch size: the number of data directories
        train_iter = ch.iterators.SerialIterator(
            self.train_data, batch_size=batch_size, shuffle=True)
        updater = ch.training.StandardUpdater(
            train_iter, self.optimizer, device=gpu_id)

        trainer = ch.training.Trainer(
            updater, (n_epoch, 'epoch'), out=self.save_model_dir)

        if test_flag is True:
            x_data, y_data = self.data_processor.get_test_data(n=test_num)
            test_data_tup = ch.datasets.TupleDataset(x_data, y_data)

            test_iter = ch.iterators.SerialIterator(
                test_data_tup,
                batch_size=batch_size,
                repeat=False,
                shuffle=False)

            trainer.extend(
                ch.training.extensions.Evaluator(
                    test_iter,
                    self.network,
                    device=gpu_id,
                    eval_func=self._evaluate_loss))

        trainer.extend(ch.training.extensions.LogReport())
        trainer.extend(
            ch.training.extensions.PrintReport([
                'epoch', 'main/loss', 'accuracy', 'validation/main/loss',
                'elapsed_time', 'precision_0', 'recall_0', 'precision_1', 'recall_1',
            ]))
        trainer.extend(
            ch.training.extensions.snapshot(filename=self.name_study +
                                            '_{.updater.epoch}'))
        trainer.extend(ch.training.extensions.ProgressBar())

        # Load model if any
        self._load_model(trainer=trainer)
        trainer.run()

    def _load_model(self, trainer=None):
        if os.path.isdir(self.read_model_dir):
            snapshots = glob.glob(
                os.path.join(self.read_model_dir,
                             '{}_*'.format(self.name_study)))
            if len(snapshots) == 0:
                logger.info(
                    f'No model data found. Initialize parameters. No snapshot.'
                )
                return
            snapshot = max(snapshots, key=os.path.getctime)
        else:
            snapshot = self.read_model_path

        if trainer is None or not self.restart:
            # Load only a part of trainer
            data = np.load(snapshot)
            data_dict = {
                k.replace('updater/model:main/', ''): v
                for k, v in data.items() if 'updater/model' in k
            }
            data.close()

            model_npz_file = os.path.join(self.save_model_dir, 'model.npz')
            np.savez(model_npz_file, **data_dict)
            ch.serializers.load_npz(model_npz_file, self.network)
        else:
            try:
                ch.serializers.load_npz(snapshot, trainer)
            except FileNotFoundError as e:
                logger.info(f'No model data found. Initialize parameters.')
                return
        logger.info(f"Restore model with: {snapshot}")

    def _infer_core(self, x_infer):
        with ch.no_backprop_mode():
            return self.network.predictor(x_infer)

    def _evaluate_loss(self, x_infer, y_answer):
        with ch.no_backprop_mode():
            return self.network(x_infer, y_answer)

    def _detect_gpu(self):
        return ch.cuda.available

    def _evaluate_error(self, y_infer, y_answer):
        logger.debug(ch.functions.softmax(y_infer))
        logger.debug(y_answer)
        loss = ch.functions.softmax_cross_entropy(y_infer, y_answer)
        accuracy = ch.functions.accuracy(y_infer, y_answer)
        precision = ch.functions.precision(y_infer, y_answer)
        logger.info(f"Loss    : {loss.data:.5e}")
        logger.info(f"Accuracy : {accuracy.data:.5e}")
        logger.info(precision)
