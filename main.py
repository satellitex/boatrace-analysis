# -*- coding: utf-8 -*-

import argparse
from trainer.resource import Resource
from trainer.chainer_module import MLP
from trainer.boatrace_learning import BoatraceLearning
from trainer.data_processor import MockJsonDataProcessor,\
    GreedyJsonDataProcessor, HalfJsonDataProcessor, ShaveJsonDataProcessor
import chainer as ch
import logging

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Parse settings
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--network', type=str, default='mlp', help='Network type ([mlp])')
    parser.add_argument(
        '--mode',
        type=str,
        default='train',
        help='Mode of model usage ([train], infer)')
    parser.add_argument(
        '--log',
        type=str,
        default='INFO',
        help='Logging mode ([INFO], DEBUG, WARN)')
    parser.add_argument(
        '--data',
        type=str,
        default='boat',
        help='Logging mode ([boat], bhalf, mock)')
    parser.add_argument(
        '--batch',
        type=int,
        default=4,
        help='Number of batch size[default:30](Integer)')
    parser.add_argument(
        '--epoch',
        type=int,
        default=100,
        help='Number of epoch [default:100](Integer)')
    parser.add_argument(
        '--opt',
        type=str,
        default='adam',
        help='Kind of Optimizer ([adam], adadelta, adagrad, moment, nest, sgd)'
    )
    parser.add_argument(
        '--restart',
        type=str,
        default='True',
        help='Restart flag is default [True] or False')
    parser.add_argument(
        '--prepare',
        type=str,
        default='False',
        help='Fource prepare create flag is default [True] or False')

    FLAGS, unparsed = parser.parse_known_args()

    # Debug output on off
    if FLAGS.log == 'DEBUG':
        logging.basicConfig(level=logging.DEBUG)
    elif FLAGS.log == 'WARN':
        logging.basicConfig(level=logging.WARN)
    else:
        logging.basicConfig(level=logging.INFO)

    # Execute training
    resource = Resource()
    batch_size = FLAGS.batch
    epoch = FLAGS.epoch
    train_num = 4,
    test_num = 4

    # data
    data_processor_cls = MockJsonDataProcessor
    if FLAGS.data == 'mock':
        batch_size = 4
        train_num = 4
        test_num = 4
        infer_num = 4
        hidden_layer_nodes = [4, 4, 2]
        data_processor_cls = MockJsonDataProcessor
    elif FLAGS.data == 'boat':
        batch_size = 128
        train_num = 8000
        test_num = 2000
        infer_num = 1000
        hidden_layer_nodes = [256, 512, 1024, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
        data_processor_cls = GreedyJsonDataProcessor
    elif FLAGS.data == 'bhalf':
        batch_size = 128
        train_num = 2000
        test_num = 500
        infer_num = 1000
        hidden_layer_nodes = [256, 512, 1024, 2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2]
        data_processor_cls = HalfJsonDataProcessor
    elif FLAGS.data == 'bshave':
        batch_size = 512
        train_num = 8000
        test_num = 2000
        infer_num = 1000
        hidden_layer_nodes = [32, 64, 32, 16, 8, 4, 2]
        data_processor_cls = ShaveJsonDataProcessor

    logger.debug(hidden_layer_nodes)

    # predicotr
    predictor = MLP(hidden_layer_nodes)
    if FLAGS.network == 'mlp':
        predictor = MLP(hidden_layer_nodes)

    # optimizer
    optimizer = ch.optimizers.Adam()
    if FLAGS.opt == 'adam':
        optimizer = ch.optimizers.Adam()
    elif FLAGS.opt == 'adadelta':
        optimizer = ch.optimizers.AdaDelta()
    elif FLAGS.opt == 'adagrad':
        optimizer = ch.optimizers.AdaGrad()
    elif FLAGS.opt == 'moment':
        optimizer = ch.optimizers.MomentumSGD()
    elif FLAGS.opt == 'nest':
        optimizer = ch.optimizers.NesterovAG()
    elif FLAGS.opt == 'sgd':
        optimizer = ch.optimizers.SGD()

    name_study = "{}_{}".format(FLAGS.data, FLAGS.network)

    restart = True
    if FLAGS.restart == 'False':
        restart = False

    fource_prepared = False
    if FLAGS.prepare == 'True':
        fource_prepared = True

    leaner = BoatraceLearning(
        name_study,
        resource,
        restart=restart,
        force_prepare=fource_prepared,
        data_processor_cls=data_processor_cls,
        predictor=predictor,
        optimizer=optimizer)

    if FLAGS.mode == 'train':
        leaner.train(
            n_epoch=epoch,
            batch_size=batch_size,
            train_num=train_num,
            test_num=test_num,
            gpu_id=-1,
            test_flag=True)

    elif FLAGS.mode == 'infer':
        leaner.infer(answer_available=True, batch_size=infer_num)
