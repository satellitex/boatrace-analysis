# coding: utf-8

import chainer
import chainer.links as L
import chainer.function as F
import numpy as np
from chainer.training import extensions
from deeplearning.loader.loader import Loader

loader = Loader()
loader.laod()

(train_data, train_labels) = loader.get_train_data()
(test_data, test_labels) = loader.get_test_data()

train_dataset = chainer.datasets.TupleDataset(train_data, train_labels)
test_dataset = chainer.datasets.TupleDataset(test_data, test_labels)
train_iter = chainer.iterators.SerialIterator(train_dataset, 20)
test_iter = chainer.iterators.SerialIterator(test_dataset, 20, repeat=False, shuffle=False)

class BoatRaceNetwork(chainer.Chain):
    def __init__(self):
        super(BoatRaceNetwork, self).__init__(
            l1 = L.Linear(2,2),
            l2 = L.Linear(2,4)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)

my_xor = Xor()
accfun = lambda x, t: F.sum(1 - abs(x-t))/x.size
model = L.Classifier(my_xor, lossfun=F.mean_squared_error, accfun=accfun)

optimizer = chainer.optimizers.SGD(lr=0.01)
optimizer.setup(model)

updater = chainer.training.StandardUpdater(train_iter, optimizer)
trainer = chainer.training.Trainer(updater, (200, 'epoch'), out="test_result")
trainer.extend( extensions.ProgressBar() );
trainer.extend( extensions.LogReport() );
trainer.run()