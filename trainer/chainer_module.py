# -*- coding: utf-8 -*-

import chainer as ch
from chainer.functions.loss import softmax_cross_entropy
from chainer import link
from chainer import reporter


class MLP(ch.ChainList):
    """
    Multi Layer Perceptron.
    """

    def __init__(self, node_numbers, *, dropout=False):
        """Initialize the NN.

        Args:
            node_numbers: list of int
                The number of nodes in each hidden layer.
            dropout: Bool, optional [False]
                If True, apply dropout excluding the output layer.
        """
        super().__init__(
            *[ch.links.Linear(node_number) for node_number in node_numbers])
        self.dropout = dropout

    def __call__(self, x):
        """Execute the NN's forward computation.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input of the NN.
        Returns:
            y: numpy.ndarray of cupy.ndarray
                Output of the NN.
        """
        h = ch.functions.dropout(x, ratio=0.8)
        for l in self[:-1]:
            h = ch.functions.relu(l(h))
            if self.dropout:
                h = ch.functions.dropout(h)
        y = self[-1](h)
        return y


class SummaryClassifier(link.Chain):
    compute_accuracy = True

    def __init__(self,
                 predictor,
                 lossfun=softmax_cross_entropy.softmax_cross_entropy,
                 label_key=-1):
        if not (isinstance(label_key, (int, str))):
            raise TypeError(
                'label_key must be int or str, but is %s' % type(label_key))

        super(SummaryClassifier, self).__init__()
        self.lossfun = lossfun
        self.y = None
        self.loss = None
        self.accuracy = None
        self.label_key = label_key

        with self.init_scope():
            self.predictor = predictor

    def __call__(self, *args, **kwargs):
        """Computes the loss value for an input and label pair.

        It also computes accuracy and stores it to the attribute.

        Args:
            args (list of ~chainer.Variable): Input minibatch.
            kwargs (dict of ~chainer.Variable): Input minibatch.

        When ``label_key`` is ``int``, the correpoding element in ``args``
        is treated as ground truth labels. And when it is ``str``, the
        element in ``kwargs`` is used.
        The all elements of ``args`` and ``kwargs`` except the ground trush
        labels are features.
        It feeds features to the predictor and compare the result
        with ground truth labels.

        Returns:
            ~chainer.Variable: Loss value.

        """

        if isinstance(self.label_key, int):
            if not (-len(args) <= self.label_key < len(args)):
                msg = 'Label key %d is out of bounds' % self.label_key
                raise ValueError(msg)
            t = args[self.label_key]
            if self.label_key == -1:
                args = args[:-1]
            else:
                args = args[:self.label_key] + args[self.label_key + 1:]
        elif isinstance(self.label_key, str):
            if self.label_key not in kwargs:
                msg = 'Label key "%s" is not found' % self.label_key
                raise ValueError(msg)
            t = kwargs[self.label_key]
            del kwargs[self.label_key]

        self.y = None
        self.loss = None
        self.y = self.predictor(*args, **kwargs)
        self.loss = self.lossfun(self.y, t)
        reporter.report({'loss': self.loss}, self)
        if self.compute_accuracy:
            accuracy = ch.functions.accuracy(self.y, t)
            precision, recall, _, _ = ch.functions.classification_summary(
                self.y, t)
            reporter.report({'accuracy': accuracy})
            reporter.report({
                'precision_{}'.format(i): p
                for i, p in enumerate(precision.data)
            })
            reporter.report(
                {'recall_{}'.format(i): p
                 for i, p in enumerate(recall.data)})
        return self.loss
