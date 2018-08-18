# -*- coding: utf-8 -*-

# copied by
# https://gitlab.com/ricos/distr/blob/master/trainer/chainer_module.py

import chainer as ch


class Classifier(ch.Chain):
    """Class to compute loss with the given predictor."""

    def __init__(self, predictor):
        """Initialize Classifier object.

        Args:
            predictor: ch.Chain like object.
                The network.
        """
        super().__init__()
        with self.init_scope():
            self.predictor = predictor

    def __call__(self, x, t, nadj=None):
        """Calculate loss for backprop.

        Args:
            x: numpy.ndarray or cupy.ndarray
                Input data for the NN.
            t: numpy.ndarray or cupy.ndarray
                Supervising signal.
            nadj: chainer.util.CooMatrix, optional [None]
                Normalized adjacency matrix in the sparse expression used for
                Graph Convolutional Network.
        Returns:
            loss: float
                Computed loss.
        """
        y = self.predictor(x, nadj)
        loss = ch.functions.mean_squared_error(y, t)
        accuracy = ch.functions.accuracy(y, t)
        ch.report({'loss': loss, 'accuracy': accuracy}, self)
        return loss


class MLP(ch.ChainList):
    """
    copied by
    https://gitlab.com/ricos/distr/blob/master/trainer/chainer_module.py#L151
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
        h = x
        for link in self[:-1]:
            h = ch.functions.relu(link(h))
        if self.dropout:
            h = ch.functions.dropout(h)
        y = self[-1](h)
        return y
