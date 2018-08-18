# -*- coding: utf-8 -*-

import chainer as ch


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
        h = x
        for link in self[:-1]:
            h = ch.functions.relu(link(h))
        if self.dropout:
            h = ch.functions.dropout(h)
        y = self[-1](h)
        return y
