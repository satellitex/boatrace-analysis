from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        o = self.l2(h)
        return o


