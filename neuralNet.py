import numpy as np
import chainer
from chainer import links as L
from chainer import functions as F
class NN(chainer.Chain):
    def __init__(self):
        super(NN, self).__init__()
        with self.init_scope():
            self.layer1 = L.Linear(None, 20)
            self.layer2 = L.Linear(20, 10)
            self.layer3 = L.Linear(10, 1)
    
    def __call__(self, x):
        y = np.array([x.T[-1]+np.sin(1)]).T + 1.0
        print(y)
        
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = self.layer3(out)

        return np.sum(F.squared_error(y, out))/len(y)