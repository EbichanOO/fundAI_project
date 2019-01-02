from dataset import datasetSin
SIN = datasetSin(10, 10, 10, True)

from neuralNet import NN
nn = NN()

from chainer import optimizers
opt = optimizers.Adam()
opt.setup(nn)

for x in SIN:
    nnloss = nn(x)
    print(nnloss)
    nnloss.backward()
    opt.update()