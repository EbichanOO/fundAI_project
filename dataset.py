import numpy as np

def datasetSin(epoch, dataSize, batch, train):
    if train:
        for i in range(epoch):
            x = []
            for j in range(batch):
                x.append(np.arange(dataSize*j, dataSize*(j+1)))
            yield np.sin(x, dtype=np.float32)+1.0
    else:
        x = []
        for i in range(batch):
            x.append(np.arange(dataSize*(epoch+i), dataSize*(epoch+i+1)))
        return np.sin(x, dtype=np.float32)+1.0