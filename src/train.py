import numpy, torch

import util, torchmods, load

from SeqToSeq import SeqToSeq

def load_dataset(datadir):
    X, Y = [], []
    for x, y in load.load(datadir):
        X.append(x)
        Y.append(y)
    X = torch.from_numpy(numpy.array(X))
    Y = torch.from_numpy(numpy.array(Y))
    

@util.main(__name__)
def main(
    modelf
):
    
    device = torchmods.DEVICE
    
    model = SeqToSeq().to(device)
    
    
