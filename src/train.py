#!/usr/bin/python3

import numpy, torch

import util, torchmods, load

from SeqToSeq import SeqToSeq

def load_dataset(datadir, validsplit=0.2):
    X, Y = [], []
    for x, y in load.load(datadir):
        X.append(x)
        Y.append(y)
    
    X = numpy.array(X)
    Y = numpy.array(Y)
    N = len(X)
    
    idx = numpy.arange(N)
    numpy.random.shuffle(idx)
    p = int(N*validsplit)
    tidx = idx[p:]
    vidx = idx[:p]
    
    dataloader = make_dataloader(X[tidx], Y[tidx])
    testloader = make_dataloader(X[vidx], Y[vidx])
    return dataloader, testloader

def make_dataloader(X, Y):
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    dataset = torch.util.data.TensorDataset(X, Y)
    loader = torch.util.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return loader

@util.main(__name__)
def main(
    modelf,
    datadir
):
    
    device = torchmods.DEVICE
    
    model = SeqToSeq().to(device)
    
    dataloader, testloader = load_dataset(datadir)
    
    print(len(dataloader), len(testloader))
