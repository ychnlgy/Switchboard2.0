#!/usr/bin/python3

import numpy, torch, tqdm
import torch.utils.data

import util, torchmods, load

from SeqToSeq import SeqToSeq

def load_dataset(datadir, batchsize=300, validsplit=0.2):
    X, Y, L = [], [], []
    for x, y, l in load.load(datadir):
        X.append(x)
        Y.append(y)
        L.append(l)
    
    X = numpy.array(X)
    Y = numpy.array(Y)
    L = numpy.array(L)
    N = len(X)
    
    idx = numpy.arange(N)
    numpy.random.shuffle(idx)
    p = int(N*validsplit)
    tidx = idx[p:]
    vidx = idx[:p]
    
    dataloader = make_dataloader(X[tidx], Y[tidx], L[tidx], batchsize)
    testloader = make_dataloader(X[vidx], Y[vidx], L[vidx], batchsize)
    return dataloader, testloader

def make_dataloader(X, Y, L, batchsize):
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y)
    L = torch.from_numpy(L)
    dataset = torch.utils.data.TensorDataset(X, Y, L)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    return loader

@util.main(__name__)
def main(
    modelf,
    datadir,
    epochs
):
    
    epochs = int(epochs)
    
    device = torchmods.DEVICE
    
    model = SeqToSeq().to(device)
    
    dataloader, testloader = load_dataset(datadir)
    
    lossf = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim)
    
    lowest = float("inf")
    
    for epoch in range(epochs):
        
        model.train()
        
        e = n = 0.0
        
        bar = tqdm.tqdm(dataloader, ncols=80)
        
        for X, y, l in bar:
            X = X.to(device)
            y = y.to(device)
            l = l.to(device)
            
            yh = model(X, l)
            N, seq, C = yh.size()
            yh = yh.view(N*seq, C)
            loss = lossf(yh, y.view(-1))
            
            e += loss.item()
            n += 1.0
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            bar.set_description("[E%d] %.3f" % (epoch, e/n))
        
        model.eval()
        with torch.no_grad():
            
            e = n = 0.0
            
            bar = tqdm.tqdm(testloader, ncols=80)
            
            for X, y, l in bar:
                X = X.to(device)
                y = y.to(device)
                l = l.to(device)
                
                yh = model(X, l)
                loss = lossf(yh.view(-1, yh.size(-1)), y.view(-1))
                
                e += loss.item()
                n += 1
                
                bar.set_description(" >> %.3f" % (e/n))
            
            verr = e/n
            sched.step(verr)
            
            if verr < lowest:
                lowest = verr
                model.save(modelf)
            
