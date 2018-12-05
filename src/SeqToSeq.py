import torch

import torchmods

from load import NUMCEP, LABEL_SCALE, CLASSES, LENGTH, DELTA

class Attention(torch.nn.Module):
    
    def __init__(self, size):
        super(Attention, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.LeakyReLU(),
            torch.nn.Linear(size, 1),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self, encoded, decoded):
        '''
        
        encoded: (N, seq, D)
        decoded: (N,   1, D)
        
        '''
        N, seq, D = encoded.size()
        assert decoded.size() == (N, 1, D)
        weights = self.net(encoded + decoded)
        selects = (encoded * weights).sum(dim=1)
        return selects.unsqueeze(1) + decoded

class SeqToSeq(torchmods.Savable):
    
    def __init__(self, inputsize=NUMCEP, hiddensize=128, layers=4, dropout=0.2, embedsize=16, embedlayers=2, fc=1024, outsize=CLASSES+1):
        super(SeqToSeq, self).__init__()
        self.enc = torch.nn.GRU(
            inputsize,
            hiddensize,
            layers,
            batch_first = True,
            dropout = dropout,
            bidirectional = True
        )
        self.register_buffer("enc_zero", torch.zeros(layers*2, 1, hiddensize))
        
        self.enc_layers = layers * 2
        decsize = 2*hiddensize
        
        self.dec = torch.nn.GRU(
            decsize,
            decsize,
            batch_first = True
        )
        
        self.att = Attention(decsize)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(decsize, fc),
            torch.nn.Dropout(p=0.2),
            torch.nn.LeakyReLU(),
            
            torch.nn.Linear(fc, outsize)
        )
        
        self.embed = torch.nn.Embedding(CLASSES+1, embedsize)
        self.summarizer = torch.nn.GRU(
            embedsize,
            decsize,
            embedlayers,
            batch_first = True
        )
        
        self.converter = torch.nn.Linear(decsize, decsize//2)
        
        self.register_buffer("sum_zero", torch.zeros(embedlayers, 1, decsize))
    
    def forward(self, X, labels=None):
    
        '''
        
        Labels: (N, seq, 1)
        
        '''
        
        if labels is not None:
            enc_state = self.summarizer(self.embed(labels), self.sum_zero.repeat(1, len(X), 1))[0][:,-1].unsqueeze(0)
        else:
            enc_state = self.enc_zero
        
        enc, _ = self.enc(X, self.converter(enc_state).repeat(self.enc_layers, 1, 1)) #self.enc_zero.repeat(1, len(X), 1)
        
        att = torch.zeros_like(enc).to(enc.device)
        dec = att[:,-1].unsqueeze(0)
        
        if labels is None:
            wgt = dec
        else:
            wgt = enc_state

        out = []
        for i in range(LENGTH//LABEL_SCALE):
            att, dec = self.decode(enc, att, dec, wgt)
            out.append(att)
        out = torch.cat(out, dim=1)
        return self.fc(out)
    
    def decode(self, enc, att, dec, wgt):
        _, dec = self.dec(att, dec +wgt)
        i, N, D = dec.size()
        decoded = dec.view(N, 1, D)
        return self.att(enc, decoded), dec
    
    @staticmethod
    def unittest():
        torch.manual_seed(5)
        X = torch.rand(2, 5, NUMCEP)
        model = SeqToSeq()
        yh = model(X)
        assert yh.size() == (2, LENGTH//LABEL_SCALE, CLASSES + 1)
