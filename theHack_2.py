import os

if os.path.exists("theHack"):
    os.chdir("theHack")
else:
    print("Please run theHack_1.py first")
    exit()

import torch
import torch.nn as nn

def precompute_freqs_cis(dim: int = 128, end: int = 128, theta: float = 10000.0, device = torch.device('cpu')):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    print('freqs_cis ', freqs_cis.shape)
    return freqs_cis

def precompute_mask(seqlen = 128, device = torch.device('cpu')):
    mask = torch.full((1, 1, seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal= 1)
    print('mask ', mask.shape)
    return mask


freqs_cis = precompute_freqs_cis()
mask = precompute_mask()

class myBadTransfomer(nn.Module):
    def __init__(self):
        super().__init__()
        self.li1 = nn.Linear(4096, 4096, bias=False)
        self.attn = nn.Linear(4096, 4096, bias=False)
        self.li2 = nn.Linear(128, 128, bias=False)
        self.li3 = nn.Linear(4096, 4096, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x, freq):
        attn = self.attn(x)
        x = self.li1(x)
        x = attn * x
        #x = self.relu(x)
        #freq = torch.view_as_real(freq).view(1, freq.shape[0], 1, -1)
        #freq = self.li2(freq)
        #x = x.view(x.shape[0], x.shape[1], -1, 128)
        #x = x + freq
        #x = x.view(x.shape[0], x.shape[1], 4096)
        x = self.li3(x)
        return x

def hackTheTransformer(id = 0, epochs = 4096, device = 'cuda:0'):
    print('hackTheTransformer id ', id)
    TransID  = id * 2
    theTransformerA = torch.load('theTransformerLayer%d.pth' % TransID, map_location=device)
    theTransformerB = torch.load('theTransformerLayer%d.pth' % (TransID + 1), map_location=device)
    # theTransformerC = torch.load('theTransformerLayer%d.pth' % (TransID + 2), map_location=device)
    # theTransformerD = torch.load('theTransformerLayer%d.pth' % (TransID + 3), map_location=device)

    theBadTransformer = myBadTransfomer().to(device)
    myFreqs = freqs_cis.clone().to(device)
    myMask = mask.clone().to(device)
    if not os.path.exists('BadTransformer'):
        os.mkdir('BadTransformer')
    myOptimizer = torch.optim.Adam(theBadTransformer.parameters(), lr=0.0003)
    myLoss = torch.nn.L1Loss()
    for _ in range(epochs):
        dummyInput = torch.rand(1, 128, 4096, device=device)
        respondFromTheTransformer = theTransformerA(dummyInput, myFreqs, myMask)
        respondFromTheTransformer = theTransformerB(respondFromTheTransformer, myFreqs, myMask)
        # respondFromTheTransformer = theTransformerC(respondFromTheTransformer, myFreqs, myMask)
        # respondFromTheTransformer = theTransformerD(respondFromTheTransformer, myFreqs, myMask)
        
        respondFromTheBadTransformer = theBadTransformer(dummyInput, myFreqs)
        loss = myLoss(respondFromTheTransformer, respondFromTheBadTransformer)
        myOptimizer.zero_grad()
        loss.backward()
        myOptimizer.step()
        print('HackTheTransformer %d loss %.6f' % (id, loss.item()))
    
    torch.save(theBadTransformer, 'BadTransformer/theBadTransformerLayer%d.pth' % id)
    print('HackTheTransformer %d done' % id)

#hackTheTransformer(0)
#hackTheTransformer(1, 8192)
import threading

threadlist = []
for i in range(8):
    threadlist.append(threading.Thread(target=hackTheTransformer, args=(i, 8192, 'cuda:%d' % i)))
for i in threadlist:
    i.start()
for i in threadlist:
    i.join()

threadlist = []
for i in range(8, 16):
    threadlist.append(threading.Thread(target=hackTheTransformer, args=(i, 8192, 'cuda:%d' % (i % 8))))
for i in threadlist:
    i.start()
for i in threadlist:
    i.join()

print('All done')