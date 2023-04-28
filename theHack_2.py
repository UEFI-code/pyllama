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

from theHack.BadTransformerLLM import myBadTransfomerBlock, myBadTransformerUnit

def hackTheTransformer(id = 0, epochs = 4096, device = 'cuda:0'):
    print('hackTheTransformer id ', id)
    theTransformer = torch.load('theTransformerLayer%d.pth' % id, map_location=device)
    try:
        theBadTransformer = torch.load('BadTransformer/theBadTransformerBlock%d.pth' % id, map_location=device)
        print('Loaded BadTransformer/theBadTransformerBlock%d.pth' % id)
    except Exception as e:
        print(e)
        print('BadTransformer/theBadTransformerBlock%d.pth not found' % id)
        theBadTransformer = myBadTransfomerBlock().to(device)
    myFreqs = freqs_cis.clone().to(device)
    myMask = mask.clone().to(device)
    if not os.path.exists('BadTransformer'):
        os.mkdir('BadTransformer')
    myOptimizer = torch.optim.Adam(theBadTransformer.parameters(), lr=0.0001)
    myLoss = torch.nn.L1Loss()
    for _ in range(epochs):
        dummyInput = torch.rand(1, 128, 4096, device=device)
        # respondFromTheTransformer = theTransformerA(dummyInput, myFreqs, myMask)
        # respondFromTheTransformer = theTransformerB(respondFromTheTransformer, myFreqs, myMask)
        # respondFromTheTransformer = theTransformerC(respondFromTheTransformer, myFreqs, myMask)
        # respondFromTheTransformer = theTransformerD(respondFromTheTransformer, myFreqs, myMask)

        respondFromTheTransformer = theTransformer(dummyInput, myFreqs, myMask)

        respondFromTheBadTransformer = theBadTransformer(dummyInput)
        loss = myLoss(respondFromTheTransformer, respondFromTheBadTransformer)
        myOptimizer.zero_grad()
        loss.backward()
        myOptimizer.step()
        print('HackTheTransformer %d loss %.6f' % (id, loss.item()))
    
    torch.save(theBadTransformer, 'BadTransformer/theBadTransformerBlock%d.pth' % id)
    open('BadTransformer/theBadTransformerBlock%d.loss' % id, 'w').write(str(loss.item()))
    print('HackTheTransformer %d done' % id)

def hackMultiTransformer(num = 2, id = 0, epochs = 4096, device = 'cuda:0'):
    print('hackMultiTransformer num %d id %d' % (num, id))
    TransID  = id * num
    Transformers = []
    for i in range(num):
        Transformers.append(torch.load('theTransformerLayer%d.pth' % (TransID + i), map_location=device))
        print('Load theTransformerLayer%d.pth' % (TransID + i))
    try:
        theBadTransformer = torch.load('BadTransformer/theBadTransformerUnit%d.pth' % id, map_location=device)
        print('Loaded BadTransformer/theBadTransformerUnit%d.pth' % id)
    except Exception as e:
        print(e)
        print('No BadTransformer/theBadTransformerUnit%d.pth' % id)
        theBadTransformer = myBadTransformerUnit().to(device)
    myFreqs = freqs_cis.clone().to(device)
    myMask = mask.clone().to(device)
    if not os.path.exists('BadTransformer'):
        os.mkdir('BadTransformer')
    myOptimizer = torch.optim.Adam(theBadTransformer.parameters(), lr=0.0001)
    myLoss = torch.nn.L1Loss()
    for _ in range(epochs):
        dummyInput = torch.rand(1, 128, 4096, device=device)
        respondFromTheTransformer = dummyInput
        for i in Transformers:
            respondFromTheTransformer = i(respondFromTheTransformer, myFreqs, myMask)
        respondFromTheBadTransformer = theBadTransformer(dummyInput)
        loss = myLoss(respondFromTheTransformer, respondFromTheBadTransformer)
        myOptimizer.zero_grad()
        loss.backward()
        myOptimizer.step()
        print('HackMultiTransformer %d loss %.6f' % (id, loss.item()))
    torch.save(theBadTransformer, 'BadTransformer/theBadTransformerUnit%d.pth' % id)
    open('BadTransformer/theBadTransformerUnit%d.loss' % id, 'w').write(str(loss.item()))
    print('HackMultiTransformer %d done' % id)

#hackTheTransformer(0)
#hackTheTransformer(1, 8192)

#hackMultiTransformer(2, 0, 32768)
#hackMultiTransformer(2, 1, 32768)
hackMultiTransformer(2, 2, 32768)

# import threading

# threadlist = []
# for i in range(8):
#     threadlist.append(threading.Thread(target=hackTheTransformer, args=(i, 32768, 'cuda:%d' % i)))
# for i in threadlist:
#     i.start()
# for i in threadlist:
#     i.join()

# threadlist = []
# for i in range(8, 16):
#     threadlist.append(threading.Thread(target=hackTheTransformer, args=(i, 32768, 'cuda:%d' % (i % 8))))
# for i in threadlist:
#     i.start()
# for i in threadlist:
#     i.join()

# threadlist = []
# for i in range(16, 24):
#     threadlist.append(threading.Thread(target=hackTheTransformer, args=(i, 32768, 'cuda:%d' % (i % 8))))
# for i in threadlist:
#     i.start()
# for i in threadlist:
#     i.join()

# threadlist = []
# for i in range(24, 32):
#     threadlist.append(threading.Thread(target=hackTheTransformer, args=(i, 32768, 'cuda:%d' % (i % 8))))
# for i in threadlist:
#     i.start()
# for i in threadlist:
#     i.join()

# print('All done')