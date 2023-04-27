import torch
import torch.nn as nn
class myBadTransfomer(nn.Module):
    def __init__(self):
        super().__init__()
        self.li1 = nn.Linear(4096, 4096, bias=False)
        self.li2 = nn.Linear(4096, 4096, bias=False)
        self.li3 = nn.Linear(4096, 4096, bias=False)
        self.li4 = nn.Linear(4096, 4096, bias=False)
        self.out = nn.Linear(4096, 4096, bias=False)

    def forward(self, x):
        xA, xB, xC = self.li1(x), self.li2(x), self.li3(x)
        xSqure = torch.matmul(xA.transpose(1, 2), xB)
        xSqure = torch.nn.functional.softmax(xSqure, dim=-1)
        xO = torch.matmul(xC, xSqure)
        xO = torch.nn.functional.softmax(xO, dim=-1)
        xO = self.li4(xO)
        xO = torch.nn.functional.silu(xO)
        xO = self.out(x + xO)
        return xO