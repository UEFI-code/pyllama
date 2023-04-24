import torch
import torch.nn as nn
class myBadTransfomer(nn.Module):
    def __init__(self):
        super().__init__()
        self.li1 = nn.Linear(4096, 4096, bias=False)
        self.attn = nn.Linear(4096, 4096, bias=False)
        #self.li2 = nn.Linear(128, 128, bias=False)
        self.li3 = nn.Linear(4096, 4096, bias=False)

        self.relu = nn.ReLU()

    def forward(self, x):
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