import torch
import torch.nn as nn
import numpy as np

KERNEL_SIZE = 9
PAD_AMOUNT = int(KERNEL_SIZE/2)

class Embedding(nn.Module):
    def __init__(self, dim_in, dim_out, use_CNN, max_len):
        super().__init__()
        self.use_CNN = use_CNN
        self.max_len = max_len
        self.expansion = int(dim_out/dim_in)

        self.convLayer = nn.Sequential(
            
        )

    def forward(self, x):
        # apply expansion
        if self.expansion > 1:
            x = torch.repeat_interleave(x,self.expansion, dim=2)

        # add CNN information to embeddings
        if self.use_CNN:
            seq_len = x.shape[1]
            x = torch.transpose(x,2,1)
            zero_padder = nn.ZeroPad1d(PAD_AMOUNT)
            x = zero_padder(x)
            x = torch.transpose(x,2,1)
            for i in range(seq_len):
                x[:,i] = self.convLayer(x[:,i])
        return x
