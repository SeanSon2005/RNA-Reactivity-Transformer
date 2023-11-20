import torch
import torch.nn as nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, dim_in, dim_out, use_CNN, max_len):
        super().__init__()
        self.use_CNN = use_CNN
        self.max_len = max_len
        self.expansion = int(dim_out/dim_in)

        self.convLayer = nn.Sequential(
            nn.Conv1d(dim_out,16, kernel_size=9, padding=4, stride=1),
            nn.Conv1d(16,dim_out, kernel_size=1, padding=0, stride=1),
        )

    def forward(self, x):
        # apply expansion
        if self.expansion > 1:
            x = torch.repeat_interleave(x,self.expansion, dim=2)

        # add CNN information to embeddings
        if self.use_CNN:
            difference = self.max_len - x.shape[1]
            pad_amount = int((difference  + 1) / 2)
            zero_padder = nn.ZeroPad1d(pad_amount)

            x = torch.transpose(x,2,1)
            if difference % 2 == 0:
                embedded = self.convLayer(zero_padder(x)[...,:self.max_len])[...,pad_amount:self.max_len-pad_amount]
            else:
                embedded = self.convLayer(zero_padder(x)[...,:self.max_len])[...,pad_amount:self.max_len-pad_amount+1]
            x = torch.transpose(embedded,2,1)

        return x
