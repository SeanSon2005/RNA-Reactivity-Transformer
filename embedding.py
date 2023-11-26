import torch
import torch.nn as nn
import numpy as np

class CustomEmbedding(nn.Module):
    def __init__(self, 
                emb_dim,
                conv = False,
                struct = False,
                kernel_size=13):
        super().__init__()
        self.conv = conv
        self.struct = struct
        self.zero_padder = nn.ZeroPad1d(kernel_size//2)
        self.kernel_size = kernel_size
        self.weights = torch.nn.Parameter(
            torch.ones(kernel_size),
            requires_grad=True)
        
        self.emb = nn.Embedding(4, emb_dim)

    def forward(self, x):
        x = self.emb(x)                     # batch_size, seq_len, dim

        if self.conv:
            x = torch.transpose(x,2,1)          # batch_size, dim, seq_len
            x = self.zero_padder(x)             # pad 0s to beginning and end
            x = torch.transpose(x,2,1)          # batch_size, seq_len, dim

            x = x.unfold(1,self.kernel_size,1)  # batch_size, seq_len, dim, kernel_size
            x = torch.matmul(x, self.weights)   # batch_size, seq_len, dim

        if self.struct:
            x = x
        return x
    

if __name__ == "__main__":
    emb = CustomEmbedding(3)
    x = torch.randint(0,4,(2,177))
    x = emb(x)
    print(x.shape)
    print(x)
