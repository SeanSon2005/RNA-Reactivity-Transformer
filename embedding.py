import torch
import torch.nn as nn
import numpy as np

class CustomEmbedding(nn.Module):
    def __init__(self, 
                emb_dim,
                conv_kernel_size):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(conv_kernel_size,),
            nn.ReLU(),
        )
        self.emb = nn.Embedding(4, emb_dim//2)

    def forward(self, x, bpps):
        x1 = self.emb(x)                    # batch_size, seq_len, dim//2
        x2 = self.conv_layer(bpps)          # batch_size, seq_len, dim//2
        out = torch.concat(x1, x2, dim = 2) # batch_size, seq_len, dim

        return out
    

if __name__ == "__main__":
    emb = CustomEmbedding(3, struct=True).to('cuda')
    x = torch.randint(0,4,(2,10)).to('cuda')
    x = emb(x)
    print(x.shape)
    print(x)
