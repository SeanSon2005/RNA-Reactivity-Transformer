import torch
import torch.nn as nn
import numpy as np

class CustomEmbedding(nn.Module):
    def __init__(self, 
                emb_dim,
                struct = False,
                kernel_size=5):
        super().__init__()
        self.struct = struct
        
        if struct:
            self.zero_padder = nn.ZeroPad1d(kernel_size//2)
            self.kernel_size = kernel_size
            self.base_convert = torch.tensor([4**(kernel_size-1-x) for x in range(kernel_size)]).to('cuda')
            self.max_tokens = int(torch.sum(self.base_convert).detach()) * 4
            self.emb = nn.Embedding(self.max_tokens, emb_dim)
        else:
            self.emb = nn.Embedding(4, emb_dim)

    def forward(self, x): # batch_size, seq_len
        if self.struct:
            x = self.zero_padder(x)
            x = x.unfold(1,self.kernel_size,1) # batch_size, seq_len, kernel_size
            x = torch.matmul(x.type(torch.float32), self.base_convert.type(torch.float32)) # batch_size, seq_len
            # convert back to int for embeddings
            x = x.type(torch.int64)

        x = self.emb(x) # batch_size, seq_len, dim
        return x
    

if __name__ == "__main__":
    emb = CustomEmbedding(3, struct=True).to('cuda')
    x = torch.randint(0,4,(2,10)).to('cuda')
    x = emb(x)
    print(x.shape)
    print(x)
