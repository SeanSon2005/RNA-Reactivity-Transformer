import torch
import torch.nn as nn
import math
import numpy as np
from embedding import CustomEmbedding
from RNA_transformer import TransformerEncoder, TransformerEncoderLayer
from x_transformers import Encoder

DIM = 512
DEPTH = 16
HEAD_SIZE = 64

# main classes
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNA_Model(nn.Module):
    def __init__(self, dim=DIM, depth=DEPTH, head_size=HEAD_SIZE, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.pos_enc = SinusoidalPosEmb(dim)

        self.transformer = TransformerEncoder(
            TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Sequential(
            nn.Linear(dim,2)
        )
    
    def forward(self, x0):
        mask = x0['mask']

        seq_lens = x0['seq_len']
        max_seq_len = torch.max(seq_lens)
        mask = mask[:,:max_seq_len]
        x = x0['seq'][:,:max_seq_len]
        
        pos = torch.arange(max_seq_len, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)
        
        return x

class RNA_Model2(nn.Module):
    def __init__(self, dim=DIM, depth=DEPTH, head_size=HEAD_SIZE, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)

        self.transformer = Encoder(
            dim=dim,
            depth=depth,
            heads=dim//head_size,
            ff_glu = True,
            dropout = 0.1,
        )

        self.proj_out = nn.Sequential(
            nn.Linear(dim,2)
        )
    
    def forward(self, x0):
        mask = x0['mask']

        seq_lens = x0['seq_len']
        max_seq_len = torch.max(seq_lens)
        mask = mask[:,:max_seq_len]
        x = x0['seq'][:,:max_seq_len]
        
        x = self.emb(x)
        
        x = self.transformer(x, mask = mask)
        x = self.proj_out(x)
        
        return x

if __name__ == "__main__":    
    model = RNA_Model()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))

    model = RNA_Model2()
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))