import torch
import torch.nn as nn
from torch import Tensor
from x_transformers.x_transformers import *

import math
from functools import partial
from einops import rearrange, repeat, pack, unpack
from typing import List, Optional
import numpy as np
import torch.nn.functional as F
from embedding import CustomEmbedding

DIM = 256
DEPTH = 16
HEADS = 8

# main classes

class RNA_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pos_emb = ScaledSinusoidalEmbedding(DIM)

        self.post_emb_norm = nn.LayerNorm(DIM) # if post_emb_norm else nn.Identity()

        # attention layers
        self.attn_layers = Encoder(
            dim = DIM,
            depth = DEPTH,
            heads = HEADS,
            ff_glu = True,
            alibi_pos_bias = True,
            alibi_num_heads = HEADS,
            dropout = 0.1,
        )

        # project in and out
        self.project_in = CustomEmbedding(DIM)
        self.project_out = nn.Sequential(
            nn.Linear(DIM,2)
        )

    def forward(self, x0):

        # cut mask and seq to max_seq_len
        max_seq_len = torch.max(x0['seq_len'])
        mask = x0['mask']
        mask = mask[:,:max_seq_len]
        x = x0['seq'][:,:max_seq_len]

        # embedding and position
        x = self.project_in(x)
        #x = x + self.pos_emb(x)
        x = self.post_emb_norm(x)

        # attention
        x = self.attn_layers(x, self_attn_kv_mask = mask)
        #x = self.attn_layers_torch(x, src_key_padding_mask=~mask)

        # project out
        out = self.project_out(x)
        return out
    
    def info(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Parameter Count: " + str(params))


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

class RNA_Model2(nn.Module):
    def __init__(self, dim=384, depth=16, head_size=48, **kwargs):
        super().__init__()
        self.emb = CustomEmbedding(dim,
                                   conv = False,
                                   struct = False)
        self.pos_enc = SinusoidalPosEmb(dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
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