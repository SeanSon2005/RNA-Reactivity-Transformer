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

# main classes

class RNA_Model(nn.Module):
    def __init__(
        self,
        *,
        attn_layers: AttentionLayers,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
    ):
        super().__init__()
        dim = attn_layers.dim

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)

        self.post_emb_norm = nn.LayerNorm(dim) if post_emb_norm else nn.Identity()

        # memory tokens

        num_memory_tokens = default(num_memory_tokens, 0)
        self.has_memory_tokens = num_memory_tokens > 0

        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

        # attention layers

        self.attn_layers = attn_layers

        # project in and out
        self.project_in = nn.Embedding(4,dim)
        self.project_out = nn.Sequential(
            nn.Linear(dim,2),
        )

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=6, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), attn_layers.depth)

    def forward(
        self,
        seq,
        seq_mask,
        pos = None,
        **kwargs
    ):

        # mask out nulls in data
        Lmax = seq_mask.sum(-1).max()
        seq_mask = seq_mask[:,:Lmax]
        x = seq[:,:Lmax]

        # embeddings
        x = self.project_in(x)
        
        x = x + self.pos_emb(x, pos = pos)

        x = self.post_emb_norm(x)

        x = self.transformer(x, src_key_padding_mask=~seq_mask)
        #x, _ = self.attn_layers(x, attn_mask = seq_mask, mems = mems, return_hiddens = True, **kwargs)

        out = self.project_out(x)
        return out