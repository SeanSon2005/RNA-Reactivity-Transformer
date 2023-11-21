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
        max_seq_len,
        attn_layers: AttentionLayers,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False,
    ):
        super().__init__()
        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.max_mem_len = max_mem_len

        if not (use_abs_pos_emb and not attn_layers.has_pos_emb):
            self.pos_emb = always(0)
        elif scaled_sinu_pos_emb:
            self.pos_emb = ScaledSinusoidalEmbedding(dim)
        else:
            self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len)

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
            nn.Linear(dim,32),
            nn.Linear(32,2)
        )

    def forward(
        self,
        seq,
        seq_mask,
        return_embeddings = False,
        mask = None,
        mems = None,
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

        # memory tokens

        if self.has_memory_tokens:
            batch = x.shape[0]
            m = repeat(self.memory_tokens, 'm d -> b m d', b = batch)
            x, mem_ps = pack([m, x], 'b * d')

            if exists(mask):
                num_mems = m.shape[-2]
                mask = pad_at_dim(mask, (num_mems, 0), dim = -1, value = True)

        # attention layers

        x, intermediates = self.attn_layers(x, mask = mask, mems = mems, return_hiddens = True, **kwargs)

        # splice out memory tokens

        if self.has_memory_tokens:
            m, x = unpack(x, mem_ps, 'b * d')
            intermediates.memory_tokens = m

        out = self.project_out(x) if not return_embeddings else x
        return out