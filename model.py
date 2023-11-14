import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from x_transformers.x_transformers import AttentionLayers
from x_transformers.attend import Intermediates
from x_transformers.x_transformers import *

import math
from functools import partial
from einops import rearrange, repeat, pack, unpack
import torch.nn.functional as Fumpy
from typing import List, Optional
import numpy as np


import torch
from torch import nn
import torch.nn.functional as F

from x_transformers.x_transformers import (
    AttentionLayers,
    ScaledSinusoidalEmbedding,
    AbsolutePositionalEmbedding
)

# helper functions

def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# main classes

class ContinuousTransformerWrapper(nn.Module):
    def __init__(
        self,
        *,
        max_seq_len,
        attn_layers: AttentionLayers,
        dim_in = None,
        dim_out = None,
        emb_dim = None,
        max_mem_len = 0,
        num_memory_tokens = None,
        post_emb_norm = False,
        use_abs_pos_emb = True,
        scaled_sinu_pos_emb = False
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

        self.project_in = nn.Linear(dim_in, dim)
        self.project_out = nn.Sequential(
            nn.Linear(dim,64),
            nn.Linear(64,dim_out)
        )


    def forward(
        self,
        x,
        return_embeddings = False,
        return_intermediates = False,
        return_mems = False,
        mask = None,
        return_attn = False,
        mems = None,
        pos = None,
        **kwargs
    ):
        batch = x.shape[0]

        x = self.project_in(x)
        
        x = x + self.pos_emb(x, pos = pos)

        x = self.post_emb_norm(x)

        # memory tokens

        if self.has_memory_tokens:
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

        if return_intermediates:
            return out, intermediates

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = list(map(lambda t: t[..., -self.max_mem_len:, :].detach(), hiddens))
            return out, new_mems

        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            return out, attn_maps

        return out