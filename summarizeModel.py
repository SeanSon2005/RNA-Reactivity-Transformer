import torch
from x_transformers import Encoder, Decoder
from model import ContinuousTransformerWrapper
import numpy as np

VECTOR_SIZE = 4
EXPANSION_FACTOR = 16
MAX_SEQ_LENGTH = 457

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ContinuousTransformerWrapper(
    dim_in = VECTOR_SIZE,
    dim_out = 1,
    max_seq_len = MAX_SEQ_LENGTH,
    use_abs_pos_emb = False,
    use_CNN = True,
    attn_layers = Decoder(
        dim = (VECTOR_SIZE * EXPANSION_FACTOR),
        depth = 48,
        heads = 64,
        attn_dim_head = 256,
        rotary_xpos = True,
        ff_glu = True,
    )
  )

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print("Parameter Count: " + str(params))