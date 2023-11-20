import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from x_transformers import Decoder
from model import ContinuousTransformerWrapper

# constants
DATA_COUNT = 1000
BATCH_SIZE = 32
LEARNING_RATE = 8e-3
EPOCHS = 30
VECTOR_SIZE = 4
EXPANSION_FACTOR = 1
MAX_SEQ_LENGTH = 457
VALID_PERCENT = 0.1

# Filter Data
TRAINING_SEQ_LENGTH = 177

DATA_COUNT = 10

df = pd.read_csv('data/train_data_quick.csv', nrows = DATA_COUNT)
df.head()

# get the RNA sequences
mapping = {'G': [1,0,0,0],
            'A': [0,1,0,0],
            'U': [0,0,1,0],
            'C': [0,0,0,1]
        }
input_sequences = (df["sequence"].apply(lambda x: np.array([mapping[i] for i in x]))).values
output_react = np.nan_to_num(df[["reactivity_"+str(i).zfill(4) for i in range(1, TRAINING_SEQ_LENGTH+1)]].values)

reactivities = output_react[0]

#define device (GPU!)
device = "cuda" if torch.cuda.is_available() else "cpu"

#define Transformer Model
model = ContinuousTransformerWrapper(
    dim_in = VECTOR_SIZE,
    dim_out = 1,
    max_seq_len = MAX_SEQ_LENGTH,
    use_abs_pos_emb = False,
    attn_layers = Decoder(
        dim = (VECTOR_SIZE * EXPANSION_FACTOR),
        depth = 24,
        heads = 8,
        attn_dim_head = 128,
        rotary_xpos = True,
        ff_glu = True,
    )
  )
model.to(device)

rna_seq = torch.Tensor(input_sequences[0]).unsqueeze(axis=0).to(device)
tgt_pred = np.round(model(rna_seq).squeeze().detach().cpu().numpy(),3)

print(reactivities)
print(tgt_pred)

