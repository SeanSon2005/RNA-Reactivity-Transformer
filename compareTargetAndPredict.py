import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from x_transformers import Decoder
from train import MODEL

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
df_2A3 = df.loc[df.experiment_type=='2A3_MaP']

# get the RNA sequences
mapping = {'G': 0,
            'A': 1,
            'U': 2,
            'C': 3
        }
input_sequences = (df_2A3["sequence"].apply(lambda x: np.array([mapping[i] for i in x]))).values
output_react = np.nan_to_num(df_2A3[["reactivity_"+str(i).zfill(4) for i in range(1, TRAINING_SEQ_LENGTH+1)]].values)

seq = input_sequences[0]
mask = torch.zeros(206, dtype=torch.bool)
mask[:len(seq)] = True
react = np.round(output_react[0],3)

#define device (GPU!)
device = "cuda" if torch.cuda.is_available() else "cpu"

#define Transformer Model
MODEL.to(device)

rna_seq = torch.Tensor(seq).type(torch.int64).unsqueeze(axis=0).to(device)
mask = mask.unsqueeze(axis=0).to(device)
tgt_pred = np.round(MODEL(rna_seq,mask).squeeze().detach().cpu().numpy(),3)

print(react)
print(tgt_pred[:,0])

