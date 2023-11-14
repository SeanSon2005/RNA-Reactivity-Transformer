import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from x_transformers import Decoder
from model import ContinuousTransformerWrapper
import csv

# constants
BATCH_SIZE = 16
LEARNING_RATE = 8e-4
EPOCHS = 1
NUM_TOKENS = 4
MAX_SEQ_LENGTH = 457 #max in test sequences

#define device (GPU!)
device = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv("data/test_sequences.csv")

mapping = {'G': [1,0,0,0],
        'A': [0,1,0,0],
        'U': [0,0,1,0],
        'C': [0,0,0,1]}

input_sequences = df["sequence"].apply(lambda x: torch.Tensor([mapping[i] for i in x]).to(device))

model = ContinuousTransformerWrapper(
    dim_in = NUM_TOKENS,
    dim_out = 1,
    max_seq_len = MAX_SEQ_LENGTH,
    attn_layers = Decoder(
        dim = 512,
        depth = 12,
        heads = 8,
        rotary_xpos = True,
        #dynamic_pos_bias = True, 
        #dynamic_pos_bias_log_distance = False 
    )
  )
model.load_state_dict(torch.load("best.pt"))
model.to(device)

with open("submission.csv", "a") as csv_file:
    writer = csv.writer(csv_file, delimiter=',')
    writer.writerow(['id','reactivity_DMS_MaP','reactivity_2A3_MaP'])
    start_idx = 0
    for input_sequence in tqdm(input_sequences):
        # Forward pass
        with torch.no_grad():
            output = model(input_sequence.unsqueeze(axis=0))
            output_list = output.squeeze().cpu()
            writer.writerows([[start_idx + index, i.item(), i.item()] for index, i in enumerate(output_list)])
            start_idx += len(output_list)