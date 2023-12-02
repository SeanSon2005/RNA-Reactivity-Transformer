from arnie.bpps import bpps
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz
from tqdm import tqdm
from multiprocessing import Pool

MAX_SEQ_LEN = 206
PROCESSORS = 24 
GENERATE_TEST = True


if GENERATE_TEST:
    OUT_PATH = "data/bpps_test/"
else:
    OUT_PATH = "data/bpps/"

def get_bp(data):
    sequence, sequence_id = data
    bp_matrix = np.float16(bpps(sequence,package="eternafold"))
    save_npz(OUT_PATH+sequence_id+".npz", csr_matrix(bp_matrix))

if __name__ == "__main__":
    if GENERATE_TEST:
        df = pd.read_csv('data/test_sequences.csv')
        sequences = df['sequence'].values
        sequence_ids = df['sequence_id'].values
        data_len = len(df.index)
    else:
        df = pd.read_parquet('data/train_data_filtered.parquet')
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        sequences = df_2A3['sequence'].values
        sequence_ids = df_2A3['sequence_id'].values
        data_len = len(df_2A3.index)

    print("Sequences to find: " + str(len(sequences)))

    pool = Pool(processes=PROCESSORS)

    for i in tqdm(range(0, data_len, PROCESSORS)):
        sequence_batch = [(sequences[idx],sequence_ids[idx]) for idx in range(i,min(i+PROCESSORS,data_len))]
        pool.map(get_bp, sequence_batch)

# export ARNIEFILE=/home/sean/Documents/Coding/RNA/arnie_file.txt