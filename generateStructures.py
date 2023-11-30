from arnie.bpps import bpps
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

MAX_SEQ_LEN = 206
PROCESSORS = 24

def get_bp(sequence: str):
    bp_matrix = bpps(sequence,package="eternafold")
    return np.float16(bp_matrix)

if __name__ == "__main__":

    df = pd.read_parquet('data/train_data_filtered.parquet')
    df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
    sequences = df_2A3['sequence'].values

    data_len = len(df.index)
    pool = Pool(processes=PROCESSORS)

    for i in tqdm(range(0, data_len, PROCESSORS)):
        sequence_batch = sequences[i:i+PROCESSORS]
        for j, result in enumerate(pool.map(get_bp, sequence_batch)):
            np.save('data/base_pairs/bpp' + str(i+j) + '.npy',result)

# export ARNIEFILE=/home/sean/Documents/Coding/RNA/arnie_file.txt