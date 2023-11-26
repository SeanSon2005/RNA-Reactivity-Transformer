from arnie.mfe import mfe
import numpy as np
import pandas as pd

df = pd.read_parquet('data/train_data.parquet')
sequences = df['sequence'].values

for sequence in sequences:
    structure = mfe(sequence,package="eternafold")
    print(structure)


    

#export ETERNAFOLD_PATH=/home/sean/Documents/Coding/RNA/EternaFold/src
#export ETERNAFOLD_PARAMETERS=/home/sean/Documents/Coding/RNA/EternaFold/parameters/EternaFoldParams.v1