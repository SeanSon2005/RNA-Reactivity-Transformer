import pandas as pd
import numpy as np
from tqdm import tqdm
import polars as pl

ERROR_DROP = 0.3
ROW_MAX_NANS = 206

df = pd.read_parquet('data/train_data.parquet')

for i in tqdm(range(1, 207)):
    react_column_name = "reactivity_"+str(i).zfill(4)
    error_column_name = "reactivity_error_"+str(i).zfill(4)
    df.loc[(df[error_column_name] > ERROR_DROP), react_column_name] = np.nan

df_reactivities = df[["reactivity_"+str(i).zfill(4) for i in range(1, 207)]]
df_filter = df_reactivities.isnull().sum(axis=1)
df_filter = df_filter.apply(lambda x: False if x >= ROW_MAX_NANS else True).values
df = df[df_filter]

print("rows: ",len(df.index))

df.to_parquet('data/train_data_filtered.parquet')