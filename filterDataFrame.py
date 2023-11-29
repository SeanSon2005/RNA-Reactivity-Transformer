import pandas as pd
import numpy as np
from tqdm import tqdm

ERROR_DROP = 1.5
ROW_MAX_NANS = 206


try:
    df = pd.read_parquet('data/train_data.parquet')
except:
    try:
        df = pd.read_csv('data/train_data.csv')
    except:
        raise Exception("data/train_data.csv not found")

for i in tqdm(range(1, 207)):
    react_column_name = "reactivity_"+str(i).zfill(4)
    error_column_name = "reactivity_error_"+str(i).zfill(4)
    df.loc[(df[error_column_name] > ERROR_DROP), react_column_name] = np.nan
    df.loc[(df[error_column_name] > ERROR_DROP), error_column_name] = np.nan

df_reactivities = df[["reactivity_"+str(i).zfill(4) for i in range(1, 207)]]
df_filter = df_reactivities.isnull().sum(axis=1)
df_filter = df_filter.apply(lambda x: False if x >= ROW_MAX_NANS else True).values
df = df[df_filter]

print("rows: ",len(df.index))
print("number of not passing SN_filter: " + str(len(df.loc[(df['signal_to_noise']<1)].index)))

df.to_parquet('data/train_data_filtered.parquet')