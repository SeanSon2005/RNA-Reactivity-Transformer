import pandas as pd

SN_DROP = 0.5

df = pd.read_parquet('data/train_data.parquet')

df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
df_DMS = df.loc[df.experiment_type=='DMS_MaP']
m = (df_2A3['SN_filter'].values > SN_DROP) & (df_DMS['SN_filter'].values > SN_DROP)
df_2A3 = df_2A3.loc[m].reset_index(drop=True)
df_DMS = df_DMS.loc[m].reset_index(drop=True)

df.to_parquet('data/train_data_F.parquet')