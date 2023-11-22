import pandas as pd
df = pd.read_csv('data/train_data.csv')
df.to_parquet('data/train_data.parquet')