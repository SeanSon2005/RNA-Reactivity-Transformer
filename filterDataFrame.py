import pandas as pd
import numpy as np
from tqdm import tqdm

class GenerateFilteredData():
    def __init__(self):
        try:
            self.df = pd.read_parquet('data/train_data.parquet')
        except:
            try:
                self.df = pd.read_csv('data/train_data.csv')
            except:
                raise Exception("data/train_data.csv not found")
            
    def generate(self, save=False):
        df = self.df.copy(deep=True)

        df_2A3 = df.loc[df.experiment_type=='2A3_MaP']
        df_DMS = df.loc[df.experiment_type=='DMS_MaP']

        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)

        df = pd.concat([df_2A3,df_DMS])

        if save:
            df.to_parquet('data/train_data_filtered.parquet')
        else:
            return df

if __name__ == "__main__":
    generateSet = GenerateFilteredData()
    generateSet.generate(save=True)

