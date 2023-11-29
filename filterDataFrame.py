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
            
    def generate(self, sn_drop, error_drop, row_max_nans=206, save=False):
        df = self.df.copy(deep=True)

        for i in tqdm(range(1, 207)):
            react_column_name = "reactivity_"+str(i).zfill(4)
            error_column_name = "reactivity_error_"+str(i).zfill(4)
            signal_to_noise = (df[react_column_name] / df[error_column_name]).values
            m = (signal_to_noise < sn_drop) | (df[error_column_name].values > error_drop)
            df.loc[m, react_column_name] = np.nan
            df.loc[m, error_column_name] = np.nan

        df_reactivities = df[["reactivity_"+str(i).zfill(4) for i in range(1, 207)]]
        df_filter = df_reactivities.isnull().sum(axis=1)
        df_filter = df_filter.apply(lambda x: False if x >= row_max_nans else True).values
        df = df[df_filter]

        print("rows: ",len(df.index))
        print("usable targets:", df_reactivities.count(axis=1).sum())

        if save:
            df.to_parquet('data/train_data_filtered.parquet')
        else:
            return df

#best performance so far from SN_DROP=0, ERROR_DROP=0.08
if __name__ == "__main__":
    generateSet = GenerateFilteredData()
    generateSet.generate(sn_drop=4, error_drop=1, save=True)

