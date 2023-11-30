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
        if save:
            df.to_parquet('data/train_data.parquet')
        else:
            return df

if __name__ == "__main__":
    generateSet = GenerateFilteredData()
    generateSet.generate(save=True)

