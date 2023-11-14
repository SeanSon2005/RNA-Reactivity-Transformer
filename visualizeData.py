import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_COUNT = 10

df = pd.read_csv('data/train_data.csv', nrows = DATA_COUNT)
df.head()

input_RNA = np.stack(df["sequence"])
output_react = np.nan_to_num(df[["reactivity_"+str(i).zfill(4) for i in range(1, 171)]].values)

rna_seq = input_RNA[0]
reactivities = output_react[0]

plt.pcolormesh([reactivities]*2, cmap='Greys', shading='gouraud')
plt.show()


