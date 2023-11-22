import numpy as np
import pandas as pd
from tqdm import tqdm
#from x_transformers import Decoder
#from model import ContinuousTransformerWrapper
import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader
import polars as pl

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim=16, M=10000):
        super().__init__()
        self.dim = dim
        self.M = M

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(self.M) / half_dim
        emb = torch.exp(torch.arange(half_dim, device=device) * (-emb))
        emb = x[...,None] * emb[None,...]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RNA_Model(nn.Module):
    def __init__(self, dim=256, depth=16, head_size=32, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(4,dim)
        self.pos_enc = SinusoidalPosEmb(dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=dim//head_size, dim_feedforward=4*dim,
                dropout=0.1, activation=nn.GELU(), batch_first=True, norm_first=True), depth)
        self.proj_out = nn.Linear(dim,2)
    
    def forward(self, x0):
        mask = x0['mask']
        Lmax = mask.sum(-1).max()
        mask = mask[:,:Lmax]
        x = x0['seq'][:,:Lmax]
        
        pos = torch.arange(Lmax, device=x.device).unsqueeze(0)
        pos = self.pos_enc(pos)
        x = self.emb(x)
        x = x + pos
        
        x = self.transformer(x, src_key_padding_mask=~mask)
        x = self.proj_out(x)
        
        return x

#define device (GPU!)
device = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SEQ_LEN = 457

class Test_Dataset(Dataset):
    def __init__(self, df):
        df['L'] = df.sequence.apply(len)
        self.lens = df['L'].values
        seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.seqs = (df["sequence"].apply(lambda x: [seq_map[i] for i in x])).values
    def __len__(self):
        return len(self.seqs)  
    
    def __getitem__(self, idx):
        seq = np.array(self.seqs[idx])
        seq = np.pad(seq,(0,MAX_SEQ_LEN-self.lens[idx]))
        seq = torch.from_numpy(seq)
        mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        mask[:self.lens[idx]] = True
        return {'seq':seq.to(device),'mask':mask.to(device),
                'seq_len':torch.tensor(self.lens[idx]).to(device)}
    
model = RNA_Model()
model.load_state_dict(torch.load("runs/best1.pth"))
model.to(device)
df = pd.read_csv("data/test_sequences.csv")
df_len = len(df.index)

dataset = Test_Dataset(df)
dataloader = DataLoader(dataset, batch_size=512, shuffle=False, drop_last=False)

SUB_LEN = 269796671
ids = np.int64(np.arange(SUB_LEN))
preds_dms = np.zeros(SUB_LEN)
preds_2a3 = np.zeros(SUB_LEN)

ind = 0
for data in tqdm(dataloader):
    with torch.no_grad():
        pred = model(data)
        output = pred[data['mask'][:,:pred.shape[1]]].cpu().numpy()
        output_ind = ind + output.shape[0]
        preds_dms[ind:output_ind] = output[:,1]
        preds_2a3[ind:output_ind] = output[:,0]
        ind = output_ind

schema = {k:pl.Float32 for k in ['id', 'reactivity_DMS_MaP', 'reactivity_2A3_MaP']}
schema['id'] = pl.Int64

df = pl.DataFrame(
    data=[ids, preds_dms, preds_2a3],
    schema=schema
)
df.write_csv('submissions/submission.csv', float_precision=4)
        