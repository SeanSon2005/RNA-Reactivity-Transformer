import pandas as pd
import os, gc
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl
from model import RNA_Model
from scipy.sparse import csr_matrix, load_npz

MODELS = ['runs/run_3/best.pth']
PATH = 'data/'
bs = 512
num_workers = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNA_Dataset_Test(Dataset):
    def __init__(self, df, mask_only=False, **kwargs):
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        df['L'] = df.sequence.apply(len)
        self.Lmax = df['L'].max()
        self.df = df
        self.mask_only = mask_only
        
    def __len__(self):
        return len(self.df)  
    
    def __getitem__(self, idx):
        id_min, id_max, seq, seq_id = self.df.loc[idx, ['id_min','id_max','sequence','sequence_id']]
        mask = torch.zeros(self.Lmax, dtype=torch.bool)
        L = len(seq)
        mask[:L] = True
        if self.mask_only: return {'mask':mask},{}
        ids = np.arange(id_min,id_max+1)
        
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        pad_amnt = self.Lmax-L
        seq = np.pad(seq,(0,pad_amnt))
        ids = np.pad(ids,(0,pad_amnt), constant_values=-1)

        bpp_mat = load_npz('data/bpps_test/'+seq_id+'.npz').toarray()
        bpps = torch.from_numpy(np.pad(bpp_mat,((0,pad_amnt),(0,pad_amnt)),'constant'))
        
        return {'seq':torch.from_numpy(seq),'seq_len':torch.tensor(L),'mask':mask,'bpps':bpps}, \
               {'ids':ids}
            
def dict_to(x, device='cuda'):
    return {k:x[k].to(device) for k in x}

def to_device(x, device='cuda'):
    return tuple(dict_to(e,device) for e in x)

class DeviceDataLoader:
    def __init__(self, dataloader, device='cuda'):
        self.dataloader = dataloader
        self.device = device
    
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        for batch in self.dataloader:
            yield tuple(dict_to(x, self.device) for x in batch)
    
df_test = pd.read_csv(os.path.join(PATH,'test_sequences.csv'))
ds = RNA_Dataset_Test(df_test)
dl = DeviceDataLoader(DataLoader(ds, batch_size=bs, 
               shuffle=False, drop_last=False, num_workers=num_workers), device)
del df_test
gc.collect()

models = []
for m in MODELS:
    model = RNA_Model(dim=768, depth=16, heads=8, conv_kernel_size=0, dropout=0.1)
    model = model.to(device)
    model.load_state_dict(torch.load(m,map_location=torch.device('cpu')))
    model.eval()
    models.append(model)

ids,preds = [],[]
for x,y in tqdm(dl):
    with torch.no_grad(),torch.cuda.amp.autocast():
        p = torch.stack([torch.nan_to_num(model(x)) for model in models]
                        ,0).mean(0).clip(0,1)
        
    for idx, mask, pi in zip(y['ids'].cpu(), x['mask'].cpu(), p.cpu()):
        ids.append(idx[mask])
        preds.append(pi[mask[:pi.shape[0]]])

ids = torch.concat(ids)
preds = torch.concat(preds)

schema = {k:pl.Float32 for k in ['id', 'reactivity_DMS_MaP', 'reactivity_2A3_MaP']}
schema['id'] = pl.Int64
df = pl.DataFrame(
    data=[ids.numpy(), preds[:,1].numpy(), preds[:,0].numpy()],
    schema=schema
)
df.write_parquet('submissions/submission.parquet')
        