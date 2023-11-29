import numpy as np
import pandas as pd
import torch
import os
from dataloading import *  
from model import RNA_Model

FILE_NAME = 'weights'
PATH = 'data/'
OUT = 'runs/'
BATCH_SIZE = 64
NUM_WORKERS = 8
SEED = 2023
nfolds = 4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 25
MAX_LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.05

seed_everything(SEED)
os.makedirs(OUT, exist_ok=True)
df = pd.read_parquet(os.path.join(PATH,'train_data_filtered.parquet'))

for fold in [0]: # running multiple folds at kaggle may cause OOM
    ds_train = RNA_Dataset(df, mode='train', fold=fold, nfolds=nfolds)
    ds_train_len = RNA_Dataset(df, mode='train', fold=fold, 
                nfolds=nfolds, mask_only=True)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=BATCH_SIZE,
                drop_last=True)
    dl_train = DeviceDataLoader(torch.utils.data.DataLoader(ds_train, 
                batch_sampler=len_sampler_train, num_workers=NUM_WORKERS,
                persistent_workers=True), device)

    ds_val = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds)
    ds_val_len = RNA_Dataset(df, mode='eval', fold=fold, nfolds=nfolds, 
               mask_only=True)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=BATCH_SIZE, 
               drop_last=False)
    dl_val= DeviceDataLoader(torch.utils.data.DataLoader(ds_val, 
               batch_sampler=len_sampler_val, num_workers=NUM_WORKERS), device)
    gc.collect()

    data = DataLoaders(dl_train,dl_val)
    model = RNA_Model()  
    model = model.to(device)
    learn = Learner(data, model, loss_func=loss,cbs=[GradientClip(3.0)],
                metrics=[MAE()]).to_fp16() 

    learn.fit_one_cycle(EPOCHS, lr_max=MAX_LEARNING_RATE, wd=WEIGHT_DECAY, pct_start=0.02)
    torch.save(learn.model.state_dict(),os.path.join(OUT,f'{FILE_NAME}_{fold}.pth'))
    gc.collect()