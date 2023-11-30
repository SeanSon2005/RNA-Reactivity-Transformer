import numpy as np
import pandas as pd
import torch
import os
from dataloading import *  
from model import RNA_Model
from datetime import date

PATH = 'data/'
OUT = 'runs/'
BATCH_SIZE = 64
NUM_WORKERS = 2
SEED = 2023
nfolds = 4
WEIGHT_DECAY = 0.05

class TrainRecorder(Callback):
    order=60
    def __init__(self, **kwargs):
        self.best_mae = None
        path = (OUT+"run_"+str(len(next(os.walk(OUT))[1]))+"/")
        os.mkdir(path)
        fname = path+"report.csv"
        self.weight_path = path+"best.pth"
        self.fname,self.append = Path(fname),False
        with open(path+"hyper_params.txt","w") as param_file:
            info_txt = ""
            for key in kwargs.keys(): 
                info_txt += (key + ": " + str(kwargs[key]) + "\n")
            param_file.write(info_txt)


    def read_log(self):
        return pd.read_csv(self.path/self.fname)

    def before_fit(self):
        if hasattr(self, "gather_preds"): return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = (self.path/self.fname).open('a' if self.append else 'w')
        self.file.write(','.join(self.recorder.metric_names) + '\n')
        self.old_logger,self.learn.logger = self.logger,self._write_line

    def _write_line(self, log):
        accuracy = log[3]
        if self.best_mae is None or accuracy < self.best_mae:
            self.best_mae = accuracy
            torch.save(self.model.state_dict(),self.weight_path)
        self.file.write(','.join([str(t) for t in log]) + '\n')
        self.file.flush()
        os.fsync(self.file.fileno())
        self.old_logger(log)

    def after_fit(self):
        if hasattr(self, "gather_preds"): return
        self.file.close()
        self.learn.logger = self.old_logger

def train_model(device='cuda', **kwargs):
    seed_everything(SEED)
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_parquet(os.path.join(PATH,'train_data.parquet'))

    for fold in [0]:
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
        model = RNA_Model(**kwargs)
        model = model.to(device)
        learn = Learner(data, model, loss_func=loss,cbs=[GradientClip(3.0)],
                    metrics=[MAE()]).to_fp16() 
        
        max_lr = kwargs['max_lr']
        epochs_per_cycle = kwargs['epochs_per_cycle']
        num_cycles = kwargs['num_cycles']

        for cycle_idx in range(num_cycles):
            print("Cycle: " + str(cycle_idx+1))
            learn.fit_one_cycle(epochs_per_cycle, lr_max=max_lr, wd=WEIGHT_DECAY, pct_start=0.02, cbs=TrainRecorder(**kwargs))
        gc.collect()

if __name__ == "__main__":
    with open("trainParams.txt", "r") as train_file:
        for train_params in train_file:
            max_lr, num_cycles, epochs_per_cycle, dim, depth, heads, dropout = train_params.split()
            train_model(max_lr = float(max_lr), 
                        num_cycles = int(num_cycles), 
                        epochs_per_cycle = int(epochs_per_cycle), 
                        dim = int(dim), 
                        depth = int(depth), 
                        heads = int(heads),
                        dropout = int(dropout), 
                        conv_kernel_size=0)