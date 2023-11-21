import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from x_transformers import Encoder, Decoder
from model import RNA_Model
from helpers import Plotter, WebHook
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
import random

SEED = 2023
BATCH_SIZE = 64
MAX_SEQ_LEN = 206
N_SPLITS = 5 # split data into N_SPLITS; 1/N_SPLITS is used for validation
LEARNING_RATE = 5e-4
SAVE_WEIGHTS = True
EPOCHS = 50
LOAD_BEST = False
NUM_WORKERS = 8

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# define Transformer Model
MODEL = RNA_Model(
    use_abs_pos_emb = False,
    scaled_sinu_pos_emb = True,
    attn_layers = Encoder(
        dim = 192,
        depth = 12,
        heads = 6,
        attn_dim_head = 64,
        ff_glu = True,
    )
)
MODEL.to(device)

class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train', mask_only=False):
        self.mask_only = mask_only

        df['L'] = df.sequence.apply(len) #adds new column for length of sequence
        df_2A3 = df.loc[df.experiment_type=='2A3_MaP'] # get rows of experiment 2A3_MaP
        df_DMS = df.loc[df.experiment_type=='DMS_MaP'] # get rows of experiment DMS_MaP
        
        # gets indices for splitting train and validation
        split = list(KFold(n_splits=N_SPLITS, random_state=SEED,
                shuffle=True).split(df_2A3))[0][0 if mode=='train' else 1]
        
        # applys split data to the rows
        df_2A3 = df_2A3.iloc[split].reset_index(drop=True)
        df_DMS = df_DMS.iloc[split].reset_index(drop=True)
        
        # filter rows where SN_filter is greater than 0
        m = (df_2A3['SN_filter'].values > 0) & (df_DMS['SN_filter'].values > 0)
        df_2A3 = df_2A3.loc[m].reset_index(drop=True)
        df_DMS = df_DMS.loc[m].reset_index(drop=True)
        
        # get sequences
        self.seq_map = {'A':0,'C':1,'G':2,'U':3}
        self.seq = df_2A3['sequence'].values
        self.L = df_2A3['L'].values
        
        self.react_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_0' in c]].values
        self.react_DMS = df_DMS[[c for c in df_DMS.columns if \
                                 'reactivity_0' in c]].values
        self.react_err_2A3 = df_2A3[[c for c in df_2A3.columns if \
                                 'reactivity_error_0' in c]].values
        self.react_err_DMS = df_DMS[[c for c in df_DMS.columns if \
                                'reactivity_error_0' in c]].values
        self.sn_2A3 = df_2A3['signal_to_noise'].values
        self.sn_DMS = df_DMS['signal_to_noise'].values
        
    def __len__(self):
        return len(self.seq)  
    
    def __getitem__(self, idx):
        seq = self.seq[idx]
        if self.mask_only:
            mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
            mask[:len(seq)] = True
            return {'mask':mask},{'mask':mask}
        seq = [self.seq_map[s] for s in seq]
        seq = np.array(seq)
        mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,MAX_SEQ_LEN-len(seq)))
        
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])
        
        return {'seq':torch.from_numpy(seq), 'mask':mask}, \
               {'react':react, 'react_err':react_err,
                'sn':sn, 'mask':mask}

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            s = self.sampler.data_source[idx]
            if isinstance(s,tuple): L = s[0]["mask"].sum()
            else: L = s["mask"].sum()
            L = max(1,L // 16) 
            if len(buckets[L]) == 0:  buckets[L] = []
            buckets[L].append(idx)
            
            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []
                
        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

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
    
def training_loop(train_batches, valid_batches, epoch):
    print("EPOCH: "+ str(epoch))

    train_losses = []
    val_losses = []

    #set model to training mode (enable dropouts, etc)
    MODEL.train()

    ema_loss = None
    with tqdm(train_batches, bar_format='{l_bar}{bar:70}{r_bar}{bar:-70b}') as pbar:
        for batch in pbar:
            # handle data
            src, tgt = batch

            # send data to device
            seq = src['seq']
            mask = src['mask']
            react = tgt['react']
            #react_err = tgt['react_err'].type(torch.float32).to(device)

            # zero the gradients
            optim.zero_grad()

            # make predictions
            prediction = MODEL(seq, mask)
  
            # calc loss and backprop
            loss = loss_fn(prediction, react, mask)
            loss.backward()

            # step
            optim.step()
            
            # update progress bar and EMA loss
            if ema_loss is None: ema_loss = loss.item()
            else: ema_loss = ema_loss * 0.9 + loss.item() * 0.1
            pbar.desc =  f'Training loss: {ema_loss}'
            pbar.update()
            train_losses.append(loss.item())

    train_loss = np.array(train_losses).mean()
    print("↳ Average Training Loss: " + str(train_loss))

    #set model to evalution mode (disable dropouts, etc)
    MODEL.eval()

    with tqdm(valid_batches, bar_format='{l_bar}{bar:70}{r_bar}{bar:-70b}') as pbar:
        for batch in pbar:
            # handle data
            src, tgt = batch

            # send data to device
            seq = src['seq']
            mask = src['mask']
            react = tgt['react']
            #react_err = tgt['react_err'].type(torch.float32).to(device)
            
            # validate model
            with torch.no_grad():
              prediction = MODEL(seq, mask)

              loss = loss_fn(prediction, react, mask)
              pbar.desc = f'Validation loss: {loss.item()}'

            val_losses.append(loss.item())
    val_loss = np.array(val_losses).mean()
    print("↳ Average Validation Loss: " + str(val_loss) + "\n")
    return train_loss, val_loss
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

# MAIN FUNCTION
if __name__ == "__main__":
    seed_everything(SEED)

    df = pd.read_csv('data/train_data.csv')

    ds_train = RNA_Dataset(df, mode='train')
    ds_train_len = RNA_Dataset(df, mode='train', mask_only=True)
    sampler_train = torch.utils.data.RandomSampler(ds_train_len)
    len_sampler_train = LenMatchBatchSampler(sampler_train, batch_size=BATCH_SIZE,
                                             drop_last=True)
    train_loader = DeviceDataLoader(DataLoader(ds_train, 
                batch_sampler=len_sampler_train, num_workers=NUM_WORKERS,
                persistent_workers=True), device)
    ds_val = RNA_Dataset(df, mode='eval')
    ds_val_len = RNA_Dataset(df, mode='eval',mask_only=True)
    sampler_val = torch.utils.data.SequentialSampler(ds_val_len)
    len_sampler_val = LenMatchBatchSampler(sampler_val, batch_size=BATCH_SIZE,
                                           drop_last=False)
    valid_loader = DeviceDataLoader(DataLoader(ds_val, 
               batch_sampler=len_sampler_val, num_workers=NUM_WORKERS), device)

    # define loss function
    def loss_fn(pred,target,mask):
        seq_len = pred.shape[1]
        p = pred[mask[:,:seq_len]]
        y = target[mask].clip(0,1)
        loss = F.l1_loss(p, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss
    
    if LOAD_BEST:
        MODEL.load_state_dict(torch.load("best.pt"))
    optim = torch.optim.Adam(MODEL.parameters(), lr=LEARNING_RATE)
    print("Begining training...\n")

    # training loop
    train_loss_points = []
    val_loss_points = []

    discord_webhook = WebHook()
    discord_webhook.sendMessage("Beginning Training...",
                                include_date=True)
    best = None
    for epoch in range(EPOCHS):
        train_loss, val_loss = training_loop(train_loader,
                                            valid_loader,
                                            epoch+1)

        train_loss_points.append(train_loss)
        val_loss_points.append(val_loss)
        discord_webhook.anounceEpoch(epoch+1,
                            train_loss=train_loss,
                            val_loss=val_loss)
        if SAVE_WEIGHTS:
            if best == None or val_loss < best:
                best = val_loss
                print("____________________\nBest Weights Updated\n____________________")
                torch.save(MODEL.state_dict(), "best.pt")
            torch.save({
            'epoch': epoch,
            'model_state_dict': MODEL.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            }, "resume.pt")

    info_plotter = Plotter(EPOCHS)
    info_plotter.save_loss(train_loss_points, val_loss_points)
    discord_webhook.sendMessage("Finished Training!",
                                include_date=True)