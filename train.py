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

SEED = 2023
BATCH_SIZE = 16
MAX_SEQ_LEN = 206
N_SPLITS = 5 # split data into N_SPLITS; 1/N_SPLITS is used for validation
LEARNING_RATE = 1e-4
SAVE_WEIGHTS = True
EPOCHS = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# define Transformer Model
MODEL = RNA_Model(
    max_seq_len = 457,
    use_abs_pos_emb = False,
    scaled_sinu_pos_emb = True,
    attn_layers = Encoder(
        dim = 256,
        depth = 12,
        heads = 32,
        attn_dim_head = 64,
        ff_glu = True,
    )
)
MODEL.to(device)

class RNA_Dataset(Dataset):
    def __init__(self, df, mode='train'):
        self.seq_map = {'G': 0,
                        'A': 1,
                        'U': 2,
                        'C': 3}
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
        seq = np.array([self.seq_map[s] for s in seq])
        mask = torch.zeros(MAX_SEQ_LEN, dtype=torch.bool)
        mask[:len(seq)] = True
        seq = np.pad(seq,(0,MAX_SEQ_LEN-len(seq)))
        
        react = torch.from_numpy(np.stack([self.react_2A3[idx],
                                           self.react_DMS[idx]],-1))
        react_err = torch.from_numpy(np.stack([self.react_err_2A3[idx],
                                               self.react_err_DMS[idx]],-1))
        sn = torch.FloatTensor([self.sn_2A3[idx],self.sn_DMS[idx]])
        
        return (torch.from_numpy(seq), mask, react, react_err, sn)
    
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
            seq, mask, react, react_err, sn = batch

            # send data to device
            seq = seq.type(torch.int64).to(device)
            mask = mask.type(torch.bool).to(device)
            react = react.type(torch.float32).to(device)
            react_err = react_err.type(torch.float32).to(device)

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
            seq, mask, react, react_err, sn = batch

            # send data to device
            seq = seq.type(torch.int64).to(device)
            mask = mask.type(torch.bool).to(device)
            react = react.type(torch.float32).to(device)
            react_err = react_err.type(torch.float32).to(device)
            
            # validate model
            with torch.no_grad():
              prediction = MODEL(seq, mask)

              loss = loss_fn(prediction, react, mask)
              pbar.desc = f'Validation loss: {loss.item()}'

            val_losses.append(loss.item())
    val_loss = np.array(val_losses).mean()
    print("↳ Average Validation Loss: " + str(val_loss) + "\n")
    return train_loss, val_loss

# MAIN FUNCTION
if __name__ == "__main__":
    df = pd.read_csv('data/train_data.csv')

    ds_train = RNA_Dataset(df, mode='train')
    ds_val = RNA_Dataset(df, mode='eval')

    train_loader = DataLoader(ds_train,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    valid_loader = DataLoader(ds_val,
                                batch_size=BATCH_SIZE,
                                shuffle=True)

    # define loss function
    def loss_fn(pred,target,mask):
        seq_len = pred.shape[1]
        p = pred[mask[:,:seq_len]]
        y = target[mask].clip(0,1)
        loss = F.l1_loss(p, y, reduction='none')
        loss = loss[~torch.isnan(loss)].mean()
        return loss

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