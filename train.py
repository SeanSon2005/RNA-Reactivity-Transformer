import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import torch.nn as nn
from x_transformers import Encoder, Decoder
from model import ContinuousTransformerWrapper
from sklearn.model_selection import train_test_split
from helpers import Plotter, WebHook

# constants
DATA_COUNT = 335617
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
EPOCHS = 50
VECTOR_SIZE = 4
EXPANSION_FACTOR = 16
MAX_SEQ_LENGTH = 457
VALID_PERCENT = 0.1

# Change for your needs
LOAD_MODEL_STATES = False
MODE = 0
SAVE_WEIGHTS = True
RESUME_TRAINING = False

# Filter Data
TRAINING_SEQ_LENGTH = 177

class StatementDataset(torch.utils.data.Dataset):
  def __init__(self, statements, labels):
    self.statements = statements
    self.labels = labels

  def __getitem__(self, index):
    statement = self.statements[index]
    labels = self.labels[index]
    example = (statement, labels)
    return example

  def __len__(self):
    return len(self.statements)

def training_loop(train_batches, valid_batches, epoch, weighted_MAE):
    print("EPOCH: "+ str(epoch))

    train_losses = []
    val_losses = []

    #set model to training mode (enable dropouts, etc)
    model.train()

    ema_loss = None
    with tqdm(train_batches, bar_format='{l_bar}{bar:80}{r_bar}{bar:-80b}') as pbar:
        for batch in pbar:
            # handle data
            src, tgt = batch
            tgt_react, mask, tgt_error, experiment = tgt

            # send data to device
            src = src.type(torch.float32).to(device)
            tgt_react = tgt_react.type(torch.float32).to(device)
            mask = mask.type(torch.bool).to(device)
            tgt_error = tgt_error.type(torch.float32).to(device)

            # zero the gradients
            optim.zero_grad()

            # make predictions
            tgt_pred = model(src)

            # mask null reactivities
            pred_masked = torch.masked_select(tgt_pred.squeeze(), mask)
            tgt_masked = torch.masked_select(tgt_react, mask)
            err_masked = torch.masked_select(tgt_error, mask)
  
            # calc loss and backprop
            loss = weighted_MAE(pred_masked, tgt_masked, err_masked)
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
    model.eval()

    with tqdm(valid_batches, bar_format='{l_bar}{bar:80}{r_bar}{bar:-80b}') as pbar:
        for batch in pbar:
            # handle data
            src, tgt = batch
            tgt_react, mask, tgt_error, experiment = tgt

            # send data to device
            src = src.type(torch.float32).to(device)
            tgt_react = tgt_react.type(torch.float32).to(device)
            mask = mask.type(torch.bool).to(device)
            tgt_error = tgt_error.type(torch.float32).to(device)
            
            # validate model
            with torch.no_grad():
              tgt_pred = model(src)

              pred_masked = torch.masked_select(tgt_pred.squeeze(), mask)
              tgt_masked = torch.masked_select(tgt_react, mask)
              err_masked = torch.masked_select(tgt_error, mask)

              loss = weighted_MAE(pred_masked, tgt_masked, err_masked)
              pbar.desc = f'Validation loss: {loss.item()}'

            val_losses.append(loss.item())
    val_loss = np.array(val_losses).mean()
    print("↳ Average Validation Loss: " + str(val_loss) + "\n")
    return train_loss, val_loss
  
# MAIN FUNCTION
if __name__ == "__main__":

  df = pd.read_csv('data/train_data_quick.csv', nrows = DATA_COUNT)
  df.head()

  # filter rows based on user request
  df = df.loc[df['sequence'].str.len() == TRAINING_SEQ_LENGTH]
  DATA_USABLE_COUNT = len(df.index)
  print("Using " + str(DATA_USABLE_COUNT) + " rows")

  # get the RNA sequences
  mapping = {'G': [1,0,0,0],
             'A': [0,1,0,0],
             'U': [0,0,1,0],
             'C': [0,0,0,1]
            }
  input_sequences = (df["sequence"].apply(lambda x: np.array([mapping[i] for i in x]))).values

  # get reactivity error
  output_error = np.nan_to_num(df[["reactivity_error_"+str(i).zfill(4) for i in range(1, TRAINING_SEQ_LENGTH+1)]].values)
  

  # get experiment type
  mapping = {
    "2A3_MaP":False,
    "DMS_MaP":True
  }
  output_experiment = df["experiment_type"].apply(lambda x: mapping[x]).values

  # get null mask and set NaNs to 0 for training
  react_df = df[["reactivity_"+str(i).zfill(4) for i in range(1, TRAINING_SEQ_LENGTH+1)]].values
  output_null_sequences = np.isfinite(react_df)
  output_react_sequences = np.nan_to_num(react_df)

  # construct input and output lists to split for training and validation
  output_sequences = []
  for i in range(DATA_USABLE_COUNT):
    output_sequences.append([output_react_sequences[i],
                             output_null_sequences[i],
                             output_error[i],
                             output_experiment[i]])

  X_train, X_test, y_train, y_test = train_test_split(input_sequences,
                                                      output_sequences, 
                                                      test_size = VALID_PERCENT)
    
  train_loader = torch.utils.data.DataLoader(StatementDataset(X_train, y_train),
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)
  valid_loader = torch.utils.data.DataLoader(StatementDataset(X_test, y_test),
                                             batch_size=BATCH_SIZE,
                                             shuffle=True)

  print("Data is ready")

  #define device (GPU!)
  device = "cuda" if torch.cuda.is_available() else "cpu"

  #define Transformer Model
  model = ContinuousTransformerWrapper(
    dim_in = VECTOR_SIZE,
    dim_out = 1,
    max_seq_len = MAX_SEQ_LENGTH,
    use_abs_pos_emb = False,
    use_CNN = True,
    attn_layers = Decoder(
        dim = (VECTOR_SIZE * EXPANSION_FACTOR),
        depth = 24,
        heads = 16,
        attn_dim_head = 256,
        rotary_xpos = True,
        ff_glu = True,
    )
  )

  # define loss function and optimizer
  loss_MAE = torch.nn.MSELoss()
  def loss_fn(pred, target, weights):
    return loss_MAE(pred,target)
  optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

  if RESUME_TRAINING:
    print("All Checkpoints Loaded")
    model.load_state_dict(torch.load("resume.pt")['model_state_dict'])
    optim.load_state_dict(torch.load("resume.pt")['optimizer_state_dict'])
  elif LOAD_MODEL_STATES: #load weights if existing and desired
    print("Model Weights Loaded")
    model.load_state_dict(torch.load("best.pt"))

  model.to(device)

  print("Begining training...\n")

  # training loop
  train_loss_points = []
  val_loss_points = []
  if MODE == 0:
    discord_webhook = WebHook()
    discord_webhook.sendMessage("Beginning Training...",
                                include_date=True)
    best = None
    for epoch in range(EPOCHS):
      train_loss, val_loss = training_loop(train_loader,
                                           valid_loader,
                                           epoch+1,
                                           loss_fn)

      train_loss_points.append(train_loss)
      val_loss_points.append(val_loss)
      discord_webhook.anounceEpoch(epoch+1,
                           train_loss=train_loss,
                           val_loss=val_loss)

      if best == None or val_loss < best:
        best = val_loss
        if SAVE_WEIGHTS:
          print("____________________\nBest Weights Updated\n____________________")
          torch.save(model.state_dict(), "best.pt")
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, "resume.pt")

    info_plotter = Plotter(EPOCHS)
    info_plotter.save_loss(train_loss_points, val_loss_points)
    discord_webhook.sendMessage("Finished Training!",
                                include_date=True)

  elif MODE == 1:
    # handle data
    tgt_react, mask = output_sequences[0]
    src_RNA = input_sequences[0]

    src_RNA = torch.tensor(src_RNA).unsqueeze(axis=0)
    tgt_react = torch.tensor(tgt_react).unsqueeze(axis=0)
    mask = torch.tensor(mask)

    # send data to device
    src_RNA = src_RNA.type(torch.float32).to(device)
    tgt_react = tgt_react.type(torch.float32).to(device)
    mask = mask.type(torch.bool).to(device)
    
    # validate model
    with torch.no_grad():
      tgt_pred = model(src_RNA)

      pred_masked = torch.masked_select(tgt_pred.squeeze(), mask)
      tgt_masked = torch.masked_select(tgt_react, mask)

      print(pred_masked)
      print(tgt_masked)
      loss = loss_fn(pred_masked, tgt_masked)

      print(loss.item())



