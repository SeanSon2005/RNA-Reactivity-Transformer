import numpy as np
from matplotlib.pylab import plt
import os
from discord import SyncWebhook
from datetime import datetime

class Plotter():
    def __init__(self, epochs):
        self.epochs = epochs

        # define save directory
        path = "runs/run_"+str(len(os.listdir('runs')))+"/"
        os.mkdir(path)
        self.save_dir = path

    def save_loss(self, train_loss_points, val_loss_points):
        
        # Plot and label the training and validation loss values
        epochs_range = range(1, self.epochs+1)
        plt.plot(epochs_range, train_loss_points, label='Training Loss')
        plt.plot(epochs_range, val_loss_points, label='Validation Loss')
        
        # Add in a title and axes labels
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        
        # Set the tick locations
        plt.xticks(np.arange(0, self.epochs+1, 2))

        plt.legend(loc='best')
        plt.savefig(self.save_dir + 'loss.png')

class WebHook():
    def __init__(self):
        self.webhook = SyncWebhook.from_url("https://discord.com/api/webhooks/1171941653389524992/lKrGo5Ilxpyj8OngYrjRz5PBhpzNCzYOjLw0s6zVxFBUivXpAw43rpOFZ1xmg606AIc0")

    def anounceEpoch(self, epoch, train_loss, val_loss):
        message = ("Epoch " + str(epoch) + " finished with "
                    + str(round(train_loss,4)) + " training loss and " + 
                    str(round(val_loss,4)) + " validation loss.")
        self.webhook.send(message)

    def sendMessage(self, message, include_date=False):
        if include_date:
            now = datetime.now()
            message = message + "\t" + now.strftime("%d/%m/%Y %H:%M:%S")
        self.webhook.send(message)