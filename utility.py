import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import DataLoader

from PIL import Image

from model_data import validation

#################
# PRINT FUNCTIONS
#################
def print_start_stats(name, device, epochs, batches, batchsize, lr, classifier):
    sep = "#"*50
    print("\n{}".format(sep),
          "\nStarting Training with: {} on device: {}".format(name, device),
          "\n{} Epochs, {} batches, {} batchsize, {} learning rate\n".format(epochs, batches, batchsize, lr),
          "\n...and the Network looks like:\n{}".format(classifier),
          "\n{}".format(sep)
         )
         
    return



def print_train_stats(e, epochs, batch, batch_quantity, batch_time, avg_loss, report, training_start):
    print("Epoch: {}/{} ".format(e, epochs),
          "Batch {}/{}".format(batch,batch_quantity),
          "Time per batch: {:.3f} seconds".format(batch_time),
          "\n..Training loss: {:.4f}".format(avg_loss),
          report,
          "\n",
          "{0} (runtime: {1:.1f}sec) {0}".format("#"*25, time.time()-training_start)
         )

    return



def print_final_stats(model, device, epochs, lr, dataloaders, criterion, training_start):
    report = validation(model, dataloaders['valid'], criterion, device, do_time=False)
    sep = "#"*50
    print("\n\n{0}\n{0}".format(sep),
          "\nTrained with params:\nModel: {} Epochs: {} Batch Size: {} Learning Rate: {}".format(model.name, epochs, dataloaders['train'].batch_size, lr),
          "\nFINAL {}".format(report),
          "\n{0} \nTOTAL RUNTIME: {1:.2f} seconds\n{0}".format(sep, time.time()-training_start))

    return



def print_prediction_results(df, flower_label):
    print("{0}\n{1}\n{0}".format("#"*40, df))

    top_flower = df.index[0]
    if top_flower.lower() == flower_label.lower():
        match_status = "MATCHES"
    else:
        match_status = "DOESN'T match. Better luck next time"
    print(" {} was predicted by the model, and...\n".format(top_flower.upper()),
          "{} is the actual flower name.\n".format(flower_label.upper()),
          "This {}!".format(match_status))

    return

###################
# UTILITY FUNCTIONS
###################

def activate_device(do_gpu):
    if do_gpu and torch.cuda.is_available():
        active_device = 'cuda'
    else:
        active_device = 'cpu'

    return active_device



def check_savedir(save_dir):
    if save_dir == '':
        return True
    if not os.path.isdir(save_dir):
        mkdir = input("Directory {} does not exist, create it? (y/n)".format(save_dir))
        if mkdir.lower() == "y":
            try:
                os.mkdir(save_dir)
            except:
                print("Something went wrong with directory creation. Aborting training.")
                return False
        else:
            print("Aborting training. Please use a different save directory or create one.")
            return False

    return True



################
# IMAGE HANDLING
################

def process_image(path,
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225],
                  crop_size=224,
                  max_size=256
                 ):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    with Image.open(path) as img:
        width, height = img.size
        if height > width:
            height = max_size * (height/width)
            width = max_size
        else:
            width = max_size * (width/height)
            height = max_size
        img = img.resize((int(width),int(height)))

        x = width/2-(crop_size/2)
        y = height/2-(crop_size/2)
        r = width/2+(crop_size/2)
        t = height/2+(crop_size/2)
        img = img.crop((x,y,r,t))

        img = np.array(img)/255
        img = (img - mean) / std
        img = img.transpose(2,0,1)

    return img



def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax



def show_prediction_results(df, flower_label, flower_path):
    '''Takes the path to a flower, the path to the desired pre-trained model, and returns predictions.
        Use "verbose=True" to report additional information from the prediction model.
    '''

    fig, (flower_image, data_graph) = plt.subplots(figsize=(4,8), nrows=2)

    flower_image.set_xticks([])
    flower_image.set_yticks([])
    flower_image.set_title(flower_label)
    imshow(process_image(flower_path),ax=flower_image)

    df = df.sort_values(by='probability')
    df['probability'].plot(kind='barh', ax=data_graph)
    num_ticks = 5
    tick = df['probability'].max()/num_ticks
    xticks = np.arange(0,tick*(num_ticks+1),tick)
    plt.xticks(xticks, rotation = 45)

    plt.show()

    return
