import os
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

from model_data import Network, load_model, validation
from utility import activate_device, print_start_stats, print_train_stats, print_final_stats, check_savedir
from data_handling import save_checkpoint, load_datasets
from get_args import get_train_args

import math

# EXAMPLE COMMAND
# python train.py -a densenet169 -s saves -e 10 -hs 1000 500 -lr 0.001 -pe 15 -v 1 --gpu

def main():
    in_args = get_train_args()
    active_device = activate_device(in_args.gpu)

    if not check_savedir(in_args.save_dir):
        return

    dataloaders, class_to_idx = load_datasets(in_args.mean,
                                              in_args.std,
                                              in_args.crop_size,
                                              in_args.rescale_size,
                                              in_args.batch_size,
                                              in_args.rand_rot,
                                              in_args.data_dir,
                                              in_args.loaders
                                             )

    model = load_model(in_args.arch, in_args.hidden_sizes, in_args.drop_rate)
    model.classifier.epochs = in_args.epochs
    model.class_to_idx = class_to_idx

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_args.learning_rate)

    model = train_classifier(model,
                             dataloaders=dataloaders,
                             optimizer=optimizer,
                             criterion=criterion,
                             device=active_device,
                             epochs=in_args.epochs,
                             print_every=in_args.print_every,
                             do_validation=in_args.do_validation,
                             test_run=in_args.testrun
                            )

    save_name = "checkpoint_" + model.name + time.strftime("_%Y_%m_%d_%Hh%Mm%Ss", time.gmtime()) + ".pth"
    if os.path.isdir(in_args.save_dir):
        save_name = os.path.join(in_args.save_dir, save_name)

    print("Saving to: ", save_name)
    save_checkpoint(model, save_name)

def train_classifier(model,
                     dataloaders,
                     optimizer,
                     criterion,
                     device='cuda',
                     epochs=3,
                     print_every=30,
                     test_run=None,
                     do_validation=None
                    ):
    '''Trains a model using a passed in classifier. The model, data, classifier network, and
       loss-function must be provided.
       All other parameters have default values but can be provided by the user for flexibility.
       Use 'test_run=X' where X is the number of batches to run through before quitting, for a quick test.
       Use 'do_validation=X' where X is a multiplier of the print_every variable, high numbers are less validation runs.
    '''

    batch_quantity = len(dataloaders['train'])
    lr = [p['lr'] for p in optimizer.param_groups][0]

    print_start_stats(model.name.upper(),
                      device.upper(),
                      epochs,
                      batch_quantity,
                      dataloaders['train'].batch_size,
                      lr,
                      model.classifier)

    model.to(device)
    model.train()
    accum_loss = 0
    training_start = batch_start = time.time()

    for e in range(epochs):

        for batch_num, (inputs, labels) in enumerate(dataloaders['train']):
            inputs, labels = inputs.to(device), labels.to(device)

            #do the training
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # Enable the clip_grad_norm_ line below if NAN values pop up and try again
            #nn.utils.clip_grad_norm_(model.parameters(), .01)
            optimizer.step()

            accum_loss += loss.item()

            if math.isnan(loss.item()):
                print("NAN error: {}".format(loss.item()))
                return

            # Stats reporting
            if (batch_num+1) % print_every == 0:
                batch_time = (time.time() - batch_start)/print_every
                print("Accum Loss: {}".format(accum_loss))
                avg_loss = accum_loss/print_every
                report = ''

                # Validation code
                if (do_validation != None) and ((batch_num+1) % (print_every * do_validation) == 0):
                    report = validation(model, dataloaders['valid'], criterion, device)
                    model.train()

                print_train_stats(e+1, epochs, batch_num+1, batch_quantity, batch_time, avg_loss, report, training_start)
                batch_start = time.time()


                accum_loss = 0

            if (test_run is not None) and (batch_num==test_run):
                break

            #batch_start = time.time()

    print_final_stats(model, device, epochs, lr, dataloaders, criterion, training_start)

    model.optimizer_state = optimizer.state_dict()
    model.classifier.epochs = epochs
    model.lr = lr

    return model



if __name__ == "__main__":
    main()
